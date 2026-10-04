from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from config.experiment_config import METRIC_DIRECTIONS
from src.evaluation.evaluation_sets import KEY_GROUP_ORDER, StorageTemperatureRecordSet


def _min_distances(from_z: np.ndarray, to_z: np.ndarray) -> np.ndarray:
    if from_z.shape[0] == 0 or to_z.shape[0] == 0:
        raise ValueError("距离计算的两个路径集合都不能为空。")
    return np.min(cdist(from_z, to_z), axis=1)


def path_coverage_error(
    record_z: np.ndarray, selected_z: np.ndarray
) -> Tuple[float, np.ndarray]:
    """PCE：每条评价参考记录到输出路径集合的最近距离均值，越小越好。"""

    distances = _min_distances(record_z, selected_z)
    return float(np.mean(distances)), distances


def key_zone_coverage_error(
    record_to_selected: np.ndarray,
    record_set: StorageTemperatureRecordSet,
) -> Tuple[float, pd.DataFrame]:
    """KCE：五个允许重叠关键温区的组内均值再等权平均，越小越好。"""

    rows = []
    group_means = []
    for group in KEY_GROUP_ORDER:
        indices = np.asarray(record_set.key_group_indices[group], dtype=int)
        if indices.size == 0:
            raise ValueError(f"储藏温度记录中关键温区集合为空：{group}")
        values = record_to_selected[indices]
        mean_distance = float(np.mean(values))
        group_means.append(mean_distance)
        rows.append(
            {
                "key_group": group,
                "path_count": int(indices.size),
                "mean_nearest_distance": mean_distance,
            }
        )
    return float(np.mean(group_means)), pd.DataFrame(rows)


def path_separation(selected_z: np.ndarray) -> float:
    """PS：输出集合内最近正距离的5%分位数，越大越好。"""

    if selected_z.shape[0] <= 1:
        return float("nan")
    distances = cdist(selected_z, selected_z)
    distances[distances <= 0] = np.inf
    nearest = np.min(distances, axis=1)
    nearest = nearest[np.isfinite(nearest)]
    if nearest.size == 0:
        return 0.0
    return float(np.quantile(nearest, 0.05))


def log_feature_space_volume(selected_z: np.ndarray, regularization: float) -> float:
    """LFV：正则化协方差矩阵log-det除以评价PCA维数，越大越好。"""

    if selected_z.shape[0] <= 1:
        return float("nan")
    covariance = np.cov(selected_z, rowvar=False)
    if covariance.ndim == 0:
        covariance = np.asarray([[float(covariance)]])
    q = covariance.shape[0]
    sign, logdet = np.linalg.slogdet(covariance + regularization * np.eye(q))
    if sign <= 0:
        return float("-inf")
    return float(logdet / q)


def evaluate_selected_design(
    selected_z: np.ndarray,
    record_z: np.ndarray,
    record_set: StorageTemperatureRecordSet,
    regularization: float,
) -> Tuple[Dict[str, float], Dict[str, pd.DataFrame]]:
    coverage_error, record_to_selected = path_coverage_error(
        record_z, selected_z
    )
    key_error, key_detail = key_zone_coverage_error(
        record_to_selected, record_set
    )
    metrics = {
        "路径覆盖误差": coverage_error,
        "关键温区覆盖误差": key_error,
        "路径间隔": path_separation(selected_z),
        "对数特征体积": log_feature_space_volume(selected_z, regularization),
    }
    tables = {
        "关键温区覆盖明细": key_detail,
    }
    return metrics, tables


def metrics_to_frame(metrics_by_method: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for method, metrics in metrics_by_method.items():
        row = {"method": method}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def rank_metrics(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for metric, direction in METRIC_DIRECTIONS.items():
        ascending = direction == "lower"
        ranked = metrics_frame[["method", metric]].copy()
        ranked["rank"] = ranked[metric].rank(method="min", ascending=ascending)
        ranked["metric"] = metric
        ranked["direction"] = "越小越好" if direction == "lower" else "越大越好"
        rows.extend(
            ranked[["method", "metric", "direction", metric, "rank"]]
            .rename(columns={metric: "value"})
            .to_dict("records")
        )
    return pd.DataFrame(rows)
