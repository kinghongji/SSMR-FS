from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from scipy.spatial.distance import cdist

from config.experiment_config import ExperimentConfig
from src.features.preprocessing import FeatureTransformer
from src.generation.risk_pool import generate_risk_pool
from src.generation.scenario_pool import generate_scenario_pool
from src.generation.stratified_pool import generate_stratified_pool
from src.utils.random_utils import deduplicate_records


@dataclass
class SSMRPoolBundle:
    scenario_pool: List[dict]
    stratified_pool: List[dict]
    risk_pool: List[dict]
    candidate_pool: List[dict]


@dataclass
class SSMRSelectionResult:
    selected_records: List[dict]
    source_counts: Dict[str, int]
    selection_log: List[dict]
    pool_bundle: SSMRPoolBundle
    transformer: FeatureTransformer
    candidate_group_summary: List[dict]
    selection_summary: Dict[str, object]

    def prefix(self, n_samples: int) -> "SSMRSelectionResult":
        """返回同一次能量距离融合选择序列的前N条。"""

        if n_samples < 1 or n_samples > len(self.selected_records):
            raise ValueError("SSMR-FS前缀数量超出已完成的选择范围。")
        selected_records = [dict(record) for record in self.selected_records[:n_samples]]
        selection_log = [dict(row) for row in self.selection_log[:n_samples]]
        summary = dict(self.selection_summary)
        summary["selected_count"] = n_samples
        summary["final_energy_distance"] = selection_log[-1][
            "energy_distance_after_selection"
        ]
        summary["mean_sequential_energy_score"] = float(
            np.mean([row["sequential_energy_score"] for row in selection_log])
        )
        return SSMRSelectionResult(
            selected_records=selected_records,
            source_counts=_count_sources(selected_records),
            selection_log=selection_log,
            pool_bundle=self.pool_bundle,
            transformer=self.transformer,
            candidate_group_summary=[
                dict(row) for row in self.candidate_group_summary
            ],
            selection_summary=summary,
        )


def build_ssmr_candidate_pool(
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> SSMRPoolBundle:
    """生成场景、分层探索和等量风险候选路径，并按完整路径去重。"""

    scenario_pool = generate_scenario_pool(config, rng)
    stratified_pool = generate_stratified_pool(config, rng)
    risk_pool = generate_risk_pool(config, rng)
    candidate_pool = deduplicate_records(scenario_pool + stratified_pool + risk_pool)
    return SSMRPoolBundle(
        scenario_pool=scenario_pool,
        stratified_pool=stratified_pool,
        risk_pool=risk_pool,
        candidate_pool=candidate_pool,
    )


def _source_key(record: dict) -> str:
    source = record.get("source_pool", "")
    if source in {"scenario", "stratified", "risk"}:
        return source
    return str(source)


def _count_sources(records: Sequence[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {"scenario": 0, "stratified": 0, "risk": 0}
    for record in records:
        key = _source_key(record)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _candidate_group_key(record: dict) -> str:
    source = _source_key(record)
    if source == "scenario":
        return str(record.get("subtype", "unspecified_scenario"))
    if source == "stratified":
        return str(record.get("stratum", "unspecified_stratum"))
    if source == "risk":
        return str(record.get("risk_type", "unspecified_risk"))
    return "unspecified"


def _candidate_group_summary(pool_bundle: SSMRPoolBundle) -> List[dict]:
    counts: Dict[tuple, int] = {}
    for record in pool_bundle.candidate_pool:
        key = (_source_key(record), _candidate_group_key(record))
        counts[key] = counts.get(key, 0) + 1
    return [
        {
            "candidate_source": source,
            "candidate_group": group,
            "record_count": count,
            "selection_role": "能量距离目标空间；不属于外部储藏温度记录",
        }
        for (source, group), count in counts.items()
    ]


def _mean_distance_to_candidate_space(
    candidate_z: np.ndarray,
    block_size: int,
) -> np.ndarray:
    """计算每条候选到完整候选空间的平均欧氏距离。"""

    if block_size < 1:
        raise ValueError("distance_block_size必须大于0。")
    candidate_count = candidate_z.shape[0]
    mean_distances = np.empty(candidate_count, dtype=float)
    for start in range(0, candidate_count, block_size):
        end = min(start + block_size, candidate_count)
        distances = cdist(candidate_z[start:end], candidate_z)
        mean_distances[start:end] = np.mean(distances, axis=1)
    return mean_distances


def select_ssmr_fs(
    config: ExperimentConfig,
    rng: np.random.Generator,
    n_samples: int,
    pool_bundle: SSMRPoolBundle = None,
) -> SSMRSelectionResult:
    """用候选约束的序贯能量距离准则选择N条温度路径。

    三类候选路径合并去重后形成完整候选空间C。选择集合S_t在第t步加入
    候选c时，最小化标准能量距离中与c有关的部分：

        mean_{e in C} d(c, e) - (1 / t) * sum_{s in S_{t-1}} d(c, s)

    第一项保持对完整候选空间的代表性，第二项减少已选路径之间的重复。
    两项系数直接来自能量距离定义。外部储藏温度记录不进入本模块。
    """

    if pool_bundle is None:
        pool_bundle = build_ssmr_candidate_pool(config, rng)
    if n_samples < 1:
        raise ValueError("SSMR-FS输出数量必须大于0。")
    if len(pool_bundle.candidate_pool) < n_samples:
        raise RuntimeError(
            f"SSMR-FS候选池只有{len(pool_bundle.candidate_pool)}条唯一路径，"
            f"少于请求的{n_samples}条。"
        )

    # 选择PCA仅由完整候选空间拟合；外部储藏温度记录完全不进入本模块。
    transformer = FeatureTransformer(config.features.pca_explained_variance)
    transformer.fit(pool_bundle.candidate_pool)
    candidate_z = transformer.transform(pool_bundle.candidate_pool)

    candidate_count = len(pool_bundle.candidate_pool)
    mean_reference_distance = _mean_distance_to_candidate_space(
        candidate_z,
        config.selection.distance_block_size,
    )
    reference_pair_mean_distance = float(np.mean(mean_reference_distance))

    selected_indices: List[int] = []
    selected_mask = np.zeros(candidate_count, dtype=bool)
    distance_sum_to_selected = np.zeros(candidate_count, dtype=float)
    selected_reference_cross_sum = 0.0
    selected_pair_distance_sum = 0.0
    selection_log: List[dict] = []

    for step in range(1, n_samples + 1):
        # 该分数与加入候选后的完整能量距离目标严格同序；除以常数后的
        # 形式更简洁，并避免为每条候选重复计算与选择无关的历史常数。
        sequential_score = (
            mean_reference_distance - distance_sum_to_selected / float(step)
        )
        sequential_score[selected_mask] = np.inf
        next_index = int(np.argmin(sequential_score))
        if not np.isfinite(sequential_score[next_index]):
            raise RuntimeError("能量距离选择没有找到可用候选路径。")

        distance_to_previous = float(distance_sum_to_selected[next_index])
        selected_reference_cross_sum += (
            mean_reference_distance[next_index] * candidate_count
        )
        selected_pair_distance_sum += 2.0 * distance_to_previous

        energy_distance = (
            2.0
            * selected_reference_cross_sum
            / float(step * candidate_count)
            - selected_pair_distance_sum / float(step * step)
            - reference_pair_mean_distance
        )

        selected_indices.append(next_index)
        selected_mask[next_index] = True
        selection_log.append(
            {
                "step": step,
                "selected_index": next_index,
                "source_pool": _source_key(
                    pool_bundle.candidate_pool[next_index]
                ),
                "candidate_group": _candidate_group_key(
                    pool_bundle.candidate_pool[next_index]
                ),
                "mean_distance_to_candidate_space": float(
                    mean_reference_distance[next_index]
                ),
                "distance_sum_to_previously_selected": distance_to_previous,
                "sequential_energy_score": float(sequential_score[next_index]),
                "energy_distance_after_selection": float(max(energy_distance, 0.0)),
                "selection_rule": "minimum_sequential_energy_distance",
            }
        )

        new_distances = cdist(candidate_z, candidate_z[[next_index]]).ravel()
        distance_sum_to_selected += new_distances

    selected_records: List[dict] = []
    for order, index in enumerate(selected_indices, start=1):
        record = dict(pool_bundle.candidate_pool[index])
        record["method"] = "SSMR-FS"
        record["selection_order"] = order
        selected_records.append(record)

    selection_summary: Dict[str, object] = {
        "selection_objective": config.selection.objective,
        "mathematical_target": (
            "greedy minimization of energy distance between selected paths "
            "and the merged unique candidate space"
        ),
        "candidate_score": (
            "mean_distance_to_candidate_space - "
            "distance_sum_to_previous_selected / current_selected_size"
        ),
        "first_path_rule": "选择到完整候选空间平均距离最小的候选路径",
        "subsequent_path_rule": "逐步选择使加入后能量距离最小的未选候选路径",
        "tie_breaking_rule": "分数相同时选择原始索引较小者",
        "reference_distribution": "三类候选路径合并去重后的完整候选空间",
        "uses_metric_feedback": False,
        "uses_storage_temperature_records": False,
        "candidate_path_count": candidate_count,
        "selected_count": n_samples,
        "final_energy_distance": selection_log[-1][
            "energy_distance_after_selection"
        ],
        "mean_sequential_energy_score": float(
            np.mean([row["sequential_energy_score"] for row in selection_log])
        ),
        "selection_pca_fitted_on": "三类路径合并去重后的完整候选空间",
    }
    return SSMRSelectionResult(
        selected_records=selected_records,
        source_counts=_count_sources(selected_records),
        selection_log=selection_log,
        pool_bundle=pool_bundle,
        transformer=transformer,
        candidate_group_summary=_candidate_group_summary(pool_bundle),
        selection_summary=selection_summary,
    )
