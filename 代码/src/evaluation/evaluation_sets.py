from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from config.experiment_config import ExperimentConfig


KEY_GROUP_ORDER = ("0-4℃", "4-8℃", ">8℃", ">=15℃", "跨0℃")


@dataclass
class StorageTemperatureRecordSet:
    """自动识别条数的储藏温度记录及五个允许重叠的关键温区集合。"""

    records: List[dict]
    key_group_indices: Dict[str, List[int]]
    audit_summary: Dict[str, object]

    @property
    def label(self) -> str:
        """用于结果表、图和日志的动态数据集名称。"""

        return f"{len(self.records)}条储藏温度记录"


def _crosses_zero(values: np.ndarray) -> bool:
    if values.size <= 1:
        return False
    return bool(np.any((values[:-1] < 0) != (values[1:] < 0)))


def _key_groups(values: np.ndarray) -> List[str]:
    groups: List[str] = []
    if np.any((values >= 0) & (values <= 4)):
        groups.append("0-4℃")
    if np.any((values > 4) & (values <= 8)):
        groups.append("4-8℃")
    if np.any(values > 8):
        groups.append(">8℃")
    if np.any(values >= 15):
        groups.append(">=15℃")
    if _crosses_zero(values):
        groups.append("跨0℃")
    return groups


def load_storage_temperature_records(
    config: ExperimentConfig,
) -> StorageTemperatureRecordSet:
    """从原始Excel只读载入任意条数的储藏温度记录。

    样本名称仅保留为唯一标识，不解析为分组、来源或方法。
    空白单元格必须只出现在路径末尾；内部空白会直接报错，防止把缺失值
    错当作0℃或静默缩短路径。记录值不会截断、替换或整数化。
    """

    settings = config.storage_temperature_records
    if not settings.file_path.exists():
        raise FileNotFoundError(f"未找到储藏温度记录文件：{settings.file_path}")

    frame = pd.read_excel(settings.file_path, sheet_name=settings.sheet_name)
    required = [settings.id_column, *settings.day_columns]
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"储藏温度记录文件缺少列：{missing_columns}")

    records: List[dict] = []
    key_group_indices: Dict[str, List[int]] = {name: [] for name in KEY_GROUP_ORDER}
    for row_number, (_, row) in enumerate(frame.iterrows(), start=2):
        sample_id = row[settings.id_column]
        raw = row[list(settings.day_columns)]
        if pd.isna(sample_id) and raw.isna().all():
            continue
        if pd.isna(sample_id):
            raise ValueError(f"Excel第{row_number}行缺少样本名称。")

        numeric = pd.to_numeric(raw, errors="coerce")
        invalid_nonblank = raw.notna() & numeric.isna()
        if invalid_nonblank.any():
            columns = list(raw.index[invalid_nonblank])
            raise ValueError(f"Excel第{row_number}行含非数值温度：{columns}")
        valid_positions = np.flatnonzero(numeric.notna().to_numpy())
        if valid_positions.size == 0:
            raise ValueError(f"Excel第{row_number}行没有有效温度。")
        last_valid = int(valid_positions[-1])
        if numeric.iloc[: last_valid + 1].isna().any():
            raise ValueError(f"Excel第{row_number}行的有效路径中存在内部空白。")

        values = numeric.iloc[: last_valid + 1].to_numpy(dtype=float)
        record_index = len(records)
        groups = _key_groups(values)
        records.append(
            {
                "record_id": str(sample_id),
                "path": tuple(float(value) for value in values),
                "method": "评价参考记录",
                "source_pool": "storage_temperature_records",
                "subtype": "storage_temperature_record",
                "key_groups": tuple(groups),
            }
        )
        for group in groups:
            key_group_indices[group].append(record_index)

    if not records:
        raise ValueError("储藏温度记录文件中没有可用温度路径。")
    if len(records) < 2:
        raise ValueError("统一评价PCA至少需要2条储藏温度记录。")
    all_values = np.concatenate([np.asarray(record["path"], dtype=float) for record in records])
    sample_ids = [record["record_id"] for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("储藏温度记录中存在重复样本名称。")
    empty_key_groups = [
        name for name in KEY_GROUP_ORDER if not key_group_indices[name]
    ]
    if empty_key_groups:
        raise ValueError(
            "储藏温度记录缺少以下关键温区，无法按既定定义计算"
            f"关键温区覆盖误差：{empty_key_groups}"
        )
    in_range_flags = [
        bool(
            np.all(
                (np.asarray(record["path"], dtype=float) >= config.temperature.temp_min)
                & (np.asarray(record["path"], dtype=float) <= config.temperature.temp_max)
            )
        )
        for record in records
    ]
    outside_point_mask = (all_values < config.temperature.temp_min) | (
        all_values > config.temperature.temp_max
    )
    length_counts = pd.Series([len(record["path"]) for record in records]).value_counts().sort_index()

    summary: Dict[str, object] = {
        "source_file": settings.file_path.name,
        "sheet_name": settings.sheet_name,
        "record_label": f"{len(records)}条储藏温度记录",
        "evaluation_role": "唯一评价参考记录集；不参与候选生成、选择PCA或序贯能量选择",
        "sample_id_rule": "样本名称仅为唯一标识，不分组、不比较、不解释",
        "path_count": len(records),
        "temperature_point_count": int(all_values.size),
        "path_length_counts": {str(int(k)): int(v) for k, v in length_counts.items()},
        "observed_min_temperature": float(np.min(all_values)),
        "observed_max_temperature": float(np.max(all_values)),
        "nominal_design_min_temperature": config.temperature.temp_min,
        "nominal_design_max_temperature": config.temperature.temp_max,
        "in_nominal_range_path_count": int(sum(in_range_flags)),
        "outside_nominal_range_path_count": int(len(in_range_flags) - sum(in_range_flags)),
        "outside_nominal_range_point_count": int(np.sum(outside_point_mask)),
        "outside_nominal_range_point_percentage": float(np.mean(outside_point_mask) * 100),
        "below_nominal_minimum_path_count": int(
            sum(
                np.any(np.asarray(record["path"], dtype=float) < config.temperature.temp_min)
                for record in records
            )
        ),
        "above_nominal_maximum_path_count": int(
            sum(
                np.any(np.asarray(record["path"], dtype=float) > config.temperature.temp_max)
                for record in records
            )
        ),
        "key_group_counts": {name: len(key_group_indices[name]) for name in KEY_GROUP_ORDER},
    }

    return StorageTemperatureRecordSet(
        records=records,
        key_group_indices=key_group_indices,
        audit_summary=summary,
    )
