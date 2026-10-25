from typing import Sequence

import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "storage_days",
    "mean_temperature",
    "median_temperature",
    "max_temperature",
    "min_temperature",
    "end_temperature",
    "days_below_0",
    "days_0_4",
    "days_4_8",
    "days_above_8",
    "days_above_equal_15",
    "auc_4",
    "auc_8",
    "auc_15",
    "temperature_range",
    "temperature_std",
    "max_abs_delta",
    "mean_abs_delta",
    "total_abs_delta",
    "warming_count",
    "cooling_count",
    "cross_0_count",
    "cross_8_count",
    "turn_count",
]


def extract_path_features(path: Sequence[float]) -> dict:
    values = np.asarray(path, dtype=float)
    n = int(values.size)
    if n == 0:
        raise ValueError("Temperature path cannot be empty.")

    diffs = np.diff(values)
    abs_diffs = np.abs(diffs)
    nonzero_diffs = diffs[diffs != 0]

    if nonzero_diffs.size >= 2:
        signs = np.sign(nonzero_diffs)
        turn_count = int(np.sum(signs[1:] != signs[:-1]))
    else:
        turn_count = 0

    return {
        "storage_days": n,
        "mean_temperature": float(np.mean(values)),
        "median_temperature": float(np.median(values)),
        "max_temperature": float(np.max(values)),
        "min_temperature": float(np.min(values)),
        "end_temperature": float(values[-1]),
        "days_below_0": int(np.sum(values < 0)),
        "days_0_4": int(np.sum((values >= 0) & (values <= 4))),
        "days_4_8": int(np.sum((values > 4) & (values <= 8))),
        "days_above_8": int(np.sum(values > 8)),
        "days_above_equal_15": int(np.sum(values >= 15)),
        "auc_4": float(np.sum(np.maximum(values - 4, 0))),
        "auc_8": float(np.sum(np.maximum(values - 8, 0))),
        "auc_15": float(np.sum(np.maximum(values - 15, 0))),
        "temperature_range": float(np.max(values) - np.min(values)),
        "temperature_std": float(np.std(values, ddof=0)),
        "max_abs_delta": float(np.max(abs_diffs)) if abs_diffs.size else 0.0,
        "mean_abs_delta": float(np.mean(abs_diffs)) if abs_diffs.size else 0.0,
        "total_abs_delta": float(np.sum(abs_diffs)) if abs_diffs.size else 0.0,
        "warming_count": int(np.sum(diffs > 0)),
        "cooling_count": int(np.sum(diffs < 0)),
        "cross_0_count": int(np.sum((values[:-1] < 0) != (values[1:] < 0))) if n > 1 else 0,
        "cross_8_count": int(np.sum((values[:-1] <= 8) != (values[1:] <= 8))) if n > 1 else 0,
        "turn_count": turn_count,
    }


def records_to_feature_frame(records: Sequence[dict]) -> pd.DataFrame:
    rows = []
    for idx, record in enumerate(records):
        row = {
            "record_id": record.get("record_id", idx),
            "method": record.get("method", ""),
            "source_pool": record.get("source_pool", ""),
            "subtype": record.get("subtype", ""),
            "stratum": record.get("stratum", ""),
            "risk_type": record.get("risk_type", ""),
            "path_key": "|".join(str(v) for v in record["path"]),
        }
        row.update(extract_path_features(record["path"]))
        rows.append(row)
    return pd.DataFrame(rows)


def feature_matrix(records: Sequence[dict]) -> np.ndarray:
    frame = records_to_feature_frame(records)
    return frame[FEATURE_NAMES].to_numpy(dtype=float)
