from typing import List

import numpy as np

from config.experiment_config import ExperimentConfig
from src.utils.random_utils import (
    q_temperature,
    sample_integer_temperature,
    sample_storage_days,
)


def _risk_record(path, risk_type: str) -> dict:
    return {
        "path": tuple(path),
        "source_pool": "risk",
        "subtype": "risk_exposure",
        "risk_type": risk_type,
    }


def _cold_boundary(rng: np.random.Generator, config: ExperimentConfig) -> dict:
    n = sample_storage_days(rng, config.temperature.min_days, config.temperature.max_days)
    values = np.arange(-2, 7)
    weights = np.array([1, 1, 2, 3, 4, 5, 5, 4, 2], dtype=float)
    weights = weights / weights.sum()
    path = tuple(int(v) for v in rng.choice(values, size=n, p=weights))
    return _risk_record(path, "cold_boundary")


def _transition_4_8(rng: np.random.Generator, config: ExperimentConfig) -> dict:
    n = sample_storage_days(rng, config.temperature.min_days, config.temperature.max_days)
    path = tuple(q_temperature(v) for v in rng.normal(6, 3, size=n))
    return _risk_record(path, "transition_4_8")


def _above_8_risk(rng: np.random.Generator, config: ExperimentConfig) -> dict:
    n = sample_storage_days(rng, config.temperature.min_days, config.temperature.max_days)
    values = [q_temperature(v) for v in rng.normal(11, 4, size=n)]
    k = max(1, n // 3)
    positions = rng.choice(n, size=k, replace=False)
    for pos in positions:
        values[int(pos)] = sample_integer_temperature(rng, 9, 15)
    return _risk_record(tuple(values), "above_8_risk")


def _above_15_abuse(rng: np.random.Generator, config: ExperimentConfig) -> dict:
    n = sample_storage_days(rng, config.temperature.min_days, config.temperature.max_days)
    values = [sample_integer_temperature(rng, -3, 12) for _ in range(n)]
    k = max(1, int(np.ceil(n * rng.uniform(0.25, 0.70))))
    positions = rng.choice(n, size=k, replace=False)
    for pos in positions:
        values[int(pos)] = sample_integer_temperature(rng, 15, 18)
    return _risk_record(tuple(values), "above_15_abuse")


def _freeze_thaw(rng: np.random.Generator, config: ExperimentConfig) -> dict:
    n = max(2, sample_storage_days(rng, config.temperature.min_days, config.temperature.max_days))
    values = []
    for idx in range(n):
        if idx % 2 == 0:
            values.append(sample_integer_temperature(rng, -18, -1))
        else:
            values.append(sample_integer_temperature(rng, 0, 12))
    if rng.random() < 0.5:
        values = list(reversed(values))
    return _risk_record(tuple(values), "freeze_thaw")


RISK_GENERATORS = {
    "cold_boundary": _cold_boundary,
    "transition_4_8": _transition_4_8,
    "above_8_risk": _above_8_risk,
    "above_15_abuse": _above_15_abuse,
    "freeze_thaw": _freeze_thaw,
}


def generate_risk_pool(config: ExperimentConfig, rng: np.random.Generator, total_count: int = None) -> List[dict]:
    target = total_count or config.source_pool.risk_pool_size
    records: List[dict] = []
    scenario_names = list(config.source_pool.risk_scenarios)
    base_count, remainder = divmod(target, len(scenario_names))
    seen_paths = set()
    for type_index, risk_type in enumerate(scenario_names):
        type_target = base_count + (1 if type_index < remainder else 0)
        type_records: List[dict] = []
        attempts = 0
        while len(type_records) < type_target:
            attempts += 1
            record = RISK_GENERATORS[risk_type](rng, config)
            path_key = tuple(record["path"])
            if path_key not in seen_paths:
                seen_paths.add(path_key)
                type_records.append(record)
            if attempts > max(1, type_target) * 100:
                raise RuntimeError(f"Too many attempts while generating risk type: {risk_type}")
        records.extend(type_records)
    return records
