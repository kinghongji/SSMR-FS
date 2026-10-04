from typing import Callable, Dict, List, Tuple

import numpy as np

from config.experiment_config import ExperimentConfig
from src.utils.random_utils import (
    deduplicate_records,
    q_temperature,
    sample_integer_temperature,
    sample_storage_days,
)


SCENARIO_NAMES = (
    "stable_frozen",
    "stable_chilled",
    "slow_warming",
    "fast_warming",
    "recooling",
    "high_frequency_fluctuation",
    "low_frequency_fluctuation",
    "late_warming",
    "cross_zero",
    "temperature_abuse",
)


def _record(path: Tuple[int, ...], subtype: str) -> dict:
    return {
        "path": tuple(path),
        "source_pool": "scenario",
        "subtype": subtype,
    }


def _stable_frozen(rng: np.random.Generator, config: ExperimentConfig) -> dict:
    n = sample_storage_days(rng, config.temperature.min_days, config.temperature.max_days)
    c = sample_integer_temperature(rng, -18, -1)
    path = tuple(q_temperature(c + rng.integers(-1, 2)) for _ in range(n))
    return _record(path, "stable_frozen")


def _stable_chilled(rng: np.random.Generator, config: ExperimentConfig) -> dict:
    n = sample_storage_days(rng, config.temperature.min_days, config.temperature.max_days)
    c = sample_integer_temperature(rng, 0, 4)
    path = tuple(q_temperature(c + rng.integers(-1, 2)) for _ in range(n))
    return _record(path, "stable_chilled")


def _slow_warming(rng: np.random.Generator, config: ExperimentConfig) -> dict:
    n = max(3, sample_storage_days(rng, config.temperature.min_days, config.temperature.max_days))
    start = sample_integer_temperature(rng, -18, 10)
    end = sample_integer_temperature(rng, min(18, start + 4), 18)
    values = []
    for i in range(n):
        base = start + i / (n - 1) * (end - start)
        values.append(q_temperature(base + rng.normal(0, 0.7)))
    return _record(tuple(values), "slow_warming")


def _fast_warming(rng: np.random.Generator, config: ExperimentConfig) -> dict:
    n = max(2, sample_storage_days(rng, config.temperature.min_days, config.temperature.max_days))
    tau = int(rng.integers(1, n))
    low = sample_integer_temperature(rng, -18, 8)
    high = sample_integer_temperature(rng, min(18, low + 8), 18)
    values = []
    for i in range(n):
        center = low if i < tau else high
        values.append(q_temperature(center + rng.integers(-1, 2)))
    return _record(tuple(values), "fast_warming")


def _recooling(rng: np.random.Generator, config: ExperimentConfig) -> dict:
    n = max(3, sample_storage_days(rng, config.temperature.min_days, config.temperature.max_days))
    tau = int(rng.integers(1, n - 1))
    start = sample_integer_temperature(rng, -18, 8)
    peak = sample_integer_temperature(rng, min(18, start + 6), 18)
    end = sample_integer_temperature(rng, -18, max(-18, peak - 3))
    values = []
    for i in range(n):
        if i <= tau:
            base = start + i / tau * (peak - start)
        else:
            denom = max(1, n - tau - 1)
            base = peak + (i - tau) / denom * (end - peak)
        values.append(q_temperature(base))
    return _record(tuple(values), "recooling")


def _high_frequency_fluctuation(rng: np.random.Generator, config: ExperimentConfig) -> dict:
    n = max(2, sample_storage_days(rng, config.temperature.min_days, config.temperature.max_days))
    low = sample_integer_temperature(rng, -18, 8)
    high = sample_integer_temperature(rng, min(18, low + 8), 18)
    values = []
    for i in range(n):
        center = low if i % 2 == 0 else high
        values.append(q_temperature(center + rng.integers(-2, 3)))
    return _record(tuple(values), "high_frequency_fluctuation")


def _low_frequency_fluctuation(rng: np.random.Generator, config: ExperimentConfig) -> dict:
    n = max(3, sample_storage_days(rng, config.temperature.min_days, config.temperature.max_days))
    low = sample_integer_temperature(rng, -18, 8)
    high = sample_integer_temperature(rng, min(18, low + 8), 18)
    block = max(2, n // 3)
    values = []
    for i in range(n):
        center = low if (i // block) % 2 == 0 else high
        values.append(q_temperature(center + rng.integers(-1, 2)))
    return _record(tuple(values), "low_frequency_fluctuation")


def _late_warming(rng: np.random.Generator, config: ExperimentConfig) -> dict:
    n = max(2, sample_storage_days(rng, config.temperature.min_days, config.temperature.max_days))
    late_len = int(rng.integers(1, min(3, n) + 1))
    tau = max(1, n - late_len)
    low = sample_integer_temperature(rng, -18, 8)
    high = sample_integer_temperature(rng, min(18, low + 6), 18)
    tail = np.linspace(high - 2, high, n - tau)
    values = [q_temperature(low) for _ in range(tau)]
    values.extend(q_temperature(v) for v in tail)
    return _record(tuple(values), "late_warming")


def _cross_zero(rng: np.random.Generator, config: ExperimentConfig) -> dict:
    n = max(2, sample_storage_days(rng, config.temperature.min_days, config.temperature.max_days))
    tau = int(rng.integers(1, n))
    low = sample_integer_temperature(rng, -18, -1)
    high = sample_integer_temperature(rng, 1, 18)
    values = []
    for i in range(n):
        center = low if i < tau else high
        values.append(q_temperature(center + rng.integers(-1, 2)))
    if rng.random() < 0.5:
        values = list(reversed(values))
    return _record(tuple(values), "cross_zero")


def _temperature_abuse(rng: np.random.Generator, config: ExperimentConfig) -> dict:
    n = sample_storage_days(rng, config.temperature.min_days, config.temperature.max_days)
    values = [sample_integer_temperature(rng, -2, 8) for _ in range(n)]
    k = int(rng.integers(1, n + 1))
    abuse_positions = rng.choice(n, size=k, replace=False)
    for pos in abuse_positions:
        values[int(pos)] = sample_integer_temperature(rng, 15, 18)
    return _record(tuple(values), "temperature_abuse")


SCENARIO_GENERATORS: Dict[str, Callable[[np.random.Generator, ExperimentConfig], dict]] = {
    "stable_frozen": _stable_frozen,
    "stable_chilled": _stable_chilled,
    "slow_warming": _slow_warming,
    "fast_warming": _fast_warming,
    "recooling": _recooling,
    "high_frequency_fluctuation": _high_frequency_fluctuation,
    "low_frequency_fluctuation": _low_frequency_fluctuation,
    "late_warming": _late_warming,
    "cross_zero": _cross_zero,
    "temperature_abuse": _temperature_abuse,
}


def generate_scenario_pool(config: ExperimentConfig, rng: np.random.Generator, count_per_type: int = None) -> List[dict]:
    target = count_per_type or config.source_pool.scenario_count_per_type
    all_records: List[dict] = []
    for name in SCENARIO_NAMES:
        records: List[dict] = []
        attempts = 0
        while len(records) < target:
            attempts += 1
            records = deduplicate_records(records + [SCENARIO_GENERATORS[name](rng, config)])
            if attempts > target * 100:
                raise RuntimeError(f"Too many attempts while generating scenario: {name}")
        all_records.extend(records[:target])
    return all_records

