from itertools import product
from typing import List, Tuple

import numpy as np

from config.experiment_config import ExperimentConfig
from src.utils.random_utils import deduplicate_records, q_temperature, sample_integer_temperature


def _ensure_temperature_layer(values: List[int], layer_low: int, layer_high: int, rng: np.random.Generator) -> List[int]:
    if any(layer_low <= v <= layer_high for v in values):
        return values
    pos = int(rng.integers(0, len(values)))
    values[pos] = sample_integer_temperature(rng, layer_low, layer_high)
    return values


def generate_path_for_stratum(
    rng: np.random.Generator,
    n_low: int,
    n_high: int,
    temp_low: int,
    temp_high: int,
    path_type: str,
    amplitude: int,
) -> Tuple[int, ...]:
    n = int(rng.integers(n_low, n_high + 1))
    center = float(rng.uniform(temp_low, temp_high))
    if path_type == "stable":
        sigma = max(0.5, amplitude / 2.0)
        values = [q_temperature(center + rng.normal(0, sigma)) for _ in range(n)]
    elif path_type == "warming":
        start = max(-18, center - amplitude - rng.uniform(0, 3))
        end = min(18, center + amplitude + rng.uniform(1, 5))
        values = [q_temperature(start + i / max(1, n - 1) * (end - start) + rng.normal(0, 0.8)) for i in range(n)]
    elif path_type == "cooling":
        start = min(18, center + amplitude + rng.uniform(1, 5))
        end = max(-18, center - amplitude - rng.uniform(0, 3))
        values = [q_temperature(start + i / max(1, n - 1) * (end - start) + rng.normal(0, 0.8)) for i in range(n)]
    elif path_type == "oscillating":
        phase = rng.uniform(0, 2 * np.pi)
        values = [
            q_temperature(center + amplitude * np.sin(i * np.pi + phase) + rng.normal(0, 0.8))
            for i in range(n)
        ]
    else:
        raise ValueError(f"Unknown path type: {path_type}")
    return tuple(_ensure_temperature_layer(values, temp_low, temp_high, rng))


def iter_strata(config: ExperimentConfig):
    for time_layer, temp_layer, path_type, amp_layer in product(
        config.source_pool.time_layers,
        config.source_pool.temperature_layers,
        config.source_pool.path_type_layers,
        config.source_pool.amplitude_layers,
    ):
        yield time_layer, temp_layer, path_type, amp_layer


def generate_stratified_pool(config: ExperimentConfig, rng: np.random.Generator, count_per_stratum: int = None) -> List[dict]:
    target = count_per_stratum or config.source_pool.stratified_count_per_stratum
    all_records: List[dict] = []
    for time_layer, temp_layer, path_type, amp_layer in iter_strata(config):
        time_name, n_low, n_high = time_layer
        temp_name, temp_low, temp_high = temp_layer
        amp_name, amplitude = amp_layer
        stratum = f"{time_name}|{temp_name}|{path_type}|{amp_name}"
        records: List[dict] = []
        attempts = 0
        while len(records) < target:
            attempts += 1
            path = generate_path_for_stratum(rng, n_low, n_high, temp_low, temp_high, path_type, amplitude)
            record = {
                "path": path,
                "source_pool": "stratified",
                "subtype": "stratified_exploration",
                "stratum": stratum,
            }
            records = deduplicate_records(records + [record])
            if attempts > target * 100:
                raise RuntimeError(f"Too many attempts while generating stratum: {stratum}")
        all_records.extend(records[:target])
    return all_records

