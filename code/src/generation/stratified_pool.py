from itertools import product
from typing import List, Tuple
import numpy as np
from config.experiment_config import ExperimentConfig
from src.generation.constraints import decreasing, increasing, has_turn
from src.utils.random_utils import deduplicate_records, q_temperature

def _matches_stratum(values, low, high, path_type):
    if not any(low <= v <= high for v in values):
        return False
    return {
        "stable": lambda: True,
        "warming": lambda: increasing(values),
        "cooling": lambda: decreasing(values),
        "oscillating": lambda: has_turn(values),
    }[path_type]()

def generate_path_for_stratum(
    rng: np.random.Generator, n_low: int, n_high: int,
    temp_low: int, temp_high: int, path_type: str, amplitude: int,
    attempt_limit: int = 20000,
) -> Tuple[int, ...]:
    minimum = {"stable": 1, "warming": 2, "cooling": 2, "oscillating": 3}
    if path_type not in minimum:
        raise ValueError(f"Unknown path type: {path_type}")
    feasible_low = max(n_low, minimum[path_type])
    if feasible_low > n_high:
        raise ValueError(f"No feasible duration for {path_type} in [{n_low}, {n_high}].")
    # Fix a feasible duration first, so trend-rejection does not silently
    # favor shorter durations within a stratum.
    n = int(rng.integers(feasible_low, n_high + 1))
    for _ in range(attempt_limit):
        center = float(rng.uniform(temp_low, temp_high))
        # Existing baseline endpoint and sinusoidal constructions are retained.
        # Every rule now adds bounded U(-a,a) daily perturbations before rounding.
        if path_type == "stable":
            baseline = np.full(n, center)
        elif path_type == "warming":
            start = max(-18, center - amplitude - rng.uniform(0, 3))
            end = min(18, center + amplitude + rng.uniform(1, 5))
            baseline = np.linspace(start, end, n)
        elif path_type == "cooling":
            start = min(18, center + amplitude + rng.uniform(1, 5))
            end = max(-18, center - amplitude - rng.uniform(0, 3))
            baseline = np.linspace(start, end, n)
        else:
            phase = rng.uniform(0, 2 * np.pi)
            baseline = center + amplitude * np.sin(np.arange(n) * np.pi + phase)
        noise = rng.uniform(-amplitude, amplitude, size=n)
        values = [q_temperature(v) for v in baseline + noise]
        # Check the target band and trend on the rounded final profile.
        # Conditional acceptance is documented; accepted daily values are
        # not claimed to remain independent uniformly distributed values.
        if _matches_stratum(values, temp_low, temp_high, path_type):
            return tuple(values)
    raise RuntimeError(f"Could not satisfy final trend and target band: {path_type}, n={n}, band={temp_low}:{temp_high}, a={amplitude}")

def iter_strata(config: ExperimentConfig):
    yield from product(
        config.source_pool.time_layers, config.source_pool.temperature_layers,
        config.source_pool.path_type_layers, config.source_pool.amplitude_layers,
    )

def generate_stratified_pool(config: ExperimentConfig, rng: np.random.Generator, count_per_stratum: int = None) -> List[dict]:
    target = count_per_stratum or config.source_pool.stratified_count_per_stratum
    all_records = []
    for time_layer, temp_layer, path_type, amp_layer in iter_strata(config):
        time_name, n_low, n_high = time_layer
        temp_name, temp_low, temp_high = temp_layer
        amp_name, amplitude = amp_layer
        stratum = f"{time_name}|{temp_name}|{path_type}|{amp_name}"
        records, attempts = [], 0
        while len(records) < target:
            attempts += 1
            path = generate_path_for_stratum(
                rng, n_low, n_high, temp_low, temp_high, path_type, amplitude,
                config.source_pool.generation_attempt_limit,
            )
            records = deduplicate_records(records + [{
                "path": path, "source_pool": "stratified",
                "subtype": "stratified_exploration", "stratum": stratum,
            }])
            if attempts > target * 100:
                raise RuntimeError(f"Too many duplicate proposals for stratum: {stratum}")
        all_records.extend(records)
    return all_records
