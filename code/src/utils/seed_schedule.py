"""Named, order-independent random streams derived from one master seed."""
from typing import List
import numpy as np

SCOPE_KEYS = {"main": 0, "repeat": 1, "size": 2}
METHOD_KEYS = {"SSMR-FS": 0, "简单随机": 1, "Sobol": 2, "OA-LHS": 3, "MaxPro": 4}

def stream_seed(master_seed: int, scope: str, method: str, repeat: int = 0, sample_size: int = 0) -> int:
    if scope not in SCOPE_KEYS or method not in METHOD_KEYS:
        raise ValueError("Unknown named random stream.")
    if repeat < 0 or sample_size < 0:
        raise ValueError("Repeat and sample size must be nonnegative.")
    sequence = np.random.SeedSequence(
        master_seed, spawn_key=(SCOPE_KEYS[scope], repeat, sample_size, METHOD_KEYS[method])
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])

def named_rng(master_seed, scope, method, repeat=0, sample_size=0):
    return np.random.default_rng(stream_seed(master_seed, scope, method, repeat, sample_size))

def seed_manifest(config) -> List[dict]:
    rows = []
    for method in METHOD_KEYS:
        rows.append(_row(config, "main", method))
    for r in range(1, config.repeated_times + 1):
        for method in METHOD_KEYS:
            rows.append(_row(config, "repeat", method, r))
    for n in config.sample_size_validation:
        for method in METHOD_KEYS:
            if method == "SSMR-FS":
                rows.append({**_row(config, "main", method), "used_for_size": n,
                             "reuse": "same main candidate pool; deterministic ordered prefix"})
            else:
                rows.append(_row(config, "size", method, sample_size=n))
    return rows

def _row(config, scope, method, repeat=0, sample_size=0):
    return {"master_seed": config.random_seed, "scope": scope, "method": method,
            "repeat": repeat, "sample_size": sample_size,
            "spawn_key": [SCOPE_KEYS[scope], repeat, sample_size, METHOD_KEYS[method]],
            "stream_seed": stream_seed(config.random_seed, scope, method, repeat, sample_size),
            "generator": "PCG64", "derivation": "SeedSequence(master_seed, spawn_key).generate_state(1, uint32)"}
