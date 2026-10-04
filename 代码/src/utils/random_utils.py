from typing import Iterable, List, Sequence, Tuple

import numpy as np


TemperaturePath = Tuple[int, ...]


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def q_temperature(value: float, temp_min: int = -18, temp_max: int = 18) -> int:
    rounded = int(np.rint(value))
    return int(min(temp_max, max(temp_min, rounded)))


def path_key(path: Sequence[int]) -> TemperaturePath:
    return tuple(int(x) for x in path)


def deduplicate_records(records: Iterable[dict]) -> List[dict]:
    seen = set()
    unique_records: List[dict] = []
    for record in records:
        key = path_key(record["path"])
        if key not in seen:
            seen.add(key)
            clean = dict(record)
            clean["path"] = key
            unique_records.append(clean)
    return unique_records


def count_duplicates(records: Sequence[dict]) -> int:
    keys = [path_key(record["path"]) for record in records]
    return len(keys) - len(set(keys))


def sample_storage_days(rng: np.random.Generator, min_days: int = 1, max_days: int = 10) -> int:
    return int(rng.integers(min_days, max_days + 1))


def sample_integer_temperature(
    rng: np.random.Generator,
    low: int,
    high: int,
) -> int:
    return int(rng.integers(low, high + 1))


def pad_path(path: Sequence[float], max_days: int = 10):
    values = [np.nan] * max_days
    for idx, value in enumerate(path[:max_days]):
        values[idx] = value
    return values


def records_to_temperature_frame(records: Sequence[dict], max_days: int = 10):
    import pandas as pd

    rows = []
    for record in records:
        row = {
            "record_id": record.get("record_id", ""),
            "method": record.get("method", ""),
            "source_pool": record.get("source_pool", ""),
            "subtype": record.get("subtype", ""),
            "stratum": record.get("stratum", ""),
            "risk_type": record.get("risk_type", ""),
            "key_groups": "|".join(record.get("key_groups", ())),
            "n_days": len(record["path"]),
            "path_key": "|".join(str(v) for v in record["path"]),
        }
        for day_idx, value in enumerate(pad_path(record["path"], max_days=max_days), start=1):
            row[f"day_{day_idx}"] = value
        rows.append(row)
    return pd.DataFrame(rows)
