from typing import Sequence


def validate_temperature_path(path: Sequence[int], temp_min: int, temp_max: int, min_days: int, max_days: int) -> None:
    if not (min_days <= len(path) <= max_days):
        raise ValueError(f"Invalid path length {len(path)}; expected {min_days}-{max_days}.")
    for value in path:
        if int(value) != value:
            raise ValueError(f"Temperature must be integer: {value}")
        if value < temp_min or value > temp_max:
            raise ValueError(f"Temperature {value} outside [{temp_min}, {temp_max}].")


def validate_records(records: Sequence[dict], temp_min: int, temp_max: int, min_days: int, max_days: int) -> None:
    for record in records:
        validate_temperature_path(record["path"], temp_min, temp_max, min_days, max_days)

