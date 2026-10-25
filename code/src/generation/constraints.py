"""Final-profile predicates; these never read evaluation records."""
from typing import Sequence
import numpy as np


def crosses_zero(path: Sequence[int]) -> bool:
    v = np.asarray(path)
    return bool(len(v) >= 2 and np.any((v[:-1] < 0) != (v[1:] < 0)))


def increasing(path: Sequence[int]) -> bool:
    v = np.asarray(path)
    return bool(len(v) >= 2 and v[-1] > v[0] and np.all(np.diff(v) >= 0))


def decreasing(path: Sequence[int]) -> bool:
    v = np.asarray(path)
    return bool(len(v) >= 2 and v[-1] < v[0] and np.all(np.diff(v) <= 0))


def has_turn(path: Sequence[int]) -> bool:
    delta = np.diff(np.asarray(path))
    signs = np.sign(delta[delta != 0])
    return bool(len(signs) >= 2 and np.any(signs[:-1] != signs[1:]))


def separated_step(path: Sequence[int], within_span: int, minimum_gap: int) -> bool:
    """Two nearly level segments with a strictly separated upward step."""
    v = np.asarray(path)
    return any(
        np.ptp(v[:cut]) <= within_span
        and np.ptp(v[cut:]) <= within_span
        and np.min(v[cut:]) - np.max(v[:cut]) >= minimum_gap
        for cut in range(1, len(v))
    )


def rise_then_fall(path: Sequence[int]) -> bool:
    signs = np.sign(np.diff(np.asarray(path)))
    signs = signs[signs != 0]
    return bool(len(signs) >= 2 and np.any(signs > 0) and np.any(signs < 0)
                and np.all(np.diff(signs) <= 0))


def alternating_daily(path: Sequence[int], minimum_gap: int = 1) -> bool:
    delta = np.diff(np.asarray(path))
    return bool(len(delta) >= 1 and np.all(np.abs(delta) >= minimum_gap)
                and np.all(np.sign(delta[:-1]) != np.sign(delta[1:])))


def alternating_blocks(path: Sequence[int]) -> bool:
    v = np.asarray(path)
    if len(v) < 3:
        return False
    block = max(2, len(v) // 3)
    chunks = [v[i:i + block] for i in range(0, len(v), block)]
    if len(chunks) < 2 or any(np.ptp(chunk) > 2 for chunk in chunks):
        return False
    return all(
        (np.min(next_chunk) - np.max(chunk) >= 6 if i % 2 == 0
         else np.min(chunk) - np.max(next_chunk) >= 6)
        for i, (chunk, next_chunk) in enumerate(zip(chunks, chunks[1:]))
    )


def late_rise(path: Sequence[int]) -> bool:
    v = np.asarray(path)
    if not increasing(v):
        return False
    return any(
        np.all(v[:cut] == v[0])
        and np.min(v[cut:]) - v[0] >= 4
        and np.all(np.diff(v[cut:]) >= 0)
        for cut in range(max(1, len(v) - 3), len(v))
    )


def feasible_exposure_counts(n: int, lower: float = 0.25, upper: float = 0.70):
    low, high = int(np.ceil(lower * n)), int(np.floor(upper * n))
    if n < 1 or low > high:
        raise ValueError(f"No integer exposure count meets [{lower}, {upper}] for n={n}.")
    return low, high


def validate_candidate_bundle(config, bundle):
    """Check all final-profile rules, source quotas, uniqueness and ordered merge."""
    from collections import Counter
    sources = (bundle.scenario_pool, bundle.stratified_pool, bundle.risk_pool)
    for expected_source, records in zip(("scenario", "stratified", "risk"), sources):
        for record in records:
            if record.get("source_pool") != expected_source:
                raise ValueError(f"Candidate source label disagrees with its pool: {expected_source}.")
            v = tuple(record["path"])
            if not (config.temperature.min_days <= len(v) <= config.temperature.max_days):
                raise ValueError("Candidate duration outside design limits.")
            if any(
                not isinstance(x, (int, np.integer))
                or isinstance(x, (bool, np.bool_))
                or not (config.temperature.temp_min <= x <= config.temperature.temp_max)
                for x in v
            ):
                raise ValueError("Candidate values must be integer and within design limits.")

    scen = Counter(r["subtype"] for r in bundle.scenario_pool)
    if len(scen) != 10 or set(scen.values()) != {config.source_pool.scenario_count_per_type}:
        raise ValueError("Scenario quotas do not match the frozen design.")
    for r in bundle.scenario_pool:
        v, kind = r["path"], r["subtype"]
        predicates = {
            "stable_frozen": lambda: all(-18 <= x <= -1 for x in v),
            "stable_chilled": lambda: all(0 <= x <= 4 for x in v),
            "slow_warming": lambda: len(v) >= 3 and increasing(v),
            "fast_warming": lambda: len(v) >= 2 and separated_step(v, 2, 6),
            "recooling": lambda: len(v) >= 3 and rise_then_fall(v),
            "high_frequency_fluctuation": lambda: len(v) >= 2 and alternating_daily(v, 4),
            "low_frequency_fluctuation": lambda: alternating_blocks(v),
            "late_warming": lambda: late_rise(v),
            "cross_zero": lambda: crosses_zero(v),
            "temperature_abuse": lambda: all(-2 <= x <= 8 or 15 <= x <= 18 for x in v)
                                         and any(15 <= x <= 18 for x in v),
        }
        if kind not in predicates or not predicates[kind]():
            raise ValueError(f"Final scenario predicate failed: {kind}, {v}")

    strata = Counter(r["stratum"] for r in bundle.stratified_pool)
    if len(strata) != 180 or set(strata.values()) != {config.source_pool.stratified_count_per_stratum}:
        raise ValueError("Stratified quotas do not match the frozen design.")
    for records, key_name in ((bundle.scenario_pool, "subtype"),
                              (bundle.stratified_pool, "stratum")):
        for group in {r[key_name] for r in records}:
            paths = [tuple(r["path"]) for r in records if r[key_name] == group]
            if len(paths) != len(set(paths)):
                raise ValueError(f"Duplicate complete sequences within {key_name}={group}.")
    risk_paths = [tuple(r["path"]) for r in bundle.risk_pool]
    if len(risk_paths) != len(set(risk_paths)):
        raise ValueError("Duplicate complete sequences across risk categories.")

    bands = {name: (lo, hi) for name, lo, hi in config.source_pool.temperature_layers}
    durations = {name: (lo, hi) for name, lo, hi in config.source_pool.time_layers}
    amplitude_names = {name for name, _ in config.source_pool.amplitude_layers}
    for r in bundle.stratified_pool:
        time, band, trend, amplitude = r["stratum"].split("|")
        if amplitude not in amplitude_names or trend not in config.source_pool.path_type_layers:
            raise ValueError("Unknown stratum trend or amplitude.")
        v = r["path"]
        lo, hi = bands[band]
        nlo, nhi = durations[time]
        if not nlo <= len(v) <= nhi or not any(lo <= x <= hi for x in v):
            raise ValueError("Final target band or duration layer failed.")
        if ((trend == "warming" and not increasing(v))
            or (trend == "cooling" and not decreasing(v))
            or (trend == "oscillating" and (len(v) < 3 or not has_turn(v)))):
            raise ValueError(f"Final stratified trend failed: {trend}")

    risks = Counter(r["risk_type"] for r in bundle.risk_pool)
    if len(risks) != 5 or set(risks.values()) != {config.source_pool.risk_pool_size // 5}:
        raise ValueError("Risk quotas do not match the frozen design.")
    for r in bundle.risk_pool:
        v, kind = r["path"], r["risk_type"]
        predicates = {
            "cold_boundary": lambda: all(-2 <= x <= 6 for x in v),
            "transition_4_8": lambda: any(4 < x <= 8 for x in v),
            "above_8_risk": lambda: any(9 <= x <= 15 for x in v),
            "above_15_abuse": lambda: len(v) >= 2
                and all(-3 <= x <= 12 or 15 <= x <= 18 for x in v)
                and .25 <= sum(15 <= x <= 18 for x in v) / len(v) <= .70,
            "freeze_thaw": lambda: len(v) >= 2
                and all(-18 <= x <= -1 or 0 <= x <= 12 for x in v)
                and all((a < 0) != (b < 0) for a, b in zip(v, v[1:])),
        }
        if kind not in predicates or not predicates[kind]():
            raise ValueError(f"Final risk predicate failed: {kind}, {v}")

    seen = set()
    ordered_records = []
    for records in sources:
        for record in records:
            key = tuple(record["path"])
            if key not in seen:
                seen.add(key)
                ordered_records.append(record)
    if len(ordered_records) != len(bundle.candidate_pool):
        raise ValueError("Merged pool count differs from ordered full-sequence deduplication.")
    for expected, actual in zip(ordered_records, bundle.candidate_pool):
        if tuple(actual["path"]) != tuple(expected["path"]):
            raise ValueError("Merged pool differs from ordered full-sequence deduplication.")
        if any(actual.get(field) != expected.get(field)
               for field in ("source_pool", "subtype", "stratum", "risk_type")):
            raise ValueError("Merged candidate source/category label differs from its first occurrence.")
    ordered = [tuple(record["path"]) for record in ordered_records]
    return {"raw_counts": [len(x) for x in sources],
            "unique_count": len(ordered), "violations": 0,
            "evaluation_records_used": False}
