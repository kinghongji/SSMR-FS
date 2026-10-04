from dataclasses import dataclass
from math import ceil, sqrt
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import qmc

from config.experiment_config import BASELINE_FAMILIES, BASELINE_METHODS, ExperimentConfig
from src.utils.random_utils import (
    path_key,
    q_temperature,
    sample_integer_temperature,
    sample_storage_days,
)


@dataclass
class BaselineSuiteResult:
    """一次公平基线生成的全部结果和可追溯信息。"""

    records_by_method: Dict[str, List[dict]]
    diagnostics: List[dict]


def _method_record(path: Sequence[int], method: str, source: str, subtype: str) -> dict:
    return {
        "path": tuple(int(value) for value in path),
        "method": method,
        "source_pool": source,
        "subtype": subtype,
    }


def generate_random_design(
    config: ExperimentConfig,
    rng: np.random.Generator,
    n_samples: int,
    method_name: str = "简单随机",
) -> List[dict]:
    """路径长度离散均匀、逐日温度独立离散均匀地生成N条唯一路径。"""

    records: List[dict] = []
    seen = set()
    attempts = 0
    while len(records) < n_samples:
        attempts += 1
        n_days = sample_storage_days(
            rng,
            config.temperature.min_days,
            config.temperature.max_days,
        )
        path = tuple(
            sample_integer_temperature(
                rng,
                config.temperature.temp_min,
                config.temperature.temp_max,
            )
            for _ in range(n_days)
        )
        key = path_key(path)
        if key not in seen:
            seen.add(key)
            records.append(
                _method_record(
                    path,
                    method_name,
                    "baseline_random",
                    "simple_random",
                )
            )
        if attempts > n_samples * 500:
            raise RuntimeError("简单随机基线在限定尝试次数内无法生成足够的唯一路径。")
    return records


def _map_unit_point(point: np.ndarray, config: ExperimentConfig) -> Tuple[int, ...]:
    """统一把11维[0,1)点映射为1个长度维度和最多10个温度维度。"""

    span_days = config.temperature.max_days - config.temperature.min_days + 1
    n_days = int(config.temperature.min_days + np.floor(span_days * point[0]))
    n_days = min(config.temperature.max_days, max(config.temperature.min_days, n_days))
    temp_span = config.temperature.temp_max - config.temperature.temp_min
    return tuple(
        q_temperature(
            config.temperature.temp_min + temp_span * point[index + 1],
            config.temperature.temp_min,
            config.temperature.temp_max,
        )
        for index in range(n_days)
    )


def _records_from_unit_points(
    points: np.ndarray,
    config: ExperimentConfig,
    rng: np.random.Generator,
    n_samples: int,
    method: str,
    source: str,
    subtype: str,
) -> Tuple[List[dict], int]:
    """统一完成离散映射、完整路径去重和不足数量补充。"""

    records: List[dict] = []
    seen = set()
    for point in points:
        path = _map_unit_point(point, config)
        key = path_key(path)
        if key in seen:
            continue
        seen.add(key)
        records.append(_method_record(path, method, source, subtype))
        if len(records) >= n_samples:
            break

    supplement_count = 0
    attempts = 0
    while len(records) < n_samples:
        attempts += 1
        n_days = sample_storage_days(
            rng,
            config.temperature.min_days,
            config.temperature.max_days,
        )
        path = tuple(
            sample_integer_temperature(
                rng,
                config.temperature.temp_min,
                config.temperature.temp_max,
            )
            for _ in range(n_days)
        )
        key = path_key(path)
        if key not in seen:
            seen.add(key)
            supplement_count += 1
            records.append(
                _method_record(
                    path,
                    method,
                    f"{source}_supplement",
                    "simple_random_supplement_after_discrete_mapping",
                )
            )
        if attempts > n_samples * 500:
            raise RuntimeError(f"{method}离散映射后无法补足{n_samples}条唯一路径。")
    return records, supplement_count


def _seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2**31 - 1))


def generate_sobol_design(
    config: ExperimentConfig,
    rng: np.random.Generator,
    n_samples: int,
) -> Tuple[List[dict], int]:
    """生成加扰Sobol二次幂点集的前N点，并映射为温度路径。"""

    sampler = qmc.Sobol(
        d=config.baseline.input_dim,
        scramble=config.baseline.sobol_scramble,
        seed=_seed(rng),
    )
    # 先生成包含N的最小二次幂点集，再取前N点，避免任意N调用random()的警告。
    power = int(ceil(np.log2(max(n_samples, 1))))
    points = sampler.random_base2(m=power)[:n_samples]
    return _records_from_unit_points(
        points,
        config,
        rng,
        n_samples,
        "Sobol",
        "baseline_sobol",
        "scrambled_sobol_power2_prefix",
    )


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def _next_prime(value: int) -> int:
    candidate = max(2, value)
    while not _is_prime(candidate):
        candidate += 1
    return candidate


def _oa_lhs_points(
    dimension: int,
    target_count: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, int]:
    """构造强度2 OA-LHS完整点集，再随机派生目标数量N。"""

    q = _next_prime(max(dimension - 1, int(ceil(sqrt(target_count)))))
    rows = [(a, b) for a in range(q) for b in range(q)]
    symbols = np.empty((q * q, dimension), dtype=int)
    for row_index, (a, b) in enumerate(rows):
        symbols[row_index, 0] = a
        symbols[row_index, 1] = b
        for column in range(2, dimension):
            symbols[row_index, column] = (a + (column - 1) * b) % q

    points = np.empty_like(symbols, dtype=float)
    for column in range(dimension):
        for symbol in range(q):
            indices = np.flatnonzero(symbols[:, column] == symbol)
            within_symbol = rng.permutation(q)
            ranks = symbol * q + within_symbol
            points[indices, column] = (ranks + rng.random(q)) / (q * q)
    order = rng.permutation(q * q)
    return points[order[:target_count]], q * q


def generate_oa_lhs_design(
    config: ExperimentConfig,
    rng: np.random.Generator,
    n_samples: int,
) -> Tuple[List[dict], int, int]:
    """生成由完整强度2 OA-LHS随机派生的N点设计。"""

    points, complete_oa_size = _oa_lhs_points(
        config.baseline.input_dim,
        n_samples,
        rng,
    )
    records, supplement_count = _records_from_unit_points(
        points,
        config,
        rng,
        n_samples,
        "OA-LHS",
        "baseline_oa_lhs",
        "strength2_oa_lhs_derived_subset",
    )
    return records, supplement_count, complete_oa_size


def _maxpro_pair_penalty(point: np.ndarray, others: np.ndarray) -> np.ndarray:
    differences = np.maximum(np.abs(others - point), 1e-14)
    log_penalty = -2.0 * np.sum(np.log(differences), axis=1)
    return np.exp(np.minimum(log_penalty, 700.0))


def _maxpro_lhs_points(
    config: ExperimentConfig,
    rng: np.random.Generator,
    n_samples: int,
) -> np.ndarray:
    """在随机LHS内用成对MaxPro准则执行可复现的坐标交换。"""

    sampler = qmc.LatinHypercube(
        d=config.baseline.input_dim,
        scramble=True,
        seed=_seed(rng),
    )
    points = sampler.random(n=n_samples)
    iterations = max(1, config.baseline.maxpro_exchange_factor * n_samples)
    all_indices = np.arange(n_samples)
    for _ in range(iterations):
        first, second = rng.choice(n_samples, size=2, replace=False)
        column = int(rng.integers(0, config.baseline.input_dim))
        others = all_indices[(all_indices != first) & (all_indices != second)]
        if others.size == 0:
            continue
        old_value = float(
            np.sum(_maxpro_pair_penalty(points[first], points[others]))
            + np.sum(_maxpro_pair_penalty(points[second], points[others]))
        )
        first_trial = points[first].copy()
        second_trial = points[second].copy()
        first_trial[column], second_trial[column] = (
            second_trial[column],
            first_trial[column],
        )
        new_value = float(
            np.sum(_maxpro_pair_penalty(first_trial, points[others]))
            + np.sum(_maxpro_pair_penalty(second_trial, points[others]))
        )
        if new_value < old_value:
            points[first, column], points[second, column] = (
                points[second, column],
                points[first, column],
            )
    return points


def generate_maxpro_design(
    config: ExperimentConfig,
    rng: np.random.Generator,
    n_samples: int,
) -> Tuple[List[dict], int]:
    """生成坐标交换MaxPro-LHS并映射为温度路径。"""

    points = _maxpro_lhs_points(config, rng, n_samples)
    return _records_from_unit_points(
        points,
        config,
        rng,
        n_samples,
        "MaxPro",
        "baseline_maxpro",
        "coordinate_exchange_maxpro_lhs",
    )


def _diagnostic(
    method: str,
    n_samples: int,
    supplement_count: int = 0,
    implementation_note: str = "",
) -> dict:
    return {
        "method": method,
        "family": BASELINE_FAMILIES[method],
        "requested_output_count": n_samples,
        "actual_output_count": n_samples,
        "discrete_mapping_supplement_count": supplement_count,
        "uses_storage_temperature_records": False,
        "implementation_note": implementation_note,
    }


def generate_baseline_suite(
    config: ExperimentConfig,
    rng: np.random.Generator,
    n_samples: int,
) -> BaselineSuiteResult:
    """生成论文正式比较的简单随机、Sobol、OA-LHS和MaxPro。"""

    seeds = rng.integers(0, 2**31 - 1, size=len(BASELINE_METHODS))
    child_rng = {
        method: np.random.default_rng(int(seed_value))
        for method, seed_value in zip(BASELINE_METHODS, seeds)
    }

    records_by_method: Dict[str, List[dict]] = {}
    diagnostics: List[dict] = []

    records_by_method["简单随机"] = generate_random_design(
        config,
        child_rng["简单随机"],
        n_samples,
    )
    diagnostics.append(_diagnostic("简单随机", n_samples))

    sobol_records, sobol_supplement = generate_sobol_design(
        config,
        child_rng["Sobol"],
        n_samples,
    )
    records_by_method["Sobol"] = sobol_records
    diagnostics.append(_diagnostic("Sobol", n_samples, sobol_supplement))

    oa_records, oa_supplement, complete_oa_size = generate_oa_lhs_design(
        config,
        child_rng["OA-LHS"],
        n_samples,
    )
    records_by_method["OA-LHS"] = oa_records
    diagnostics.append(
        _diagnostic(
            "OA-LHS",
            n_samples,
            oa_supplement,
            implementation_note=(
                f"由完整{complete_oa_size}点强度2 OA-LHS随机派生N={n_samples}子集；"
                "任意N子集不再声称保持完整强度2正交性"
            ),
        )
    )

    maxpro_records, maxpro_supplement = generate_maxpro_design(
        config,
        child_rng["MaxPro"],
        n_samples,
    )
    records_by_method["MaxPro"] = maxpro_records
    diagnostics.append(_diagnostic("MaxPro", n_samples, maxpro_supplement))

    missing = [method for method in BASELINE_METHODS if method not in records_by_method]
    if missing:
        raise RuntimeError(f"基线套件遗漏方法：{missing}")
    ordered = {method: records_by_method[method] for method in BASELINE_METHODS}
    return BaselineSuiteResult(records_by_method=ordered, diagnostics=diagnostics)
