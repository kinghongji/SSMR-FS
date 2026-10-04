from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from config.experiment_config import METRIC_DIRECTIONS, METHOD_ORDER


def summarize_repeated_metrics(repeat_metrics: pd.DataFrame) -> pd.DataFrame:
    """按方法和指标汇总30次重复的平均值与样本标准差。"""

    required = {"repeat", "method", *METRIC_DIRECTIONS}
    missing = required.difference(repeat_metrics.columns)
    if missing:
        raise ValueError(f"重复结果缺少必要字段：{sorted(missing)}")

    rows = []
    for method in METHOD_ORDER:
        method_frame = repeat_metrics[repeat_metrics["method"] == method]
        if method_frame.empty:
            raise ValueError(f"重复结果缺少方法：{method}")
        for metric in METRIC_DIRECTIONS:
            values = method_frame[metric].to_numpy(dtype=float)
            mean_value = float(np.mean(values))
            standard_deviation = float(np.std(values, ddof=1))
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "repeat_count": int(values.size),
                    "mean": mean_value,
                    "standard_deviation": standard_deviation,
                    "mean_plus_minus_sd": (
                        f"{mean_value:.6f} ± {standard_deviation:.6f}"
                    ),
                }
            )
    return pd.DataFrame(rows)


def count_ssmr_simultaneous_best(
    repeat_metrics: pd.DataFrame,
    baseline_methods: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """严格统计SSMR-FS在同一次重复中四项指标同时排名第一的次数。

    越小越好的指标必须严格小于全部正式基线；越大越好的指标必须
    严格大于全部正式基线。相等不计为严格最优。
    """

    detail_rows = []
    repeat_ids = sorted(repeat_metrics["repeat"].unique())
    for repeat_id in repeat_ids:
        current = repeat_metrics[repeat_metrics["repeat"] == repeat_id]
        ssmr_rows = current[current["method"] == "SSMR-FS"]
        if len(ssmr_rows) != 1:
            raise ValueError(
                f"第{repeat_id}次重复中的SSMR-FS结果数量应为1，实际为{len(ssmr_rows)}。"
            )
        baseline_rows = current[current["method"].isin(baseline_methods)]
        if set(baseline_rows["method"]) != set(baseline_methods):
            raise ValueError(f"第{repeat_id}次重复缺少正式基线结果。")

        metric_flags = {}
        for metric, direction in METRIC_DIRECTIONS.items():
            ssmr_value = float(ssmr_rows.iloc[0][metric])
            baseline_values = baseline_rows[metric].to_numpy(dtype=float)
            metric_flags[metric] = bool(
                ssmr_value < np.min(baseline_values)
                if direction == "lower"
                else ssmr_value > np.max(baseline_values)
            )

        detail_rows.append(
            {
                "repeat": int(repeat_id),
                "pce_best": metric_flags["路径覆盖误差"],
                "kce_best": metric_flags["关键温区覆盖误差"],
                "ps_best": metric_flags["路径间隔"],
                "lfv_best": metric_flags["对数特征体积"],
                "simultaneously_best": bool(all(metric_flags.values())),
            }
        )

    detail = pd.DataFrame(detail_rows)
    simultaneous_count = int(detail["simultaneously_best"].sum())
    repeat_count = int(len(detail))
    summary = pd.DataFrame(
        [
            {
                "repeat_count": repeat_count,
                "simultaneous_best_count": simultaneous_count,
                "simultaneous_best_proportion": (
                    simultaneous_count / repeat_count if repeat_count else float("nan")
                ),
                "strict_rule": "四项指标在同一次重复中均严格优于四种正式基线",
            }
        ]
    )
    return detail, summary
