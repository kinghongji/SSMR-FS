import json
import re
from pathlib import Path
from typing import Dict, Hashable

import numpy as np
import pandas as pd


# These labels are applied only when data are exported. Internal field names,
# metric definitions, values, and calculation paths are unchanged.
COLUMN_LABELS = {
    "record_id": "Record ID",
    "method": "Method",
    "source_pool": "Candidate source",
    "subtype": "Path category",
    "stratum": "Stratum",
    "risk_type": "Exposure category",
    "key_groups": "Key-condition groups",
    "n_days": "Storage duration (d)",
    "path_key": "Complete temperature profile",
    "selection_order": "Selection order",
    "storage_days": "Storage duration (d)",
    "mean_temperature": "Mean temperature (°C)",
    "median_temperature": "Median temperature (°C)",
    "max_temperature": "Maximum temperature (°C)",
    "min_temperature": "Minimum temperature (°C)",
    "end_temperature": "Final temperature (°C)",
    "days_below_0": "Days below 0 °C",
    "days_0_4": "Days at 0–4 °C",
    "days_4_8": "Days at 4 < T ≤ 8 °C",
    "days_above_8": "Exposure above 8 °C (d)",
    "days_above_equal_15": "Days at or above 15 °C",
    "auc_4": "AUC4 (°C·d)",
    "auc_8": "AUC8 (°C·d)",
    "auc_15": "AUC15 (°C·d)",
    "temperature_range": "Temperature range (°C)",
    "temperature_std": "Temperature standard deviation (°C)",
    "max_abs_delta": "Maximum adjacent-day temperature change (°C)",
    "mean_abs_delta": "Mean adjacent-day temperature change (°C)",
    "total_abs_delta": "Total temperature variation (°C)",
    "warming_count": "Warming steps (count)",
    "cooling_count": "Cooling steps (count)",
    "cross_0_count": "0 °C crossings (count)",
    "cross_8_count": "8 °C crossings (count)",
    "turn_count": "Direction changes (count)",
    "mean_distance_to_candidate_space": "Mean distance to candidate set",
    "distance_sum_to_previously_selected": "Distance sum to previously selected paths",
    "sequential_energy_score": "Sequential energy score",
    "energy_distance_after_selection": "Energy distance after selection",
    "explained_variance_ratio": "Explained variance ratio",
    "cumulative_explained_variance": "Cumulative explained variance",
    "mean_plus_minus_sd": "Mean ± SD",
    "pce_best": "PCE strictly best",
    "kce_best": "KCE strictly best",
    "ps_best": "PS strictly best",
    "lfv_best": "LFV strictly best",
    "simultaneously_best": "All four metrics strictly best",
    "simultaneous_best_count": "Runs with all four metrics strictly best",
    "simultaneous_best_proportion": "Proportion with all four metrics strictly best",
    "ssmr_better": "SSMR-FS strictly better",
    "路径覆盖误差": "Path coverage error (PCE)",
    "关键温区覆盖误差": "Key-condition coverage error (KCE)",
    "路径间隔": "Path separation (PS)",
    "对数特征体积": "Log feature volume (LFV)",
    "方法": "Method",
    "简单随机": "SRS",
    "重复序号": "Repeat",
    "拟合数据": "Data used to fit PCA",
    "是否加权": "Weighted",
    "是否使用储藏温度记录": "Measured records used",
    "主成分数": "Retained principal components",
    "累计解释方差": "Cumulative explained variance",
    "PCA用途": "PCA purpose",
    "变量": "Variable",
}

VALUE_LABELS = {
    "简单随机": "SRS",
    "路径覆盖误差": "Path coverage error (PCE)",
    "关键温区覆盖误差": "Key-condition coverage error (KCE)",
    "路径间隔": "Path separation (PS)",
    "对数特征体积": "Log feature volume (LFV)",
    "实际每日温度": "Actual daily temperatures",
    "评价参考记录": "Measured records",
    "storage_temperature_records": "Measured records",
    "storage_temperature_record": "Measured record",
    "scenario": "Typical storage scenarios",
    "stratified": "Stratified exploration",
    "risk": "Risk exposure",
    "stable_frozen": "Stable frozen storage",
    "stable_chilled": "Stable chilled storage",
    "slow_warming": "Gradual warming",
    "fast_warming": "Rapid warming",
    "recooling": "Warming followed by recooling",
    "high_frequency_fluctuation": "High-frequency fluctuations",
    "low_frequency_fluctuation": "Low-frequency fluctuations",
    "late_warming": "Late-stage warming",
    "cross_zero": "0 °C crossings",
    "temperature_abuse": "Elevated-temperature exposure",
    "freezing": "Freezing",
    "chilled": "Chilling",
    "transition": "Transition",
    "abuse": "High-temperature exposure",
    "stable": "Stable",
    "warming": "Warming",
    "cooling": "Cooling",
    "oscillating": "Oscillating",
    "low": "Low (a = 1 °C)",
    "medium": "Medium (a = 3 °C)",
    "high": "High (a = 6 °C)",
    "stratified_exploration": "Stratified exploration",
    "risk_exposure": "Risk exposure",
    "cold_boundary": "Chilling boundary",
    "transition_4_8": "4 < T ≤ 8 °C exposure",
    "above_8_risk": "Above 8 °C exposure",
    "above_15_abuse": "At or above 15 °C exposure",
    "freeze_thaw": "Daily 0 °C boundary alternation",
    "0-4℃": "0–4 °C",
    "4-8℃": "4 < T ≤ 8 °C",
    ">8℃": ">8 °C",
    ">=15℃": "≥15 °C",
    "跨0℃": "0 °C crossings",
    "lower": "Lower is better",
    "越小越好": "Lower is better",
    "越大越好": "Higher is better",
    "随机抽样": "Random sampling",
    "低差异序列": "Low-discrepancy sequence",
    "正交分层空间抽样": "Orthogonal-array-based Latin hypercube",
    "投影填充设计": "Maximum projection design",
    "higher": "Higher is better",
    "baseline_random": "SRS",
    "baseline_sobol": "Sobol",
    "baseline_oa_lhs": "OA-LHS",
    "baseline_maxpro": "MaxPro",
    "simple_random": "SRS",
    "scrambled_sobol_power2_prefix": "Scrambled Sobol power-of-two prefix",
    "strength2_oa_lhs_derived_subset": "Subset of a strength-two OA-LHS",
    "coordinate_exchange_maxpro_lhs": "MaxPro-oriented coordinate-exchange LHS",
    "simple_random_supplement_after_discrete_mapping": "Unique SRS supplementation after discrete mapping",
    "candidate_constrained_sequential_energy_distance": "Candidate-constrained sequential energy distance",
    "minimum_sequential_energy_distance": "Minimum sequential energy distance",
    "greedy minimization of energy distance between selected paths and the merged unique candidate space": "Sequential minimization of energy distance to the merged unique candidate set",
    "mean_distance_to_candidate_space - distance_sum_to_previous_selected / current_selected_size": "Mean distance to the candidate set minus the distance sum to previously selected paths divided by the current selected-set size",
    "五类风险路径等量生成，不表示实际发生概率": "The five risk-exposure categories have equal design quotas; these do not represent occurrence probabilities.",
    "唯一评价参考记录集；不参与任何方法的生成、参数设置或选择": "Measured records are reserved for the common evaluation space and evaluation; they are excluded from candidate generation and selection.",
    "唯一评价参考记录集；不参与候选生成、选择PCA或序贯能量选择": "Measured records are reserved for evaluation and are not used in candidate generation, selection-space PCA, or sequential selection.",
    "三类候选路径合并去重后的完整候选空间": "Complete merged and deduplicated candidate set",
    "样本名称仅为唯一标识，不分组、不比较、不解释": "Sample names are used only as unique identifiers.",
    "按方法自身规则固定；不读取外部储藏温度记录": "Method-specific generation without measured-record feedback",
    "输出当前配置的四种对照方法": "Results are reported for the four configured comparators.",
    "SSMR-FS内部选择": "SSMR-FS selection",
    "5种方法统一评价": "Common evaluation of five methods",
}

SHEET_LABELS = {
    "命名随机流": "Named random streams",
    "场景候选路径": "Scenario profiles",
    "分层探索候选路径": "Stratified profiles",
    "风险候选路径": "Risk-exposure profiles",
    "合并去重候选集合": "Unique merged candidates",
    "储藏温度记录24维特征": "Measured record features",
    "最终温度序列": "Selected profiles",
    "最终24维特征": "Selected path features",
    "能量距离融合选择日志": "Sequential selection log",
    "来源构成": "Source composition",
    "选择方法摘要": "Selection summary",
    "候选目标空间构成": "Candidate set composition",
    "选择PCA摘要": "Selection PCA summary",
    "四项指标": "Four metrics",
    "指标排名": "Metric ranks",
    "SSMR逐指标方向比较": "Metric comparisons",
    "四种基线运行诊断": "Comparator diagnostics",
    "四种基线定义与文献": "Comparator definitions",
    "SSMR来源构成": "SSMR-FS source composition",
    "SSMR能量距离选择日志": "SSMR-FS selection log",
    "SSMR选择方法摘要": "SSMR-FS selection summary",
    "SSMR候选目标空间构成": "Candidate set composition",
    "两套PCA摘要": "Selection and evaluation PCA",
    "PCA白化坐标": "Whitened PCA coordinates",
    "PCA解释方差": "PCA explained variance",
    "PCA摘要": "PCA summary",
    "30次四项指标原始值": "Metrics by repeat",
    "各方法平均值与标准差": "Means and standard deviations",
    "四项严格占优逐次核验": "Metric ranks by repeat",
    "四项严格占优汇总": "Metric rank summary",
    "每次候选与选择PCA": "Candidate sets and PCA",
    "不同输出数量四项指标": "Metrics by design size",
    "固定基础对象摘要": "Fixed evaluation settings",
    "运行诊断": "Run diagnostics",
    "基线定义与文献": "Comparator definitions",
    "温度路径": "Temperature profiles",
    "生成规则": "Generation rules",
    "Table6": "Table6",
    "Table7": "Table7",
}

JSON_KEY_LABELS = {
    "路径覆盖误差": "PCE",
    "关键温区覆盖误差": "KCE",
    "路径间隔": "PS",
    "对数特征体积": "LFV",
    "0-4℃": "0–4 °C",
    "4-8℃": "4 < T ≤ 8 °C",
    ">8℃": ">8 °C",
    ">=15℃": "≥15 °C",
    "跨0℃": "0 °C crossings",
}


def _translate_excel_column(column: Hashable) -> Hashable:
    if not isinstance(column, str):
        return column
    if column in COLUMN_LABELS:
        return COLUMN_LABELS[column]
    day_match = re.fullmatch(r"day_(\d+)", column)
    if day_match:
        return f"Day {day_match.group(1)} temperature (°C)"
    if re.fullmatch(r"PC\d+", column):
        return column
    return column.replace("_", " ").capitalize()


def _translate_excel_value(value):
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if bool(value) else "No"
    if not isinstance(value, str):
        return value
    if value in VALUE_LABELS:
        return VALUE_LABELS[value]
    if re.fullmatch(r"\d+_measured_temperature_records", value):
        return value.replace("_", " ")
    if "|" in value:
        part_labels = {**VALUE_LABELS, "risk": "Elevated temperature"}
        return " | ".join(part_labels.get(part, part) for part in value.split("|"))
    if value.startswith("baseline_") and value.endswith("_supplement"):
        source = value[: -len("_supplement")]
        return f"{VALUE_LABELS.get(source, source)} supplementary profile"
    if value.startswith("baseline_"):
        subtype = value[len("baseline_"):]
        return f"{VALUE_LABELS.get(subtype, subtype)} baseline profile"
    return value


def _translate_sheet_name(sheet_name: str) -> str:
    if sheet_name in SHEET_LABELS:
        return SHEET_LABELS[sheet_name]
    if re.fullmatch(r"\d+_measured_temperature_records", sheet_name):
        return "Measured records"
    if re.fullmatch(r"R\d{2}_五方法路径", sheet_name):
        return sheet_name.split("_", 1)[0] + "_profiles"
    for suffix, replacement in (
        ("_温度序列", "_profiles"),
        ("_24维特征", "_24_features"),
        ("_选择日志", "_selection_log"),
    ):
        if sheet_name.endswith(suffix):
            return str(_translate_excel_value(sheet_name[:-len(suffix)])) + replacement
    if sheet_name.endswith("条储藏温度记录"):
        return "Measured records"
    if re.search(r"[\u4e00-\u9fff]", sheet_name):
        raise ValueError(f"No English sheet name defined for {sheet_name!r}")
    return sheet_name


def localize_excel_frame(frame: pd.DataFrame) -> pd.DataFrame:
    displayed = frame.copy()
    for column in displayed.columns:
        if displayed[column].dtype == object or pd.api.types.is_bool_dtype(displayed[column]):
            displayed[column] = displayed[column].map(_translate_excel_value)
    return displayed.rename(columns=_translate_excel_column)


def _localize_json(data, field=None):
    if isinstance(data, dict):
        return {JSON_KEY_LABELS.get(str(k), str(k)): _localize_json(v, str(k))
                for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        if field in {"temperature_layers", "temperature_bands"}:
            band_names = {"freezing": "Freezing", "chilled": "Chilling",
                          "transition": "Transition", "risk": "Elevated temperature",
                          "abuse": "High-temperature exposure"}
            return [[band_names.get(item[0], item[0]), *[_localize_json(v) for v in item[1:]]]
                    for item in data]
        return [_localize_json(v) for v in data]
    if isinstance(data, (bool, np.bool_)):
        return bool(data)
    return _translate_excel_value(data)


def write_excel(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            english_name = _translate_sheet_name(sheet_name)
            localize_excel_frame(frame).to_excel(writer, sheet_name=english_name[:31], index=False)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_localize_json(data), f, ensure_ascii=False, indent=2, default=str)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    localize_excel_frame(frame).to_csv(path, index=False, encoding="utf-8-sig")
