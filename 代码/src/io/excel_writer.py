import json
import re
from pathlib import Path
from typing import Dict, Hashable

import numpy as np
import pandas as pd


EXCEL_COLUMN_TRANSLATIONS = {
    # 路径基本信息
    "record_id": "记录编号",
    "method": "方法",
    "source_pool": "路径来源",
    "subtype": "路径子类型",
    "stratum": "分层组合",
    "risk_type": "风险类型",
    "key_groups": "所属关键温区",
    "n_days": "储藏天数",
    "path_key": "完整温度路径",
    "selection_order": "选择顺序",
    # 24维温度路径特征
    "storage_days": "储藏天数",
    "mean_temperature": "平均温度",
    "median_temperature": "温度中位数",
    "max_temperature": "最高温度",
    "min_temperature": "最低温度",
    "end_temperature": "终点温度",
    "days_below_0": "低于0℃天数",
    "days_0_4": "0～4℃天数",
    "days_4_8": "4～8℃天数",
    "days_above_8": "高于8℃天数",
    "days_above_equal_15": "不低于15℃天数",
    "auc_4": "4℃以上累积暴露",
    "auc_8": "8℃以上累积暴露",
    "auc_15": "15℃以上累积暴露",
    "temperature_range": "温度极差",
    "temperature_std": "温度标准差",
    "max_abs_delta": "最大相邻日绝对温差",
    "mean_abs_delta": "平均相邻日绝对温差",
    "total_abs_delta": "累计温度变化量",
    "warming_count": "升温次数",
    "cooling_count": "降温次数",
    "cross_0_count": "跨0℃次数",
    "cross_8_count": "跨8℃次数",
    "turn_count": "方向转换次数",
    # SSMR-FS候选空间与能量距离选择
    "step": "选择步骤",
    "selected_index": "候选路径索引",
    "candidate_source": "候选路径来源",
    "candidate_group": "候选路径类别",
    "record_count": "路径数量",
    "selection_role": "选择作用",
    "mean_distance_to_candidate_space": "到完整候选空间的平均距离",
    "distance_sum_to_previously_selected": "到此前已选路径的距离和",
    "sequential_energy_score": "序贯能量距离分数",
    "energy_distance_after_selection": "选择后的能量距离",
    "selection_rule": "选择规则",
    "selection_objective": "选择目标",
    "mathematical_target": "数学目标",
    "candidate_score": "候选路径评分公式",
    "first_path_rule": "首条路径选择规则",
    "subsequent_path_rule": "后续路径选择规则",
    "tie_breaking_rule": "并列分数处理规则",
    "reference_distribution": "能量距离目标空间",
    "uses_metric_feedback": "是否使用评价指标反馈",
    "uses_storage_temperature_records": "是否使用储藏温度记录",
    "candidate_path_count": "候选路径数量",
    "selected_count": "最终选择数量",
    "final_energy_distance": "最终能量距离",
    "mean_sequential_energy_score": "平均序贯能量距离分数",
    "selection_pca_fitted_on": "选择PCA拟合数据",
    "scenario": "典型场景路径数",
    "stratified": "分层探索路径数",
    "risk": "风险暴露路径数",
    # 基线定义与运行诊断
    "order": "序号",
    "family": "方法类别",
    "classic_reference": "经典文献",
    "reporting_rule": "结果报告规则",
    "requested_output_count": "要求输出数量",
    "actual_output_count": "实际输出数量",
    "discrete_mapping_supplement_count": "离散映射后补充数量",
    "implementation_note": "实现说明",
    # 评价、排名和方向比较
    "metric": "评价指标",
    "direction": "优劣方向",
    "value": "指标值",
    "rank": "排名",
    "baseline": "基线方法",
    "ssmr_value": "SSMR-FS指标值",
    "baseline_value": "基线指标值",
    "ssmr_better": "SSMR-FS是否更优",
    # PCA输出
    "component": "主成分序号",
    "explained_variance_ratio": "解释方差比例",
    "cumulative_explained_variance": "累计解释方差比例",
    "retained_for_distance": "是否用于距离计算",
    "pca_explained_variance_threshold": "PCA累计解释方差阈值",
    "pca_components": "PCA保留主成分数",
    "pca_explained_variance": "PCA累计解释方差",
    "pca_fitted_on": "PCA拟合数据",
    "coordinate_source": "坐标数据来源",
    # 关键温区
    "key_group": "关键温区组",
    "path_count": "路径数量",
    "mean_nearest_distance": "平均最近距离",
    # 重复分析
    "repeat": "重复序号",
    "repeat_count": "重复次数",
    "mean": "平均值",
    "standard_deviation": "标准差",
    "mean_plus_minus_sd": "平均值±标准差",
    "pce_best": "路径覆盖误差是否严格占优",
    "kce_best": "关键温区覆盖误差是否严格占优",
    "ps_best": "路径间隔是否严格占优",
    "lfv_best": "对数特征体积是否严格占优",
    "simultaneously_best": "四项指标是否均严格占优",
    "simultaneous_best_count": "四项指标均严格占优次数",
    "simultaneous_best_proportion": "四项指标均严格占优比例",
    "strict_rule": "严格判定规则",
    # 不同输出数量
    "sample_size": "输出数量N",
    "count": "数量",
    "proportion": "比例",
    "max_sample_size": "最大输出数量",
    "ssmr_unique_candidate_pool_size": "SSMR-FS唯一候选路径数量",
    "ssmr_energy_reference_space": "SSMR-FS能量距离目标空间",
    "selection_pca_components": "选择PCA主成分数",
    "selection_pca_explained_variance": "选择PCA累计解释方差",
    "ssmr_selection_objective": "SSMR-FS选择目标",
    "evaluation_pca_components": "评价PCA主成分数",
    "evaluation_pca_explained_variance": "评价PCA累计解释方差",
    "storage_temperature_record_count": "储藏温度记录数量",
    "storage_temperature_point_count": "储藏温度点数量",
    # 其他可能写入表格的审计字段
    "evaluation_role": "评价数据作用",
    "sample_id_rule": "样品编号规则",
    "temperature_point_count": "温度点数量",
    "path_length_counts": "路径长度分布",
    "observed_min_temperature": "记录最低温度",
    "observed_max_temperature": "记录最高温度",
    "nominal_design_min_temperature": "名义设计最低温度",
    "nominal_design_max_temperature": "名义设计最高温度",
    "in_nominal_range_path_count": "名义范围内路径数量",
    "outside_nominal_range_path_count": "含超范围值路径数量",
    "outside_nominal_range_point_count": "超范围温度点数量",
    "outside_nominal_range_point_percentage": "超范围温度点比例",
    "below_nominal_minimum_path_count": "低于名义下限路径数量",
    "above_nominal_maximum_path_count": "高于名义上限路径数量",
    "key_group_counts": "关键温区路径数量",
    "variable": "变量",
}


EXCEL_VALUE_TRANSLATIONS = {
    # 指标方向与布尔含义
    "lower": "越小越好",
    "higher": "越大越好",
    # 三类候选来源
    "scenario": "典型场景候选",
    "stratified": "分层探索候选",
    "risk": "风险暴露候选",
    "storage_temperature_records": "储藏温度记录",
    # 典型储藏场景
    "stable_frozen": "稳定冷冻",
    "stable_chilled": "稳定冷藏",
    "slow_warming": "缓慢升温",
    "fast_warming": "快速升温",
    "recooling": "降温复冷",
    "high_frequency_fluctuation": "高频温度波动",
    "low_frequency_fluctuation": "低频温度波动",
    "late_warming": "后期升温",
    "cross_zero": "跨0℃转换",
    "temperature_abuse": "温度滥用",
    # 分层探索组成
    "1-3 d": "1～3天",
    "4-6 d": "4～6天",
    "7-10 d": "7～10天",
    "freezing": "冻结温区",
    "chilled": "冷藏温区",
    "transition": "过渡温区",
    "abuse": "温度滥用区",
    "stable": "稳定",
    "warming": "升温",
    "cooling": "降温",
    "oscillating": "往复波动",
    "low": "低波动",
    "medium": "中波动",
    "high": "高波动",
    "stratified_exploration": "分层探索",
    # 风险暴露类别
    "risk_exposure": "风险暴露",
    "cold_boundary": "冷藏边界",
    "transition_4_8": "4～8℃过渡",
    "above_8_risk": "高于8℃暴露",
    "above_15_abuse": "不低于15℃温度滥用",
    "freeze_thaw": "冻融转换",
    "storage_temperature_record": "储藏温度记录",
    # SSMR-FS选择规则
    "candidate_constrained_sequential_energy_distance": "候选约束的序贯能量距离",
    "minimum_sequential_energy_distance": "最小序贯能量距离",
    "greedy minimization of energy distance between selected paths and the merged unique candidate space": "逐步最小化所选路径与合并去重候选空间之间的能量距离",
    "mean_distance_to_candidate_space - distance_sum_to_previous_selected / current_selected_size": "到完整候选空间的平均距离－到此前已选路径的距离和÷当前选择数量",
    # 直接生成型基线来源与实现
    "baseline_random": "简单随机基线",
    "baseline_sobol": "Sobol基线",
    "baseline_oa_lhs": "OA-LHS基线",
    "baseline_maxpro": "MaxPro基线",
    "simple_random": "简单随机",
    "scrambled_sobol_power2_prefix": "加扰Sobol二次幂点集前缀",
    "strength2_oa_lhs_derived_subset": "由强度2 OA-LHS派生的子集",
    "coordinate_exchange_maxpro_lhs": "坐标交换MaxPro-LHS",
    "simple_random_supplement_after_discrete_mapping": "离散映射去重后的简单随机补充",
}


def _translate_excel_column(column: Hashable) -> Hashable:
    """只转换Excel展示表头，不改变程序内部DataFrame字段。"""

    if not isinstance(column, str):
        return column
    if column in EXCEL_COLUMN_TRANSLATIONS:
        return EXCEL_COLUMN_TRANSLATIONS[column]
    day_match = re.fullmatch(r"day_(\d+)", column)
    if day_match:
        return f"第{day_match.group(1)}天温度"
    pc_match = re.fullmatch(r"PC(\d+)", column)
    if pc_match:
        return f"主成分{pc_match.group(1)}"
    return column


def _translate_excel_value(value):
    """转换Excel中的英文类别标签；数值及计算结果保持原样。"""

    if isinstance(value, (bool, np.bool_)):
        return "是" if bool(value) else "否"
    if not isinstance(value, str):
        return value
    if value in EXCEL_VALUE_TRANSLATIONS:
        return EXCEL_VALUE_TRANSLATIONS[value]
    if "|" in value:
        compound_translations = {
            **EXCEL_VALUE_TRANSLATIONS,
            "risk": "风险温区",
        }
        translated_parts = [
            compound_translations.get(part, part) for part in value.split("|")
        ]
        return "｜".join(translated_parts)
    if value.startswith("baseline_") and value.endswith("_supplement"):
        original_source = value[: -len("_supplement")]
        source_name = EXCEL_VALUE_TRANSLATIONS.get(original_source, original_source)
        return f"{source_name}补充路径"
    if value.startswith("baseline_"):
        subtype = value[len("baseline_") :]
        subtype_name = EXCEL_VALUE_TRANSLATIONS.get(subtype, subtype)
        return f"{subtype_name}基线路径"
    return value


def localize_excel_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """返回仅用于Excel输出的中文展示副本。"""

    localized = frame.copy()
    for column in localized.columns:
        if localized[column].dtype == object or pd.api.types.is_bool_dtype(
            localized[column]
        ):
            localized[column] = localized[column].map(_translate_excel_value)
    return localized.rename(columns=_translate_excel_column)


def write_excel(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            safe_name = sheet_name[:31]
            localize_excel_frame(frame).to_excel(
                writer,
                sheet_name=safe_name,
                index=False,
            )


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
