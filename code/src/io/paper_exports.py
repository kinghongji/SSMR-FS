"""Manuscript tables and single-sheet S1–S16 computed from the main run."""
import json
import shutil
import numpy as np
import pandas as pd
from config.experiment_config import METHOD_ORDER
from src.features.feature_extraction import records_to_feature_frame
from src.generation.rules import generation_rules
from src.io.excel_writer import write_excel, write_json
from src.utils.random_utils import records_to_temperature_frame

FEATURE_ROWS = [
 ("Storage duration (d)", "storage_days"), ("Mean temperature (°C)", "mean_temperature"),
 ("Maximum temperature (°C)", "max_temperature"), ("Exposure above 8 °C (d)", "days_above_8"),
 ("AUC8 (°C·d)", "auc_8"), ("Total temperature variation (°C)", "total_abs_delta"),
 ("0 °C crossings (count)", "cross_0_count"), ("Direction changes (count)", "turn_count"),
]

def export_submission_tables(config, records_by_method, ssmr_result, reference, metrics, baseline_diagnostics=None):
    out = config.result_root / "10_manuscript_tables_and_supplements"
    rules = generation_rules(config)
    write_json(out / "generation_and_comparator_rules.json", rules)
    frames = {"Reference": records_to_feature_frame(reference.records)}
    frames.update({m: records_to_feature_frame(records_by_method[m]) for m in METHOD_ORDER})
    rows = []
    for label, key in FEATURE_ROWS:
        row = {"Feature": label}
        for method, frame in frames.items():
            v = frame[key].to_numpy(dtype=float)
            row[method] = f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}"
        rows.append(row)
    write_excel(out / "Table5_descriptive_statistics_and_Table6_metrics.xlsx",
                {"Table5": pd.DataFrame(rows), "Table6": metrics})
    b = ssmr_result.pool_bundle
    for path_number, rule_number, name, records, rule in [
        (1, 2, "typical_storage_scenarios", b.scenario_pool, rules["scenario"]),
        (3, 4, "stratified_exploration", b.stratified_pool, [rules["stratified"]]),
        (5, 6, "risk_exposure", b.risk_pool, rules["risk"]),
    ]:
        display_rows = [dict(row) for row in rule]
        if name == "stratified_exploration":
            band_names = {
                "freezing": "Freezing",
                "chilled": "Chilling",
                "transition": "Transition",
                "risk": "Elevated temperature",
                "abuse": "High-temperature exposure",
            }
            for display_row in display_rows:
                display_row["target_temperature_bands"] = [
                    {"band": band_names[label], "lower_C": low, "upper_C": high}
                    for label, low, high in display_row.pop("temperature_bands")
                ]
        normalized = [
            {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
             for k, v in row.items()}
            for row in display_rows
        ]
        write_excel(out / f"S{path_number}_{name}_profiles.xlsx",
                    {"Temperature profiles": records_to_temperature_frame(records)})
        write_excel(out / f"S{rule_number}_{name}_generation_rules.xlsx",
                    {"Generation rules": pd.DataFrame(normalized)})
    write_excel(out / "S7_SSMR-FS_selected_profiles.xlsx",
                {"Temperature profiles": records_to_temperature_frame(ssmr_result.selected_records)})
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.storage_temperature_records.file_path,
                 out / "S8_measured_temperature_records.xlsx")
    for path_number, rule_number, method, label in [
        (9, 10, "简单随机", "SRS"),
        (11, 12, "Sobol", "Sobol"),
        (13, 14, "OA-LHS", "OA-LHS"),
        (15, 16, "MaxPro", "MaxPro"),
    ]:
        write_excel(out / f"S{path_number}_{label}_profiles.xlsx",
                    {"Temperature profiles": records_to_temperature_frame(records_by_method[method])})
        write_excel(out / f"S{rule_number}_{label}_generation_rules.xlsx",
                    {"Generation rules": pd.DataFrame([{
                        "method": method,
                        "rule": rules["baselines"][label],
                        "mapping": ("direct discrete uniform sampling of n and daily temperatures"
                                    if method == "简单随机" else rules["baselines"]["mapping"]),
                        "dedup_and_supplement": ("reject exact duplicate full paths and redraw until N"
                                                 if method == "简单随机" else rules["baselines"]["dedup_and_supplement"]),
                    }])})
    ssmr = metrics[metrics["method"]=="SSMR-FS"].iloc[0]
    comparisons = []
    for _, row in metrics[metrics["method"]!="SSMR-FS"].iterrows():
        comparisons.append({"baseline":row["method"],
            "PCE_reduction_percent":100*(row["路径覆盖误差"]-ssmr["路径覆盖误差"])/row["路径覆盖误差"],
            "KCE_reduction_percent":100*(row["关键温区覆盖误差"]-ssmr["关键温区覆盖误差"])/row["关键温区覆盖误差"],
            "PS_ratio":ssmr["路径间隔"]/row["路径间隔"] if row["路径间隔"] else None})
    write_json(out / "abstract_comparison_ratios.json", comparisons)
