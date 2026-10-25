from pathlib import Path
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config.experiment_config import (
    CODE_ROOT,
    BASELINE_FAMILIES,
    BASELINE_METHODS,
    METRIC_DIRECTIONS,
    METHOD_ORDER,
    ExperimentConfig,
    config_to_dict,
    ensure_result_dirs,
    get_result_dir,
)
from src.evaluation.evaluation_sets import (
    StorageTemperatureRecordSet,
    load_storage_temperature_records,
)
from src.evaluation.metrics import evaluate_selected_design, metrics_to_frame, rank_metrics
from src.evaluation.statistics import (
    count_ssmr_simultaneous_best,
    summarize_repeated_metrics,
)
from src.features.feature_extraction import records_to_feature_frame
from src.features.preprocessing import FeatureTransformer
from src.generation.baselines import (
    BaselineSuiteResult,
    generate_baseline_suite,
)
from src.io.excel_writer import write_excel, write_json, localize_excel_frame
from src.plotting.figures import (
    plot_design_quality_dimensions,
    plot_evaluation_space_distribution,
    plot_method_temperature_heatmaps,
    plot_pca_cumulative_variance,
    plot_repeated_stability,
    plot_sample_size_metrics,
    plot_storage_temperature_heatmap,
)
from src.selection.ssmr_fs import SSMRSelectionResult, build_ssmr_candidate_pool, select_ssmr_fs
from src.utils.random_utils import records_to_temperature_frame
from src.utils.seed_schedule import named_rng, stream_seed, seed_manifest
from src.generation.rules import generation_rules
from src.generation.constraints import validate_candidate_bundle
from src.io.paper_exports import export_submission_tables
from src.utils.validation import validate_records


BASELINE_REFERENCE_ROWS = (
    ("简单随机", "Robert and Casella, Monte Carlo Statistical Methods, 2004"),
    ("Sobol", "Sobol, USSR Computational Mathematics and Mathematical Physics, 1967"),
    ("OA-LHS", "Tang, Journal of the American Statistical Association, 1993"),
    ("MaxPro", "Joseph, Gul and Ba, Biometrika, 2015"),
)


def _rng(config, scope, method, repeat=0, sample_size=0):
    return named_rng(config.random_seed, scope, method, repeat, sample_size)


def _baseline_suite(config, n_samples, scope="main", repeat=0, sample_size=0):
    result = generate_baseline_suite(
        config, None, n_samples,
        method_rngs={m: _rng(config, scope, m, repeat, sample_size) for m in BASELINE_METHODS},
    )
    for row in result.diagnostics:
        row.update({"master_seed": config.random_seed, "scope": scope, "repeat": repeat,
                    "stream_sample_size": sample_size,
                    "stream_seed": stream_seed(config.random_seed, scope, row["method"], repeat, sample_size)})
    return result


def _result_path_for_log(config: ExperimentConfig, result_path: Path) -> str:
    return Path(os.path.relpath(result_path, CODE_ROOT)).as_posix()


def _baseline_definition_frame() -> pd.DataFrame:
    rows = []
    for order, (method, reference) in enumerate(BASELINE_REFERENCE_ROWS, start=1):
        rows.append(
            {
                "order": order,
                "method": method,
                "family": BASELINE_FAMILIES[method],
                "classic_reference": reference,
                "selection_rule": "按方法自身规则固定；不读取外部储藏温度记录",
                "reporting_rule": "输出当前配置的四种对照方法",
            }
        )
    return pd.DataFrame(rows)


def initialize_outputs(config: ExperimentConfig) -> None:
    ensure_result_dirs(config)
    config_dir = get_result_dir(config, "config")
    write_json(config_dir / "run_configuration.json", config_to_dict(config))
    write_json(config_dir / "random_stream_manifest.json", seed_manifest(config))
    write_excel(config_dir / "random_stream_manifest.xlsx", {"Named random streams": pd.DataFrame(seed_manifest(config))})
    write_json(config_dir / "temperature_profile_generation_rules.json", generation_rules(config))


def _fit_evaluation_transformer(
    config: ExperimentConfig,
    evaluation_set: StorageTemperatureRecordSet,
) -> Tuple[FeatureTransformer, np.ndarray]:
    # 评价PCA只由当前载入的储藏温度记录拟合，不反馈给任何设计方法。
    transformer = FeatureTransformer(config.features.pca_explained_variance)
    transformer.fit(evaluation_set.records)
    evaluation_z = transformer.transform(evaluation_set.records)
    return transformer, evaluation_z


def _validate_method_records(
    config: ExperimentConfig,
    records_by_method: Dict[str, List[dict]],
) -> None:
    missing = [method for method in METHOD_ORDER if method not in records_by_method]
    if missing:
        raise RuntimeError(f"完整比较遗漏方法：{missing}")
    expected_count = len(next(iter(records_by_method.values())))
    for method in METHOD_ORDER:
        records = records_by_method[method]
        if len(records) != expected_count:
            raise RuntimeError(
                f"{method}输出{len(records)}条，与统一输出数量{expected_count}不一致。"
            )
        for record in records:
            record["method"] = method
        validate_records(
            records,
            config.temperature.temp_min,
            config.temperature.temp_max,
            config.temperature.min_days,
            config.temperature.max_days,
        )


def _generate_all_methods(
    config: ExperimentConfig,
    n_samples: int,
    seed_scope: str = "main",
    repeat_index: int = 0,
    logger=None,
) -> Tuple[Dict[str, List[dict]], SSMRSelectionResult, BaselineSuiteResult]:
    if logger:
        logger.info(
            f"Generating {len(METHOD_ORDER)} designs at N={n_samples}."
        )
    baseline_result = _baseline_suite(config, n_samples, seed_scope, repeat_index)
    ssmr_result = select_ssmr_fs(
        config,
        _rng(config, seed_scope, "SSMR-FS", repeat_index),
        n_samples,
    )
    validate_candidate_bundle(config, ssmr_result.pool_bundle)
    records_by_method = {
        **baseline_result.records_by_method,
        "SSMR-FS": ssmr_result.selected_records,
    }
    records_by_method = {method: records_by_method[method] for method in METHOD_ORDER}
    _validate_method_records(config, records_by_method)
    return records_by_method, ssmr_result, baseline_result


def _evaluate_methods(
    config: ExperimentConfig,
    records_by_method: Dict[str, List[dict]],
    evaluation_set: StorageTemperatureRecordSet,
    transformer: FeatureTransformer,
    evaluation_z: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, pd.DataFrame]], Dict[str, np.ndarray]]:
    metrics_by_method = {}
    group_tables_by_method: Dict[str, Dict[str, pd.DataFrame]] = {}
    transformed_by_method = {}
    for method in METHOD_ORDER:
        selected_z = transformer.transform(records_by_method[method])
        transformed_by_method[method] = selected_z
        metrics, group_tables = evaluate_selected_design(
            selected_z,
            evaluation_z,
            evaluation_set,
            regularization=config.features.logdet_regularization,
        )
        metrics_by_method[method] = metrics
        group_tables_by_method[method] = group_tables
    return metrics_to_frame(metrics_by_method), group_tables_by_method, transformed_by_method


def _main_directional_comparison(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    """逐项列出SSMR-FS与四种对照方法的指标比较结果。"""

    rows = []
    for metric, direction in METRIC_DIRECTIONS.items():
        ssmr_value = float(
            metrics_frame.loc[metrics_frame["method"] == "SSMR-FS", metric].iloc[0]
        )
        for baseline in BASELINE_METHODS:
            baseline_value = float(
                metrics_frame.loc[metrics_frame["method"] == baseline, metric].iloc[0]
            )
            rows.append(
                {
                    "metric": metric,
                    "direction": "lower" if direction == "lower" else "higher",
                    "baseline": baseline,
                    "ssmr_value": ssmr_value,
                    "baseline_value": baseline_value,
                    "ssmr_better": (
                        ssmr_value < baseline_value
                        if direction == "lower"
                        else ssmr_value > baseline_value
                    ),
                }
            )
    return pd.DataFrame(rows)


def _save_design_tables(
    config: ExperimentConfig,
    records_by_method: Dict[str, List[dict]],
    output_path: Path,
) -> None:
    sheets = {}
    for method in METHOD_ORDER:
        if method not in records_by_method:
            continue
        records = records_by_method[method]
        sheets[f"{method}_温度序列"] = records_to_temperature_frame(
            records,
            config.temperature.max_days,
        )
        sheets[f"{method}_24维特征"] = records_to_feature_frame(records)
    write_excel(output_path, sheets)


def _save_pca_outputs(
    records_by_method: Dict[str, List[dict]],
    transformed_by_method: Dict[str, np.ndarray],
    evaluation_set: StorageTemperatureRecordSet,
    evaluation_z: np.ndarray,
    transformer: FeatureTransformer,
    output_path: Path,
) -> None:
    record_label = evaluation_set.label
    if (
        transformer.explained_variance_ratio_ is None
        or transformer.cumulative_explained_variance_ is None
    ):
        raise RuntimeError("评价PCA尚未拟合，无法输出坐标。")
    metadata_columns = [
        "record_id",
        "method",
        "source_pool",
        "subtype",
        "stratum",
        "risk_type",
        "path_key",
        "storage_days",
    ]
    coordinate_frames = []
    for method in METHOD_ORDER:
        feature_frame = records_to_feature_frame(records_by_method[method])
        metadata_frame = feature_frame[metadata_columns].reset_index(drop=True)
        pc_values = transformed_by_method[method]
        pc_frame = pd.DataFrame(
            pc_values,
            columns=[f"PC{i}" for i in range(1, pc_values.shape[1] + 1)],
        )
        coordinate_frames.append(pd.concat([metadata_frame, pc_frame], axis=1))

    record_metadata = records_to_feature_frame(evaluation_set.records)[
        metadata_columns
    ].reset_index(drop=True)
    record_metadata["record_id"] = [record["record_id"] for record in evaluation_set.records]
    record_pc_frame = pd.DataFrame(
        evaluation_z,
        columns=[f"PC{i}" for i in range(1, evaluation_z.shape[1] + 1)],
    )
    coordinate_frames.append(pd.concat([record_metadata, record_pc_frame], axis=1))
    variance_frame = pd.DataFrame(
        {
            "component": np.arange(1, len(transformer.explained_variance_ratio_) + 1),
            "explained_variance_ratio": transformer.explained_variance_ratio_,
            "cumulative_explained_variance": transformer.cumulative_explained_variance_,
        }
    )
    variance_frame["retained_for_distance"] = (
        variance_frame["component"] <= int(transformer.n_components_)
    )
    summary_frame = pd.DataFrame(
        [
            {
                "pca_explained_variance_threshold": transformer.explained_variance_threshold,
                "pca_components": transformer.n_components_,
                "pca_explained_variance": transformer.explained_variance_ratio_sum_,
                "pca_fitted_on": record_label,
                "coordinate_source": f"{record_label} and {len(METHOD_ORDER)} design methods",
            }
        ]
    )
    write_excel(
        output_path,
        {
            "PCA白化坐标": pd.concat(coordinate_frames, ignore_index=True),
            "PCA解释方差": variance_frame,
            "PCA摘要": summary_frame,
        },
    )


def save_candidate_pool_outputs(
    config: ExperimentConfig,
    ssmr_result: SSMRSelectionResult,
) -> None:
    pool_dir = get_result_dir(config, "candidate_pools")
    sheets = {
        "场景候选路径": records_to_temperature_frame(
            ssmr_result.pool_bundle.scenario_pool,
            config.temperature.max_days,
        ),
        "分层探索候选路径": records_to_temperature_frame(
            ssmr_result.pool_bundle.stratified_pool,
            config.temperature.max_days,
        ),
        "风险候选路径": records_to_temperature_frame(
            ssmr_result.pool_bundle.risk_pool,
            config.temperature.max_days,
        ),
        "合并去重候选集合": records_to_temperature_frame(
            ssmr_result.pool_bundle.candidate_pool,
            config.temperature.max_days,
        ),
    }
    write_excel(pool_dir / "candidate_profiles_and_merged_set.xlsx", sheets)
    write_json(
        pool_dir / "candidate_pool_summary.json",
        {
            "scenario_pool_raw": len(ssmr_result.pool_bundle.scenario_pool),
            "stratified_pool_raw": len(ssmr_result.pool_bundle.stratified_pool),
            "risk_pool_raw": len(ssmr_result.pool_bundle.risk_pool),
            "merged_unique_candidate_pool": len(ssmr_result.pool_bundle.candidate_pool),
            "risk_generation_rule": "五类风险路径等量生成，不表示实际发生概率",
            "source_counts_in_selected": ssmr_result.source_counts,
        },
    )


def save_storage_temperature_record_outputs(
    config: ExperimentConfig,
    evaluation_set: StorageTemperatureRecordSet,
) -> None:
    eval_dir = get_result_dir(config, "evaluation_sets")
    record_label = evaluation_set.label
    sheets = {
        record_label: records_to_temperature_frame(
            evaluation_set.records,
            config.temperature.max_days,
        ),
        "储藏温度记录24维特征": records_to_feature_frame(evaluation_set.records),
    }
    write_excel(eval_dir / f"{record_label}_and_evaluation_features.xlsx", sheets)
    write_json(
        eval_dir / f"{record_label}_summary.json",
        {
            "evaluation_role": "唯一评价参考记录集；不参与任何方法的生成、参数设置或选择",
            "ssmr_internal_reference": "三类候选路径合并去重后的完整候选空间",
            "storage_temperature_records": evaluation_set.audit_summary,
        },
    )


def _ssmr_workbook_sheets(
    config: ExperimentConfig,
    ssmr_result: SSMRSelectionResult,
) -> Dict[str, pd.DataFrame]:
    return {
        "最终温度序列": records_to_temperature_frame(
            ssmr_result.selected_records,
            config.temperature.max_days,
        ),
        "最终24维特征": records_to_feature_frame(ssmr_result.selected_records),
        "能量距离融合选择日志": pd.DataFrame(ssmr_result.selection_log),
        "来源构成": pd.DataFrame([ssmr_result.source_counts]),
        "选择方法摘要": pd.DataFrame([ssmr_result.selection_summary]),
        "候选目标空间构成": pd.DataFrame(ssmr_result.candidate_group_summary),
        "选择PCA摘要": pd.DataFrame(
            [
                {
                    "拟合数据": "Complete merged and deduplicated candidate set",
                    "是否加权": False,
                    "是否使用储藏温度记录": False,
                    "主成分数": ssmr_result.transformer.n_components_,
                    "累计解释方差": ssmr_result.transformer.explained_variance_ratio_sum_,
                }
            ]
        ),
    }


def run_main_comparison(config: ExperimentConfig, logger=None) -> pd.DataFrame:
    initialize_outputs(config)
    if logger:
        logger.info(f"Running main comparison at N={config.main_sample_size}.")
    records_by_method, ssmr_result, baseline_result = _generate_all_methods(
        config,
        config.main_sample_size,
        seed_scope="main",
        logger=logger,
    )
    evaluation_set = load_storage_temperature_records(config)
    record_label = evaluation_set.label
    transformer, evaluation_z = _fit_evaluation_transformer(config, evaluation_set)
    metrics_frame, _, transformed_by_method = _evaluate_methods(
        config,
        records_by_method,
        evaluation_set,
        transformer,
        evaluation_z,
    )
    rank_frame = rank_metrics(metrics_frame)
    directional_frame = _main_directional_comparison(metrics_frame)

    main_dir = get_result_dir(config, "main")
    figure_dir = get_result_dir(config, "figures")
    ssmr_dir = get_result_dir(config, "ssmr")

    _save_design_tables(
        config,
        records_by_method,
        main_dir / "N300_method_profiles_and_24_features.xlsx",
    )
    _save_pca_outputs(
        records_by_method,
        transformed_by_method,
        evaluation_set,
        evaluation_z,
        transformer,
        main_dir / "evaluation_PCA_coordinates_and_variance.xlsx",
    )
    save_candidate_pool_outputs(config, ssmr_result)
    save_storage_temperature_record_outputs(config, evaluation_set)
    export_submission_tables(config, records_by_method, ssmr_result, evaluation_set, metrics_frame, baseline_result.diagnostics)
    write_excel(
        main_dir / config.excel_main_name,
        {
            "四项指标": metrics_frame,
            "指标排名": rank_frame,
            "SSMR逐指标方向比较": directional_frame,
            "四种基线运行诊断": pd.DataFrame(baseline_result.diagnostics),
            "四种基线定义与文献": _baseline_definition_frame(),
            "SSMR来源构成": pd.DataFrame([ssmr_result.source_counts]),
            "SSMR能量距离选择日志": pd.DataFrame(ssmr_result.selection_log),
            "SSMR选择方法摘要": pd.DataFrame([ssmr_result.selection_summary]),
            "SSMR候选目标空间构成": pd.DataFrame(
                ssmr_result.candidate_group_summary
            ),
            "两套PCA摘要": pd.DataFrame(
                [
                    {
                        "PCA用途": "SSMR-FS内部选择",
                        "拟合数据": "Complete merged and deduplicated candidate set",
                        "是否加权": False,
                        "主成分数": ssmr_result.transformer.n_components_,
                        "累计解释方差": ssmr_result.transformer.explained_variance_ratio_sum_,
                    },
                    {
                        "PCA用途": f"Common evaluation of {len(METHOD_ORDER)} methods",
                        "拟合数据": record_label,
                        "是否加权": False,
                        "主成分数": transformer.n_components_,
                        "累计解释方差": transformer.explained_variance_ratio_sum_,
                    },
                ]
            ),
        },
    )
    write_json(
        main_dir / "N300_main_comparison_summary.json",
        {
            "method_count": len(METHOD_ORDER),
            "baseline_count": len(BASELINE_METHODS),
            "metrics": metrics_frame.to_dict("records"),
            "ssmr_directional_comparison": directional_frame.to_dict("records"),
            "baseline_diagnostics": baseline_result.diagnostics,
            "reporting_rule": "输出当前配置的四种对照方法",
            "ssmr_source_counts": ssmr_result.source_counts,
            "ssmr_selection": ssmr_result.selection_summary,
            "selection_pca_components": ssmr_result.transformer.n_components_,
            "selection_pca_explained_variance": (
                ssmr_result.transformer.explained_variance_ratio_sum_
            ),
            "evaluation_pca_components": transformer.n_components_,
            "evaluation_pca_explained_variance": transformer.explained_variance_ratio_sum_,
            "storage_temperature_records": evaluation_set.audit_summary,
        },
    )
    write_excel(
        ssmr_dir / "SSMR-FS_sequential_selection.xlsx",
        _ssmr_workbook_sheets(config, ssmr_result),
    )

    plot_storage_temperature_heatmap(
        evaluation_set.records,
        figure_dir / f"Figure2_{len(evaluation_set.records)}_measured_temperature_records",
        config.figure_dpi,
        config.temperature.max_days,
    )
    plot_method_temperature_heatmaps(
        records_by_method,
        figure_dir / "Figure3_N300_method_temperature_profiles",
        config.figure_dpi,
        config.temperature.max_days,
    )
    plot_evaluation_space_distribution(
        evaluation_z,
        transformed_by_method,
        transformer.explained_variance_ratio_,
        figure_dir / "Figure5_common_evaluation_space",
        config.figure_dpi,
    )
    plot_pca_cumulative_variance(
        transformer.cumulative_explained_variance_,
        int(transformer.n_components_),
        float(transformer.explained_variance_ratio_sum_),
        figure_dir / "Figure4_evaluation_PCA_explained_variance",
        config.figure_dpi,
    )
    plot_design_quality_dimensions(
        metrics_frame,
        figure_dir / "Figure6_N300_coverage_and_dispersion",
        config.figure_dpi,
    )
    if logger:
        logger.info(
            f"Main comparison completed: {_result_path_for_log(config, main_dir)}"
        )
        logger.info(localize_excel_frame(metrics_frame).to_string(index=False))
    return metrics_frame


def run_repeated_analysis(config: ExperimentConfig, logger=None) -> pd.DataFrame:
    initialize_outputs(config)
    if logger:
        logger.info(f"Running {config.repeated_times} repeated comparisons.")
    evaluation_set = load_storage_temperature_records(config)
    transformer, evaluation_z = _fit_evaluation_transformer(config, evaluation_set)
    repeat_rows = []
    repeat_candidate_rows = []
    repeat_temperature_sheets = {}
    baseline_diagnostic_rows = []
    for repeat_index in range(1, config.repeated_times + 1):
        if logger:
            logger.info(f"Repeat {repeat_index}/{config.repeated_times}")
        records_by_method, repeated_ssmr, baseline_result = _generate_all_methods(
            config,
            config.main_sample_size,
            seed_scope="repeat",
            repeat_index=repeat_index,
            logger=None,
        )
        metrics_frame, _, _ = _evaluate_methods(
            config,
            records_by_method,
            evaluation_set,
            transformer,
            evaluation_z,
        )
        all_repeat_records = [record for method in METHOD_ORDER for record in records_by_method[method]]
        repeat_temperature_sheets[f"R{repeat_index:02d}_五方法路径"] = records_to_temperature_frame(all_repeat_records)
        metrics_frame["repeat"] = repeat_index
        repeat_rows.extend(metrics_frame.to_dict("records"))
        repeat_candidate_rows.append({"repeat": repeat_index,
            "stream_seed": stream_seed(config.random_seed, "repeat", "SSMR-FS", repeat_index),
            "candidate_path_count": len(repeated_ssmr.pool_bundle.candidate_pool),
            "selection_pca_components": repeated_ssmr.transformer.n_components_,
            "selection_pca_explained_variance": repeated_ssmr.transformer.explained_variance_ratio_sum_})
        for row in baseline_result.diagnostics:
            baseline_diagnostic_rows.append({"repeat": repeat_index, **row})

    repeat_frame = pd.DataFrame(repeat_rows)
    repeat_summary_frame = summarize_repeated_metrics(repeat_frame)
    best_detail_frame, best_summary_frame = count_ssmr_simultaneous_best(
        repeat_frame,
        BASELINE_METHODS,
    )

    repeat_dir = get_result_dir(config, "repeat")
    figure_dir = get_result_dir(config, "figures")
    repeat_sheets = {
        "30次四项指标原始值": repeat_frame,
        "各方法平均值与标准差": repeat_summary_frame,
        "四项严格占优逐次核验": best_detail_frame,
        "四项严格占优汇总": best_summary_frame,
        "四种基线运行诊断": pd.DataFrame(baseline_diagnostic_rows),
        "每次候选与选择PCA": pd.DataFrame(repeat_candidate_rows),
    }
    write_excel(repeat_dir / config.excel_repeat_name, repeat_sheets)
    write_excel(repeat_dir / "N300_profiles_by_repeat.xlsx", repeat_temperature_sheets)
    write_json(
        repeat_dir / "N300_repeated_runs_summary.json",
        {
            "method_count": len(METHOD_ORDER),
            "candidate_regeneration_audit": repeat_candidate_rows,
            "mean_and_standard_deviation": repeat_summary_frame.to_dict("records"),
            "ssmr_simultaneous_best": best_summary_frame.to_dict("records")[0],
            "statistical_evidence": (
                "Means and sample standard deviations across runs; counts and proportions "
                "of runs where SSMR-FS is strictly best on all four metrics."
            ),
        },
    )
    plot_repeated_stability(
        repeat_frame,
        figure_dir / "Figure7_N300_repeated_runs",
        config.figure_dpi,
    )
    if logger:
        logger.info(
            "Repeated comparisons completed: "
            f"{_result_path_for_log(config, repeat_dir)}"
        )
    return repeat_frame


def run_sample_size_validation(config: ExperimentConfig, logger=None) -> pd.DataFrame:
    initialize_outputs(config)
    if logger:
        logger.info("Running comparisons across design sizes.")
    max_sample_size = max(config.sample_size_validation)
    ssmr_pool_bundle = build_ssmr_candidate_pool(config, _rng(config, "main", "SSMR-FS"))
    validate_candidate_bundle(config, ssmr_pool_bundle)
    ssmr_full_result = select_ssmr_fs(
        config,
        _rng(config, "main", "SSMR-FS"),
        max_sample_size,
        pool_bundle=ssmr_pool_bundle,
    )
    evaluation_set = load_storage_temperature_records(config)
    transformer, evaluation_z = _fit_evaluation_transformer(config, evaluation_set)

    metric_rows = []
    directional_comparison_rows = []
    baseline_diagnostic_rows = []
    sample_temperature_sheets = {}
    sample_feature_sheets = {}
    ssmr_source_rows = []
    ssmr_selection_log_sheets = {}
    for sample_size in config.sample_size_validation:
        if logger:
            logger.info(f"Current design size N={sample_size}")
        # 每个N使用同一基础随机种子；四种对照方法均按各自N相关规则构造。
        baseline_result = _baseline_suite(config, sample_size, "size", sample_size=sample_size)
        ssmr_result = ssmr_full_result.prefix(sample_size)
        records_by_method = {
            **baseline_result.records_by_method,
            "SSMR-FS": ssmr_result.selected_records,
        }
        records_by_method = {method: records_by_method[method] for method in METHOD_ORDER}
        _validate_method_records(config, records_by_method)

        all_records_for_n = []
        for method, method_records in records_by_method.items():
            for record in method_records:
                all_records_for_n.append({**record, "method": method})
        sample_temperature_sheets[f"N{sample_size}_温度序列"] = records_to_temperature_frame(
            all_records_for_n,
            config.temperature.max_days,
        )
        sample_feature_sheets[f"N{sample_size}_24维特征"] = records_to_feature_frame(
            all_records_for_n
        )
        for row in baseline_result.diagnostics:
            baseline_diagnostic_rows.append({"sample_size": sample_size, **row})
        for source_name, count in ssmr_result.source_counts.items():
            ssmr_source_rows.append(
                {
                    "sample_size": sample_size,
                    "source_pool": source_name,
                    "count": count,
                    "proportion": count / sample_size,
                }
            )
        selection_log = pd.DataFrame(ssmr_result.selection_log)
        selection_log["sample_size"] = sample_size
        ssmr_selection_log_sheets[f"N{sample_size}_选择日志"] = selection_log

        metrics_frame, _, _ = _evaluate_methods(
            config,
            records_by_method,
            evaluation_set,
            transformer,
            evaluation_z,
        )
        metrics_frame["sample_size"] = sample_size
        metric_rows.extend(metrics_frame.to_dict("records"))
        for metric, direction in METRIC_DIRECTIONS.items():
            ssmr_value = float(
                metrics_frame.loc[metrics_frame["method"] == "SSMR-FS", metric].iloc[0]
            )
            for baseline in BASELINE_METHODS:
                baseline_value = float(
                    metrics_frame.loc[metrics_frame["method"] == baseline, metric].iloc[0]
                )
                is_better = (
                    ssmr_value < baseline_value
                    if direction == "lower"
                    else ssmr_value > baseline_value
                )
                directional_comparison_rows.append(
                    {
                        "sample_size": sample_size,
                        "metric": metric,
                        "baseline": baseline,
                        "ssmr_value": ssmr_value,
                        "baseline_value": baseline_value,
                        "ssmr_better": is_better,
                    }
                )

    sample_frame = pd.DataFrame(metric_rows)
    directional_frame = pd.DataFrame(directional_comparison_rows)
    ssmr_source_frame = pd.DataFrame(ssmr_source_rows)
    fixed_base_frame = pd.DataFrame(
        [
            {
                "max_sample_size": max_sample_size,
                "ssmr_unique_candidate_pool_size": len(ssmr_pool_bundle.candidate_pool),
                "ssmr_energy_reference_space": (
                    "三类候选路径合并去重后的完整候选空间"
                ),
                "selection_pca_components": ssmr_full_result.transformer.n_components_,
                "selection_pca_explained_variance": (
                    ssmr_full_result.transformer.explained_variance_ratio_sum_
                ),
                "ssmr_selection_objective": ssmr_full_result.selection_summary[
                    "selection_objective"
                ],
                "evaluation_pca_components": transformer.n_components_,
                "evaluation_pca_explained_variance": transformer.explained_variance_ratio_sum_,
                "storage_temperature_record_count": len(evaluation_set.records),
                "storage_temperature_point_count": evaluation_set.audit_summary[
                    "temperature_point_count"
                ],
            }
        ]
    )
    sample_dir = get_result_dir(config, "sample_size")
    figure_dir = get_result_dir(config, "figures")
    write_excel(
        sample_dir / config.excel_sample_size_name,
        {
            "不同输出数量四项指标": sample_frame,
            "SSMR逐指标方向比较": directional_frame,
            "四种基线运行诊断": pd.DataFrame(baseline_diagnostic_rows),
            "SSMR来源构成": ssmr_source_frame,
            "固定基础对象摘要": fixed_base_frame,
        },
    )
    write_excel(
        sample_dir / "profiles_by_design_size.xlsx",
        sample_temperature_sheets,
    )
    write_excel(
        sample_dir / "features_by_design_size.xlsx",
        sample_feature_sheets,
    )
    write_excel(
        sample_dir / "SSMR-FS_selection_logs_by_size.xlsx",
        ssmr_selection_log_sheets,
    )
    write_json(
        sample_dir / "sample_size_comparison_summary.json",
        {
            "metrics": sample_frame.to_dict("records"),
            "directional_comparison": directional_frame.to_dict("records"),
            "baseline_diagnostics": baseline_diagnostic_rows,
            "ssmr_source_composition": ssmr_source_frame.to_dict("records"),
            "ssmr_selection": ssmr_full_result.selection_summary,
            "fixed_base_design": fixed_base_frame.to_dict("records")[0],
            "storage_temperature_records": evaluation_set.audit_summary,
        },
    )
    plot_sample_size_metrics(
        sample_frame,
        figure_dir / "Figure8_metrics_by_design_size",
        config.figure_dpi,
    )
    if logger:
        logger.info(
            "Design-size comparisons completed: "
            f"{_result_path_for_log(config, sample_dir)}"
        )
    return sample_frame


def generate_baseline_outputs(config: ExperimentConfig, logger=None) -> None:
    initialize_outputs(config)
    n_samples = config.main_sample_size
    if logger:
        logger.info("Generating the four comparator designs.")
    baseline_result = _baseline_suite(config, n_samples, "main")
    main_dir = get_result_dir(config, "main")
    _save_design_tables(
        config,
        baseline_result.records_by_method,
        main_dir / "N300_comparator_profiles_and_features.xlsx",
    )
    write_excel(
        main_dir / "N300_comparator_generation_diagnostics.xlsx",
        {
            "运行诊断": pd.DataFrame(baseline_result.diagnostics),
            "基线定义与文献": _baseline_definition_frame(),
        },
    )
    write_json(
        main_dir / "N300_comparator_generation_summary.json",
        {
            "baseline_count": len(BASELINE_METHODS),
            "diagnostics": baseline_result.diagnostics,
            "reporting_rule": "输出当前配置的四种对照方法",
        },
    )


def generate_ssmr_outputs(config: ExperimentConfig, logger=None) -> None:
    initialize_outputs(config)
    ssmr_result = select_ssmr_fs(
        config,
        _rng(config, "main", "SSMR-FS"),
        config.main_sample_size,
    )
    validate_candidate_bundle(config, ssmr_result.pool_bundle)
    save_candidate_pool_outputs(config, ssmr_result)
    write_excel(
        get_result_dir(config, "ssmr") / "SSMR-FS_sequential_selection_only.xlsx",
        _ssmr_workbook_sheets(config, ssmr_result),
    )
    if logger:
        logger.info("SSMR-FS selection output saved.")


def generate_storage_temperature_record_outputs(
    config: ExperimentConfig, logger=None
) -> None:
    initialize_outputs(config)
    evaluation_set = load_storage_temperature_records(config)
    save_storage_temperature_record_outputs(config, evaluation_set)
    if logger:
        logger.info(f"Measured records and evaluation features saved: {evaluation_set.label}")
