from dataclasses import asdict, dataclass, field
from pathlib import Path
import os
from typing import Dict, Tuple


CODE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CODE_ROOT.parent
# Results are written beside the code directory, independently of the local absolute path.
RESULT_ROOT = WORKSPACE_ROOT / "results"
# Read measured records from a path relative to this code directory.
STORAGE_TEMPERATURE_FILE = CODE_ROOT / "measured_storage_temperature_records.xlsx"


@dataclass(frozen=True)
class TemperatureSpaceConfig:
    temp_min: int = -18
    temp_max: int = 18
    min_days: int = 1
    max_days: int = 10


@dataclass(frozen=True)
class BaselineConfig:
    # Sobol, OA-LHS, and MaxPro use one duration and ten daily-temperature coordinates; SRS samples integers directly.
    input_dim: int = 11
    sobol_scramble: bool = True
    # The 4N value is the number of proposed swaps; accepted swaps are reported separately.
    maxpro_exchange_factor: int = 4


@dataclass(frozen=True)
class SourcePoolConfig:
    scenario_count_per_type: int = 180
    stratified_count_per_stratum: int = 10
    risk_pool_size: int = 1800
    generation_attempt_limit: int = 20000
    time_layers: Tuple[Tuple[str, int, int], ...] = (
        ("1-3 d", 1, 3),
        ("4-6 d", 4, 6),
        ("7-10 d", 7, 10),
    )
    temperature_layers: Tuple[Tuple[str, int, int], ...] = (
        ("freezing", -18, -1),
        ("chilled", 0, 4),
        ("transition", 5, 8),
        ("risk", 9, 14),
        ("abuse", 15, 18),
    )
    path_type_layers: Tuple[str, ...] = (
        "stable",
        "warming",
        "cooling",
        "oscillating",
    )
    amplitude_layers: Tuple[Tuple[str, int], ...] = (
        ("low", 1),
        ("medium", 3),
        ("high", 6),
    )
    risk_scenarios: Tuple[str, ...] = (
        "cold_boundary",
        "transition_4_8",
        "above_8_risk",
        "above_15_abuse",
        "freeze_thaw",
    )


@dataclass(frozen=True)
class SelectionConfig:
    # Select from the candidate pool by sequential energy-distance minimization.
    # Measured records are excluded from the selection transformation and process.
    objective: str = "candidate_constrained_sequential_energy_distance"
    # Distance blocks control memory use without changing the selected paths.
    distance_block_size: int = 256


@dataclass(frozen=True)
class StorageTemperatureRecordConfig:
    """Import settings for measured records; counts and summaries are computed from the data."""

    file_path: Path = STORAGE_TEMPERATURE_FILE
    sheet_name: str = "Measured daily temperatures"
    id_column: str = "Sample ID"
    day_columns: Tuple[str, ...] = tuple(f"Day {i} measured temperature (°C)" for i in range(1, 11))


@dataclass(frozen=True)
class FeatureConfig:
    pca_explained_variance: float = 0.95
    logdet_regularization: float = 1e-6


@dataclass(frozen=True)
class ExperimentConfig:
    # Master seed. SeedSequence derives named streams without arithmetic offsets.
    random_seed: int = 42
    main_sample_size: int = 300
    # The size comparison holds measured records and their evaluation transform fixed.
    sample_size_validation: Tuple[int, ...] = (200, 500, 800, 1000, 1300, 1500)
    repeated_times: int = 30
    figure_dpi: int = 600
    temperature: TemperatureSpaceConfig = field(default_factory=TemperatureSpaceConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    source_pool: SourcePoolConfig = field(default_factory=SourcePoolConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    storage_temperature_records: StorageTemperatureRecordConfig = field(
        default_factory=StorageTemperatureRecordConfig
    )
    features: FeatureConfig = field(default_factory=FeatureConfig)
    result_root: Path = RESULT_ROOT
    excel_main_name: str = "N300_main_comparison.xlsx"
    excel_repeat_name: str = "N300_30_repeated_runs.xlsx"
    excel_sample_size_name: str = "sample_size_comparison.xlsx"


RESULT_SUBDIRS: Dict[str, str] = {
    "config": "01_configuration_and_rules",
    "candidate_pools": "02_candidate_profiles_and_selection_space",
    "ssmr": "03_ssmr_fs_sequential_selection",
    "evaluation_sets": "04_measured_records_and_evaluation_space",
    "main": "05_N300_main_comparison",
    "repeat": "06_N300_repeated_runs",
    "sample_size": "07_sample_size_comparison",
    "figures": "08_manuscript_figures",
    "logs": "09_run_logs",
}


BASELINE_METHODS: Tuple[str, ...] = (
    "简单随机",
    "Sobol",
    "OA-LHS",
    "MaxPro",
)


METHOD_ORDER: Tuple[str, ...] = (
    *BASELINE_METHODS,
    "SSMR-FS",
)


BASELINE_FAMILIES: Dict[str, str] = {
    "简单随机": "随机抽样",
    "Sobol": "低差异序列",
    "OA-LHS": "正交分层空间抽样",
    "MaxPro": "投影填充设计",
}


METRIC_DIRECTIONS: Dict[str, str] = {
    "路径覆盖误差": "lower",
    "关键温区覆盖误差": "lower",
    "路径间隔": "higher",
    "对数特征体积": "higher",
}


METRIC_ENGLISH_NAMES: Dict[str, str] = {
    "路径覆盖误差": "path coverage error (PCE)",
    "关键温区覆盖误差": "key-condition coverage error (KCE)",
    "路径间隔": "path separation (PS)",
    "对数特征体积": "log feature volume (LFV)",
}


def get_config() -> ExperimentConfig:
    return ExperimentConfig()


def config_to_dict(config: ExperimentConfig) -> Dict[str, object]:
    data = asdict(config)
    # Store portable relative paths in the configuration export.
    data["result_root"] = Path(os.path.relpath(config.result_root, CODE_ROOT)).as_posix()
    data["storage_temperature_records"]["file_path"] = (
        config.storage_temperature_records.file_path.name
    )
    return data


def get_result_dir(config: ExperimentConfig, key: str) -> Path:
    if key not in RESULT_SUBDIRS:
        raise KeyError(f"Unknown result directory key: {key}")
    return config.result_root / RESULT_SUBDIRS[key]


def ensure_result_dirs(config: ExperimentConfig) -> None:
    config.result_root.mkdir(parents=True, exist_ok=True)
    for dirname in RESULT_SUBDIRS.values():
        (config.result_root / dirname).mkdir(parents=True, exist_ok=True)

