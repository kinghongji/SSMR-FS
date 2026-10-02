from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Tuple


CODE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CODE_ROOT.parent
# 结果目录与“代码”目录并列，位置不依赖运行者的本机绝对路径。
RESULT_ROOT = WORKSPACE_ROOT / "结果"
# 评价数据随代码一同发布，审稿人下载后无需修改路径即可运行。
STORAGE_TEMPERATURE_FILE = CODE_ROOT / "210条实验室冰箱储藏温度记录.xlsx"


@dataclass(frozen=True)
class TemperatureSpaceConfig:
    temp_min: int = -18
    temp_max: int = 18
    min_days: int = 1
    max_days: int = 10


@dataclass(frozen=True)
class BaselineConfig:
    # 1个路径长度维度+10个逐日温度维度。所有直接生成型基线使用同一映射。
    input_dim: int = 11
    sobol_scramble: bool = True
    maxpro_exchange_factor: int = 4


@dataclass(frozen=True)
class SourcePoolConfig:
    scenario_count_per_type: int = 180
    stratified_count_per_stratum: int = 10
    risk_pool_size: int = 1800
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
    # 候选约束的序贯能量距离：同时保持候选空间代表性并减少路径重复。
    # 系数来自能量距离定义；外部储藏温度记录不进入选择模块。
    objective: str = "candidate_constrained_sequential_energy_distance"
    # 仅控制距离矩阵的分块内存，不改变选择结果或论文方法定义。
    distance_block_size: int = 256


@dataclass(frozen=True)
class StorageTemperatureRecordConfig:
    """储藏温度记录的只读导入配置；记录条数和统计特征由程序自动识别。"""

    file_path: Path = STORAGE_TEMPERATURE_FILE
    sheet_name: str = "实际每日温度---菌落"
    id_column: str = "样本名称"
    day_columns: Tuple[str, ...] = tuple(f"第{i}天实际温度" for i in range(1, 11))


@dataclass(frozen=True)
class FeatureConfig:
    pca_explained_variance: float = 0.95
    logdet_regularization: float = 1e-6


@dataclass(frozen=True)
class ExperimentConfig:
    # 全局随机种子。不同模块会在此基础上加固定 offset，保证可复现且互不干扰。
    random_seed: int = 42
    main_sample_size: int = 300
    # 不同输出数量验证固定储藏温度记录和评价变换，只改变输出数量N。
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
    excel_main_name: str = "N300_五种方法主比较.xlsx"
    excel_repeat_name: str = "N300_五种方法30次重复.xlsx"
    excel_sample_size_name: str = "五种方法不同输出数量验证.xlsx"


RESULT_SUBDIRS: Dict[str, str] = {
    "config": "01_运行配置与方法参数",
    "candidate_pools": "02_SSMR候选路径与选择空间",
    "ssmr": "03_SSMR序贯能量选择结果",
    "evaluation_sets": "04_储藏温度记录与评价空间",
    "main": "05_N300五种方法主比较",
    "repeat": "06_N300五种方法30次重复",
    "sample_size": "07_五种方法不同输出数量验证",
    "figures": "08_论文图",
    "logs": "09_运行日志",
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
    "关键温区覆盖误差": "key-zone coverage error (KCE)",
    "路径间隔": "path separation (PS)",
    "对数特征体积": "log feature-space volume (LFV)",
}


def get_config() -> ExperimentConfig:
    return ExperimentConfig()


def config_to_dict(config: ExperimentConfig) -> Dict[str, object]:
    data = asdict(config)
    # 配置快照只记录可移植的相对位置，不写入运行者本机的绝对路径。
    data["result_root"] = "../结果"
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
