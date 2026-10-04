from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.experiment_config import get_config, get_result_dir
from src.io.logger import setup_logger
from src.workflow import run_main_comparison, run_repeated_analysis, run_sample_size_validation


if __name__ == "__main__":
    config = get_config()
    logger = setup_logger(get_result_dir(config, "logs") / "07_五种方法一键运行全部分析.log")
    logger.info("开始运行：五种方法N300主比较、30次重复和不同输出数量验证")
    run_main_comparison(config, logger)
    run_repeated_analysis(config, logger)
    run_sample_size_validation(config, logger)
    logger.info("全部流程运行完成")
