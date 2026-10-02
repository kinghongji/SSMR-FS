from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.experiment_config import get_config, get_result_dir
from src.io.logger import setup_logger
from src.workflow import run_repeated_analysis


if __name__ == "__main__":
    config = get_config()
    logger = setup_logger(get_result_dir(config, "logs") / "05_N300五种方法30次重复.log")
    logger.info("正在运行30次随机重复分析")
    run_repeated_analysis(config, logger)
    logger.info("随机重复分析运行完成")
