from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.experiment_config import get_config, get_result_dir
from src.io.logger import setup_logger
from src.workflow import run_main_comparison


if __name__ == "__main__":
    config = get_config()
    logger = setup_logger(get_result_dir(config, "logs") / "04_N300_main_comparison.log")
    logger.info("Running the five-method main comparison at N=300.")
    run_main_comparison(config, logger)
    logger.info("Main comparison completed.")
