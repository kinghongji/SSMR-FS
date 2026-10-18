from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.experiment_config import get_config, get_result_dir
from src.io.logger import setup_logger
from src.workflow import run_sample_size_validation


if __name__ == "__main__":
    config = get_config()
    logger = setup_logger(get_result_dir(config, "logs") / "06_sample_size_comparison.log")
    logger.info("Running design-size comparisons at N={200,500,800,1000,1300,1500}.")
    run_sample_size_validation(config, logger)
    logger.info("Design-size comparisons completed.")
