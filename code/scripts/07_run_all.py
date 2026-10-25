from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.experiment_config import get_config, get_result_dir
from src.io.logger import setup_logger
from src.workflow import run_main_comparison, run_repeated_analysis, run_sample_size_validation


if __name__ == "__main__":
    config = get_config()
    logger = setup_logger(get_result_dir(config, "logs") / "07_complete_analysis.log")
    logger.info("Starting the N=300 main comparison, 30 repeated comparisons, and six design-size comparisons.")
    run_main_comparison(config, logger)
    run_repeated_analysis(config, logger)
    run_sample_size_validation(config, logger)
    logger.info("All requested analyses completed.")
