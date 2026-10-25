from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.experiment_config import get_config, get_result_dir
from src.io.logger import setup_logger
from src.workflow import generate_baseline_outputs


if __name__ == "__main__":
    config = get_config()
    logger = setup_logger(get_result_dir(config, "logs") / "01_generate_comparators.log")
    logger.info("Generating the SRS, Sobol, OA-LHS, and MaxPro comparator designs.")
    generate_baseline_outputs(config, logger)
    logger.info("Comparator designs completed.")
