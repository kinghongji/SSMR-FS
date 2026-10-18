from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.experiment_config import get_config, get_result_dir
from src.io.logger import setup_logger
from src.workflow import generate_ssmr_outputs


if __name__ == "__main__":
    config = get_config()
    logger = setup_logger(get_result_dir(config, "logs") / "02_ssmr_fs_selection.log")
    logger.info("Running SSMR-FS sequential energy-distance selection.")
    generate_ssmr_outputs(config, logger)
    logger.info("SSMR-FS selection completed.")
