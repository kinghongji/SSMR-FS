from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.experiment_config import get_config, get_result_dir
from src.io.logger import setup_logger
from src.workflow import generate_storage_temperature_record_outputs


if __name__ == "__main__":
    config = get_config()
    logger = setup_logger(get_result_dir(config, "logs") / "03_measured_records.log")
    logger.info("Loading measured storage-temperature records for evaluation.")
    generate_storage_temperature_record_outputs(config, logger)
    logger.info("Measured records and evaluation features saved.")
