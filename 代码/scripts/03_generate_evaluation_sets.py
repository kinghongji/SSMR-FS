from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.experiment_config import get_config, get_result_dir
from src.io.logger import setup_logger
from src.workflow import generate_storage_temperature_record_outputs


if __name__ == "__main__":
    config = get_config()
    logger = setup_logger(get_result_dir(config, "logs") / "03_储藏温度记录.log")
    logger.info("正在只读载入储藏温度记录，记录条数由程序自动识别")
    generate_storage_temperature_record_outputs(config, logger)
    logger.info("储藏温度记录及评价特征输出完成")
