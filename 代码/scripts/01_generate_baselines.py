from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.experiment_config import get_config, get_result_dir
from src.io.logger import setup_logger
from src.workflow import generate_baseline_outputs


if __name__ == "__main__":
    config = get_config()
    logger = setup_logger(get_result_dir(config, "logs") / "01_生成四种正式基线.log")
    logger.info("正在生成简单随机、Sobol、OA-LHS和MaxPro四种对照方法")
    generate_baseline_outputs(config, logger)
    logger.info("四种对照方法生成完成")
