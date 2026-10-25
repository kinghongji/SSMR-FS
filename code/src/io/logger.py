import logging
from pathlib import Path


def setup_logger(log_path: Path = None) -> logging.Logger:
    logger = logging.getLogger("ssmr_fs")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # 每次运行单独记录日志，避免混入上次运行的信息。
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
