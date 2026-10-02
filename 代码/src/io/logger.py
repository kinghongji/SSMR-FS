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
        # 每次运行生成一份独立、干净的复现日志，避免保留旧的本机路径记录。
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
