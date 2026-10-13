"""Configure console and optional file logging from a stage config."""

import logging
import os
import sys
from pathlib import Path


def configure_logging(output_path, level="INFO", mode="w"):
    """Configure root logging; only distributed rank 0 writes the shared log file."""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    level_value = getattr(logging, str(level).upper(), None)
    if not isinstance(level_value, int):
        raise ValueError(f"Unknown logging level: {level}")
    if mode not in {"w", "a"}:
        raise ValueError("logging.mode must be 'w' or 'a'")

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )
    handlers = []
    is_global_main = os.environ.get("RANK", "0") == "0"
    if output_path and is_global_main:
        log_path = Path(output_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode=mode, encoding="utf-8"))

    handlers.append(logging.StreamHandler(sys.stdout))
    for handler in handlers:
        handler.setFormatter(formatter)

    root_logger.setLevel(level_value)
    root_logger.handlers = handlers
