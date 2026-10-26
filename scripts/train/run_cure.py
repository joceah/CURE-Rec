"""CURE Phase 1 训练入口；配置见 configs/cure.yaml。"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from models.cure.trainer import run_cure
from utils.logging_config import configure_logging


@hydra.main(config_path="../../configs", config_name="cure", version_base=None)
def main(cfg: DictConfig):
    log_path = cfg.logging.get("output_path") or cfg.logging.train_output_path
    configure_logging(log_path, cfg.logging.level, cfg.logging.mode)
    logging.getLogger(__name__).info("CURE 训练日志: %s", Path(log_path).resolve())
    run_cure(cfg)


if __name__ == "__main__":
    main()
