"""CURE 评估入口；沿用 SFT 的 leave-two-out test 协议。"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from inference.cure_evaluator import evaluate_cure
from utils.logging_config import configure_logging


@hydra.main(config_path="../../configs", config_name="cure", version_base=None)
def main(cfg: DictConfig):
    log_path = cfg.logging.get("output_path") or cfg.logging.eval_output_path
    configure_logging(log_path, cfg.logging.level, cfg.logging.mode)
    logger = logging.getLogger(__name__)
    logger.info("CURE 评估日志: %s", Path(log_path).resolve())
    metrics = evaluate_cure(cfg)
    if metrics:
        logger.info("评估指标: %s", metrics)


if __name__ == "__main__":
    main()
