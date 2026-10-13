"""
Stage 3 SFT 训练启动脚本。

双卡 + DeepSpeed ZeRO-2：
  CUDA_VISIBLE_DEVICES=1,2 accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 scripts/train/run_sft.py

覆盖超参：
  python scripts/train/run_sft.py train.epochs=5 train.learning_rate=1e-4
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from utils.logging_config import configure_logging


@hydra.main(config_path="../../configs", config_name="sft", version_base=None)
def main(cfg: DictConfig) -> None:
    configure_logging(cfg.logging.output_path, cfg.logging.level, cfg.logging.mode)
    from models.llm.trainer import run_sft
    run_sft(cfg)


if __name__ == "__main__":
    main()
