"""
Stage 2：训练 RQ-VAE 并生成 item.index.json。

用法：
  conda activate cure_rec
  python scripts/train/run_rqvae.py
  CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num_processes 2 scripts/train/run_rqvae.py

  # 覆盖超参
  python scripts/train/run_rqvae.py train.epochs=3000 train.learning_rate=5e-4

  # 切换数据域并覆盖模型结构
  python scripts/train/run_rqvae.py data.domain=sports model.rqvae.codebook_size=512
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from utils.logging_config import configure_logging


@hydra.main(config_path="../../configs", config_name="rqvae", version_base=None)
def main(cfg: DictConfig) -> None:
    configure_logging(cfg.logging.output_path, cfg.logging.level, cfg.logging.mode)
    from models.rqvae.trainer import run_rqvae
    run_rqvae(cfg)


if __name__ == "__main__":
    main()
