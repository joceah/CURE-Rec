"""
Stage 2：商品文本 → embedding 向量。

用法：
  # 默认配置（title + features + categories）
  conda activate cure_rec
  CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num_processes 2 scripts/train/run_text2emb.py

  # 命令行覆盖字段
  accelerate launch --num_processes 2 scripts/train/run_text2emb.py \
    "data.text_fields=[title,description]"
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from utils.logging_config import configure_logging



@hydra.main(
    config_path="../../configs",
    config_name="text2emb",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    configure_logging(cfg.logging.output_path, cfg.logging.level, cfg.logging.mode)
    from models.text2emb import run_text2emb
    run_text2emb(cfg)


if __name__ == "__main__":
    main()
