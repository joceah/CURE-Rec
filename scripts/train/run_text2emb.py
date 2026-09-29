"""
Stage 2 Step 1：商品文本 → embedding 向量。

用法：
  # 默认配置（title + features + categories）
  conda activate MiniOneRec
  accelerate launch --num_processes 2 scripts/train/run_text2emb.py

  # 命令行覆盖字段
  accelerate launch --num_processes 2 scripts/train/run_text2emb.py \
    "data.text_fields=[title,description]"
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # accelerate 多进程下 stdout 才能被捕获
    ],
    force=True,
)


@hydra.main(
    config_path="../../configs",
    config_name="stage2",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    from models.text2emb import run_text2emb
    run_text2emb(cfg)


if __name__ == "__main__":
    main()
