"""
Stage 3 SFT 训练启动脚本。

单卡：
  python scripts/train/run_sft.py

覆盖超参：
  python scripts/train/run_sft.py train.epochs=5 train.learning_rate=1e-4
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

log_file = Path(__file__).resolve().parents[2] / "logs" / "sft.log"
log_file.parent.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)


@hydra.main(config_path="../../configs", config_name="stage3", version_base=None)
def main(cfg: DictConfig) -> None:
    from models.llm.trainer import run_sft
    run_sft(cfg)


if __name__ == "__main__":
    main()
