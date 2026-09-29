"""
Stage 2 Step 2+3：训练 RQ-VAE + 生成 item.index.json。

用法：
  conda activate MiniOneRec
  python scripts/train/run_rqvae.py

  # 覆盖超参
  python scripts/train/run_rqvae.py train.epochs=3000 train.learning_rate=5e-4

  # 覆盖模型结构
  python scripts/train/run_rqvae.py model/rqvae=rqvae model.rqvae.codebook_size=512
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

LOG_FILE = Path(__file__).resolve().parents[2] / "logs" / "rqvae.log"


@hydra.main(config_path="../../configs", config_name="stage2", version_base=None)
def main(cfg: DictConfig) -> None:
    # 在 Hydra 接管 logging 之后重新配置，确保日志写入固定文件
    LOG_FILE.parent.mkdir(exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    fh = logging.FileHandler(LOG_FILE, mode='w')
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root_logger.handlers = [fh, sh]

    from models.rqvae.trainer import run_rqvae
    run_rqvae(cfg)


if __name__ == "__main__":
    main()
