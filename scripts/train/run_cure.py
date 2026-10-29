"""Run variants in order: residual_ce -> cure_g -> cure_gp -> cure_full.

Two GPUs:
  CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 scripts/train/run_cure.py
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))


@hydra.main(config_path='../../configs', config_name='cure', version_base=None)
def main(cfg: DictConfig):
    from utils.logging_config import configure_logging
    if cfg.train.get('overwrite_output_dir', False):
        cfg.logging.mode = 'w'
    configure_logging(cfg.logging.output_path, cfg.logging.level, cfg.logging.mode)
    from models.cure.trainer import run_cure
    run_cure(cfg)


if __name__ == '__main__':
    main()
