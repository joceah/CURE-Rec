"""
Hydra 启动脚本，调用 evaluator.evaluate()。
  conda activate cure_rec
  python scripts/eval/run_eval.py
  CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 scripts/eval/run_eval.py

"""
import logging
import sys
from pathlib import Path

# 添加 src 到 PYTHONPATH
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from utils.logging_config import configure_logging

logger = logging.getLogger(__name__)

import hydra
from omegaconf import DictConfig, OmegaConf
from inference.evaluator import evaluate


@hydra.main(config_path="../../configs", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    configure_logging(cfg.logging.output_path, cfg.logging.level, cfg.logging.mode)
    logger.info("Stage 4: SFT Evaluation\\n%s", OmegaConf.to_yaml(cfg))

    # 构造输出路径
    output_path = str(Path(cfg.eval.output_dir) / f"{cfg.eval.output_name}.json")

    # 调用 evaluate()
    metrics = evaluate(
        model_path=cfg.eval.model_path,
        tokenizer_path=cfg.eval.tokenizer_path,
        item_index_path=cfg.eval.item_index_path,
        user_sequences_path=cfg.eval.user_sequences_path,
        test_csv_path=cfg.eval.test_csv_path,
        valid_csv_path=cfg.eval.get("valid_csv_path"),
        max_history_len=cfg.eval.max_history_len,
        num_beams=cfg.eval.num_beams,
        batch_size=cfg.eval.batch_size,
        num_sid_layers=cfg.eval.num_sid_layers,
        sample_size=cfg.eval.sample_size,
        random_seed=cfg.eval.random_seed,
        k_values=cfg.eval.k_values,
        output_path=output_path,
        decode_strategy=cfg.eval.get("decode_strategy", "beam_search"),
        diversity_penalty=cfg.eval.get("diversity_penalty", 0.5),
        num_beam_groups=cfg.eval.get("num_beam_groups", 5),
        temperature=cfg.eval.get("temperature", 1.0),
    )

    if metrics:  # 主进程
        logger.info("=" * 80)
        logger.info("Evaluation Complete!")
        logger.info("=" * 80)
        for k, v in sorted(metrics.items()):
            logger.info("%-12s: %.4f", k, v)


if __name__ == "__main__":
    main()
