"""Hydra 启动脚本，调用 evaluator.evaluate()。"""
import sys
from pathlib import Path

# 添加 src 到 PYTHONPATH
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

import hydra
from omegaconf import DictConfig, OmegaConf
from inference.evaluator import evaluate


@hydra.main(config_path="../../configs", config_name="stage5", version_base=None)
def main(cfg: DictConfig):
    print("=" * 80)
    print("Stage 5: SFT Evaluation")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))

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
        print("\n" + "=" * 80)
        print("Evaluation Complete!")
        print("=" * 80)
        for k, v in sorted(metrics.items()):
            print(f"{k:12s}: {v:.4f}")


if __name__ == "__main__":
    main()
