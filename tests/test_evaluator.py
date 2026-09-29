"""Evaluator 端到端 smoke test。

运行：
cd /data/users/juxian/genrec
conda activate MiniOneRec
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python tests/test_evaluator.py
"""
import sys
import logging
from inference.evaluator import evaluate

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def test_evaluator_smoke():
    print("=" * 80)
    print("Evaluator Smoke Test (100 samples)")
    print("=" * 80)

    metrics = evaluate(
        model_path="data/checkpoints/sft/merged",
        tokenizer_path="data/checkpoints/sft/tokenizer",
        item_index_path="data/processed/beauty_2021_2023/item.index.json",
        user_sequences_path="data/processed/beauty_2021_2023/user_sequences.json",
        test_csv_path="data/processed/beauty_2021_2023/test.csv",
        valid_csv_path="data/processed/beauty_2021_2023/valid.csv",
        max_history_len=20,
        num_beams=10,
        batch_size=4,
        num_sid_layers=3,
        sample_size=100,
        random_seed=42,
        k_values=[1, 3, 5, 10, 20, 50],
        output_path="data/eval_results/smoke_test.json",
    )

    if not metrics:  # 非主进程
        print("Non-main process finished")
        return

    # 验证指标在合理范围
    print("\n" + "=" * 80)
    print("Metrics Validation")
    print("=" * 80)

    for k, v in sorted(metrics.items()):
        print(f"{k:12s}: {v:.4f}")
        assert 0.0 <= v <= 1.0, f"{k} out of range: {v}"

    # HR@K 应该单调递增（K 越大，命中率越高）
    hr_keys = [f"HR@{k}" for k in [1, 3, 5, 10, 20, 50]]
    hr_values = [metrics[k] for k in hr_keys]
    for i in range(len(hr_values) - 1):
        assert hr_values[i] <= hr_values[i+1], f"HR not monotonic: {hr_keys[i]}={hr_values[i]} > {hr_keys[i+1]}={hr_values[i+1]}"

    print("\n✅ Smoke test PASS")


if __name__ == "__main__":
    test_evaluator_smoke()
