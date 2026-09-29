"""从GT rank JSON中生成HR@K和NDCG@K完整曲线"""
import json
import numpy as np
from pathlib import Path
import sys

def compute_hr_ndcg(ranks, k_values):
    """从rank列表计算HR@K和NDCG@K"""
    n = len(ranks)
    hr = {}
    ndcg = {}
    for k in k_values:
        hits = sum(1 for r in ranks if r <= k)
        hr[k] = hits / n
        # NDCG: 命中位置r的贡献是 1/log2(r+1)
        ndcg_sum = sum(1.0 / np.log2(r + 1) for r in ranks if r <= k)
        ndcg[k] = ndcg_sum / n
    return hr, ndcg


def main():
    if len(sys.argv) < 2:
        print("Usage: python compute_hr_curve.py <gt_rank_json>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path) as f:
        data = json.load(f)

    ranks_b = data["ranks_baseline"]
    ranks_c = data["ranks_cot"]
    n = len(ranks_b)

    k_values = [1, 3, 5, 10, 20, 30, 40, 50]
    hr_b, ndcg_b = compute_hr_ndcg(ranks_b, k_values)
    hr_c, ndcg_c = compute_hr_ndcg(ranks_c, k_values)

    print(f"\n=== 完整HR/NDCG曲线（n={n}） ===\n")
    print(f"{'K':<5} {'HR_Base':<12} {'HR_CoT':<12} {'HR_Diff':<12} {'NDCG_Base':<12} {'NDCG_CoT':<12} {'NDCG_Diff':<12}")
    print("-" * 90)
    for k in k_values:
        hr_diff = hr_c[k] - hr_b[k]
        ndcg_diff = ndcg_c[k] - ndcg_b[k]
        hr_arrow = "↑" if hr_diff > 0.0005 else "↓" if hr_diff < -0.0005 else "="
        ndcg_arrow = "↑" if ndcg_diff > 0.0005 else "↓" if ndcg_diff < -0.0005 else "="
        print(f"{k:<5} {hr_b[k]*100:>6.2f}%     {hr_c[k]*100:>6.2f}%     {hr_diff*100:+.2f}% {hr_arrow}    "
              f"{ndcg_b[k]*100:>6.2f}%     {ndcg_c[k]*100:>6.2f}%     {ndcg_diff*100:+.2f}% {ndcg_arrow}")

    # Entropy估计（用rank位置作为proxy分布）
    print(f"\n=== Rank分布均匀度（越高越均匀，即模型输出越'不sharp'） ===")
    # 用命中样本的rank分布计算entropy
    hit_ranks_b = [r for r in ranks_b if r <= 50]
    hit_ranks_c = [r for r in ranks_c if r <= 50]
    if hit_ranks_b:
        print(f"Baseline命中平均rank: {np.mean(hit_ranks_b):.2f} (n={len(hit_ranks_b)})")
    if hit_ranks_c:
        print(f"CoT命中平均rank:      {np.mean(hit_ranks_c):.2f} (n={len(hit_ranks_c)})")

    # 保存曲线数据（画图用）
    curve = {
        "k_values": k_values,
        "hr_baseline": [hr_b[k] for k in k_values],
        "hr_cot": [hr_c[k] for k in k_values],
        "ndcg_baseline": [ndcg_b[k] for k in k_values],
        "ndcg_cot": [ndcg_c[k] for k in k_values],
    }
    out = Path(path).with_stem(Path(path).stem + "_curve")
    with open(out, "w") as f:
        json.dump(curve, f, indent=2)
    print(f"\n✅ 曲线数据已保存到: {out}")


if __name__ == "__main__":
    main()
