"""HR@K / NDCG@K 评估指标计算。"""

import math


def hr_at_k(rank: int, k: int) -> float:
    """Hit Rate @K：rank 从 0 开始，rank < k 即命中。"""
    return 1.0 if rank < k else 0.0


def ndcg_at_k(rank: int, k: int) -> float:
    """NDCG @K：rank < k 时为 1/log2(rank+2)，否则 0。

    NDCG = DCG / IDCG。这里 ground truth 只有 1 个相关项，
    IDCG = 1/log2(0+2) = 1.0，所以归一化的 DCG 就是 1/log2(rank+2)。
    """
    if rank >= k:
        return 0.0
    return 1.0 / math.log2(rank + 2)


def compute_metrics(
    all_predictions: list[list[tuple[int, ...]]],
    all_targets: list[tuple[int, ...]],
    k_values: list[int] = (1, 3, 5, 10, 20, 50),
) -> dict[str, float]:
    """对所有测试样本汇总 HR@K 和 NDCG@K。

    Args:
        all_predictions: 每个样本的 Top-N 预测 SID 列表（每个 SID 是 token id tuple）
        all_targets: 每个样本的目标 SID
        k_values: 要计算的 K 值列表

    Returns:
        {"HR@1": ..., "HR@3": ..., ..., "NDCG@50": ...}
    """
    assert len(all_predictions) == len(all_targets)

    metrics = {f"HR@{k}": 0.0 for k in k_values}
    metrics.update({f"NDCG@{k}": 0.0 for k in k_values})

    for preds, target in zip(all_predictions, all_targets):
        try:
            rank = preds.index(target)
        except ValueError:
            rank = float('inf')  # 不在预测列表中，rank 设为无穷大确保不命中
        for k in k_values:
            metrics[f"HR@{k}"] += hr_at_k(rank, k)
            metrics[f"NDCG@{k}"] += ndcg_at_k(rank, k)

    n = len(all_targets)
    return {key: val / n for key, val in metrics.items()}
