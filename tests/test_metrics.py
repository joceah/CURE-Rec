import math
import pytest

from inference.metrics import hr_at_k, ndcg_at_k, compute_metrics


class TestHrAtK:
    def test_rank_zero_hit(self):
        assert hr_at_k(0, 1) == 1.0
        assert hr_at_k(0, 10) == 1.0

    def test_rank_below_k_hit(self):
        assert hr_at_k(4, 5) == 1.0

    def test_rank_at_k_miss(self):
        assert hr_at_k(5, 5) == 0.0

    def test_rank_above_k_miss(self):
        assert hr_at_k(10, 5) == 0.0


class TestNdcgAtK:
    def test_rank_zero(self):
        # 1 / log2(0+2) = 1 / log2(2) = 1.0
        assert ndcg_at_k(0, 1) == pytest.approx(1.0)

    def test_rank_one(self):
        # 1 / log2(1+2) = 1 / log2(3) ≈ 0.6309
        assert ndcg_at_k(1, 5) == pytest.approx(1.0 / math.log2(3))

    def test_rank_at_or_above_k(self):
        assert ndcg_at_k(5, 5) == 0.0
        assert ndcg_at_k(100, 10) == 0.0


class TestComputeMetrics:
    def test_perfect_predictions(self):
        # 三个样本都在 rank 0 命中
        preds = [[(1, 2, 3), (4, 5, 6)], [(7, 8, 9), (1, 1, 1)], [(0, 0, 0)]]
        targets = [(1, 2, 3), (7, 8, 9), (0, 0, 0)]
        result = compute_metrics(preds, targets, k_values=[1, 5])
        assert result["HR@1"] == pytest.approx(1.0)
        assert result["HR@5"] == pytest.approx(1.0)
        assert result["NDCG@1"] == pytest.approx(1.0)

    def test_target_not_in_predictions(self):
        preds = [[(1, 2, 3), (4, 5, 6)]]
        targets = [(99, 99, 99)]
        result = compute_metrics(preds, targets, k_values=[1, 5])
        assert result["HR@1"] == 0.0
        assert result["HR@5"] == 0.0
        assert result["NDCG@5"] == 0.0

    def test_mixed_ranks(self):
        # 样本0: rank 0; 样本1: rank 2; 样本2: 不命中
        preds = [
            [(1, 2, 3), (4, 5, 6), (7, 8, 9)],
            [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4)],
            [(0, 0, 0), (1, 1, 1)],
        ]
        targets = [(1, 2, 3), (3, 3, 3), (9, 9, 9)]
        result = compute_metrics(preds, targets, k_values=[1, 3, 5])
        assert result["HR@1"] == pytest.approx(1.0 / 3)  # 只有样本 0 命中
        assert result["HR@3"] == pytest.approx(2.0 / 3)  # 样本 0 和 2 命中（rank 0,2）
        assert result["HR@5"] == pytest.approx(2.0 / 3)
