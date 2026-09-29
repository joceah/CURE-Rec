import pytest
import torch

from inference.prefix_tree import SidPrefixTree, ConstrainedLogitsProcessor


@pytest.fixture
def simple_item_sid_ids():
    """3 个商品，2 层 SID（token_id 直接用整数模拟）。
    商品 0: <a_5> <b_12>
    商品 1: <a_5> <b_99>
    商品 2: <a_7> <b_12>
    """
    return {
        0: [5, 12],
        1: [5, 99],
        2: [7, 12],
    }


@pytest.fixture
def collision_item_sid_ids():
    """碰撞场景：商品 0 和 1 共享同一个 SID。"""
    return {
        0: [5, 12, 100],
        1: [5, 12, 100],   # 和 0 完全相同（碰撞）
        2: [5, 12, 200],
        3: [7, 50, 80],
    }


class TestSidPrefixTree:
    def test_root_returns_layer0_tokens(self, simple_item_sid_ids):
        tree = SidPrefixTree(simple_item_sid_ids)
        assert tree.get_allowed_tokens(()) == {5, 7}

    def test_layer1_tokens_under_a5(self, simple_item_sid_ids):
        tree = SidPrefixTree(simple_item_sid_ids)
        assert tree.get_allowed_tokens((5,)) == {12, 99}

    def test_layer1_tokens_under_a7(self, simple_item_sid_ids):
        tree = SidPrefixTree(simple_item_sid_ids)
        assert tree.get_allowed_tokens((7,)) == {12}

    def test_leaf_returns_empty(self, simple_item_sid_ids):
        tree = SidPrefixTree(simple_item_sid_ids)
        assert tree.get_allowed_tokens((5, 12)) == set()

    def test_invalid_prefix_returns_empty(self, simple_item_sid_ids):
        tree = SidPrefixTree(simple_item_sid_ids)
        assert tree.get_allowed_tokens((999,)) == set()
        assert tree.get_allowed_tokens((5, 999)) == set()

    def test_collision_handled(self, collision_item_sid_ids):
        """商品 0 和 1 SID 完全相同 → trie 不重复，但 leaf 仍合法。"""
        tree = SidPrefixTree(collision_item_sid_ids)
        # 第 2 层 prefix=(5, 12) 后允许 {100, 200}
        assert tree.get_allowed_tokens((5, 12)) == {100, 200}
        # 第 1 层 prefix=() → {5, 7}
        assert tree.get_allowed_tokens(()) == {5, 7}


class TestConstrainedLogitsProcessor:
    def test_first_step_only_layer0_allowed(self, simple_item_sid_ids):
        """第一步生成时（已生成长度 0），只允许 layer 0 的 token（5 和 7）。"""
        tree = SidPrefixTree(simple_item_sid_ids)
        # 模拟 batch_size=1, num_beams=1, prompt_length=3
        prompt_lengths = torch.tensor([3])
        proc = ConstrainedLogitsProcessor(tree, prompt_lengths, num_sid_layers=2)

        # input_ids: prompt 部分 [10, 20, 30]，未开始生成
        input_ids = torch.tensor([[10, 20, 30]])
        scores = torch.zeros(1, 100)  # vocab_size=100，全部初始化为 0

        out = proc(input_ids, scores)
        # 只有 token 5 和 7 不是 -inf
        assert out[0, 5].item() == 0.0
        assert out[0, 7].item() == 0.0
        assert out[0, 12].item() == float("-inf")
        assert out[0, 0].item() == float("-inf")

    def test_second_step_after_a5(self, simple_item_sid_ids):
        """已生成 [5]，下一步只允许 {12, 99}。"""
        tree = SidPrefixTree(simple_item_sid_ids)
        prompt_lengths = torch.tensor([3])
        proc = ConstrainedLogitsProcessor(tree, prompt_lengths, num_sid_layers=2)

        input_ids = torch.tensor([[10, 20, 30, 5]])  # prompt + 已生成 [5]
        scores = torch.zeros(1, 100)

        out = proc(input_ids, scores)
        assert out[0, 12].item() == 0.0
        assert out[0, 99].item() == 0.0
        assert out[0, 5].item() == float("-inf")
        assert out[0, 7].item() == float("-inf")

    def test_batch_with_different_prompt_lengths(self, simple_item_sid_ids):
        """batch 内不同样本 prompt 长度不同。"""
        tree = SidPrefixTree(simple_item_sid_ids)
        # 样本 0: prompt_length=3, 样本 1: prompt_length=5
        prompt_lengths = torch.tensor([3, 5])
        proc = ConstrainedLogitsProcessor(tree, prompt_lengths, num_sid_layers=2)

        # 样本 0 已生成 [5]，样本 1 还在 prompt（用 pad 0 填充）
        # 假设 padded 长度为 6
        input_ids = torch.tensor([
            [10, 20, 30, 5, 0, 0],   # prompt_length=3, 已生成 [5]
            [10, 20, 30, 40, 50, 0], # prompt_length=5, 已生成 [] （还没开始）
        ])
        scores = torch.zeros(2, 100)
        out = proc(input_ids, scores)

        # 样本 0：在 [5] 之后，允许 {12, 99}
        assert out[0, 12].item() == 0.0
        assert out[0, 99].item() == 0.0
        assert out[0, 5].item() == float("-inf")
        # 样本 1：还没生成，允许 {5, 7}
        assert out[1, 5].item() == 0.0
        assert out[1, 7].item() == 0.0
        assert out[1, 12].item() == float("-inf")
