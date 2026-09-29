"""k-core 过滤和数据处理的单元测试。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.amazon_process import (
    build_user_sequences,
    kcore_filter,
    load_interactions,
    remap_ids,
    split_by_time,
)


def make_interactions(pairs: list[tuple], base_ts: int = 1000) -> list[dict]:
    """辅助函数：从 (user, item) 列表生成交互记录，时间戳递增。"""
    return [
        {"user_id": u, "item_id": i, "timestamp": base_ts + idx}
        for idx, (u, i) in enumerate(pairs)
    ]


class TestKcoreFilter:
    def test_basic_convergence(self):
        """所有用户和商品交互数 >= k 时应收敛。"""
        # 构造 5 个用户各与 5 个商品交互（每个商品被 5 个用户交互）
        pairs = [(f"u{u}", f"i{i}") for u in range(5) for i in range(5)]
        interactions = make_interactions(pairs)
        result = kcore_filter(interactions, k=5)
        assert len(result) == len(interactions)

    def test_removes_low_activity_users(self):
        """交互数 < k 的用户应被过滤。"""
        # u0 只有 1 次交互，k=5 时应被过滤
        pairs = [(f"u{u}", f"i{i}") for u in range(1, 6) for i in range(5)]
        pairs += [("u0", "i0")]  # u0 只有 1 次
        interactions = make_interactions(pairs)
        result = kcore_filter(interactions, k=5)
        user_ids = {r["user_id"] for r in result}
        assert "u0" not in user_ids

    def test_iterative_cascade(self):
        """删除用户后导致商品交互数不足，应继续迭代过滤。"""
        # u0 只与 i_rare 交互，i_rare 只被 u0 交互 -> 两者都应被过滤
        pairs = [(f"u{u}", f"i{i}") for u in range(5) for i in range(5)]
        pairs += [("u0_extra", "i_rare")]
        interactions = make_interactions(pairs)
        result = kcore_filter(interactions, k=5)
        item_ids = {r["item_id"] for r in result}
        assert "i_rare" not in item_ids

    def test_empty_input(self):
        result = kcore_filter([], k=5)
        assert result == []

    def test_all_filtered(self):
        """所有交互数都 < k 时应返回空列表。"""
        pairs = [("u0", "i0"), ("u1", "i1")]
        interactions = make_interactions(pairs)
        result = kcore_filter(interactions, k=5)
        assert result == []


class TestRemapIds:
    def test_contiguous_ids(self):
        pairs = [("userA", "itemX"), ("userB", "itemX"), ("userA", "itemY")]
        interactions = make_interactions(pairs)
        remapped, user2id, item2id = remap_ids(interactions)
        assert set(user2id.values()) == {0, 1}
        assert set(item2id.values()) == {0, 1}

    def test_original_preserved(self):
        pairs = [("u0", "i0"), ("u1", "i1")]
        interactions = make_interactions(pairs)
        remapped, user2id, item2id = remap_ids(interactions)
        assert len(remapped) == len(interactions)
        for r in remapped:
            assert isinstance(r["user_id"], int)
            assert isinstance(r["item_id"], int)


class TestSplitByTime:
    def test_split_ratio(self):
        """每个用户最后 1 条为 test，倒数第 2 条为 valid，其余为 train。"""
        # 5 个用户，每人 5 次交互
        pairs = [(u, i) for u in range(5) for i in range(5)]
        interactions = make_interactions(pairs)
        interactions, _, _ = remap_ids(interactions)
        train, valid, test = split_by_time(interactions)
        assert len(valid) == 5
        assert len(test) == 5
        assert len(train) == 15  # 5 * (5-2)

    def test_users_with_less_than_3_skipped(self):
        """交互数 < 3 的用户不出现在任何集合中。"""
        pairs = [("u0", "i0"), ("u0", "i1")]  # u0 只有 2 次
        pairs += [(f"u{u}", f"i{i}") for u in range(1, 4) for i in range(3)]
        interactions = make_interactions(pairs)
        interactions, user2id, _ = remap_ids(interactions)
        train, valid, test = split_by_time(interactions)
        u0_id = user2id["u0"]
        all_users = {r["user_id"] for r in train + valid + test}
        assert u0_id not in all_users

    def test_time_order_respected(self):
        """test 的 timestamp 应大于 valid，valid 应大于 train 中同用户最大 timestamp。"""
        pairs = [("u0", f"i{i}") for i in range(5)]
        interactions = make_interactions(pairs, base_ts=100)
        interactions, _, _ = remap_ids(interactions)
        train, valid, test = split_by_time(interactions)
        u0_train_max_ts = max(r["timestamp"] for r in train if r["user_id"] == 0)
        u0_valid_ts = next(r["timestamp"] for r in valid if r["user_id"] == 0)
        u0_test_ts = next(r["timestamp"] for r in test if r["user_id"] == 0)
        assert u0_train_max_ts < u0_valid_ts < u0_test_ts


class TestBuildUserSequences:
    def test_sequence_order(self):
        """用户序列应按时间排序。"""
        pairs = [("u0", f"i{i}") for i in range(5)]
        interactions = make_interactions(pairs, base_ts=0)
        interactions, _, item2id = remap_ids(interactions)
        train, valid, test = split_by_time(interactions)
        seq_df, _, _ = build_user_sequences(train, valid, test)

        u0_seq = seq_df.filter(pl.col("user_id") == 0)["item_ids"][0]
        # polars list 列取出来是 list，转一下确保
        u0_seq = list(u0_seq)
        # train 有 3 条（5 条中去掉最后 2 条），序列长度应为 3
        assert len(u0_seq) == 3
        # 序列应是递增的（因为 item_id 按时间顺序分配）
        assert u0_seq == sorted(u0_seq)

    def test_valid_test_only_target(self):
        """valid/test 只包含 user_id 和 item_id 两列。"""
        pairs = [("u0", f"i{i}") for i in range(5)]
        interactions = make_interactions(pairs, base_ts=0)
        interactions, _, _ = remap_ids(interactions)
        train, valid, test = split_by_time(interactions)
        _, valid_df, test_df = build_user_sequences(train, valid, test)

        assert set(valid_df.columns) == {"user_id", "item_id"}
        assert set(test_df.columns) == {"user_id", "item_id"}

    def test_one_row_per_user_in_sequences(self):
        """user_sequences 每个用户只有一行。"""
        pairs = [(f"u{u}", f"i{i}") for u in range(5) for i in range(5)]
        interactions = make_interactions(pairs)
        interactions, _, _ = remap_ids(interactions)
        train, valid, test = split_by_time(interactions)
        seq_df, _, _ = build_user_sequences(train, valid, test)

        assert seq_df["user_id"].n_unique() == len(seq_df)


    def test_start_date_filters_early(self, tmp_path):
        """start_date 之前的交互应被过滤。"""
        import json
        f = tmp_path / "reviews.jsonl"
        rows = [
            {"user_id": "u0", "parent_asin": "i0", "timestamp": 1420070400000},  # 2015-01-01
            {"user_id": "u0", "parent_asin": "i1", "timestamp": 1609459200000},  # 2021-01-01
        ]
        f.write_text("\n".join(json.dumps(r) for r in rows))
        result = load_interactions(str(f), start_date="2020-01-01")
        assert len(result) == 1
        assert result[0]["item_id"] == "i1"

    def test_end_date_filters_late(self, tmp_path):
        """end_date 之后的交互应被过滤。"""
        import json
        f = tmp_path / "reviews.jsonl"
        rows = [
            {"user_id": "u0", "parent_asin": "i0", "timestamp": 1420070400000},  # 2015-01-01
            {"user_id": "u0", "parent_asin": "i1", "timestamp": 1609459200000},  # 2021-01-01
        ]
        f.write_text("\n".join(json.dumps(r) for r in rows))
        result = load_interactions(str(f), end_date="2019-12-31")
        assert len(result) == 1
        assert result[0]["item_id"] == "i0"

    def test_date_range(self, tmp_path):
        """start_date 和 end_date 同时使用时只保留范围内的交互。"""
        import json
        f = tmp_path / "reviews.jsonl"
        rows = [
            {"user_id": "u0", "parent_asin": "i0", "timestamp": 1420070400000},  # 2015-01-01
            {"user_id": "u0", "parent_asin": "i1", "timestamp": 1546300800000},  # 2019-01-01
            {"user_id": "u0", "parent_asin": "i2", "timestamp": 1609459200000},  # 2021-01-01
        ]
        f.write_text("\n".join(json.dumps(r) for r in rows))
        result = load_interactions(str(f), start_date="2018-01-01", end_date="2020-12-31")
        assert len(result) == 1
        assert result[0]["item_id"] == "i1"

    def test_no_filter_returns_all(self, tmp_path):
        """不传日期参数时返回全部数据。"""
        import json
        f = tmp_path / "reviews.jsonl"
        rows = [
            {"user_id": "u0", "parent_asin": "i0", "timestamp": 1420070400000},
            {"user_id": "u0", "parent_asin": "i1", "timestamp": 1609459200000},
        ]
        f.write_text("\n".join(json.dumps(r) for r in rows))
        result = load_interactions(str(f))
        assert len(result) == 2


# polars 在测试中需要导入
import polars as pl
