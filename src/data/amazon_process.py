"""
Amazon Reviews 2023 原始数据处理：k-core 过滤 + 用户/商品重映射 + 时间序切分。
输出：
  data/processed/{domain}/
    ├── train.csv
    ├── valid.csv
    ├── test.csv
    └── item_meta.json   (item_id -> {title, description})
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def _parse_date(date_str: str | None) -> int | None:
    """将 'YYYY-MM-DD' 字符串转为毫秒时间戳，None 表示不限制。"""
    if date_str is None:
        return None
    return int(datetime.strptime(date_str, "%Y-%m-%d").timestamp() * 1000)


def load_interactions(
    review_path: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict]:
    """流式读取 Amazon Reviews jsonl，提取 (user_id, parent_asin, timestamp)。

    Args:
        start_date: 起始日期（含），格式 'YYYY-MM-DD'，None 表示不限
        end_date:   截止日期（含），格式 'YYYY-MM-DD'，None 表示不限
    """
    ts_start = _parse_date(start_date)
    ts_end = _parse_date(end_date)
    if ts_end is not None:
        # end_date 当天最后一毫秒
        ts_end = ts_end + 86400 * 1000 - 1

    if start_date or end_date:
        logger.info(f"日期过滤: {start_date or '不限'} ~ {end_date or '不限'}")

    interactions = []
    with open(review_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                user_id = obj.get("user_id")
                item_id = obj.get("parent_asin")
                timestamp = obj.get("timestamp", 0)
                if not user_id or not item_id:
                    continue
                if ts_start is not None and timestamp < ts_start:
                    continue
                if ts_end is not None and timestamp > ts_end:
                    continue
                interactions.append(
                    {"user_id": user_id, "item_id": item_id, "timestamp": timestamp}
                )
            except json.JSONDecodeError:
                continue
    logger.info(f"加载原始交互数: {len(interactions)}")
    return interactions


def load_item_meta(meta_path: str) -> dict[str, dict]:
    """流式读取商品元数据 jsonl，保留用于 SID 编码和数据增强的字段。"""
    item_meta = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                parent_asin = obj.get("parent_asin")
                if not parent_asin:
                    continue

                def to_str(v) -> str:
                    if isinstance(v, list):
                        return " ".join(str(x) for x in v if x)
                    return str(v).strip() if v else ""

                item_meta[parent_asin] = {
                    "title":       obj.get("title", "").strip(),
                    "description": to_str(obj.get("description", "")),
                    "features":    to_str(obj.get("features", "")),
                    "categories":  to_str(obj.get("categories", "")),
                    "store":       obj.get("store", "").strip() if obj.get("store") else "",
                    "details":     to_str(obj.get("details", "")),
                }
            except json.JSONDecodeError:
                continue
    logger.info(f"加载商品元数据数: {len(item_meta)}")
    return item_meta


def kcore_filter(interactions: list[dict], k: int = 5) -> list[dict]:
    """迭代 k-core 过滤，直到所有用户和商品的交互数均 >= k。"""
    data = interactions[:]
    round_num = 0
    while True:
        round_num += 1
        user_count: dict[str, int] = defaultdict(int)
        item_count: dict[str, int] = defaultdict(int)
        for r in data:
            user_count[r["user_id"]] += 1
            item_count[r["item_id"]] += 1

        filtered = [
            r
            for r in data
            if user_count[r["user_id"]] >= k and item_count[r["item_id"]] >= k
        ]

        logger.info(
            f"k-core 第 {round_num} 轮: {len(data)} -> {len(filtered)} 条交互, "
            f"{len(user_count)} 用户 / {len(item_count)} 商品"
        )

        if len(filtered) == len(data):
            break
        data = filtered

    return data


def remap_ids(interactions: list[dict]) -> tuple[list[dict], dict, dict]:
    """将 user_id / item_id 字符串重映射为连续整数，返回映射表。"""
    users = sorted({r["user_id"] for r in interactions})
    items = sorted({r["item_id"] for r in interactions})
    user2id = {u: i for i, u in enumerate(users)}
    item2id = {it: i for i, it in enumerate(items)}

    remapped = [
        {
            "user_id": user2id[r["user_id"]],
            "item_id": item2id[r["item_id"]],
            "timestamp": r["timestamp"],
        }
        for r in interactions
    ]
    return remapped, user2id, item2id


def split_by_time(
    interactions: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    按时间序 leave-two-out 切分。
    每个用户的交互按时间排序后，最后 1 条为 test，倒数第 2 条为 valid，其余为 train。
    用户交互数 < 3 的跳过（无法切出 valid/test）。
    """
    user_records: dict[int, list[dict]] = defaultdict(list)
    for r in interactions:
        user_records[r["user_id"]].append(r)

    train, valid, test = [], [], []
    for uid, records in user_records.items():
        records.sort(key=lambda x: x["timestamp"])
        if len(records) < 3:
            continue
        train.extend(records[:-2])
        valid.append(records[-2])
        test.append(records[-1])

    logger.info(
        f"数据切分完成: train={len(train)}, valid={len(valid)}, test={len(test)}"
    )
    return train, valid, test


def build_user_sequences(
    train: list[dict], valid: list[dict], test: list[dict]
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    将交互列表转为 DataFrame。

    设计思路：不在这里展开历史列表（O(n²) 内存），而是：
    - user_sequences.csv：每个用户一行，存完整的有序 item 序列（train 部分）
    - valid.csv / test.csv：每行只存 (user_id, item_id)，训练时从 user_sequences 动态切历史

    这样 320 万条 train 不需要各自复制历史，内存和时间都是 O(n)。
    """
    # 按时间排序，构建每个用户的 train item 序列
    user_seq: dict[int, list[int]] = defaultdict(list)
    for r in sorted(train, key=lambda x: x["timestamp"]):
        user_seq[r["user_id"]].append(r["item_id"])

    # user_sequences：每个用户一行
    seq_df = pl.DataFrame(
        {
            "user_id": list(user_seq.keys()),
            "item_ids": list(user_seq.values()),  # 完整 train 序列
        }
    )

    # valid / test：只存目标 item
    def to_df(records: list[dict]) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "user_id": [r["user_id"] for r in records],
                "item_id": [r["item_id"] for r in records],
            }
        )

    logger.info(
        f"用户序列构建完成: {len(seq_df)} 个用户, "
        f"valid={len(valid)}, test={len(test)}"
    )
    return seq_df, to_df(valid), to_df(test)


def process(
    review_path: str,
    meta_path: str,
    output_dir: str,
    k: int = 5,
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    """完整 Stage 1 流程入口。

    Args:
        start_date: 只保留该日期（含）之后的交互，格式 'YYYY-MM-DD'
        end_date:   只保留该日期（含）之前的交互，格式 'YYYY-MM-DD'
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. 加载（含日期过滤）
    interactions = load_interactions(review_path, start_date=start_date, end_date=end_date)
    item_meta = load_item_meta(meta_path)

    # 2. k-core 过滤
    interactions = kcore_filter(interactions, k=k)

    # 3. 重映射 ID
    interactions, user2id, item2id = remap_ids(interactions)
    logger.info(f"最终: {len(user2id)} 用户, {len(item2id)} 商品, {len(interactions)} 交互")

    # 4. 时间序切分
    train, valid, test = split_by_time(interactions)

    # 5. 构建用户序列（O(n)，不展开历史）
    seq_df, valid_df, test_df = build_user_sequences(train, valid, test)

    # 6. 保存文件
    # user_sequences 含 list 列，用 json 格式保存（csv 不支持 list 列）
    user_sequences = {
        str(row["user_id"]): row["item_ids"]
        for row in seq_df.iter_rows(named=True)
    }
    with open(out / "user_sequences.json", "w") as f:
        json.dump(user_sequences, f)
    valid_df.write_csv(out / "valid.csv")
    test_df.write_csv(out / "test.csv")
    logger.info(f"文件已保存到 {out}")

    # 7. 保存 item_meta（只保留过滤后存在的商品）
    id2raw_item = {v: k for k, v in item2id.items()}
    filtered_meta = {
        str(new_id): item_meta.get(raw_id, {"title": "", "description": ""})
        for new_id, raw_id in id2raw_item.items()
    }
    with open(out / "item_meta.json", "w", encoding="utf-8") as f:
        json.dump(filtered_meta, f, ensure_ascii=False, indent=2)
    logger.info(f"item_meta.json 已保存，共 {len(filtered_meta)} 条")

    # 8. 保存 ID 映射（便于后续追溯）
    with open(out / "user2id.json", "w") as f:
        json.dump(user2id, f)
    with open(out / "item2id.json", "w") as f:
        json.dump(item2id, f)

    logger.info("Stage 1 数据处理完成！")
