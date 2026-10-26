"""在 SFT 样本上补充历史 SID 的显式位置。

记号：L 为单样本 token 数，N 为历史商品数，每个商品固定有 3 个 SID token。
单样本返回 input_ids/labels: [L]，history_positions: [N, 3]。
"""

import csv
import json

import torch
from torch.utils.data import Dataset

from models.llm.dataset import SFTDataset, ItemReconstructionDataset, MixedSFTDataset


class CUREDataset(SFTDataset):
    """主推荐任务；复用 SFT 的 prompt 和 label，仅增加 latent 所需字段。"""

    def __init__(self, user_sequences_path, item_sid_ids, tokenizer,
                 max_history_len=20, max_seq_len=512, split_path=None):
        super().__init__(user_sequences_path, item_sid_ids, tokenizer,
                         max_history_len, max_seq_len, split_path)
        # SFTDataset.samples 只保存 (history, target)，这里按相同顺序补回 user_id。
        with open(user_sequences_path) as f:
            sequences = json.load(f)
        if split_path is None:
            self.user_ids = [int(uid) for uid, seq in sequences.items()
                             for _ in range(max(0, sum(iid in item_sid_ids for iid in seq) - 1))]
        else:
            with open(split_path) as f:
                self.user_ids = [int(row["user_id"]) for row in csv.DictReader(f)
                                 if row["user_id"] in sequences
                                 and int(row["item_id"]) in item_sid_ids]
        if len(self.user_ids) != len(self.samples):
            raise ValueError("CURE user IDs do not match SFT samples")

    def __getitem__(self, idx):
        """返回历史 SID 在未填充 input_ids 中的位置 [N, 3]。"""
        history, target = self.samples[idx]
        history = history[-self.max_history_len:]
        # 历史 SID 连续接在中文前缀后；位置不依赖模型内的固定偏移猜测。
        prefix_len = len(self.tokenizer.encode("用户历史：", add_special_tokens=False))
        positions = []
        cursor = prefix_len
        for item_id in history:
            width = len(self.item_sid_ids[item_id])
            positions.append(list(range(cursor, cursor + width)))
            cursor += width
        sample = super().__getitem__(idx)
        if not positions or any(p[-1] >= len(sample["input_ids"]) for p in positions):
            raise ValueError("History was truncated or is empty; increase max_seq_len")
        sample.update(history_positions=torch.tensor(positions, dtype=torch.long),
                      history_item_mask=torch.ones(len(positions), dtype=torch.bool),
                      latent_active=torch.tensor(True),
                      user_id=torch.tensor(self.user_ids[idx]),
                      target_item_id=torch.tensor(target),
                      history_len=torch.tensor(len(history)))
        return sample


class CUREItemReconstructionDataset(ItemReconstructionDataset):
    """辅助重建任务没有用户历史，因此关闭 latent 路径。"""

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        sample.update(history_positions=torch.empty((0, 3), dtype=torch.long),
                      history_item_mask=torch.empty(0, dtype=torch.bool),
                      latent_active=torch.tensor(False),
                      user_id=torch.tensor(-1),
                      target_item_id=torch.tensor(self.items[idx][0]),
                      history_len=torch.tensor(0))
        return sample


def collate_cure(batch, pad_token_id):
    """右填充序列与商品维。

    B 为 batch 大小，T 为本批最长序列，N 为最多历史商品数：
    input_ids/labels/attention_mask: [B, T]；history_positions: [B, N, 3]；
    history_item_mask: [B, N]；latent_active/user_id 等元数据: [B]。
    历史商品维中的填充值为 0，只在 mask=True 时使用。
    """
    batch_size = len(batch)
    seq_len = max(len(x["input_ids"]) for x in batch)
    max_items = max(len(x["history_positions"]) for x in batch)
    sid_width = next((x["history_positions"].shape[1] for x in batch
                      if len(x["history_positions"])), 3)
    result = {
        "input_ids": torch.full((batch_size, seq_len), pad_token_id, dtype=torch.long),
        "attention_mask": torch.zeros((batch_size, seq_len), dtype=torch.long),
        "labels": torch.full((batch_size, seq_len), -100, dtype=torch.long),
        "history_positions": torch.zeros((batch_size, max_items, sid_width), dtype=torch.long),
        "history_item_mask": torch.zeros((batch_size, max_items), dtype=torch.bool),
    }
    for i, sample in enumerate(batch):
        n = len(sample["input_ids"])
        m = len(sample["history_positions"])
        result["input_ids"][i, :n] = sample["input_ids"]
        result["attention_mask"][i, :n] = 1
        result["labels"][i, :n] = sample["labels"]
        result["history_positions"][i, :m] = sample["history_positions"]
        result["history_item_mask"][i, :m] = sample["history_item_mask"]
    for key in ("latent_active", "user_id", "target_item_id", "history_len"):
        result[key] = torch.stack([x[key] for x in batch])
    return result


__all__ = ["CUREDataset", "CUREItemReconstructionDataset", "MixedSFTDataset", "collate_cure"]
