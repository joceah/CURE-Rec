"""SFT Dataset：将用户历史序列转换为 LLM 输入格式。

输入格式（prompt）：
  "用户历史：<a_1><b_2><c_3> <a_4><b_5><c_6> ... 请推荐下一个商品："

目标格式（label）：
  "<a_7><b_8><c_9>"

辅助任务（item reconstruction）：
  输入："商品：<a_1><b_2><c_3>"
  目标："<a_1><b_2><c_3>"
"""

import json
import random
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class SFTDataset(Dataset):
    """主任务：给定用户历史，预测下一个商品的 SID。

    训练集：从 user_sequences.json 展开所有 (history, target) 对。
    验证/测试集：从 valid.csv / test.csv 读取目标商品，历史从 user_sequences.json 取。
    """

    def __init__(
        self,
        user_sequences_path: str,
        item_sid_ids: dict[int, list[int]],
        tokenizer: PreTrainedTokenizer,
        max_history_len: int = 20,
        max_seq_len: int = 512,
        split_path: str | None = None,
    ):
        """
        Args:
            user_sequences_path: user_sequences.json 路径
            item_sid_ids: item_id -> SID token id 列表
            tokenizer: 扩展后的 tokenizer
            max_history_len: 最多使用多少个历史商品
            max_seq_len: 最大序列长度（超出则截断）
            split_path: valid.csv / test.csv 路径。None 则从 user_sequences 展开训练集。
        """
        self.item_sid_ids = item_sid_ids
        self.tokenizer = tokenizer
        self.max_history_len = max_history_len
        self.max_seq_len = max_seq_len

        with open(user_sequences_path) as f:
            raw = json.load(f)
        user_sequences = {int(k): v for k, v in raw.items()}

        if split_path is None:
            # 训练集：展开每个用户序列的所有 (history, target) 对
            # 序列长度 >= 2 才能构成至少一个样本
            self.samples = []
            for user_id, seq in user_sequences.items():
                valid_seq = [iid for iid in seq if iid in item_sid_ids]
                for i in range(1, len(valid_seq)):
                    history = valid_seq[:i]
                    target = valid_seq[i]
                    self.samples.append((history, target))
        else:
            # 验证/测试集：从 split_path 读取目标商品
            self.samples = []
            with open(split_path) as f:
                next(f)  # 跳过 header
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    user_id, item_id = line.split(",")
                    user_id, item_id = int(user_id), int(item_id)
                    if user_id not in user_sequences or item_id not in item_sid_ids:
                        continue
                    history = [
                        iid for iid in user_sequences[user_id]
                        if iid in item_sid_ids
                    ]
                    self.samples.append((history, item_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        history, target = self.samples[idx]

        # 取最近 max_history_len 个历史
        history = history[-self.max_history_len:]

        prompt_ids = self._build_prompt(history)
        target_ids = self.item_sid_ids[target]

        eos_id = self.tokenizer.eos_token_id
        input_ids = prompt_ids + target_ids + [eos_id]
        labels = [-100] * len(prompt_ids) + target_ids + [eos_id]

        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _build_prompt(self, history: list[int]) -> list[int]:
        prefix_ids = self.tokenizer.encode("用户历史：", add_special_tokens=False)
        history_ids = []
        for iid in history:
            history_ids.extend(self.item_sid_ids[iid])
        suffix_ids = self.tokenizer.encode(" 请推荐下一个商品：", add_special_tokens=False)
        return prefix_ids + history_ids + suffix_ids


class ItemReconstructionDataset(Dataset):
    """辅助任务：给定商品 SID，重建该商品的 SID（帮助模型学习 SID 语义）。"""

    def __init__(
        self,
        item_sid_ids: dict[int, list[int]],
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int = 64,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.items = list(item_sid_ids.items())

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        _, sid_ids = self.items[idx]

        prefix_ids = self.tokenizer.encode("商品：", add_special_tokens=False)
        eos_id = self.tokenizer.eos_token_id

        input_ids = prefix_ids + sid_ids + [eos_id]
        labels = [-100] * len(prefix_ids) + sid_ids + [eos_id]

        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class MixedSFTDataset(Dataset):
    """混合主任务和辅助任务的 Dataset。"""

    def __init__(
        self,
        main_dataset: SFTDataset,
        aux_dataset: ItemReconstructionDataset,
        aux_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.main_dataset = main_dataset
        self.aux_dataset = aux_dataset

        total = len(main_dataset)
        n_aux = int(total * aux_ratio / (1 - aux_ratio))
        n_aux = min(n_aux, len(aux_dataset))

        rng = random.Random(seed)
        aux_indices = rng.sample(range(len(aux_dataset)), n_aux)

        self.index_map = [(False, i) for i in range(total)]
        self.index_map += [(True, i) for i in aux_indices]
        rng.shuffle(self.index_map)

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        is_aux, original_idx = self.index_map[idx]
        if is_aux:
            return self.aux_dataset[original_idx]
        return self.main_dataset[original_idx]


def collate_fn(
    batch: list[dict[str, torch.Tensor]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """动态 padding 到 batch 内最长序列。"""
    max_len = max(x["input_ids"].size(0) for x in batch)

    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, sample in enumerate(batch):
        seq_len = sample["input_ids"].size(0)
        input_ids[i, :seq_len] = sample["input_ids"]
        labels[i, :seq_len] = sample["labels"]
        attention_mask[i, :seq_len] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }



