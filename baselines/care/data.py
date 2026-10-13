"""Read frozen CURE-Rec samples and encode histories for CARE."""

import bisect
import csv
import json
import random
from pathlib import Path

from huggingface_hub import snapshot_download
import torch
from torch.utils.data import Dataset


ANSWER_TOKEN = "|start_of_answer|"


def resolve_model_source(model_name: str, local_files_only: bool) -> str:
    """Use a local snapshot path so Transformers cannot probe the Hub offline."""
    if not local_files_only:
        return model_name
    path = Path(model_name).expanduser()
    if path.is_dir():
        return str(path.resolve())
    return snapshot_download(repo_id=model_name, local_files_only=True)


def load_item_sids(path: str, tokenizer, sid_layers: int) -> dict[int, tuple[int, ...]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    result = {}
    for key, tokens in raw.items():
        if len(tokens) != sid_layers:
            raise ValueError(f"Item {key} has {len(tokens)} SID tokens; expected {sid_layers}")
        ids = tokenizer.convert_tokens_to_ids(tokens)
        if any(i == tokenizer.unk_token_id for i in ids):
            raise ValueError(f"Item {key} has an unknown SID token")
        result[int(key)] = tuple(ids)
    return result


def load_sequences(path: str) -> dict[int, list[int]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(key): value for key, value in raw.items()}


def load_split_samples(
    split: str,
    sequences: dict[int, list[int]],
    split_path: str,
    ids_path: str,
    valid_path: str,
    item_sids: dict[int, tuple[int, ...]],
    sample_size: int | None = None,
    seed: int = 42,
) -> list[dict]:
    if split not in {"valid", "test"}:
        raise ValueError(split)
    with open(split_path, newline="", encoding="utf-8") as handle:
        split_rows = list(csv.DictReader(handle))
    with open(ids_path, newline="", encoding="utf-8") as handle:
        id_rows = list(csv.DictReader(handle))
    if len(split_rows) != len(id_rows):
        raise ValueError("Frozen sample IDs and split rows have different lengths")
    valid_items = {}
    if split == "test":
        with open(valid_path, newline="", encoding="utf-8") as handle:
            valid_items = {int(r["user_id"]): int(r["item_id"]) for r in csv.DictReader(handle)}
    chosen = list(range(len(split_rows)))
    if sample_size is not None and 0 < sample_size < len(chosen):
        chosen = sorted(random.Random(seed).sample(chosen, sample_size))
    samples = []
    for row_index in chosen:
        row, frozen = split_rows[row_index], id_rows[row_index]
        user_id, target = int(row["user_id"]), int(row["item_id"])
        if (int(frozen["row_index"]) != row_index
                or int(frozen["user_id"]) != user_id
                or int(frozen["target_item_id"]) != target):
            raise ValueError(f"Frozen sample mismatch at row {row_index}")
        history = list(sequences[user_id])
        if split == "test":
            history.append(valid_items[user_id])
        if not history or target not in item_sids or any(i not in item_sids for i in history):
            raise ValueError(f"Missing history or SID for sample {frozen['sample_id']}")
        samples.append({"sample_id": frozen["sample_id"], "user_id": user_id,
                        "target_item_id": target, "history": history})
    return samples


class CAREDataset(Dataset):
    def __init__(self, sequences, item_sids, answer_id, max_history_len, split_samples=None):
        self.sequences = sequences
        self.item_sids = item_sids
        self.answer_id = answer_id
        self.max_history_len = max_history_len
        self.split_samples = split_samples
        if split_samples is None:
            self.users = list(sequences)
            self.offsets = [0]
            for user in self.users:
                self.offsets.append(self.offsets[-1] + max(0, len(sequences[user]) - 1))

    def __len__(self):
        return len(self.split_samples) if self.split_samples is not None else self.offsets[-1]

    def __getitem__(self, index):
        if self.split_samples is None:
            user_index = bisect.bisect_right(self.offsets, index) - 1
            user = self.users[user_index]
            sequence = self.sequences[user]
            target_pos = index - self.offsets[user_index] + 1
            history, target = sequence[:target_pos], sequence[target_pos]
            sample_id = None
        else:
            sample = self.split_samples[index]
            history, target = sample["history"], sample["target_item_id"]
            sample_id = sample["sample_id"]
        history = history[-self.max_history_len:]
        prompt = [tid for item in history for tid in self.item_sids[item]] + [self.answer_id]
        return {"input_ids": prompt, "history_lengths": len(history),
                "labels": self.item_sids[target], "sample_id": sample_id,
                "target_item_id": target}


def collate_care(batch: list[dict], pad_id: int) -> dict:
    width = max(len(sample["input_ids"]) for sample in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for row, sample in enumerate(batch):
        n = len(sample["input_ids"])
        input_ids[row, width-n:] = torch.tensor(sample["input_ids"])
        attention_mask[row, width-n:] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask,
            "history_lengths": torch.tensor([x["history_lengths"] for x in batch]),
            "labels": torch.tensor([x["labels"] for x in batch]),
            "sample_ids": [x["sample_id"] for x in batch],
            "target_item_ids": [x["target_item_id"] for x in batch]}
