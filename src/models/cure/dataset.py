"""Recommendation-only data with explicit history spans and frozen split identities."""

import csv
import json
from collections import Counter
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler


class CUREDataset(Dataset):
    def __init__(self, user_sequences_path, item_sid_ids, tokenizer,
                 max_history_len=20, max_seq_len=512, split_path=None,
                 valid_history_path=None, sample_ids_path=None, sample_size=None, seed=42):
        if max_history_len < 1:
            raise ValueError("max_history_len must be positive")
        self.item_sid_ids = item_sid_ids
        if any(len(sid) != 3 for sid in item_sid_ids.values()):
            raise ValueError("CURE v1 requires exactly three SID tokens per item")
        self.prefix = tokenizer.encode("用户历史：", add_special_tokens=False)
        self.suffix = tokenizer.encode(" 请推荐下一个商品：", add_special_tokens=False)
        self.history_limit = min(max_history_len, (max_seq_len - len(self.prefix) - len(self.suffix) - 3) // 3)
        if self.history_limit < 1:
            raise ValueError("max_seq_len cannot fit a complete history item and target")
        with open(user_sequences_path) as f:
            self.sequences = {int(k): v for k, v in json.load(f).items()}
        # Popularity is computed only from train interactions, never held-out targets.
        self.popularity = Counter(i for seq in self.sequences.values() for i in seq)
        if any(i not in item_sid_ids for i in self.popularity):
            raise ValueError("Train history contains an item missing from the SID artifact")
        self.valid_targets = {}
        if valid_history_path:
            if not split_path:
                raise ValueError("valid_history_path is only meaningful for test evaluation")
            with open(valid_history_path) as f:
                self.valid_targets = {int(r['user_id']): int(r['item_id']) for r in csv.DictReader(f)}
        self.samples = []
        if split_path:
            identities = {}
            if sample_ids_path:
                with open(sample_ids_path) as f:
                    identities = {(int(r['user_id']), int(r['target_item_id'])): r['sample_id']
                                  for r in csv.DictReader(f)}
            with open(split_path) as f:
                for row in csv.DictReader(f):
                    user, target = int(row['user_id']), int(row['item_id'])
                    if user not in self.sequences or target not in item_sid_ids:
                        raise ValueError("Frozen split is not covered by history/SID artifacts")
                    if sample_ids_path and (user, target) not in identities:
                        raise ValueError("Split sample is missing from frozen sample IDs")
                    self.samples.append((user, len(self.sequences[user]), target,
                                         identities.get((user, target), f"eval:{user}:{target}")))
        else:
            for user, seq in self.sequences.items():
                self.samples.extend((user, t, seq[t], f"train:{user}:{t}:{seq[t]}")
                                    for t in range(1, len(seq)))
        if sample_size is not None and sample_size < len(self.samples):
            import random
            indices = sorted(random.Random(seed).sample(range(len(self.samples)), sample_size))
            self.samples = [self.samples[i] for i in indices]
        if not self.samples:
            raise ValueError("No recommendation samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        user, end, target, sample_id = self.samples[index]
        # Store sequence offsets rather than copying all prefixes (quadratic memory).
        history = self.sequences[user][max(0, end - self.history_limit):end]
        if user in self.valid_targets:
            history = (history + [self.valid_targets[user]])[-self.history_limit:]
        prompt = list(self.prefix)
        positions = []
        for item in history:
            start = len(prompt)
            prompt.extend(self.item_sid_ids[item])
            positions.append([start, start + 1, start + 2])
        if not positions:
            raise ValueError("A nonempty history is required")
        prompt.extend(self.suffix)
        target_ids = self.item_sid_ids[target]
        return {
            'input_ids': torch.tensor(prompt + target_ids),
            'labels': torch.tensor([-100] * len(prompt) + target_ids),
            'prompt_length': len(prompt),
            'history_positions': torch.tensor(positions),
            'user_id': user, 'target_item_id': target,
            'history_len': len(history), 'target_popularity': self.popularity[target],
            'sample_id': sample_id, 'sample_index': index,
        }


def collate_cure(samples, pad_token_id):
    b, width = len(samples), max(len(s['input_ids']) for s in samples)
    items = max(len(s['history_positions']) for s in samples)
    batch = {
        'input_ids': torch.full((b, width), pad_token_id, dtype=torch.long),
        'labels': torch.full((b, width), -100, dtype=torch.long),
        'attention_mask': torch.zeros(b, width, dtype=torch.long),
        'history_positions': torch.full((b, items, 3), -1, dtype=torch.long),
        'history_mask': torch.zeros(b, items, dtype=torch.bool),
    }
    for i, sample in enumerate(samples):
        length, n = len(sample['input_ids']), len(sample['history_positions'])
        batch['input_ids'][i, :length] = sample['input_ids']
        batch['labels'][i, :length] = sample['labels']
        batch['attention_mask'][i, :length] = 1
        batch['history_positions'][i, :n] = sample['history_positions']
        batch['history_mask'][i, :n] = True
    for key in ('prompt_length', 'user_id', 'target_item_id', 'history_len',
                'target_popularity', 'sample_index'):
        batch[key] = torch.tensor([s[key] for s in samples], dtype=torch.long)
    batch['sample_id'] = [s['sample_id'] for s in samples]
    return batch


def make_loader(dataset, batch_size, pad_token_id, shuffle=False, seed=42):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed),
                      collate_fn=partial(collate_cure, pad_token_id=pad_token_id))


def to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


class MergeSingletonBatchSampler(BatchSampler):
    """Single-process personalized training keeps its final sample as a donor."""
    def __iter__(self):
        previous = None
        for current in super().__iter__():
            if previous is not None:
                if len(current) == 1:
                    yield previous + current
                    return
                yield previous
            previous = current
        if previous is not None:
            yield previous

    def __len__(self):
        length = super().__len__()
        return length - int(length > 1 and len(self.sampler) % self.batch_size == 1)


def make_train_loader(dataset, batch_size, pad_token_id, needs_donor=False, world_size=1, seed=42):
    # Equal complete local batches keep DDP collectives aligned and donors distinct.
    # With drop_last=True Accelerate also drops incomplete groups of world_size batches.
    if world_size > 1 and len(dataset) < batch_size * world_size:
        raise ValueError('Training samples must cover one global batch; reduce train.per_device_batch_size')
    if needs_donor and (batch_size < 2 or len(dataset) < 2):
        raise ValueError('Personalized variants require per-device batch_size >= 2')
    generator = torch.Generator().manual_seed(seed)
    collate = partial(collate_cure, pad_token_id=pad_token_id)
    if needs_donor and world_size == 1:
        sampler = MergeSingletonBatchSampler(RandomSampler(dataset, generator=generator), batch_size, False)
        return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator,
                      drop_last=world_size > 1, collate_fn=collate)
