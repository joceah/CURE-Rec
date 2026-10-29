"""Precomputed train-only matching covariates and deterministic non-self donors."""

import hashlib
import logging
from pathlib import Path

import torch

from .dataset import to_device
from .distributed import gather_records, main_process_call, shard_loader, valid_rows

logger = logging.getLogger(__name__)
MATCHING_PROTOCOL = 'history-popularity-baseline-score-v1'


def bucketize(history_len, target_popularity, baseline_score,
              history_edges=(3, 5, 10, 20), popularity_edges=(5, 20, 100, 500),
              difficulty_edges=(-8., -5., -3., -1.)):
    values = (history_len, target_popularity, baseline_score)
    edges = (history_edges, popularity_edges, difficulty_edges)
    if any(list(e) != sorted(set(e)) for e in edges):
        raise ValueError("Bucket edges must be strictly increasing")
    return torch.stack([torch.bucketize(torch.as_tensor(v).cpu().contiguous(), torch.tensor(e, dtype=torch.float32))
                        for v, e in zip(values, edges)], dim=-1)


def match_donors(buckets, generator=None, sample_ids=None):
    """Relaxation: all -> difficulty+history -> difficulty -> random non-self.

    Donors may be reused. This is matching, not a permutation or global shuffle.
    """
    buckets = torch.as_tensor(buckets).cpu()
    n = len(buckets)
    if n < 2:
        raise ValueError("Matched donor selection requires at least two distinct samples")
    identities = list(range(n)) if sample_ids is None else list(sample_ids)
    if len(identities) != n:
        raise ValueError("Matching identities have the wrong length")
    groups = [{}, {}, {}]
    keys = [(0, 1, 2), (0, 2), (2,)]
    for level, columns in enumerate(keys):
        for j, row in enumerate(buckets.tolist()):
            key = tuple(row[c] for c in columns)
            groups[level].setdefault(key, []).append(j)
    donors, levels = [], []
    for i, row in enumerate(buckets.tolist()):
        for level, columns in enumerate(keys):
            candidates = [j for j in groups[level][tuple(row[c] for c in columns)]
                          if identities[j] != identities[i]]
            if candidates:
                break
        else:
            level = 3
            candidates = [j for j in range(n) if identities[j] != identities[i]]
        if not candidates:
            raise ValueError("No distinct donor identity exists")
        pick = torch.randint(len(candidates), (), generator=generator).item()
        donors.append(candidates[pick])
        levels.append(level)
    return torch.tensor(donors), torch.tensor(levels)


def sample_fingerprint(dataset):
    digest = hashlib.sha256()
    for i in range(len(dataset)):
        digest.update(dataset.samples[i][3].encode())
        digest.update(b'\n')
    return digest.hexdigest()


@torch.no_grad()
def prepare_matching_cache(model, loader, path, identity, bucket_config=None, accelerator=None):
    bucket_config = {key: list(value) for key, value in (bucket_config or {}).items()}
    expected = {'protocol': MATCHING_PROTOCOL, 'identity': identity,
                'samples_sha256': sample_fingerprint(loader.dataset), 'buckets': bucket_config}
    path = Path(path)
    n = len(loader.dataset)

    def existing_cache():
        if not path.exists():
            return False
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload['metadata'] != expected or len(payload['scores']) != n:
            raise ValueError(f"Stale matching cache: {path}. Use a new path or remove it explicitly.")
        return True

    if main_process_call(existing_cache, accelerator):
        return torch.load(path, map_location='cpu', weights_only=True)
    local_indices, scores, lengths, popularity = [], [], [], []
    local_loader = shard_loader(loader, accelerator)
    for step, batch in enumerate(local_loader):
        moved = to_device(batch, model.baseline.device)
        baseline = model.score(model.baseline.prepare(moved), disabled=True)
        keep = valid_rows(batch)
        local_indices.extend(batch['sample_index'][keep].tolist())
        scores.append(baseline.sequence_score.cpu()[keep])
        lengths.append(batch['history_len'][keep].float())
        popularity.append(batch['target_popularity'][keep].float())
        if step % 100 == 0 and (accelerator is None or accelerator.is_main_process):
            logger.info("Frozen difficulty cache: %d/%d local batches", step, len(local_loader))
    local = {'indices': local_indices,
             'scores': torch.cat(scores) if scores else torch.empty(0),
             'lengths': torch.cat(lengths) if lengths else torch.empty(0),
             'popularity': torch.cat(popularity) if popularity else torch.empty(0)}
    records = gather_records([local], accelerator)

    def save_cache():
        all_scores, all_lengths, all_popularity = torch.empty(n), torch.empty(n), torch.empty(n)
        seen = []
        for record in records:
            index = record['indices']
            all_scores[index] = record['scores']
            all_lengths[index] = record['lengths']
            all_popularity[index] = record['popularity']
            seen.extend(index)
        if sorted(seen) != list(range(n)):
            raise ValueError('Distributed cache samples must appear exactly once')
        payload = {'metadata': expected, 'scores': all_scores,
                   'buckets': bucketize(all_lengths, all_popularity, all_scores, **bucket_config)}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)

    main_process_call(save_cache, accelerator)
    return torch.load(path, map_location='cpu', weights_only=True)
