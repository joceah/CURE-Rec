"""Candidate-normalized beam decoding and paired CURE mechanism evaluation."""

import json
from pathlib import Path

import torch

from inference.metrics import compute_metrics
from transformers import LogitsProcessor
from models.cure.dataset import make_loader, to_device
from models.cure.distributed import gather_records, main_process_call, shard_loader, valid_rows, zero_stage
from models.cure.losses import safe_kl
from models.cure.matching import MATCHING_PROTOCOL, bucketize, match_donors
from models.cure.runtime import DECODE_PROTOCOL


class _CURELogitsProcessor(LogitsProcessor):
    def __init__(self, model, prompt_length, latent, hidden, disabled):
        self.model = model
        self.prompt_length = prompt_length
        self.latent = latent
        self.hidden = hidden
        self.disabled = disabled

    def __call__(self, input_ids, scores):
        result = torch.full_like(scores, float('-inf'))
        # One CPU prefix transfer and one residual call for the entire beam batch.
        prefixes = [row[self.prompt_length:] for row in input_ids.tolist()]
        packed = self.model.baseline.pack_candidates(self.hidden['last'], prefixes, scores)
        latent = None if self.latent is None else self.latent.expand(len(input_ids), -1)
        corrected, _, _ = self.model.residual(
            packed.hidden, latent, packed.basis, packed.logits, disabled=self.disabled,
            candidate_indices=packed.basis_indices, candidate_mask=packed.mask)
        rows = torch.arange(len(input_ids), device=scores.device)[:, None].expand_as(packed.ids)
        result[rows[packed.mask], packed.ids[packed.mask]] = corrected[packed.mask]
        return result


@torch.no_grad()
def generate_cure_sids(model, prompt, latent=None, num_beams=50, disabled=False):
    """Use SFT's own HF beam/cache implementation, normalized on legal children.

    The hook observes the frozen current-prefix state; the processor only adds
    candidate-space residuals. A disabled correction preserves the SFT path.
    """
    if num_beams < 2:
        raise ValueError("Beam decoding requires num_beams >= 2")
    if model.training:
        raise ValueError("Generation requires model.eval() for deterministic residuals")
    if latent is None and not disabled:
        raise ValueError('Enabled residual decoding requires a personalized latent')
    baseline = model.baseline
    ids = torch.tensor([prompt], device=baseline.device, dtype=torch.long)
    hidden = {}

    def observe(module, args, output):
        hidden['last'] = output.last_hidden_state[:, -1]

    hook = baseline.model.base_model.register_forward_hook(observe)
    try:
        processor = _CURELogitsProcessor(model, len(prompt), latent, hidden, disabled)
        output = baseline.model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=3, min_new_tokens=3, num_beams=num_beams,
            num_return_sequences=num_beams, do_sample=False, early_stopping=False,
            logits_processor=[processor], renormalize_logits=True,
            pad_token_id=baseline.model.config.pad_token_id,
            synced_gpus=any(hasattr(p, 'ds_id') for p in baseline.model.parameters()))
    finally:
        hook.remove()
    return [tuple(row) for row in output[:, len(prompt):].tolist()]


@torch.no_grad()
def evaluate_cure(model, dataset, pad_token_id, batch_size=8, num_beams=50,
                  k_values=(1, 5, 10, 20, 50), seed=42, bucket_config=None,
                  metadata=None, output_path=None, accelerator=None):
    # Generation and uneven inference shards must not enter DDP forward collectives.
    if accelerator is not None:
        model = accelerator.unwrap_model(model)
    if len(dataset) < 2:
        raise ValueError("Paired evaluation needs at least two samples")
    if num_beams < max(k_values):
        raise ValueError("num_beams must cover every reported K")
    was_training = model.training
    model.eval()
    loader = shard_loader(make_loader(dataset, batch_size, pad_token_id), accelerator)
    records = []
    # Gather evidence once so matching uses the same global pool at any world size.
    for batch in loader:
        features = model.baseline.prepare(to_device(batch, model.baseline.device))
        keep = valid_rows(batch)
        z = model.encode(features).float().cpu()
        scores = model.score(features, disabled=True).sequence_score.cpu()
        records.append({'indices': batch['sample_index'][keep].tolist(),
                        'latent': z[keep], 'scores': scores[keep],
                        'lengths': batch['history_len'][keep], 'popularity': batch['target_popularity'][keep]})
    records = gather_records(records, accelerator)
    n = len(dataset)
    latents = torch.empty(n, model.architecture['hidden_dim'])
    baseline_scores, lengths, popularity = torch.empty(n), torch.empty(n), torch.empty(n)
    seen = []
    for record in records:
        index = record['indices']
        latents[index] = record['latent']
        baseline_scores[index] = record['scores']
        lengths[index] = record['lengths'].float()
        popularity[index] = record['popularity'].float()
        seen.extend(index)
    if sorted(seen) != list(range(n)):
        raise ValueError('Distributed evaluation samples must appear exactly once')
    buckets = bucketize(lengths, popularity, baseline_scores, **(bucket_config or {}))
    donors, levels = match_donors(buckets, torch.Generator().manual_seed(seed),
                                  [s[3] for s in dataset.samples])
    local_results = []
    for batch in loader:
        features = model.baseline.prepare(to_device(batch, model.baseline.device))
        index = batch['sample_index']
        z = latents[index].to(model.baseline.device)
        wrong_z = latents[donors[index]].to(model.baseline.device)
        base = model.score(features, disabled=True)
        correct = model.score(features, z)
        matched = model.score(features, wrong_z)
        batch_stats = {
            'mean_target_score_gain': (correct.sequence_score - base.sequence_score).tolist(),
            'mean_personalized_score_gap': (correct.sequence_score - matched.sequence_score).tolist(),
            'mean_safe_KL': safe_kl(base, matched).tolist(),
            'mean_gate': correct.gates.mean(-1).tolist(),
            'mean_correction_norm': correct.correction_norms.mean(-1).tolist()}
        for i in range(len(index)):
            prompt = batch['input_ids'][i, :batch['prompt_length'][i]].tolist()
            target = tuple(features.targets[i].tolist())
            predictions = {}
            for mode, latent in (('baseline', None), ('correct', z[i]), ('matched', wrong_z[i])):
                predictions[mode] = generate_cure_sids(model, prompt, latent, num_beams, disabled=mode == 'baseline')
            if not valid_rows(batch)[i]:
                continue
            ranks = [predictions[mode].index(target) + 1 if target in predictions[mode] else num_beams + 1
                     for mode in ('baseline', 'correct')]
            local_results.append({'index': index[i].item(), 'target': target,
                                  'sample_id': batch['sample_id'][i], 'predictions': predictions,
                                  'stats': {**{key: values[i] for key, values in batch_stats.items()},
                                            'mean_target_rank_shift': ranks[0] - ranks[1]}})
    results = sorted(gather_records(local_results, accelerator), key=lambda record: record['index'])
    if [record['index'] for record in results] != list(range(n)):
        raise ValueError('Distributed predictions must cover each sample exactly once')
    targets = [record['target'] for record in results]
    sample_ids = [record['sample_id'] for record in results]
    metrics = {mode: compute_metrics([record['predictions'][mode] for record in results], targets, k_values)
               for mode in ('baseline', 'correct', 'matched')}
    stats = {key: sum(record['stats'][key] for record in results) / n for key in results[0]['stats']}
    result = dict(metadata or {})
    result.update({'decode_protocol': DECODE_PROTOCOL, 'seed': seed, 'beam_size': num_beams,
                   'sample_ids': sample_ids, 'metrics': metrics,
                   'num_processes': accelerator.num_processes if accelerator is not None else 1,
                   'evaluation_sharding': 'sample-index-stride; results count each sample once',
                   'zero3_collective_padding': zero_stage(accelerator) == 3,
                   'RG': {f'HR@{k}': metrics['correct'][f'HR@{k}'] - metrics['baseline'][f'HR@{k}'] for k in k_values},
                   'PG': {f'HR@{k}': metrics['correct'][f'HR@{k}'] - metrics['matched'][f'HR@{k}'] for k in k_values},
                   'FD': {f'HR@{k}': metrics['baseline'][f'HR@{k}'] - metrics['matched'][f'HR@{k}'] for k in k_values},
                   'donor_sample_ids': [sample_ids[j] for j in donors.tolist()],
                   'matching_protocol': MATCHING_PROTOCOL,
                   'matching_buckets': {k: list(v) for k, v in (bucket_config or {}).items()},
                   'matching_relaxation_counts': torch.bincount(levels, minlength=4).tolist(),
                   'rank_shift_protocol': 'baseline-minus-correct; absent target rank=beam_size+1',
                   'correction_norm_protocol': 'L2 norm of gated correction over legal candidates; mean over 3 stages',
                   **stats})
    if output_path:
        def write_result():
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
        main_process_call(write_result, accelerator)
    model.train(was_training)
    return result
