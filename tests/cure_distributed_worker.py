"""Opt-in torchrun worker validating real DDP, shared artifacts, and global matching.

Used by test_cure_distributed.py; synthetic data only, no research test split.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from inference.cure_evaluator import evaluate_cure
from inference.prefix_tree import SidPrefixTree
from models.cure import CUREModel, FrozenSFT
from models.cure.dataset import CUREDataset, make_loader
from models.cure.distributed import create_accelerator, gather_records, main_process_call
from models.cure.matching import prepare_matching_cache
from models.cure import trainer

SID = {0: [10, 20, 30], 1: [10, 20, 31], 2: [10, 21, 30],
       3: [11, 20, 30], 4: [11, 21, 31], 5: [11, 21, 32]}


class Tokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [1, 2] if text == '用户历史：' else [3, 4]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--precision', choices=['no', 'bf16'], default='no')
    parser.add_argument('--variant', default='cure_full')
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    root = Path(args.output)
    cfg = OmegaConf.load('configs/cure.yaml')
    cfg.model.device = 'cpu' if args.device == 'cpu' else 'auto'
    cfg.distributed.mixed_precision = args.precision
    cfg.train.epochs = 3
    cfg.train.per_device_batch_size = 2
    cfg.train.gradient_accumulation_steps = 2
    cfg.train.max_steps = 4  # early stop must terminate before this cap
    cfg.train.eval_steps = 1
    cfg.train.early_stop.patience_steps = 2
    cfg.train.early_stop.min_delta = 1.0
    cfg.train.wandb_log_steps = 1
    cfg.train.wandb_enabled = args.wandb
    cfg.train.wandb_mode = 'offline'
    cfg.train.output_dir = str(root / 'run')
    cfg.matching.cache_path = str(root / 'difficulty.pt')
    cfg.eval.num_beams = 10
    cfg.eval.k_values = [1, 5, 10]
    cfg.eval.batch_size = 2
    cfg.method.variant = args.variant
    accelerator = create_accelerator(cfg)

    def write_data():
        root.mkdir(parents=True, exist_ok=True)
        (root / 'sequences.json').write_text(json.dumps({str(i): [i % 6, (i + 1) % 6, (i + 2) % 6, (i + 3) % 6] for i in range(5)}))
        (root / 'valid.csv').write_text('user_id,item_id\n' + ''.join(f'{i},{(i + 4) % 6}\n' for i in range(5)))

    main_process_call(write_data, accelerator)
    tokenizer = Tokenizer()
    train = CUREDataset(root / 'sequences.json', SID, tokenizer)
    valid = CUREDataset(root / 'sequences.json', SID, tokenizer, split_path=root / 'valid.csv')
    assert len(train) == 15 and len(valid) == 5
    torch.manual_seed(11)
    backbone = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        attention_dropout=0., pad_token_id=0, eos_token_id=2, bos_token_id=1))
    model = CUREModel(FrozenSFT(backbone, SidPrefixTree(SID)), hidden_dim=16,
                       num_heads=2, dropout=0., residual_hidden_dim=24).to(accelerator.device)
    initial_base = [p.detach().cpu().clone() for p in model.baseline.parameters()]
    initial_branch = torch.cat([p.detach().flatten().cpu() for p in model.parameters() if p.requires_grad])
    consumed = []
    forward = model.forward

    def observed_forward(batch, donor_index=None, method=None):
        consumed.extend(batch['sample_index'].tolist())
        if donor_index is not None:
            ids = batch['sample_id']
            assert all(ids[i] != ids[j] for i, j in enumerate(donor_index.tolist()))
        return forward(batch, donor_index, method=method)

    model.forward = observed_forward
    synchronized_updates = []
    clip = accelerator.clip_grad_norm_

    def checked_clip(parameters, norm):
        params = list(parameters)
        gradients = torch.cat([p.grad.detach().float().flatten() for p in params])
        gathered = accelerator.gather(gradients[None])
        for row in gathered:
            torch.testing.assert_close(row, gathered[0], rtol=1e-6, atol=1e-7)
        assert gradients.abs().sum() > 0
        synchronized_updates.append(True)
        return clip(params, norm)

    accelerator.clip_grad_norm_ = checked_clip
    saved_paths = []
    save = model.save_checkpoint

    def observed_save(path, metadata=None):
        saved_paths.append(str(path))
        return save(path, metadata)

    model.save_checkpoint = observed_save
    trainer.load_components = lambda *unused: (model, tokenizer, SID, {})
    trainer.make_dataset = lambda *a, **kw: train if len(a) == 3 else valid
    identity = {key: 'synthetic' for key in ('sft_checkpoint_identity', 'tokenizer_identity', 'item_index_hash',
                                            'dataset_manifest', 'train_sequences_identity', 'decode_protocol')}
    trainer.provenance = lambda cfg: identity
    result = trainer._train(cfg, accelerator)
    assert result['steps'] == 3 and len(synchronized_updates) == 3
    assert result['stopped_early'] and result['early_stop']['num_evaluations'] == 3
    assert all(torch.equal(before, after.detach().cpu()) for before, after in zip(initial_base, model.baseline.parameters()))
    final_branch = torch.cat([p.detach().flatten() for p in model.parameters() if p.requires_grad])
    assert not torch.equal(initial_branch, final_branch.cpu())
    gathered = accelerator.gather(final_branch[None])
    for row in gathered:
        torch.testing.assert_close(row, gathered[0], rtol=1e-6, atol=1e-7)
    summaries = gather_records([{'rank': accelerator.process_index, 'samples': consumed, 'saves': len(saved_paths)}], accelerator)
    assert all(record['saves'] == 0 for record in summaries if record['rank'] != 0)
    # First epoch: 3 complete local batches, 6 different samples per rank, no overlap.
    assert len({i for record in summaries for i in record['samples'][:6]}) == 12
    paired = evaluate_cure(model, valid, 0, batch_size=2, num_beams=4, k_values=[1, 4],
                           output_path=root / 'parallel_eval.json', accelerator=accelerator)

    def check_single_process_parity():
        reference = evaluate_cure(model, valid, 0, batch_size=2, num_beams=4, k_values=[1, 4])
        for key in ('metrics', 'sample_ids', 'donor_sample_ids', 'matching_relaxation_counts', 'RG', 'PG', 'FD'):
            assert paired[key] == reference[key], key
        for key in (k for k in reference if k.startswith('mean_')):
            assert abs(paired[key] - reference[key]) < 1e-5, key
        if args.variant in ('cure_gp', 'cure_full'):
            cached = torch.load(root / 'difficulty.pt', weights_only=True)
            expected = prepare_matching_cache(model, make_loader(train, 2, 0), root / 'reference_cache.pt',
                                               {**identity, 'history_limit': train.history_limit}, dict(cfg.matching.buckets))
            torch.testing.assert_close(cached['scores'], expected['scores'], atol=1e-5, rtol=1e-5)
            assert torch.equal(cached['buckets'], expected['buckets'])
        checkpoint = torch.load(result['best_checkpoint'], weights_only=True)
        assert checkpoint['metadata']['distributed']['num_processes'] == 2
        report = {'world_size': 2, 'device': args.device, 'precision': args.precision,
                  'variant': args.variant, 'optimizer_steps': result['steps'],
                  'gradient_and_parameter_sync': True, 'frozen_baseline': True,
                  'global_matching_and_metrics_parity': True, 'evaluation_samples': len(paired['sample_ids']),
                  'rank_summaries': summaries, 'wandb_enabled': args.wandb}
        (root / 'report.json').write_text(json.dumps(report, indent=2))

    main_process_call(check_single_process_parity, accelerator)
    if accelerator.is_main_process and args.wandb:
        summary = dict(accelerator.get_tracker('wandb', unwrap=True).summary)
        required = {'train/rec', 'train/gain', 'train/personal', 'train/safe',
                    'valid/correct/NDCG@10', 'valid/FD/HR@10'}
        assert required.issubset(summary), summary.keys()
    accelerator.end_training()
    if accelerator.is_main_process and args.wandb:
        runs = list((root / 'run' / 'wandb').glob('offline-run-*'))
        assert len(runs) == 1, runs
        # Offline data remains local; no WANDB API call or upload is performed.
        report = json.loads((root / 'report.json').read_text())
        report['wandb_offline_runs'] = len(runs)
        (root / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
