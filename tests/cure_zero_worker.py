"""CUDA-only two-rank ZeRO integration worker; synthetic data only."""

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cure_distributed_worker import SID, Tokenizer
from inference.prefix_tree import SidPrefixTree
from inference.cure_evaluator import evaluate_cure
from models.cure import CUREModel, FrozenSFT
from models.cure import trainer
from models.cure.dataset import CUREDataset
from models.cure.distributed import create_accelerator, main_process_call, gather_parameters, gather_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage', type=int, choices=[2, 3], required=True)
    parser.add_argument('--offload', choices=['none', 'cpu'], default='none')
    parser.add_argument('--optimizer-offload', choices=['none', 'cpu'], default='none')
    parser.add_argument('--variant', default='cure_full')
    args = parser.parse_args()
    root = Path(args.output)
    cfg = OmegaConf.load('configs/cure.yaml')
    cfg.distributed.use_deepspeed = True
    cfg.distributed.mixed_precision = 'bf16'
    cfg.distributed.deepspeed.zero_optimization.stage = args.stage
    cfg.distributed.deepspeed.zero_optimization.offload_param.device = args.offload
    cfg.distributed.deepspeed.zero_optimization.offload_optimizer.device = args.optimizer_offload
    cfg.distributed.deepspeed.zero_optimization.stage3_param_persistence_threshold = 0
    cfg.distributed.deepspeed.zero_optimization.reduce_bucket_size = 10000
    cfg.distributed.deepspeed.zero_optimization.stage3_prefetch_bucket_size = 10000
    cfg.train.epochs = 3
    cfg.train.per_device_batch_size = 2
    cfg.train.gradient_accumulation_steps = 2
    cfg.train.max_steps = 4  # three local batches, last update is a partial window
    cfg.train.eval_steps = 1
    cfg.train.early_stop.patience_steps = 1
    cfg.train.early_stop.min_delta = 1.0
    cfg.train.wandb_log_steps = 1
    cfg.train.output_dir = str(root / 'run')
    cfg.matching.cache_path = str(root / 'difficulty.pt')
    cfg.eval.num_beams = 10
    cfg.eval.k_values = [1, 5, 10]
    cfg.eval.batch_size = 2
    cfg.method.variant = args.variant
    accelerator = create_accelerator(cfg)

    def data_files():
        root.mkdir(parents=True, exist_ok=True)
        (root / 'sequences.json').write_text(json.dumps({str(i): [i % 6, (i + 1) % 6, (i + 2) % 6, (i + 3) % 6] for i in range(5)}))
        (root / 'valid.csv').write_text('user_id,item_id\n' + ''.join(f'{i},{(i + 4) % 6}\n' for i in range(5)))
        torch.manual_seed(11)
        tiny = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=48,
             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
             max_position_embeddings=256, attention_dropout=0., pad_token_id=0, eos_token_id=2, bos_token_id=1))
        tiny.save_pretrained(root / 'sft')
    main_process_call(data_files, accelerator)
    tokenizer = Tokenizer()
    train = CUREDataset(root / 'sequences.json', SID, tokenizer)
    valid = CUREDataset(root / 'sequences.json', SID, tokenizer, split_path=root / 'valid.csv')
    torch.manual_seed(11)
    import deepspeed
    import logging
    deepspeed.utils.logger.setLevel(logging.WARNING)
    backbone = AutoModelForCausalLM.from_pretrained(root / 'sft', torch_dtype=torch.bfloat16, local_files_only=True).to(accelerator.device)
    model = CUREModel(FrozenSFT(backbone, SidPrefixTree(SID)), hidden_dim=16, num_heads=2,
                       dropout=0., residual_hidden_dim=24).to(accelerator.device)
    with gather_parameters(model.baseline.parameters()):
        original_base = [p.detach().cpu().clone() for p in model.baseline.parameters()]
    with gather_parameters(list(model.reasoner.parameters()) + list(model.residual.parameters())):
        original_branch = torch.cat([p.detach().float().flatten().cpu() for p in model.parameters() if p.requires_grad])
    trainer.load_components = lambda *unused: (model, tokenizer, SID, {})
    trainer.make_dataset = lambda *a, **kw: train if len(a) == 3 else valid
    identity = {key: 'synthetic' for key in ('sft_checkpoint_identity', 'tokenizer_identity', 'item_index_hash',
                                            'dataset_manifest', 'train_sequences_identity', 'decode_protocol')}
    trainer.provenance = lambda cfg: identity
    result = trainer._train(cfg, accelerator)
    engine = accelerator._models[-1]
    assert result['steps'] == engine.global_steps == 2
    assert result['stopped_early'] and result['early_stop']['num_evaluations'] == 2
    assert float(engine.get_global_grad_norm()) > 0
    with gather_parameters(model.baseline.parameters()):
        assert all(torch.equal(before, after.detach().cpu()) for before, after in zip(original_base, model.baseline.parameters()))
    branch = list(model.reasoner.parameters()) + list(model.residual.parameters())
    with gather_parameters(branch):
        final = torch.cat([p.detach().float().flatten() for p in branch])
        assert not torch.equal(original_branch, final.cpu())
        collected = accelerator.gather(final[None])
        for row in collected:
            torch.testing.assert_close(row, collected[0], atol=0, rtol=0)
        payload = torch.load(root / 'run' / 'last.pt', map_location='cpu', weights_only=True)
        assert all(v.numel() > 0 for part in ('reasoner', 'residual') for v in payload[part].values())
        for name, module in (('reasoner', model.reasoner), ('residual', model.residual)):
            for key, value in module.state_dict().items():
                assert torch.equal(value.cpu(), payload[name][key]), (name, key)
    evaluated = evaluate_cure(model, valid, 0, batch_size=2, num_beams=4, k_values=[1, 4], accelerator=accelerator)
    # Odd number of samples exercises collective-only dummy work on rank 1.
    assert len(evaluated['sample_ids']) == len(set(evaluated['sample_ids'])) == 5
    local = {'rank': accelerator.process_index, 'partitioned': sum(hasattr(p, 'ds_id') for p in model.baseline.parameters()),
             'parameter_elements': sum(getattr(p, 'ds_numel', p.numel()) for p in model.baseline.parameters()),
             'local_shard_elements': sum(p.ds_tensor.numel() if hasattr(p, 'ds_tensor') else p.numel() for p in model.baseline.parameters())}
    ranks = gather_records([local], accelerator)
    if args.stage == 3:
        assert all(r['partitioned'] > 0 and r['local_shard_elements'] < r['parameter_elements'] for r in ranks)
    main_process_call(lambda: (root / 'report.json').write_text(json.dumps({
        'stage': args.stage, 'offload': args.offload, 'optimizer_offload': args.optimizer_offload,
        'steps': result['steps'], 'engine_steps': engine.global_steps, 'frozen_baseline': True,
        'checkpoint_full_branch': True, 'parameters_synchronized': True,
        'valid_samples': evaluated['sample_ids'], 'ranks': ranks}, indent=2)), accelerator)
    accelerator.end_training()


if __name__ == '__main__':
    main()
