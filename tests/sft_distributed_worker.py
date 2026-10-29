"""Synthetic LoRA early-stop/save/restore smoke under DDP and ZeRO."""
import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import InitProcessGroupKwargs
from omegaconf import OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

from models.llm import trainer
from test_sft_training import Tokenizer, examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage', type=int, default=0)
    args = parser.parse_args()
    root = Path(args.output)
    cfg = OmegaConf.load('configs/sft.yaml')
    cfg.train.output_dir = str(root / 'run')
    cfg.train.epochs = 3
    cfg.train.max_steps = 4
    cfg.train.gradient_accumulation_steps = 2
    cfg.train.wandb_enabled = False
    cfg.train.eval_steps = 1
    cfg.train.early_stop.patience_steps = 2
    cfg.train.learning_rate = .01
    plugin = None
    if args.stage:
        config = OmegaConf.to_container(cfg.distributed.deepspeed, resolve=True)
        config['train_micro_batch_size_per_gpu'] = 2
        config['zero_optimization'].update(stage=args.stage, reduce_bucket_size=10000,
            allgather_bucket_size=10000, stage3_prefetch_bucket_size=10000,
            stage3_param_persistence_threshold=0)
        plugin = DeepSpeedPlugin(hf_ds_config=config, zero3_init_flag=args.stage == 3)
    accelerator = Accelerator(cpu=not args.stage, mixed_precision='bf16' if args.stage else 'no',
        gradient_accumulation_steps=2, deepspeed_plugin=plugin, step_scheduler_with_optimizer=False,
        kwargs_handlers=[InitProcessGroupKwargs(backend='nccl' if args.stage else 'gloo')])
    dtype = torch.bfloat16 if args.stage else torch.float32
    if accelerator.is_main_process:
        root.mkdir(parents=True, exist_ok=True)
        with trainer._cpu_model_init(accelerator) if plugin else nullcontext():
            torch.manual_seed(9)
            Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=48,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=64, pad_token_id=0, eos_token_id=2, bos_token_id=1)).save_pretrained(root / 'base')
    accelerator.wait_for_everyone()
    base = AutoModelForCausalLM.from_pretrained(root / 'base', torch_dtype=dtype, local_files_only=True)
    model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=4,
        lora_dropout=0., target_modules=['q_proj'], modules_to_save=['embed_tokens', 'lm_head']))
    original_evaluate = trainer._evaluate
    seen, snapshots = [], []
    probe = torch.stack([x['input_ids'] for x in examples(2)]).to(accelerator.device)

    def evaluate(model, loader, acc):
        # Exercise actual distributed token weighting and sampler-padding removal.
        loss = original_evaluate(model, loader, acc)
        values = acc.gather(torch.tensor([loss], device=acc.device))
        assert torch.allclose(values, values[0].expand_as(values), atol=0, rtol=0)
        seen.append(loss)
        training = model.training
        model.eval()
        with torch.no_grad():
            snapshots.append(model(input_ids=probe).logits.float().cpu())
        model.train(training)
        return .5 + .1 * (len(seen) - 1)

    trainer._evaluate = evaluate
    result = trainer._train_sft(model, Tokenizer(), DataLoader(examples(32), batch_size=2),
        DataLoader(examples(5), batch_size=2), cfg, accelerator)
    assert result['steps'] == 3 and result['stopped_early']
    assert result['best_checkpoint_step'] == 1 and len(seen) == 3
    decisions = accelerator.gather(torch.tensor([result['steps']], device=accelerator.device))
    assert decisions.tolist() == [3, 3]
    if accelerator.is_main_process:
        with trainer._cpu_model_init(accelerator) if plugin else nullcontext():
            merged = AutoModelForCausalLM.from_pretrained(root / 'run/merged', local_files_only=True).eval()
        with torch.no_grad():
            actual = merged(probe.cpu()).logits.float()
        torch.testing.assert_close(actual, snapshots[0], atol=.025 if args.stage else 1e-5, rtol=.025 if args.stage else 1e-5)
        assert not torch.allclose(actual, snapshots[-1], atol=.01, rtol=.01)
        (root / 'report.json').write_text(json.dumps(result))
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == '__main__':
    main()
