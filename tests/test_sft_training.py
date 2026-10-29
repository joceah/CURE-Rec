"""Real tiny LoRA updates, early stop, and best-checkpoint merged export."""

import json
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import Qwen2Config, Qwen2ForCausalLM, AutoModelForCausalLM

from models.llm import trainer


class Tokenizer:
    def __len__(self):
        return 64

    def save_pretrained(self, path):
        from pathlib import Path
        (Path(path) / 'tokenizer_smoke.json').write_text('{}')


def examples(n):
    result = []
    for i in range(n):
        ids = torch.tensor([1, 3, 10 + i % 2, 20 + i % 2, 30 + i % 2, 2])
        labels = ids.clone()
        labels[:2] = -100
        result.append({'input_ids': ids, 'labels': labels, 'attention_mask': torch.ones_like(ids)})
    return result


def tiny_model():
    torch.manual_seed(9)
    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, pad_token_id=0, eos_token_id=2, bos_token_id=1))
    return get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=4,
                         lora_dropout=0., target_modules=['q_proj'], modules_to_save=['embed_tokens', 'lm_head']))


@pytest.mark.parametrize('enabled,expected_steps', [(True, 3), (False, 4)])
def test_sft_stop_and_export_best(tmp_path, monkeypatch, enabled, expected_steps):
    cfg = OmegaConf.load('configs/sft.yaml')
    cfg.train.output_dir = str(tmp_path / 'run')
    cfg.train.epochs = 3
    cfg.train.max_steps = 4
    cfg.train.gradient_accumulation_steps = 2
    cfg.train.wandb_enabled = False
    cfg.train.eval_steps = 1
    cfg.train.early_stop.enabled = enabled
    cfg.train.early_stop.patience_steps = 2
    cfg.train.learning_rate = .01
    model = tiny_model()
    probe = torch.stack([x['input_ids'] for x in examples(2)])
    seen, first, last = [], [], []

    def validate(model, loader, accelerator):
        seen.append(len(seen) + 1)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            values = accelerator.unwrap_model(model)(input_ids=probe).logits.clone()
        model.train(was_training)
        if not first:
            first.append(values)
        last[:] = [values]
        return .5 + (len(seen) - 1) * .1

    monkeypatch.setattr(trainer, '_evaluate', validate)
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2, mixed_precision='no', step_scheduler_with_optimizer=False)
    try:
        result = trainer._train_sft(model, Tokenizer(), DataLoader(examples(16), batch_size=2),
                                   DataLoader(examples(5), batch_size=2), cfg, accelerator)
    finally:
        accelerator.end_training()
    assert result['steps'] == expected_steps and result['best_checkpoint_step'] == 1
    assert result['stopped_early'] == enabled
    assert seen == list(range(1, expected_steps + 1))
    merged = AutoModelForCausalLM.from_pretrained(tmp_path / 'run' / 'merged', local_files_only=True).eval()
    with torch.no_grad():
        actual = merged(input_ids=probe).logits
    torch.testing.assert_close(actual, first[0], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(actual, last[0], atol=1e-4, rtol=1e-4)
    assert json.loads((tmp_path / 'run' / 'training_summary.json').read_text())['best_checkpoint_step'] == 1


def test_validation_uses_target_token_weighting():
    class Fixed(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            logits = torch.nn.functional.one_hot(input_ids, num_classes=64).float() * 3
            return SimpleNamespace(logits=logits)
    data = examples(5)
    for i, sample in enumerate(data):
        sample['labels'][:2 + i % 4] = -100
    accelerator = Accelerator(cpu=True, mixed_precision='no')
    loader = accelerator.prepare_data_loader(DataLoader(data, batch_size=2))
    model = Fixed().train()
    actual = trainer._evaluate(model, loader, accelerator)
    ids = torch.stack([x['input_ids'] for x in data])
    labels = torch.stack([x['labels'] for x in data])[:, 1:]
    mask = labels != -100
    expected = torch.nn.functional.cross_entropy(model(ids).logits[:, :-1][mask], labels[mask]).item()
    assert actual == pytest.approx(expected, abs=1e-6)
    assert model.training
