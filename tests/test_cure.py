"""CURE Phase 1 invariants on a small random Qwen model."""

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from models.cure.model import CUREForCausalLM
from models.cure.dataset import CUREDataset, CUREItemReconstructionDataset, collate_cure


def _models():
    torch.manual_seed(7)
    config = Qwen2Config(vocab_size=96, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=128,
                         pad_token_id=0, eos_token_id=2)
    base = Qwen2ForCausalLM(config).eval()
    cure = CUREForCausalLM(base, {"hidden_dim": 16, "num_heads": 4,
                                 "max_history_len": 4, "dropout": 0.0}).eval()
    return base, cure


def _batch():
    ids = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11, 12, 13]])
    mask = torch.ones_like(ids)
    positions = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
    item_mask = torch.tensor([[True, True]])
    labels = torch.full_like(ids, -100)
    labels[0, 7:10] = ids[0, 7:10]
    return ids, mask, positions, item_mask, labels


def test_zero_matches_sft_and_normal_changes_logits():
    base, cure = _models()
    ids, mask, positions, item_mask, _ = _batch()
    with torch.no_grad():
        expected = base(ids, attention_mask=mask).logits
        zero = cure(ids, attention_mask=mask, latent_mode="zero").logits
        normal = cure(ids, attention_mask=mask, history_positions=positions,
                      history_item_mask=item_mask).logits
    torch.testing.assert_close(zero, expected, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(normal, zero)


def test_latent_has_no_target_leakage():
    _, cure = _models()
    ids, mask, positions, item_mask, _ = _batch()
    altered = ids.clone()
    altered[:, 7:] = torch.tensor([[20, 21, 22]])
    with torch.no_grad():
        first = cure.encode_latent(ids, mask, positions, item_mask)
        second = cure.encode_latent(altered, mask, positions, item_mask)
    torch.testing.assert_close(first, second)


def test_frozen_backbone_and_trainable_latent_gradients():
    _, cure = _models()
    ids, mask, positions, item_mask, labels = _batch()
    cure.train()
    loss = cure(ids, attention_mask=mask, labels=labels, history_positions=positions,
                history_item_mask=item_mask).loss
    loss.backward()
    assert all(p.grad is None for p in cure.backbone.parameters())
    for module in (cure.reasoner, cure.latent_projection, cure.fusion_gate):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_checkpoint_round_trip(tmp_path):
    base, cure = _models()
    ids, mask, positions, item_mask, _ = _batch()
    base.save_pretrained(tmp_path / "base")
    cure.save_cure(tmp_path / "cure")
    loaded = CUREForCausalLM.from_cure_pretrained(tmp_path / "base", tmp_path / "cure",
                                                   torch_dtype=torch.float32).eval()
    with torch.no_grad():
        expected = cure(ids, attention_mask=mask, history_positions=positions,
                        history_item_mask=item_mask).logits
        actual = loaded(ids, attention_mask=mask, history_positions=positions,
                        history_item_mask=item_mask).logits
    torch.testing.assert_close(actual, expected)


def test_train_generation_fusion_parity():
    _, cure = _models()
    ids, mask, positions, item_mask, _ = _batch()
    prompt = ids[:, :7]
    prompt_mask = mask[:, :7]
    z = cure.encode_latent(prompt, prompt_mask, positions, item_mask)
    with torch.no_grad():
        train_logits = cure(ids, attention_mask=mask, history_positions=positions,
                            history_item_mask=item_mask).logits[:, 6]
        generation_logits = cure(prompt, attention_mask=prompt_mask,
                                 latent_z=z).logits[:, -1]
    torch.testing.assert_close(train_logits, generation_logits, atol=1e-5, rtol=1e-5)
    output = cure.generate(prompt, attention_mask=prompt_mask, latent_z=z,
                           max_new_tokens=2, do_sample=False, num_beams=2)
    assert output.shape == (1, 9)


def test_dataset_positions_and_auxiliary_mask(tmp_path):
    import json

    class Tokenizer:
        eos_token_id = 2
        def encode(self, text, add_special_tokens=False):
            return [3, 4] if text.startswith("用户") else [5, 6]

    sequences = tmp_path / "users.json"
    sequences.write_text(json.dumps({"11": [1, 2]}))
    sids = {1: [7, 8, 9], 2: [10, 11, 12]}
    main = CUREDataset(str(sequences), sids, Tokenizer())
    auxiliary = CUREItemReconstructionDataset(sids, Tokenizer())
    sample = main[0]
    assert sample["history_positions"].tolist() == [[2, 3, 4]]
    assert sample["user_id"].item() == 11
    batch = collate_cure([sample, auxiliary[0]], pad_token_id=0)
    assert batch["latent_active"].tolist() == [True, False]
    assert batch["history_item_mask"].tolist() == [[True], [False]]


def test_bf16_history_with_float32_reasoner():
    _, cure = _models()
    _, _, positions, item_mask, _ = _batch()
    hidden = torch.randn(1, 10, 32, dtype=torch.bfloat16)
    latent = cure._history_latent(hidden, positions, item_mask)
    assert latent.dtype == torch.float32
    assert torch.isfinite(latent).all()


def test_generation_projects_only_last_position():
    _, cure = _models()
    ids, mask, positions, item_mask, _ = _batch()
    z = cure.encode_latent(ids, mask, positions, item_mask)
    with torch.no_grad():
        full = cure(ids, attention_mask=mask, latent_z=z).logits
        short = cure(ids, attention_mask=mask, latent_z=z, logits_to_keep=1).logits
    assert short.shape == (1, 1, full.size(-1))
    torch.testing.assert_close(short[:, 0], full[:, -1])

    lengths = []
    original_fuse = cure._fuse

    def record_fuse(hidden, *args, **kwargs):
        lengths.append(hidden.size(1))
        return original_fuse(hidden, *args, **kwargs)

    cure._fuse = record_fuse
    cure.generate(ids, attention_mask=mask, latent_z=z,
                  num_beams=2, max_new_tokens=2, do_sample=False)
    assert lengths and all(length == 1 for length in lengths)
