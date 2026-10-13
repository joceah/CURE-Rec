"""CARE stage semantics and frozen leave-two-out sample handling."""

import csv

import torch
from transformers import Qwen2Config

from baselines.care.data import CAREDataset, load_split_samples
from baselines.care.model import CARE


def test_stage_visibility_and_train_inference_logits():
    config = Qwen2Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=4, max_position_embeddings=64)
    config._attn_implementation = "sdpa"
    config.care_query_list = [1, 1, 1]
    config.care_progressive_list = [True, True, True]
    config.care_sid_layers = 3
    model = CARE(config).eval()
    prompt = torch.tensor([[0, 0, 0, 4, 5, 6, 3], [4, 5, 6, 7, 8, 9, 3]])
    prompt_mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]])
    lengths = torch.tensor([1, 2])
    targets = torch.tensor([[10, 11, 12], [13, 14, 15]])
    embeds, mask, positions = model._stage_inputs(prompt, prompt_mask, lengths, targets[:, :-1])
    stage_mask = mask["full_attention"]
    assert embeds.shape[1] == 12
    assert positions == [7, 9, 11]
    assert stage_mask[0, 0, 7, 4] < 0  # stage 1 cannot see history level 2
    assert stage_mask[0, 0, 9, 4] == 0  # stage 2 can see history level 2
    assert stage_mask[0, 0, 7, 6] == 0  # answer token is always visible
    with torch.no_grad():
        train_logits = model._last_query_logits(prompt, prompt_mask, lengths, targets[:, :-1])
        for stage in range(3):
            infer_logits = model.score_next(prompt, prompt_mask, lengths, targets[:, :stage])
            torch.testing.assert_close(infer_logits, train_logits[:, stage], atol=1e-5, rtol=1e-5)
        assert torch.isfinite(model(prompt, prompt_mask, targets, lengths).loss)


def test_frozen_test_history_includes_valid_item(tmp_path):
    sequences = {1: [1, 2]}
    sids = {i: (i, i + 10, i + 20) for i in range(1, 5)}
    valid_path = tmp_path / "valid.csv"
    test_path = tmp_path / "test.csv"
    ids_path = tmp_path / "test_ids.csv"
    for path, rows in [
        (valid_path, [{"user_id": 1, "item_id": 3}]),
        (test_path, [{"user_id": 1, "item_id": 4}]),
    ]:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["user_id", "item_id"])
            writer.writeheader()
            writer.writerows(rows)
    with ids_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_index", "sample_id", "user_id", "target_item_id"])
        writer.writeheader()
        writer.writerow({"row_index": 0, "sample_id": "toy:test:1:4", "user_id": 1, "target_item_id": 4})
    samples = load_split_samples("test", sequences, str(test_path), str(ids_path), str(valid_path), sids)
    dataset = CAREDataset(sequences, sids, answer_id=31, max_history_len=2, split_samples=samples)
    assert samples[0]["history"] == [1, 2, 3]
    assert dataset[0]["input_ids"] == [2, 12, 22, 3, 13, 23, 31]
    assert dataset[0]["sample_id"] == "toy:test:1:4"


def test_care_lora_checkpoint_preserves_query_scores(tmp_path):
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    config = Qwen2Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=4,
                         num_key_value_heads=4, max_position_embeddings=64)
    config._attn_implementation = "sdpa"
    config.care_query_list = [1, 1, 1]
    config.care_progressive_list = [True, True, True]
    config.care_sid_layers = 3
    import copy
    original_config = copy.deepcopy(config)
    base = CARE(config)
    base_weights = {key: value.clone() for key, value in base.state_dict().items()}
    base.resize_token_embeddings(36, mean_resizing=False)
    adapter_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=4,
        target_modules=["q_proj"],
        modules_to_save=["query_embeddings", "embed_tokens", "lm_head"],
        ensure_weight_tying=True,
    )
    wrapped = get_peft_model(base, adapter_config).eval()
    prompt = torch.tensor([[4, 5, 6, 3]])
    mask = torch.ones_like(prompt)
    lengths = torch.tensor([1])
    labels = torch.tensor([[10, 11, 12]])
    with torch.no_grad():
        assert torch.isfinite(wrapped(input_ids=prompt, attention_mask=mask,
                                      labels=labels, history_lengths=lengths).loss)
        expected = wrapped.get_base_model().score_next(prompt, mask, lengths, labels[:, :1])
    wrapped.save_pretrained(tmp_path)
    restored_base = CARE(original_config)
    restored_base.load_state_dict(base_weights)
    restored_base.resize_token_embeddings(36, mean_resizing=False)
    restored = PeftModel.from_pretrained(restored_base, tmp_path).eval()
    with torch.no_grad():
        actual = restored.get_base_model().score_next(prompt, mask, lengths, labels[:, :1])
    torch.testing.assert_close(actual, expected)


def test_beam_search_returns_only_existing_sids():
    from baselines.care.evaluate import beam_search_item_sids
    from inference.prefix_tree import SidPrefixTree

    config = Qwen2Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=4,
                         num_key_value_heads=4, max_position_embeddings=64)
    config._attn_implementation = "sdpa"
    config.care_query_list = [1, 1, 1]
    config.care_progressive_list = [True, True, True]
    config.care_sid_layers = 3
    model = CARE(config).eval()
    item_sids = {1: [4, 5, 6], 2: [4, 5, 7], 3: [8, 9, 10]}
    tree = SidPrefixTree(item_sids)
    sample = {"input_ids": [4, 5, 6, 3], "history_lengths": 1}
    predictions = beam_search_item_sids(model, sample, tree, 2, 3, torch.device("cpu"))
    assert len(predictions) == 2
    assert len(set(predictions)) == 2
    assert set(predictions) <= {tuple(sid) for sid in item_sids.values()}
