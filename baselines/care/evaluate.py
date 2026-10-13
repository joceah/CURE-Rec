"""Evaluate CARE with exact SID-trie beam search on frozen sample IDs."""

import json
import logging
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from transformers import AutoConfig, AutoTokenizer

from .data import ANSWER_TOKEN, CAREDataset, load_item_sids, load_sequences, load_split_samples, resolve_model_source
from .model import CARE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from inference.metrics import compute_metrics  # noqa: E402
from inference.prefix_tree import SidPrefixTree  # noqa: E402

logger = logging.getLogger(__name__)


@torch.no_grad()
def beam_search_item_sids(model, sample, tree, num_beams, sid_layers, device):
    prompt = torch.tensor(sample["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
    prompt_mask = torch.ones_like(prompt)
    history_count = sample["history_lengths"]
    beams = [((), 0.0)]
    for _ in range(sid_layers):
        prefixes = torch.tensor([prefix for prefix, _ in beams], dtype=torch.long, device=device)
        scores = model.score_next(
            prompt.expand(len(beams), -1), prompt_mask.expand(len(beams), -1),
            torch.full((len(beams),), history_count, dtype=torch.long, device=device), prefixes,
        ).float().log_softmax(dim=-1)
        candidates = []
        for beam_index, (prefix, prior_score) in enumerate(beams):
            allowed = sorted(tree.get_allowed_tokens(prefix))
            if not allowed:
                raise ValueError(f"Invalid SID prefix: {prefix}")
            for token_id in allowed:
                candidates.append((prefix + (token_id,), prior_score + scores[beam_index, token_id].item()))
        beams = sorted(candidates, key=lambda entry: entry[1], reverse=True)[:num_beams]
    return [prefix for prefix, _ in beams]


def evaluate(cfg: DictConfig):
    if cfg.eval.num_beams < max(cfg.eval.k_values):
        raise ValueError("num_beams must cover the largest requested K")
    checkpoint = Path(cfg.eval.checkpoint_dir)
    saved_cfg = OmegaConf.load(checkpoint / "care_run_config.yaml")
    if saved_cfg.data.domain != cfg.data.domain:
        raise ValueError("Checkpoint domain differs from requested evaluation domain")
    if saved_cfg.data.item_index_path != cfg.data.item_index_path:
        raise ValueError("Checkpoint SID index differs from requested evaluation index")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    item_sids = load_item_sids(cfg.data.item_index_path, tokenizer, saved_cfg.model.sid_layers)
    answer_id = tokenizer.convert_tokens_to_ids(ANSWER_TOKEN)
    sequences = load_sequences(cfg.data.user_sequences_path)
    split = str(cfg.eval.split)
    split_path = cfg.data.valid_path if split == "valid" else cfg.data.test_path
    ids_path = cfg.data.valid_ids_path if split == "valid" else cfg.data.test_ids_path
    samples = load_split_samples(
        split, sequences, split_path, ids_path, cfg.data.valid_path,
        item_sids, cfg.eval.sample_size, cfg.train.seed,
    )
    dataset = CAREDataset(sequences, item_sids, answer_id, saved_cfg.train.max_history_len, samples)
    care_config = AutoConfig.from_pretrained(checkpoint / "care_model_config.json")
    base_source = resolve_model_source(
        saved_cfg.model.base_model, saved_cfg.model.local_files_only
    )
    base_config = AutoConfig.from_pretrained(
        base_source, local_files_only=saved_cfg.model.local_files_only
    )
    for name in ("care_query_list", "care_progressive_list", "care_sid_layers", "care_query_div_scale"):
        setattr(base_config, name, getattr(care_config, name))
    base = CARE.from_pretrained(
        base_source, config=base_config,
        local_files_only=saved_cfg.model.local_files_only,
        dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    base.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model = PeftModel.from_pretrained(base, checkpoint, is_trainable=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    care = model.get_base_model()
    tree = SidPrefixTree({item: list(sid) for item, sid in item_sids.items()})
    sid_to_items = {}
    for item, sid in item_sids.items():
        sid_to_items.setdefault(sid, []).append(item)
    predictions, targets, records = [], [], []
    for index in range(len(dataset)):
        example = dataset[index]
        predicted = beam_search_item_sids(
            care, example, tree, cfg.eval.num_beams, saved_cfg.model.sid_layers, device
        )
        target = tuple(example["labels"])
        predictions.append(predicted)
        targets.append(target)
        records.append({
            "sample_id": example["sample_id"], "target_item_id": example["target_item_id"],
            "predicted_sids": [list(sid) for sid in predicted],
            "predicted_item_ids": [sid_to_items[sid] for sid in predicted],
        })
        if (index + 1) % 100 == 0:
            logger.info("Evaluated %s/%s CARE samples", index + 1, len(dataset))
    metrics = compute_metrics(predictions, targets, list(cfg.eval.k_values))
    output_path = Path(cfg.eval.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "model": "CARE", "domain": cfg.data.domain, "split": split,
        "checkpoint_dir": str(checkpoint), "sample_count": len(dataset),
        "num_beams": cfg.eval.num_beams, "metric_unit": "SID",
        "ambiguous_sid_count": sum(len(items) > 1 for items in sid_to_items.values()),
        "metrics": metrics, "samples": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("CARE %s %s: %s", cfg.data.domain, split, metrics)
    return metrics


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    evaluate(cfg)


if __name__ == "__main__":
    main()
