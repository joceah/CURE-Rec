"""Train CARE against the frozen Beauty/Sports SID artifacts."""

import json
import logging
from functools import partial
from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, get_cosine_schedule_with_warmup

from .data import ANSWER_TOKEN, CAREDataset, collate_care, load_item_sids, load_sequences, load_split_samples, resolve_model_source
from .model import CARE

logger = logging.getLogger(__name__)


def prepare_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_model_source(cfg.model.base_model, cfg.model.local_files_only),
        local_files_only=cfg.model.local_files_only,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    prefixes = "abcd"
    if cfg.model.sid_layers > len(prefixes):
        raise ValueError("Unsupported SID layer count")
    tokens = [f"<{prefixes[layer]}_{code}>"
              for layer in range(cfg.model.sid_layers)
              for code in range(cfg.model.codebook_size)]
    tokenizer.add_special_tokens({"additional_special_tokens": tokens + [ANSWER_TOKEN]})
    return tokenizer


def load_base_model(cfg, vocab_size):
    source = resolve_model_source(cfg.model.base_model, cfg.model.local_files_only)
    config = AutoConfig.from_pretrained(
        source, local_files_only=cfg.model.local_files_only,
        trust_remote_code=True,
    )
    config.care_query_list = list(cfg.model.query_list)
    config.care_progressive_list = list(cfg.model.progressive_list)
    config.care_sid_layers = int(cfg.model.sid_layers)
    config.care_query_div_scale = float(cfg.model.query_div_scale)
    model = CARE.from_pretrained(
        source, config=config, local_files_only=cfg.model.local_files_only,
        dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model.resize_token_embeddings(vocab_size, mean_resizing=False)
    model.config.use_cache = False
    return model


def train(cfg: DictConfig):
    torch.manual_seed(cfg.train.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        mixed_precision="bf16",
    )
    tokenizer = prepare_tokenizer(cfg)
    item_sids = load_item_sids(cfg.data.item_index_path, tokenizer, cfg.model.sid_layers)
    sequences = load_sequences(cfg.data.user_sequences_path)
    answer_id = tokenizer.convert_tokens_to_ids(ANSWER_TOKEN)
    train_dataset = CAREDataset(sequences, item_sids, answer_id, cfg.train.max_history_len)
    valid_samples = load_split_samples(
        "valid", sequences, cfg.data.valid_path, cfg.data.valid_ids_path,
        cfg.data.valid_path, item_sids, cfg.train.valid_sample_size, cfg.train.seed,
    )
    valid_dataset = CAREDataset(
        sequences, item_sids, answer_id, cfg.train.max_history_len, valid_samples
    )
    collate = partial(collate_care, pad_id=tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.per_device_batch_size,
                              shuffle=True, collate_fn=collate, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.train.per_device_batch_size,
                              shuffle=False, collate_fn=collate, num_workers=0)
    model = load_base_model(cfg, len(tokenizer))
    lora = cfg.model.lora
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=lora.r, lora_alpha=lora.alpha,
        lora_dropout=lora.dropout, target_modules=list(lora.target_modules),
        modules_to_save=["query_embeddings", "embed_tokens", "lm_head"],
        ensure_weight_tying=True,
    ))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    model, optimizer, train_loader, valid_loader = accelerator.prepare(
        model, optimizer, train_loader, valid_loader
    )
    steps = (len(train_loader) * cfg.train.epochs + cfg.train.gradient_accumulation_steps - 1) // cfg.train.gradient_accumulation_steps
    scheduler = accelerator.prepare(get_cosine_schedule_with_warmup(
        optimizer, max(1, int(steps * 0.03)), max(1, steps)
    ))
    output_dir = Path(cfg.train.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(output_dir)
        (output_dir / "care_run_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
        (output_dir / "care_model_config.json").write_text(
            accelerator.unwrap_model(model).get_base_model().config.to_json_string(), encoding="utf-8"
        )
    accelerator.wait_for_everyone()
    best_valid = float("inf")
    for epoch in range(cfg.train.epochs):
        model.train()
        for batch in train_loader:
            with accelerator.accumulate(model):
                output = model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    history_lengths=batch["history_lengths"], labels=batch["labels"],
                )
                accelerator.backward(output.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in valid_loader:
                output = model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    history_lengths=batch["history_lengths"], labels=batch["labels"],
                )
                losses.append(accelerator.gather_for_metrics(output.loss.detach().repeat(len(batch["labels"]))))
        valid_loss = torch.cat(losses).mean().item()
        accelerator.print(f"CARE epoch {epoch + 1}: valid_loss={valid_loss:.6f}")
        if valid_loss < best_valid:
            best_valid = valid_loss
            accelerator.wait_for_everyone()
            accelerator.unwrap_model(model).save_pretrained(
                output_dir, is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )
            if accelerator.is_main_process:
                (output_dir / "training_summary.json").write_text(json.dumps(
                    {"best_epoch": epoch + 1, "valid_loss": valid_loss,
                     "train_samples": len(train_dataset), "valid_samples": len(valid_dataset)},
                    indent=2), encoding="utf-8")
            accelerator.wait_for_everyone()


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    train(cfg)


if __name__ == "__main__":
    main()
