"""Stage 3 SFT 训练器：LoRA 微调 + DeepSpeed ZeRO-2（via accelerate）。"""

import logging
from functools import partial
from pathlib import Path

import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup

from .dataset import (
    ItemReconstructionDataset,
    MixedSFTDataset,
    SFTDataset,
    collate_fn,
)
from .tokenizer import get_item_sid_token_ids, load_tokenizer

logger = logging.getLogger(__name__)


def run_sft(cfg: DictConfig) -> None:
    """SFT 训练主流程。"""
    mcfg = cfg.model.llm
    tcfg = cfg.train
    dcfg = cfg.data
    accelerator = Accelerator(gradient_accumulation_steps=tcfg.gradient_accumulation_steps)

    # ── wandb 初始化 ─────────────────────────────────────────────────────────
    if accelerator.is_main_process and hasattr(tcfg, "wandb_project"):
        import wandb
        wandb.init(
            project=tcfg.wandb_project,
            name=tcfg.wandb_run_name or None,
            config={
                "model": mcfg.base_model,
                "epochs": tcfg.epochs,
                "batch_size": tcfg.per_device_batch_size,
                "gradient_accumulation_steps": tcfg.gradient_accumulation_steps,
                "learning_rate": tcfg.learning_rate,
                "lora_r": mcfg.lora.r,
                "lora_alpha": mcfg.lora.alpha,
                "max_seq_len": tcfg.max_seq_len,
                "aux_ratio": tcfg.aux_ratio,
            },
        )
        logger.info(f"wandb 已初始化: {tcfg.wandb_project}")

    # ── tokenizer ──────────────────────────────────────────────────────────
    logger.info("加载并扩展 tokenizer...")
    tokenizer = load_tokenizer(
        model_name=mcfg.base_model,
        num_layers=mcfg.rqvae.num_layers,
        codebook_size=mcfg.rqvae.codebook_size,
        save_dir=tcfg.tokenizer_save_dir,
        tokenizer_path=getattr(mcfg, "tokenizer_path", None),  # 如果配置了预扩展 tokenizer，直接加载
    )
    logger.info(f"tokenizer 词表大小: {len(tokenizer)}")

    item_sid_ids = get_item_sid_token_ids(tokenizer, dcfg.item_index_path)
    logger.info(f"加载 item SID 映射: {len(item_sid_ids)} 个商品")

    # ── dataset ────────────────────────────────────────────────────────────
    logger.info("构建 Dataset...")
    train_main = SFTDataset(
        user_sequences_path=dcfg.user_sequences_path,
        item_sid_ids=item_sid_ids,
        tokenizer=tokenizer,
        max_history_len=tcfg.max_history_len,
        max_seq_len=tcfg.max_seq_len,
        split_path=None,
    )
    train_aux = ItemReconstructionDataset(
        item_sid_ids=item_sid_ids,
        tokenizer=tokenizer,
        max_seq_len=tcfg.max_seq_len,
    )
    train_dataset = MixedSFTDataset(
        main_dataset=train_main,
        aux_dataset=train_aux,
        aux_ratio=tcfg.aux_ratio,
        seed=tcfg.seed,
    )
    logger.info(f"训练集大小: {len(train_dataset)} (主任务: {len(train_main)}, 辅助: {len(train_aux)})")

    valid_dataset = SFTDataset(
        user_sequences_path=dcfg.user_sequences_path,
        item_sid_ids=item_sid_ids,
        tokenizer=tokenizer,
        max_history_len=tcfg.max_history_len,
        max_seq_len=tcfg.max_seq_len,
        split_path=dcfg.valid_path,
    )
    logger.info(f"验证集大小: {len(valid_dataset)}")

    pad_id = tokenizer.pad_token_id
    train_loader = DataLoader(
        train_dataset,
        batch_size=tcfg.per_device_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=partial(collate_fn, pad_token_id=pad_id),
        pin_memory=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=tcfg.per_device_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(collate_fn, pad_token_id=pad_id),
        pin_memory=False,
    )

    # ── 模型 ───────────────────────────────────────────────────────────────
    logger.info(f"加载基座模型: {mcfg.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        mcfg.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=mcfg.lora.r,
        lora_alpha=mcfg.lora.alpha,
        lora_dropout=mcfg.lora.dropout,
        target_modules=list(mcfg.lora.target_modules),
        modules_to_save=["embed_tokens", "lm_head"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    import sys; sys.stdout.flush()

    # ── optimizer / scheduler ──────────────────────────────────────────────
    ga_steps = tcfg.gradient_accumulation_steps
    print(f"[DEBUG] ga_steps={ga_steps}", flush=True)
    total_steps = len(train_loader) * tcfg.epochs // ga_steps
    print(f"[DEBUG] total_steps={total_steps}", flush=True)
    warmup_steps = int(total_steps * tcfg.warmup_ratio)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg.learning_rate,
        weight_decay=tcfg.weight_decay,
    )
    print("[DEBUG] optimizer created", flush=True)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print("[DEBUG] scheduler created", flush=True)

    # ── accelerate 接管分布式 / DeepSpeed ──────────────────────────────────
    print("[DEBUG] calling accelerator.prepare...", flush=True)
    model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, scheduler
    )
    print("[DEBUG] accelerator.prepare done", flush=True)

    # ── 训练循环 ───────────────────────────────────────────────────────────
    import faulthandler, signal
    faulthandler.register(signal.SIGUSR1, file=__import__('sys').stdout, all_threads=True)

    best_valid_loss = float("inf")
    output_dir = Path(tcfg.output_dir)
    print(f"[DEBUG] output_dir={output_dir.resolve()}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("[DEBUG] output_dir created", flush=True)

    log_steps = getattr(tcfg, "wandb_log_steps", 50)
    global_step = 0

    for epoch in range(1, tcfg.epochs + 1):
        print(f"[DEBUG] epoch {epoch} start", flush=True)
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        print("[DEBUG] entering train_loader loop", flush=True)

        for step, batch in enumerate(train_loader):
            if step == 0:
                print(f"[DEBUG] first batch received, input_ids shape={batch['input_ids'].shape}", flush=True)
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            total_loss += loss.item()

            # 步级 wandb 日志
            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process and global_step % log_steps == 0:
                    try:
                        import wandb
                        wandb.log({
                            "train/step_loss": loss.item(),
                            "train/lr": scheduler.get_last_lr()[0],
                            "step": global_step,
                        })
                    except Exception:
                        pass

        avg_train_loss = total_loss / len(train_loader)
        valid_loss = _evaluate(model, valid_loader, accelerator)

        accelerator.print(
            f"Epoch {epoch}/{tcfg.epochs}  "
            f"train_loss={avg_train_loss:.4f}  valid_loss={valid_loss:.4f}"
        )
        if accelerator.is_main_process:
            logger.info(
                f"Epoch {epoch}/{tcfg.epochs}  "
                f"train_loss={avg_train_loss:.4f}  valid_loss={valid_loss:.4f}"
            )
            try:
                import wandb
                wandb.log({
                    "train/epoch_loss": avg_train_loss,
                    "valid/loss": valid_loss,
                    "epoch": epoch,
                    "step": global_step,
                })
            except Exception:
                pass

        # DeepSpeed save 是集合操作，所有 rank 必须一起调，否则 NCCL 死锁
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            accelerator.save_state(str(output_dir / "best_ckpt"))
            if accelerator.is_main_process:
                logger.info(f"  → 保存最优 checkpoint (valid_loss={valid_loss:.4f})")

    # ── wandb 收尾 ───────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    # ── 合并 LoRA 权重并保存 ───────────────────────────────────────────────
    if accelerator.is_main_process:
        logger.info("合并 LoRA 权重...")
        unwrapped = accelerator.unwrap_model(model)
        merged_model = unwrapped.merge_and_unload()
        merged_model.save_pretrained(str(output_dir / "merged"))
        tokenizer.save_pretrained(str(output_dir / "merged"))
        logger.info(f"合并后模型已保存: {output_dir / 'merged'}")


@torch.no_grad()
def _evaluate(model, loader: DataLoader, accelerator: Accelerator) -> float:
    model.eval()
    total_loss = 0.0
    for batch in loader:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        total_loss += outputs.loss.item()
    model.train()
    return total_loss / len(loader)
