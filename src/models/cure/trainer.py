"""CURE Phase 1 训练：冻结 merged SFT backbone，仅优化 latent 路径。"""

import logging
import math
from functools import partial
from pathlib import Path

import torch
from accelerate import Accelerator, DeepSpeedPlugin
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from models.llm.tokenizer import get_item_sid_token_ids
from .dataset import CUREDataset, CUREItemReconstructionDataset, MixedSFTDataset, collate_cure
from .model import CUREForCausalLM

logger = logging.getLogger(__name__)
_MODEL_INPUTS = ("input_ids", "attention_mask", "labels", "history_positions",
                 "history_item_mask", "latent_active")


def _forward_batch(model, batch):
    """只把模型需要的张量传给 forward，元数据留在 batch 中。

    input_ids/labels/attention_mask: [B, T]；history_positions: [B, N, 3]；
    history_item_mask: [B, N]；latent_active: [B]。
    """
    return model(**{key: batch[key] for key in _MODEL_INPUTS})


def run_cure(cfg):
    """执行 Latent-CE 训练与验证；多卡通信交由 Accelerate/DeepSpeed。"""
    if cfg.method.variant != "latent_ce" or cfg.method.lambda_util or cfg.method.lambda_anchor:
        raise NotImplementedError("Phase 1 supports Latent-CE only")

    tcfg = cfg.train
    distributed_cfg = cfg.get("distributed", {})
    deepspeed_cfg = distributed_cfg.get("deepspeed")
    # 与 SFT 相同：可选 ZeRO-2；wandb 只在全局主进程初始化。
    deepspeed_plugin = None
    if deepspeed_cfg is not None:
        deepspeed_plugin = DeepSpeedPlugin(
            hf_ds_config=OmegaConf.to_container(deepspeed_cfg, resolve=True)
        )
    accelerator = Accelerator(
        gradient_accumulation_steps=tcfg.gradient_accumulation_steps,
        mixed_precision=distributed_cfg.get("mixed_precision", "no"),
        deepspeed_plugin=deepspeed_plugin,
    )
    torch.manual_seed(tcfg.seed)

    wandb_run = None
    if accelerator.is_main_process and tcfg.get("wandb_project"):
        import wandb
        wandb_run = wandb.init(
            project=tcfg.wandb_project,
            name=tcfg.get("wandb_run_name") or None,
            config={
                "method": cfg.method.variant,
                "domain": cfg.data.domain,
                "sft_model_path": cfg.model.sft_model_path,
                "epochs": tcfg.epochs,
                "per_device_batch_size": tcfg.per_device_batch_size,
                "gradient_accumulation_steps": tcfg.gradient_accumulation_steps,
                "num_processes": accelerator.num_processes,
                "learning_rate": tcfg.learning_rate,
                "max_history_len": tcfg.max_history_len,
                "max_seq_len": tcfg.max_seq_len,
                "aux_ratio": tcfg.aux_ratio,
                "deepspeed_stage": deepspeed_cfg.zero_optimization.stage if deepspeed_cfg else None,
            },
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.sft_model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    item_sids = get_item_sid_token_ids(tokenizer, cfg.data.item_index_path)
    common = dict(user_sequences_path=cfg.data.user_sequences_path, item_sid_ids=item_sids,
                  tokenizer=tokenizer, max_history_len=tcfg.max_history_len,
                  max_seq_len=tcfg.max_seq_len)
    # 主任务显式携带历史位置；辅助商品重建设置 latent_active=False。
    train_main = CUREDataset(**common)
    train_aux = CUREItemReconstructionDataset(item_sids, tokenizer, tcfg.max_seq_len)
    train_data = MixedSFTDataset(train_main, train_aux, tcfg.aux_ratio, tcfg.seed)
    valid_data = CUREDataset(**common, split_path=cfg.data.valid_path)
    logger.info("CURE samples: train=%s (recommendation=%s, auxiliary=%s), valid=%s",
                len(train_data), len(train_main), len(train_data) - len(train_main), len(valid_data))
    collate = partial(collate_cure, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(train_data, batch_size=tcfg.per_device_batch_size,
                              shuffle=True, num_workers=0, collate_fn=collate)
    valid_loader = DataLoader(valid_data, batch_size=tcfg.per_device_batch_size,
                              shuffle=False, num_workers=0, collate_fn=collate)

    backbone = AutoModelForCausalLM.from_pretrained(
        cfg.model.sft_model_path, torch_dtype=torch.bfloat16, local_files_only=True
    )
    model = CUREForCausalLM(backbone, OmegaConf.to_container(cfg.model.reasoner, resolve=True))
    # backbone/LM head/SID embedding 均冻结，优化器只接收新增模块参数。
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=tcfg.learning_rate,
                                  weight_decay=tcfg.weight_decay)
    # Accelerate 会按进程切分 DataLoader；scheduler 按每张卡的更新次数计算。
    batches_per_rank = math.ceil(len(train_loader) / accelerator.num_processes)
    total_steps = max(1, math.ceil(batches_per_rank / tcfg.gradient_accumulation_steps)
                      * tcfg.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * tcfg.warmup_ratio), total_steps
    )
    model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, scheduler
    )
    logger.info("Accelerate prepared: processes=%s, mixed_precision=%s, steps=%s",
                accelerator.num_processes, accelerator.mixed_precision, total_steps)

    output_dir = Path(tcfg.output_dir)
    best = float("inf")
    global_step = 0
    optimizer.zero_grad()
    for epoch in range(1, tcfg.epochs + 1):
        model.train()
        # [loss 之和, batch 数]，epoch 末跨卡 reduce 后再求均值。
        train_stats = torch.zeros(2, device=accelerator.device)
        for batch in train_loader:
            with accelerator.accumulate(model):
                loss = _forward_batch(model, batch).loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            train_stats[0] += loss.detach()
            train_stats[1] += 1
            if accelerator.sync_gradients:
                global_step += 1
                if global_step % tcfg.wandb_log_steps == 0:
                    step_loss = loss.detach().item()
                    learning_rate = scheduler.get_last_lr()[0]
                    if accelerator.is_main_process:
                        logger.info("step=%s train_loss=%.4f lr=%.3g",
                                    global_step, step_loss, learning_rate)
                    if wandb_run is not None:
                        wandb_run.log({"train/step_loss": step_loss,
                                       "train/lr": learning_rate,
                                       "step": global_step})

        train_stats = accelerator.reduce(train_stats, reduction="sum")
        train_loss = (train_stats[0] / train_stats[1]).item()
        model.eval()
        valid_stats = torch.zeros(2, device=accelerator.device)
        with torch.no_grad():
            for batch in valid_loader:
                loss = _forward_batch(model, batch).loss
                valid_stats[0] += loss.detach()
                valid_stats[1] += 1
        valid_stats = accelerator.reduce(valid_stats, reduction="sum")
        valid_loss = (valid_stats[0] / valid_stats[1]).item()
        # logger 同时写主进程本地文件和终端；accelerator.print 只写终端。
        if accelerator.is_main_process:
            logger.info("epoch=%s train_loss=%.4f valid_loss=%.4f",
                        epoch, train_loss, valid_loss)
        if wandb_run is not None:
            wandb_run.log({"train/epoch_loss": train_loss, "valid/loss": valid_loss,
                           "epoch": epoch, "step": global_step})

        if valid_loss < best:
            best = valid_loss
            # DeepSpeed 的训练状态保存是集合操作，所有 rank 必须一起进入。
            accelerator.save_state(str(output_dir / "best_ckpt"))
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                # 另存轻量 CURE 权重供评估加载；best_ckpt 用于恢复训练。
                accelerator.unwrap_model(model).save_cure(output_dir / "best")
                logger.info("Saved best CURE checkpoint: valid_loss=%.4f", best)
            accelerator.wait_for_everyone()

    if wandb_run is not None:
        wandb_run.finish()
    if accelerator.is_main_process:
        logger.info("Best validation loss: %.4f", best)
