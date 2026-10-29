"""Stage 3 SFT 训练器：LoRA 微调 + DeepSpeed ZeRO-2（via accelerate）。"""

import logging
import json
import math
import random
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import DistributedType
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from peft.utils.other import ModulesToSaveWrapper
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup, set_seed
from torch.nn import functional as F
from utils.early_stopping import ValidationControl, update_early_stop

from .dataset import (
    ItemReconstructionDataset,
    MixedSFTDataset,
    SFTDataset,
    collate_fn,
)
from .tokenizer import get_item_sid_token_ids, load_tokenizer, resolve_model_source

logger = logging.getLogger(__name__)


def run_sft(cfg: DictConfig) -> None:
    """SFT 训练主流程。"""
    mcfg = cfg.model.llm
    tcfg = cfg.train
    dcfg = cfg.data
    distributed_cfg = cfg.get("distributed", {})
    deepspeed_cfg = distributed_cfg.get("deepspeed")
    deepspeed_plugin = None
    if deepspeed_cfg is not None:
        deepspeed_plugin = DeepSpeedPlugin(
            hf_ds_config=OmegaConf.to_container(deepspeed_cfg, resolve=True)
        )
    accelerator = Accelerator(
        gradient_accumulation_steps=tcfg.gradient_accumulation_steps,
        mixed_precision=distributed_cfg.get("mixed_precision"),
        deepspeed_plugin=deepspeed_plugin,
        step_scheduler_with_optimizer=False,
    )

    set_seed(tcfg.seed)
    # Validate stopping settings before loading the model.
    _validation_control(tcfg)

    # ── wandb 初始化 ─────────────────────────────────────────────────────────
    if accelerator.is_main_process and tcfg.get("wandb_enabled", True) and hasattr(tcfg, "wandb_project"):
        import wandb
        wandb.init(
            project=tcfg.wandb_project,
            name=tcfg.wandb_run_name or None,
            mode=tcfg.get("wandb_mode", "online"),
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
    local_files_only = getattr(mcfg, "local_files_only", False)
    model_source = resolve_model_source(mcfg.base_model, local_files_only=local_files_only)
    tokenizer = load_tokenizer(
        model_name=model_source,
        num_layers=mcfg.rqvae.num_layers,
        codebook_size=mcfg.rqvae.codebook_size,
        save_dir=tcfg.tokenizer_save_dir,
        tokenizer_path=getattr(mcfg, "tokenizer_path", None),  # 如果配置了预扩展 tokenizer，直接加载
        local_files_only=local_files_only,
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
    ecfg = cfg.get('eval', {})
    sample_size = ecfg.get('sample_size', 1000)
    if sample_size is not None:
        if sample_size < 1:
            raise ValueError('eval.sample_size must be positive or null')
        if sample_size < len(valid_dataset):
            indices = sorted(random.Random(ecfg.get('random_seed', tcfg.seed)).sample(range(len(valid_dataset)), sample_size))
            valid_dataset = Subset(valid_dataset, indices)
    logger.info(f"固定验证集大小: {len(valid_dataset)}")

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
        batch_size=cfg.get('eval', {}).get('batch_size', tcfg.per_device_batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=partial(collate_fn, pad_token_id=pad_id),
        pin_memory=False,
    )

    # ── 模型 ───────────────────────────────────────────────────────────────
    logger.info(f"加载基座模型: {model_source}")
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=local_files_only,
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

    try:
        return _train_sft(model, tokenizer, train_loader, valid_loader, cfg, accelerator)
    finally:
        if accelerator.is_main_process and tcfg.get('wandb_enabled', True):
            try:
                import wandb
                wandb.finish()
            except ImportError:
                pass
        accelerator.end_training()


def _validation_control(train):
    control = ValidationControl.from_config(train, 'loss', 'min')
    if control.monitor != 'loss':
        raise ValueError('SFT early_stop.monitor currently supports loss')
    return control


def _log_wandb(cfg, accelerator, values):
    if accelerator.is_main_process and cfg.train.get('wandb_enabled', True):
        import wandb
        if wandb.run is not None:
            wandb.log(values)


@contextmanager
def _cpu_model_init(accelerator):
    # Accelerate's zero3_init_context_manager(False) still advertises stage 3
    # to Transformers in some versions. Explicitly suspend its weak reference.
    from transformers.integrations.deepspeed import unset_hf_deepspeed_config, set_hf_deepspeed_config
    config = accelerator.state.deepspeed_plugin.dschf
    unset_hf_deepspeed_config()
    try:
        with torch.device('cpu'):
            yield
    finally:
        set_hf_deepspeed_config(config)


def _export_best(model, tokenizer, accelerator, cfg):
    output = Path(cfg.train.output_dir) / 'merged'
    raw = accelerator.unwrap_model(model)
    partitioned = any(hasattr(p, 'ds_id') for p in raw.parameters())
    if accelerator.is_main_process:
        if partitioned:
            # Consolidate on CPU rather than gathering the whole frozen model on GPU.
            from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
            state = get_fp32_state_dict_from_zero_checkpoint(str(Path(cfg.train.output_dir) / 'best_ckpt'))
            with _cpu_model_init(accelerator):
                cpu_base = AutoModelForCausalLM.from_config(raw.config, torch_dtype=next(raw.parameters()).dtype)
                cpu_base.resize_token_embeddings(len(tokenizer))
                cpu_model = get_peft_model(cpu_base, raw.peft_config['default'])
                cpu_model.load_state_dict(state, strict=True)
                merged = cpu_model.merge_and_unload()
        else:
            merged = raw.merge_and_unload()
        merged.save_pretrained(str(output))
        tokenizer.save_pretrained(str(output))
        logger.info('验证最优权重已合并保存: %s', output)
    accelerator.wait_for_everyone()


class _Zero3ModulesToSaveWrapper(ModulesToSaveWrapper):
    """Keep ZeRO backward hooks local instead of forwarding them to the active child.

    PEFT delegates unknown attributes to modules_to_save. ZeRO-3 checks these
    names while installing hooks; borrowing the child's hook captures the wrong
    module and breaks the next forward/backward. State-dict keys stay unchanged.
    """

    def __getattr__(self, name):
        if name in ('pre_bwd_fn', 'post_bwd_fn'):
            return torch.nn.Module.__getattr__(self, name)
        return super().__getattr__(name)


def _train_sft(model, tokenizer, train_loader, valid_loader, cfg, accelerator):
    train = cfg.train
    control = _validation_control(train)
    if min(train.epochs, train.gradient_accumulation_steps, train.wandb_log_steps) < 1:
        raise ValueError('epochs, gradient_accumulation_steps and wandb_log_steps must be positive')
    if train.get('max_steps') is not None and train.max_steps < 1:
        raise ValueError('train.max_steps must be positive or null')
    output = Path(train.output_dir)
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output / 'config.yaml', resolve=True)
    accelerator.wait_for_everyone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=train.learning_rate, weight_decay=train.weight_decay)
    plugin = accelerator.state.deepspeed_plugin
    if plugin and plugin.deepspeed_config.get('zero_optimization', {}).get('stage') == 3:
        for module in model.modules():
            if type(module) is ModulesToSaveWrapper:
                module.__class__ = _Zero3ModulesToSaveWrapper
    model, optimizer, train_loader, valid_loader = accelerator.prepare(model, optimizer, train_loader, valid_loader)
    ga_steps = train.gradient_accumulation_steps
    total_steps = math.ceil(len(train_loader) / ga_steps) * train.epochs
    if train.get('max_steps') is not None:
        total_steps = min(total_steps, train.max_steps)
    scheduler = accelerator.prepare_scheduler(get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * train.warmup_ratio), num_training_steps=total_steps))
    is_deepspeed = accelerator.distributed_type == DistributedType.DEEPSPEED
    best_valid_loss, best_checkpoint_step, global_step = float('inf'), None, 0
    logger.info('SFT update 上限=%d eval_steps=%s early_stop=%s patience_steps=%d',
                total_steps, control.eval_steps, control.enabled, control.patience_steps)

    def validate(epoch, avg_train_loss):
        nonlocal best_valid_loss, best_checkpoint_step
        value = _evaluate(model, valid_loader, accelerator)
        update_early_stop(control, value, global_step, accelerator)
        if value < best_valid_loss:
            best_valid_loss, best_checkpoint_step = value, global_step
            # Every rank must enter DeepSpeed save/load collectives together.
            accelerator.save_state(str(output / 'best_ckpt'))
        _log_wandb(cfg, accelerator, {'train/epoch_loss': avg_train_loss, 'valid/loss': value,
                                    'epoch': epoch, 'step': global_step, **control.logs()})
        if accelerator.is_main_process:
            logger.info('Validation step=%d loss=%.6f best_loss=%.6f stale_steps=%d/%d',
                        global_step, value, best_valid_loss, control.steps_without_improvement, control.patience_steps)
            if control.stopped:
                logger.info('Early stopping at update %d: validation loss did not improve by more than %.6g',
                            global_step, control.min_delta)

    optimizer.zero_grad()
    for epoch in range(1, train.epochs + 1):
        train_loader.set_epoch(epoch - 1)
        model.train()
        total_loss, batches = 0., 0
        for batch_index, batch in enumerate(train_loader):
            with accelerator.accumulate(*(() if is_deepspeed else (model,))):
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                loss = outputs.loss
                window_start = batch_index // ga_steps * ga_steps
                window_size = min(ga_steps, len(train_loader) - window_start)
                accelerator.backward(loss * ga_steps / window_size)
                if accelerator.sync_gradients:
                    if not is_deepspeed:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    if not accelerator.optimizer_step_was_skipped:
                        scheduler.step()
                        global_step += 1
                    optimizer.zero_grad()
            total_loss += loss.detach().item()
            batches += 1
            if accelerator.sync_gradients:
                if global_step % train.wandb_log_steps == 0 and not accelerator.optimizer_step_was_skipped:
                    avg_loss = accelerator.reduce(loss.detach().float(), reduction='mean').item()
                    _log_wandb(cfg, accelerator, {'train/step_loss': avg_loss, 'train/lr': scheduler.get_last_lr()[0], 'step': global_step})
                if control.should_evaluate(global_step):
                    validate(epoch, total_loss / batches)
                if control.stopped or global_step >= total_steps:
                    break
        if control.should_evaluate(global_step, end_of_epoch=True):
            validate(epoch, total_loss / max(1, batches))
        if control.stopped or global_step >= total_steps:
            break
    if best_checkpoint_step is None:
        raise RuntimeError('SFT produced no successful optimizer update/validation checkpoint')
    reason = 'early_stop' if control.stopped else 'max_steps' if train.get('max_steps') is not None and global_step >= train.max_steps else 'epochs'
    summary = {'steps': global_step, 'best_checkpoint_step': best_checkpoint_step,
               'best_valid_loss': best_valid_loss, 'stop_reason': reason,
               'stopped_early': control.stopped, 'early_stop': control.state_dict(),
               'best_checkpoint': str(output / 'best_ckpt'), 'merged_path': str(output / 'merged')}
    if accelerator.is_main_process:
        (output / 'training_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    accelerator.wait_for_everyone()
    # Restore before exporting even if training ended by epochs/max_steps, not only early stop.
    accelerator.load_state(str(output / 'best_ckpt'))
    accelerator.wait_for_everyone()
    _export_best(model, tokenizer, accelerator, cfg)
    return summary


@torch.no_grad()
def _evaluate(model, loader: DataLoader, accelerator: Accelerator) -> float:
    """Global token-weighted loss, trimming distributed sampler padding exactly."""
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0., 0.
    for batch in loader:
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                        use_cache=False, return_dict=True)
        targets = batch['labels'][:, 1:]
        mask = targets != -100
        per_sample = torch.zeros(len(targets), dtype=torch.float64, device=targets.device)
        if mask.any():
            # Only materialize FP32 vocabulary logits for labeled tokens, not the full prompt.
            token_loss = F.cross_entropy(outputs.logits[:, :-1][mask].float(), targets[mask], reduction='none')
            per_sample.scatter_add_(0, mask.nonzero()[:, 0], token_loss.double())
        pairs = torch.stack((per_sample, mask.sum(-1).double()), dim=-1)
        gathered = accelerator.gather_for_metrics(pairs)
        total_loss += gathered[:, 0].sum().item()
        total_tokens += gathered[:, 1].sum().item()
    model.train(was_training)
    if total_tokens == 0:
        raise ValueError('Validation has no labeled target tokens')
    return total_loss / total_tokens
