"""Accelerate DDP/DeepSpeed training with branch-only checkpoints and W&B."""

import json
import logging
import math
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import set_seed
from accelerate.utils import DistributedType

from inference.cure_evaluator import evaluate_cure
from utils.early_stopping import ValidationControl, update_early_stop
from .dataset import make_loader, make_train_loader, to_device
from .distributed import create_accelerator, init_tracking, main_process_call, validation_logs, save_branch_checkpoint, zero_stage, gather_records, configure_zero3_model
from .losses import VARIANTS
from .matching import match_donors, prepare_matching_cache
from .runtime import load_components, make_dataset, provenance, assert_checkpoint_identity

logger = logging.getLogger(__name__)


def selection_key(result):
    return (result['metrics']['correct']['NDCG@10'], result['metrics']['correct']['HR@10'],
            result['mean_target_score_gain'], -result['FD']['HR@10'])


def _initialize_output(cfg):
    output = Path(cfg.train.output_dir)
    artifacts = [output / name for name in
                 ('best.pt', 'last.pt', 'best_validation.json', 'training_summary.json')]
    artifacts += list(output.glob('valid_step_*.json')) + list(output.glob('valid_epoch_*.json'))
    existing = [path for path in artifacts if path.exists() or path.is_symlink()]
    if existing and not cfg.train.get('overwrite_output_dir', False):
        raise FileExistsError(
            f'Refusing to overwrite an existing run: {output}. '
            'Set train.overwrite_output_dir=true to restart in this directory.')
    output.mkdir(parents=True, exist_ok=True)
    if cfg.train.get('overwrite_output_dir', False):
        # Delete only generated run files; preserve unrelated files and W&B history.
        for path in existing:
            path.unlink()
    OmegaConf.save(cfg, output / 'config.yaml', resolve=True)


def run_cure(cfg):
    if cfg.data.domain == 'sports' and (not cfg.eval.design_frozen or not cfg.eval.recipe_id):
        raise ValueError('Sports confirmation requires a frozen recipe_id and design_frozen=true')
    if cfg.method.variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {cfg.method.variant}")
    if min(cfg.train.epochs, cfg.train.gradient_accumulation_steps, cfg.train.wandb_log_steps, cfg.train.per_device_batch_size) < 1:
        raise ValueError('epochs, gradient_accumulation_steps, wandb_log_steps and per_device_batch_size must be positive')
    if cfg.train.max_steps is not None and cfg.train.max_steps < 1:
        raise ValueError('max_steps must be positive or null')
    if 10 not in cfg.eval.k_values or cfg.eval.num_beams < max(cfg.eval.k_values):
        raise ValueError('Checkpoint selection needs HR/NDCG@10 and beams covering all K')
    accelerator = create_accelerator(cfg)
    try:
        return _train(cfg, accelerator)
    finally:
        accelerator.end_training()


def _train(cfg, accelerator):
    control = ValidationControl.from_config(cfg.train, 'NDCG@10', 'max')
    allowed_metrics = {f'{metric}@{k}' for metric in ('HR', 'NDCG') for k in cfg.eval.k_values}
    if control.monitor not in allowed_metrics:
        raise ValueError(f'early_stop.monitor must be a configured correct-branch metric: {control.monitor}')
    output = Path(cfg.train.output_dir)

    set_seed(cfg.train.seed)
    # Read an in-place initialization checkpoint before removing the old run files.
    init = cfg.train.init_checkpoint
    preload = bool(cfg.train.get('overwrite_output_dir', False) and init
                   and Path(init).absolute().parent.resolve() == output.resolve())
    if preload:
        components = load_components(cfg, init)
        accelerator.wait_for_everyone()
    main_process_call(lambda: _initialize_output(cfg), accelerator)
    model, tokenizer, sid_ids, saved = components if preload else load_components(cfg, init)
    identity = main_process_call(lambda: provenance(cfg), accelerator)
    if saved:
        assert_checkpoint_identity(saved, identity)
    metadata = main_process_call(lambda: {
        **identity, 'model_variant': cfg.method.variant,
        'method': OmegaConf.to_container(cfg.method, resolve=True),
        'config': OmegaConf.to_container(cfg, resolve=True),
        'distributed': {'num_processes': accelerator.num_processes,
                        'per_device_batch_size': cfg.train.per_device_batch_size,
                        'effective_batch_size': cfg.train.per_device_batch_size * accelerator.num_processes * cfg.train.gradient_accumulation_steps,
                        'mixed_precision': accelerator.mixed_precision,
                        'train_drop_global_tail': accelerator.num_processes > 1}}, accelerator)
    init_tracking(cfg, accelerator)
    train = make_dataset(cfg, tokenizer, sid_ids, sample_size=cfg.train.sample_size)
    valid = make_dataset(cfg, tokenizer, sid_ids, 'valid', cfg.eval.sample_size)
    needs_donor = cfg.method.variant in ('cure_gp', 'cure_full')
    loader = make_train_loader(train, cfg.train.per_device_batch_size, tokenizer.pad_token_id,
                               needs_donor, accelerator.num_processes, cfg.train.seed)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    # Let Accelerate shard the actual loader. Iterating its batch_sampler manually
    # would bypass both sharding and end-of-loader gradient accumulation tracking.
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    raw_model = accelerator.unwrap_model(model)
    configure_zero3_model(raw_model, accelerator)
    is_deepspeed = accelerator.distributed_type == DistributedType.DEEPSPEED
    metadata['distributed']['engine'] = 'deepspeed' if is_deepspeed else 'ddp'
    metadata['distributed']['zero_stage'] = zero_stage(accelerator)
    if accelerator.device.type == 'cuda':
        metadata['distributed']['memory_after_prepare'] = gather_records([{
            'rank': accelerator.process_index,
            'allocated_bytes': torch.cuda.memory_allocated(accelerator.device),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(accelerator.device)}], accelerator)
        torch.cuda.reset_peak_memory_stats(accelerator.device)
        metadata['distributed']['memory_peak_scope'] = 'after prepare: difficulty cache + train + validation; CUDA allocated bytes, excludes driver/NCCL allocations'
    buckets = None
    if needs_donor:
        cache_identity = {k: identity[k] for k in ('sft_checkpoint_identity', 'tokenizer_identity',
                                                  'item_index_hash', 'dataset_manifest', 'train_sequences_identity', 'decode_protocol')}
        cache_identity['history_limit'] = train.history_limit
        cache_identity['scoring_engine'] = {'engine': metadata['distributed']['engine'],
                                            'zero_stage': zero_stage(accelerator),
                                            'model_dtype': str(cfg.model.dtype),
                                            'deepspeed_config': OmegaConf.to_container(cfg.distributed.deepspeed, resolve=True) if is_deepspeed else None}
        cache = prepare_matching_cache(raw_model, make_loader(train, cfg.train.cache_per_device_batch_size, tokenizer.pad_token_id),
                                       cfg.matching.cache_path, cache_identity, dict(cfg.matching.buckets), accelerator)
        buckets = cache['buckets']
    accumulation = cfg.train.gradient_accumulation_steps
    num_batches = len(loader)
    total_steps = math.ceil(num_batches / accumulation) * cfg.train.epochs
    if cfg.train.max_steps is not None:
        total_steps = min(total_steps, cfg.train.max_steps)
    warmup_steps = int(total_steps * cfg.train.warmup_ratio)

    def lr_scale(step):
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0., (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = accelerator.prepare_scheduler(torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale))
    donor_rng = torch.Generator().manual_seed(cfg.train.seed + accelerator.process_index)
    best_key, global_step, best_checkpoint_step = None, 0, None
    log_started, log_step = time.perf_counter(), 0
    if accelerator.is_main_process:
        logger.info('%s ranks=%d per-device batch=%d effective batch=%d updates=%d',
                    metadata['distributed']['engine'], accelerator.num_processes, cfg.train.per_device_batch_size, metadata['distributed']['effective_batch_size'], total_steps)
        logger.info('Validation eval_steps=%s early_stop=%s monitor=%s mode=%s patience_steps=%d min_delta=%.6g',
                    control.eval_steps, control.enabled, control.monitor, control.mode, control.patience_steps, control.min_delta)

    def validate(epoch):
        nonlocal best_key, best_checkpoint_step, log_started
        validation_started = time.perf_counter()
        result = evaluate_cure(raw_model, valid, tokenizer.pad_token_id, batch_size=cfg.eval.batch_size,
                               num_beams=cfg.eval.num_beams, k_values=list(cfg.eval.k_values),
                               seed=cfg.eval.random_seed, bucket_config=dict(cfg.matching.buckets),
                               metadata={**metadata, 'split': 'valid', 'epoch': epoch, 'step': global_step},
                               output_path=output / f'valid_step_{global_step}.json', accelerator=accelerator)
        if accelerator.device.type == 'cuda':
            metadata['distributed']['memory_after_validation'] = gather_records([{
                'rank': accelerator.process_index,
                'allocated_bytes': torch.cuda.memory_allocated(accelerator.device),
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(accelerator.device)}], accelerator)
        update_early_stop(control, result['metrics']['correct'][control.monitor], global_step, accelerator)
        key = selection_key(result)
        improved = best_key is None or key > best_key
        checkpoint_metadata = {**metadata, 'epoch': epoch, 'step': global_step,
                               'validation_selection': list(key), 'early_stop': control.state_dict()}
        save_branch_checkpoint(raw_model, accelerator, output / 'last.pt', checkpoint_metadata)
        if improved:
            best_key, best_checkpoint_step = key, global_step
            save_branch_checkpoint(raw_model, accelerator, output / 'best.pt', checkpoint_metadata)
            main_process_call(lambda: (output / 'best_validation.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n'), accelerator)
        accelerator.log({**validation_logs(result), **control.logs(), 'epoch': epoch}, step=global_step)
        if accelerator.is_main_process:
            logger.info('Validation step=%d NDCG@10=%.6f HR@10=%.6f monitor=%s best_step=%d stale_steps=%d/%d',
                        global_step, result['metrics']['correct']['NDCG@10'], result['metrics']['correct']['HR@10'],
                        control.monitor, control.best_step, control.steps_without_improvement, control.patience_steps)
        if control.stopped and accelerator.is_main_process:
            logger.info('Early stopping at update %d: %s did not improve by more than %.6g for %d updates',
                        global_step, control.monitor, control.min_delta, control.steps_without_improvement)
        log_started += time.perf_counter() - validation_started

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(cfg.train.epochs):
        model.train()
        loader.set_epoch(epoch)
        window_logs, window_count = None, 0
        for batch_index, batch in enumerate(loader):
            donor = None
            if needs_donor:
                indices = batch['sample_index'].cpu()
                donor, _ = match_donors(buckets[indices], donor_rng, batch['sample_id'])
                donor = donor.to(accelerator.device)
            with accelerator.accumulate(*(() if is_deepspeed else (model,))):
                logs = model(to_device(batch, accelerator.device), donor, method=dict(cfg.method))
                detached = torch.stack([value.detach().float() for value in logs.values()])
                window_logs = detached if window_logs is None else window_logs + detached
                window_count += 1
                # Accelerate divides by K; compensate when the final window has < K batches.
                window_start = (batch_index // accumulation) * accumulation
                window_size = min(accumulation, num_batches - window_start)
                # Accelerate forwards sync_gradients to its DeepSpeed wrapper.
                accelerator.backward(logs['loss'] * accumulation / window_size)
                if accelerator.sync_gradients:
                    if not is_deepspeed:
                        accelerator.clip_grad_norm_(params, cfg.train.max_grad_norm)
                    optimizer.step()
                    if not accelerator.optimizer_step_was_skipped:
                        scheduler.step()
                        global_step += 1
                    optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                if global_step % cfg.train.wandb_log_steps == 0 and not accelerator.optimizer_step_was_skipped:
                    reduced = accelerator.reduce(window_logs / window_count, reduction='mean')
                    logged = {f'train/{key}': value for key, value in zip(logs, reduced.tolist())}
                    logged.update({'train/lr': scheduler.get_last_lr()[0], 'epoch': epoch + 1,
                                   'train/seconds_per_update': (time.perf_counter() - log_started) / max(1, global_step - log_step)})
                    accelerator.log(logged, step=global_step)
                    if accelerator.is_main_process:
                        logger.info('epoch=%d step=%d %s', epoch + 1, global_step, logged)
                    log_started, log_step = time.perf_counter(), global_step
                window_logs, window_count = None, 0
                if control.should_evaluate(global_step):
                    validate(epoch + 1)
                if control.stopped or global_step >= total_steps:
                    break
        if control.should_evaluate(global_step, end_of_epoch=True):
            validate(epoch + 1)
        if control.stopped or global_step >= total_steps:
            break
    if best_checkpoint_step is None:
        raise RuntimeError('CURE produced no successful optimizer update/validation checkpoint')
    reason = 'early_stop' if control.stopped else 'max_steps' if cfg.train.max_steps is not None and global_step >= cfg.train.max_steps else 'epochs'
    summary = {'best_checkpoint': str(output / 'best.pt'), 'steps': global_step, 'selection': best_key,
               'best_checkpoint_step': best_checkpoint_step, 'stop_reason': reason,
               'stopped_early': control.stopped, 'early_stop': control.state_dict()}
    main_process_call(lambda: (output / 'training_summary.json').write_text(json.dumps(summary, indent=2) + '\n'), accelerator)
    return summary
