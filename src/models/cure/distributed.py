"""Accelerate DDP/ZeRO setup, collective-safe inference, and branch checkpoints."""

import os
from contextlib import nullcontext
from functools import partial

import torch
from accelerate import DeepSpeedPlugin, Accelerator, DataLoaderConfiguration, DistributedDataParallelKwargs
from accelerate.utils import DistributedType, InitProcessGroupKwargs, broadcast_object_list, gather_object, is_wandb_available
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset


def deepspeed_config(cfg, training=True):
    distributed = cfg.get('distributed', {})
    if not distributed.get('use_deepspeed', False):
        return None
    config = OmegaConf.to_container(distributed.deepspeed, resolve=True)
    stage = config.get('zero_optimization', {}).get('stage', 0)
    if stage not in (1, 2, 3):
        raise ValueError('CURE supports ZeRO stages 1, 2 and 3')
    if 'optimizer' in config or 'scheduler' in config:
        raise ValueError('Configure optimizer/scheduler using train.*, not duplicate DeepSpeed sections')
    expected = {'train_micro_batch_size_per_gpu': cfg.train.per_device_batch_size,
                'gradient_accumulation_steps': cfg.train.gradient_accumulation_steps if training else 1,
                'gradient_clipping': cfg.train.max_grad_norm}
    expected['train_batch_size'] = expected['train_micro_batch_size_per_gpu'] * expected['gradient_accumulation_steps'] * int(os.environ.get('WORLD_SIZE', '1'))
    for key, value in expected.items():
        current = config.get(key, 'auto')
        if training and current != 'auto' and current != value:
            raise ValueError(f'DeepSpeed {key} must agree with train configuration: {current} != {value}')
        config[key] = value
    if stage < 3 and config['zero_optimization'].get('offload_param', {}).get('device', 'none') != 'none':
        raise ValueError('Parameter offload requires ZeRO-3')
    return config


def zero_stage(accelerator=None):
    plugin = getattr(getattr(accelerator, 'state', None), 'deepspeed_plugin', None)
    return plugin.deepspeed_config.get('zero_optimization', {}).get('stage', 0) if plugin else 0


def gather_parameters(parameters, modifier_rank=None):
    partitioned = [p for p in parameters if hasattr(p, 'ds_id')]
    if not partitioned:
        return nullcontext()
    from deepspeed import zero
    return zero.GatheredParameters(partitioned, modifier_rank=modifier_rank)


def configure_zero3_model(model, accelerator):
    if zero_stage(accelerator) != 3:
        return
    from deepspeed import zero
    # PyTorch MHA calls out_proj via functional linear rather than its forward.
    attention = model.reasoner.attention
    zero.register_external_parameter(attention, attention.out_proj.weight)
    if attention.out_proj.bias is not None:
        zero.register_external_parameter(attention, attention.out_proj.bias)


def save_branch_checkpoint(model, accelerator, path, metadata):
    # All ranks gather ONLY the small trainable branch, never the frozen backbone.
    parameters = list(model.reasoner.parameters()) + list(model.residual.parameters())
    with gather_parameters(parameters, modifier_rank=0):
        main_process_call(lambda: model.save_checkpoint(path, metadata), accelerator)


def create_accelerator(cfg, training=True):
    distributed = cfg.get('distributed', {})
    if cfg.train.get('wandb_enabled', False) and not is_wandb_available():
        raise ImportError('train.wandb_enabled=true requires the wandb package')
    cpu = str(cfg.model.device) == 'cpu' or not torch.cuda.is_available()
    ds_config = deepspeed_config(cfg, training)
    if ds_config is not None and cpu:
        raise ValueError('DeepSpeed CURE requires CUDA; use DDP for CPU tests')
    plugin = DeepSpeedPlugin(hf_ds_config=ds_config, zero3_init_flag=bool(distributed.get('zero3_init', True) and ds_config['zero_optimization']['stage'] == 3)) if ds_config is not None else None
    precision = distributed.get('mixed_precision', 'no')
    if ds_config is not None:
        precision = 'bf16' if ds_config.get('bf16', {}).get('enabled') else 'fp16' if ds_config.get('fp16', {}).get('enabled') else 'no'
    handlers = [DistributedDataParallelKwargs(find_unused_parameters=False, broadcast_buffers=False)]
    launched = int(os.environ.get('LOCAL_RANK', '-1')) >= 0 or int(os.environ.get('WORLD_SIZE', '1')) > 1
    if launched:
        handlers.append(InitProcessGroupKwargs(backend='gloo' if cpu else 'nccl'))
    accelerator = Accelerator(
        cpu=cpu,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps if training else 1,
        mixed_precision=precision if training or plugin is not None else 'no',
        deepspeed_plugin=plugin,
        step_scheduler_with_optimizer=False,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True, data_seed=cfg.train.seed),
        kwargs_handlers=handlers,
        log_with='wandb' if cfg.train.get('wandb_enabled', False) else None,
        project_dir=str(cfg.train.output_dir),
    )
    if accelerator.distributed_type not in (DistributedType.NO, DistributedType.MULTI_GPU, DistributedType.MULTI_CPU, DistributedType.DEEPSPEED):
        raise ValueError('CURE supports Accelerate DDP and DeepSpeed ZeRO, not FSDP')
    if accelerator.distributed_type == DistributedType.DEEPSPEED and ds_config is None:
        raise ValueError('Enable distributed.use_deepspeed=true or select --config-name cure_zero2/cure_zero3')
    requested = str(cfg.model.device)
    if accelerator.num_processes == 1 and requested not in ('auto', 'cpu', str(accelerator.device)):
        # torch.device('cuda') and torch.device('cuda:0') name the same current device.
        if not (requested == 'cuda' and accelerator.device.type == 'cuda'):
            raise ValueError('Select GPUs using CUDA_VISIBLE_DEVICES and model.device=auto')
    cfg.model.device = str(accelerator.device)
    return accelerator


def main_process_call(function, accelerator=None):
    """Propagate a rank-0 failure before other ranks enter the next collective."""
    if accelerator is None or accelerator.num_processes == 1:
        return function()
    message = [None]
    if accelerator.is_main_process:
        try:
            message[0] = (True, function())
        except Exception as error:
            message[0] = (False, f'{type(error).__name__}: {error}')
    broadcast_object_list(message)
    success, result = message[0]
    if not success:
        raise RuntimeError(f'Main-process operation failed: {result}')
    return result


def gather_records(records, accelerator=None):
    # Accelerate gather_object flattens one list level; callers always pass a list.
    return gather_object(records) if accelerator is not None and accelerator.num_processes > 1 else records


class _InferenceShard(Dataset):
    def __init__(self, dataset, indices, count):
        self.dataset, self.indices, self.count = dataset, list(indices), count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        real = index < len(self.indices)
        sample = dict(self.dataset[self.indices[index] if real else 0])
        sample['_valid_sample'] = real
        return sample


def _collate_inference(samples, collate):
    batch = collate(samples)
    batch['_valid_rows'] = torch.tensor([sample['_valid_sample'] for sample in samples])
    return batch


def valid_rows(batch):
    return batch.get('_valid_rows', torch.ones(len(batch['sample_index']), dtype=torch.bool)).bool()


def shard_loader(loader, accelerator=None):
    """ZeRO-3 pads work (not results) to align collectives; all samples count once."""
    if accelerator is None or accelerator.num_processes == 1:
        return loader
    indices = range(accelerator.process_index, len(loader.dataset), accelerator.num_processes)
    if zero_stage(accelerator) == 3:
        count = (len(loader.dataset) + accelerator.num_processes - 1) // accelerator.num_processes
        dataset = _InferenceShard(loader.dataset, indices, count)
        collate = partial(_collate_inference, collate=loader.collate_fn)
    else:
        dataset, collate = Subset(loader.dataset, indices), loader.collate_fn
    return DataLoader(dataset, batch_size=loader.batch_size, collate_fn=collate, shuffle=False, num_workers=0)


def init_tracking(cfg, accelerator, suffix=''):
    if not cfg.train.get('wandb_enabled', False):
        return
    kwargs = {'name': str(cfg.train.wandb_run_name) + suffix,
              'mode': cfg.train.get('wandb_mode', 'online'),
              'dir': str(cfg.train.output_dir),
              'group': cfg.train.get('wandb_group'),
              'entity': cfg.train.get('wandb_entity'),
              'tags': list(cfg.train.get('wandb_tags', [])),
              'config': {**OmegaConf.to_container(cfg, resolve=True),
                         'runtime': {'num_processes': accelerator.num_processes,
                                     'effective_batch_size': cfg.train.per_device_batch_size * accelerator.num_processes * cfg.train.gradient_accumulation_steps}}}
    # Supply config directly to wandb.init. Accelerate's store_init_configuration
    # restarts an offline run, otherwise creating an extra empty run.
    main_process_call(lambda: accelerator.init_trackers(
        project_name=cfg.train.wandb_project, init_kwargs={'wandb': kwargs}), accelerator)


def evaluation_logs(result, split='valid'):
    logs = {}
    for mode, metrics in result['metrics'].items():
        logs.update({f'{split}/{mode}/{key}': value for key, value in metrics.items()})
    for key in ('RG', 'PG', 'FD'):
        logs.update({f'{split}/{key}/{metric}': value for metric, value in result[key].items()})
    logs.update({f'{split}/{key}': value for key, value in result.items() if key.startswith('mean_')})
    return logs


def validation_logs(result):
    return evaluation_logs(result, 'valid')
