"""ZeRO config consistency and opt-in real two-GPU DeepSpeed integration."""

import json
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf
import pytest
import torch

from models.cure.distributed import deepspeed_config
from types import SimpleNamespace
from accelerate.utils import DistributedType


def test_deepspeed_batch_and_accumulation_consistency(monkeypatch):
    cfg = OmegaConf.load('configs/cure.yaml')
    assert deepspeed_config(cfg) is None
    cfg.distributed.use_deepspeed = True
    monkeypatch.setenv('WORLD_SIZE', '2')
    config = deepspeed_config(cfg)
    assert config['train_batch_size'] == cfg.train.per_device_batch_size * 2 * cfg.train.gradient_accumulation_steps
    cfg.distributed.deepspeed.gradient_accumulation_steps = 99
    with pytest.raises(ValueError, match='gradient_accumulation_steps'):
        deepspeed_config(cfg)
    inference = deepspeed_config(cfg, training=False)
    assert inference['gradient_accumulation_steps'] == 1
    assert inference['train_batch_size'] == cfg.train.per_device_batch_size * 2


def test_zero2_rejects_parameter_offload():
    cfg = OmegaConf.load('configs/cure.yaml')
    cfg.distributed.use_deepspeed = True
    cfg.distributed.deepspeed.zero_optimization.stage = 2
    cfg.distributed.deepspeed.zero_optimization.offload_param.device = 'cpu'
    with pytest.raises(ValueError, match='Parameter offload requires ZeRO-3'):
        deepspeed_config(cfg)


def test_deepspeed_launcher_requires_explicit_config(monkeypatch):
    import models.cure.distributed as distributed
    cfg = OmegaConf.load('configs/cure.yaml')
    fake = SimpleNamespace(distributed_type=DistributedType.DEEPSPEED, num_processes=2,
                           device=torch.device('cuda:0'))
    monkeypatch.setattr(distributed, 'Accelerator', lambda **kwargs: fake)
    with pytest.raises(ValueError, match='distributed.use_deepspeed=true'):
        distributed.create_accelerator(cfg)


@pytest.mark.skipif(os.environ.get('CURE_ZERO_TEST') != '1', reason='Opt-in two-GPU DeepSpeed test')
@pytest.mark.parametrize('stage,offload,optimizer_offload,variant', [
    (2, 'none', 'none', 'residual_ce'),
    (3, 'none', 'none', 'cure_full'),
    (3, 'cpu', 'cpu', 'cure_full'),
])
def test_two_gpu_zero_training(tmp_path, stage, offload, optimizer_offload, variant):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip('Requires two CUDA GPUs')
    pytest.importorskip('deepspeed')
    root = Path(__file__).resolve().parents[1]
    args = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
            str(root / 'tests/cure_zero_worker.py'), '--output', str(tmp_path), '--stage', str(stage),
            '--offload', offload, '--optimizer-offload', optimizer_offload, '--variant', variant]
    result = subprocess.run(args, cwd=root, env=dict(os.environ, OMP_NUM_THREADS='2'),
                            capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, (result.stdout + '\n' + result.stderr)[-16000:]
    report = json.loads((tmp_path / 'report.json').read_text())
    assert report['steps'] == report['engine_steps'] == 2
    assert report['frozen_baseline'] and report['checkpoint_full_branch'] and report['parameters_synchronized']
    assert len(report['valid_samples']) == len(set(report['valid_samples'])) == 5
