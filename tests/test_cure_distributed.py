"""Two-process DDP integration: CPU by default, CUDA opt-in with CURE_DDP_DEVICE=cuda."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('variant', ['residual_ce', 'cure_full'])
def test_two_process_training_and_evaluation(tmp_path, variant):
    root = Path(__file__).resolve().parents[1]
    device = os.environ.get('CURE_DDP_DEVICE', 'cpu')
    precision = os.environ.get('CURE_DDP_PRECISION', 'no')
    args = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
            str(root / 'tests/cure_distributed_worker.py'), '--output', str(tmp_path),
            '--device', device, '--precision', precision, '--variant', variant]
    if variant == 'cure_full':
        args.append('--wandb')
    env = dict(os.environ, OMP_NUM_THREADS='2', WANDB_MODE='offline')
    result = subprocess.run(args, cwd=root, env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    report = json.loads((tmp_path / 'report.json').read_text())
    assert report['world_size'] == 2 and report['optimizer_steps'] == 3
    assert report['global_matching_and_metrics_parity']
    assert report['gradient_and_parameter_sync'] and report['frozen_baseline']
    if variant == 'cure_full':
        assert report['wandb_offline_runs'] == 1
