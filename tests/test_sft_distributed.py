"""Two-process SFT validation, early stopping and best merged-model export."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


@pytest.mark.parametrize('stage', [0, 2, 3])
def test_distributed_sft_early_stop(tmp_path, stage):
    if stage and (os.environ.get('CURE_ZERO_TEST') != '1' or torch.cuda.device_count() < 2):
        pytest.skip('Opt-in two-GPU DeepSpeed test')
    root = Path(__file__).resolve().parents[1]
    args = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
        str(root / 'tests/sft_distributed_worker.py'), '--output', str(tmp_path), '--stage', str(stage)]
    result = subprocess.run(args, cwd=root, env=dict(os.environ, OMP_NUM_THREADS='2'),
        capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, (result.stdout + '\n' + result.stderr)[-18000:]
    report = json.loads((tmp_path / 'report.json').read_text())
    assert report['steps'] == 3 and report['best_checkpoint_step'] == 1 and report['stopped_early']
