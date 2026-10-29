"""Optimizer-step patience and validation scheduling are shared by SFT/CURE."""

import pytest

from utils.early_stopping import ValidationControl


def test_patience_counts_updates_not_validation_calls():
    control = ValidationControl(eval_steps=500, mode='max', patience_steps=1000, min_delta=.1)
    assert not control.update(1., 500)
    assert not control.update(1.05, 999)  # epoch-end validation does not consume 500 steps
    assert control.best_step == 500
    assert control.update(1.05, 1500)
    assert control.steps_without_improvement == 1000


def test_min_mode_significant_improvement_resets_window():
    control = ValidationControl(mode='min', patience_steps=3, min_delta=.01)
    assert not control.update(1., 1)
    assert not control.update(.98, 3)
    assert control.best_step == 3
    assert not control.update(.975, 4)
    assert control.update(.979, 6)


def test_schedule_and_disable():
    control = ValidationControl(eval_steps=2, enabled=False, patience_steps=1)
    assert not control.should_evaluate(0, end_of_epoch=True)
    assert not control.should_evaluate(1)
    assert control.should_evaluate(2)
    control.update(1., 2)
    assert not control.should_evaluate(2, end_of_epoch=True)
    assert control.should_evaluate(3, end_of_epoch=True)
    assert not control.update(2., 3)
    epoch_only = ValidationControl(eval_steps=None)
    assert not epoch_only.should_evaluate(100)
    assert epoch_only.should_evaluate(100, end_of_epoch=True)


def test_invalid_metric_and_config():
    with pytest.raises(ValueError, match='Non-finite'):
        ValidationControl().update(float('nan'), 1)
    for kwargs in ({'patience_steps': 0}, {'eval_steps': 0}, {'min_delta': -1}, {'mode': 'bad'}):
        with pytest.raises(ValueError):
            ValidationControl(**kwargs)
    control = ValidationControl()
    control.update(1., 1)
    with pytest.raises(ValueError, match='increasing'):
        control.update(.9, 1)
