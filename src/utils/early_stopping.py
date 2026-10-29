"""Shared validation scheduling and optimizer-step-based early stopping."""

import math
from dataclasses import dataclass


@dataclass
class ValidationControl:
    eval_steps: int | None = 500
    enabled: bool = True
    monitor: str = 'loss'
    mode: str = 'min'
    patience_steps: int = 2500
    min_delta: float = 0.0001
    best_value: float | None = None
    best_step: int | None = None
    last_eval_step: int | None = None
    num_evaluations: int = 0
    stopped: bool = False

    def __post_init__(self):
        if self.eval_steps is not None and (isinstance(self.eval_steps, bool) or not isinstance(self.eval_steps, int) or self.eval_steps < 1):
            raise ValueError('train.eval_steps must be a positive integer or null')
        if isinstance(self.patience_steps, bool) or not isinstance(self.patience_steps, int) or self.patience_steps < 1:
            raise ValueError('train.early_stop.patience_steps must be a positive integer')
        if self.mode not in ('min', 'max'):
            raise ValueError('train.early_stop.mode must be min or max')
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError('train.early_stop.min_delta must be finite and nonnegative')

    @classmethod
    def from_config(cls, train, monitor, mode):
        early = train.get('early_stop', {})
        return cls(eval_steps=train.get('eval_steps', 500), enabled=early.get('enabled', True),
                   monitor=early.get('monitor', monitor), mode=early.get('mode', mode),
                   patience_steps=early.get('patience_steps', 2500), min_delta=early.get('min_delta', 0.0001))

    def should_evaluate(self, step, end_of_epoch=False):
        return step > 0 and step != self.last_eval_step and (
            end_of_epoch or (self.eval_steps is not None and step % self.eval_steps == 0))

    def update(self, value, step):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f'Non-finite validation metric {self.monitor}: {value}')
        if step < 1 or (self.last_eval_step is not None and step <= self.last_eval_step):
            raise ValueError('Validation updates require strictly increasing optimizer steps')
        self.last_eval_step = step
        self.num_evaluations += 1
        delta = None if self.best_value is None else (value - self.best_value if self.mode == 'max' else self.best_value - value)
        if delta is None or delta > self.min_delta:
            self.best_value, self.best_step = value, step
        self.stopped = bool(self.enabled and step - self.best_step >= self.patience_steps)
        return self.stopped

    @property
    def steps_without_improvement(self):
        if self.best_step is None or self.last_eval_step is None:
            return 0
        return self.last_eval_step - self.best_step

    def state_dict(self):
        return {'enabled': self.enabled, 'monitor': self.monitor, 'mode': self.mode,
                'eval_steps': self.eval_steps, 'patience_steps': self.patience_steps,
                'min_delta': self.min_delta, 'best_value': self.best_value,
                'best_step': self.best_step, 'last_eval_step': self.last_eval_step,
                'num_evaluations': self.num_evaluations,
                'steps_without_improvement': self.steps_without_improvement, 'stopped': self.stopped}

    def logs(self):
        return {'early_stop/best_value': self.best_value, 'early_stop/best_step': self.best_step,
                'early_stop/steps_without_improvement': self.steps_without_improvement,
                'early_stop/stopped': int(self.stopped)}


def update_early_stop(control, value, step, accelerator=None):
    """Rank 0 owns the decision; every rank receives the same complete state."""
    if accelerator is None or accelerator.num_processes == 1:
        return control.update(value, step)
    from accelerate.utils import broadcast_object_list
    message = [None]
    if accelerator.is_main_process:
        try:
            control.update(value, step)
            message[0] = (True, control.state_dict())
        except Exception as error:
            message[0] = (False, f'{type(error).__name__}: {error}')
    broadcast_object_list(message)
    success, state = message[0]
    if not success:
        raise RuntimeError(f'Early stopping failed: {state}')
    for key in ('best_value', 'best_step', 'last_eval_step', 'num_evaluations', 'stopped'):
        setattr(control, key, state[key])
    return control.stopped
