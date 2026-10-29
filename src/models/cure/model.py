"""Safe personalized residual model and a single three-SID target scorer."""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import torch
from torch import nn

from .baseline import FrozenSFT
from .reasoner import BaselineAwareReasoner
from .residual import ResidualCandidateScorer


def validate_donor_indices(index, size):
    if index is None or index.ndim != 1 or len(index) != size or index.dtype != torch.long:
        raise ValueError("Donors must be a one-dimensional LongTensor with one index per sample")
    if ((index < 0) | (index >= size)).any() or (index == torch.arange(size, device=index.device)).any():
        raise ValueError("Donors must be valid non-self sample indices")


@dataclass
class TargetScores:
    sequence_score: torch.Tensor
    stage_log_probs: torch.Tensor
    gates: torch.Tensor
    correction_norms: torch.Tensor
    padded_logits: torch.Tensor
    padded_log_probs: torch.Tensor
    padded_ids: torch.Tensor
    candidate_mask: torch.Tensor
    lengths: list[list[int]]

    def _unpack(self, tensor):
        return [[tensor[i, t, :n] for t, n in enumerate(row)] for i, row in enumerate(self.lengths)]

    @cached_property
    def stage_distributions(self):
        # Preserve the exact unpadded log_softmax debug/reference view.
        return [[logits.log_softmax(-1) for logits in row] for row in self.candidate_logits]

    @cached_property
    def candidate_logits(self):
        return self._unpack(self.padded_logits)

    @cached_property
    def candidate_ids(self):
        return self._unpack(self.padded_ids)


class CUREModel(nn.Module):
    FORMAT = 'safe-personalized-residual-v1'

    def __init__(self, baseline: FrozenSFT, hidden_dim=256, num_heads=4, dropout=0.1,
                 residual_hidden_dim=256, gate_bias=-3.0):
        super().__init__()
        self.baseline = baseline
        self.architecture = dict(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout,
                                 residual_hidden_dim=residual_hidden_dim, gate_bias=gate_bias)
        self.reasoner = BaselineAwareReasoner(baseline.hidden_dim, hidden_dim, num_heads, dropout)
        self.residual = ResidualCandidateScorer(baseline.hidden_dim, hidden_dim,
                                                 residual_hidden_dim, gate_bias)

    def encode(self, features):
        return self.reasoner(features.decision, features.history, features.history_mask)

    def candidate_step(self, step, latent=None, disabled=False):
        return self.residual(step.hidden, latent, step.candidate_basis,
                             step.candidate_logits, disabled=disabled)

    def score(self, features, latent=None, disabled=False):
        packed = features.candidates
        expanded = None if latent is None else latent[:, None].expand(-1, packed.hidden.shape[1], -1)
        logits, gates, correction = self.residual(
            packed.hidden, expanded, packed.basis, packed.logits, disabled=disabled,
            candidate_indices=packed.basis_indices, candidate_mask=packed.mask)
        distributions = logits.log_softmax(-1)
        stage_lp = distributions.gather(-1, packed.target_indices.unsqueeze(-1)).squeeze(-1)
        return TargetScores(stage_lp.mean(-1), stage_lp, gates, correction.norm(dim=-1),
                            logits, distributions, packed.ids, packed.mask, packed.lengths)

    def forward(self, batch, donor_index=None, method=None):
        features = self.baseline.prepare(batch)
        latent = self.encode(features)
        result = {'baseline': self.score(features, disabled=True),
                  'correct': self.score(features, latent)}
        if donor_index is not None:
            validate_donor_indices(donor_index, len(latent))
            result['matched'] = self.score(features, latent[donor_index])
        if method is not None:
            from .losses import cure_loss
            losses = cure_loss(result, **method)
            logs = {**losses, 'mean_gate': result['correct'].gates.mean(),
                    'mean_correction_norm': result['correct'].correction_norms.mean()}
            return {key: value if key == 'loss' else value.detach() for key, value in logs.items()}
        return result

    def save_checkpoint(self, path, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # No frozen Qwen weights in CURE checkpoints.
        torch.save({'format': self.FORMAT, 'architecture': self.architecture,
                    'reasoner': self.reasoner.state_dict(), 'residual': self.residual.state_dict(),
                    'metadata': metadata or {}}, path)

    @classmethod
    def load_checkpoint(cls, path, baseline):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('format') != cls.FORMAT:
            raise ValueError("Not a new safe personalized residual CURE checkpoint")
        model = cls(baseline, **payload['architecture']).to(baseline.device)
        model.reasoner.to(dtype=next(iter(payload['reasoner'].values())).dtype)
        model.residual.to(dtype=next(iter(payload['residual'].values())).dtype)
        model.reasoner.load_state_dict(payload['reasoner'])
        model.residual.load_state_dict(payload['residual'])
        return model, payload['metadata']


def score_target(baseline_model, cure_model, batch, latent_mode='correct', donor_index=None, features=None, latent=None):
    if baseline_model is not cure_model.baseline:
        raise ValueError("Scoring must use the model's frozen baseline")
    features = features if features is not None else baseline_model.prepare(batch)
    if latent_mode == 'baseline':
        return cure_model.score(features, disabled=True)
    if latent is None:
        latent = cure_model.encode(features)
    if latent_mode == 'matched':
        validate_donor_indices(donor_index, len(latent))
        latent = latent[donor_index]
    elif latent_mode != 'correct':
        raise ValueError(f"Unknown latent_mode: {latent_mode}")
    return cure_model.score(features, latent)
