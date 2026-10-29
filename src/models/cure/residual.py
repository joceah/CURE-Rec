"""Scalar-gated residual projected onto frozen legal SID output rows."""

import math

import torch
from torch import nn


class ResidualCandidateScorer(nn.Module):
    def __init__(self, backbone_dim, latent_dim=256, hidden_dim=256, gate_bias=-3.0):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(backbone_dim + latent_dim, hidden_dim), nn.SiLU())
        self.projection = nn.Linear(hidden_dim, backbone_dim, bias=False)
        self.gate = nn.Linear(backbone_dim + latent_dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias)

    def forward(self, hidden, latent, candidate_basis, baseline_logits, disabled=False,
                candidate_indices=None, candidate_mask=None):
        if disabled:
            # Structural equality; bypass every learned operation.
            return baseline_logits, baseline_logits.new_zeros(hidden.shape[:-1]), torch.zeros_like(baseline_logits)
        features = torch.cat((hidden, latent.to(hidden.dtype)), dim=-1).to(self.mlp[0].weight.dtype)
        shape = features.shape[:-1]
        # One set of linear calls for all sample/stage pairs; ZeRO sees 2D inputs.
        flat = features.reshape(-1, features.shape[-1])
        query = self.projection(self.mlp(flat)).reshape(*shape, hidden.shape[-1])
        gate = torch.sigmoid(self.gate(flat).float()).reshape(shape)
        # matmul is autocast-eligible (unlike the old mv); keep SID projection FP32.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            residual = torch.matmul(query.float(), candidate_basis.float().T) / math.sqrt(query.shape[-1])
        if candidate_indices is not None:
            residual = residual.gather(-1, candidate_indices)
        correction = gate.unsqueeze(-1) * residual
        if candidate_mask is not None:
            correction = correction.masked_fill(~candidate_mask, 0.)
        return baseline_logits + correction, gate, correction
