"""Shared frozen forward and prefix candidate space for training and decoding."""

from dataclasses import dataclass
from functools import cached_property

import torch
from torch import nn

from .distributed import gather_parameters


@dataclass
class BaselineStepOutput:
    hidden: torch.Tensor
    logits: torch.Tensor | None
    candidate_ids: torch.Tensor
    candidate_logits: torch.Tensor
    candidate_basis: torch.Tensor


@dataclass
class PackedCandidates:
    hidden: torch.Tensor
    logits: torch.Tensor
    ids: torch.Tensor
    basis_indices: torch.Tensor
    mask: torch.Tensor
    lengths: list[list[int]]
    basis: torch.Tensor
    target_indices: torch.Tensor | None = None


@dataclass
class FrozenFeatures:
    decision: torch.Tensor
    history: torch.Tensor
    history_mask: torch.Tensor
    candidates: PackedCandidates
    targets: torch.Tensor

    @cached_property
    def steps(self):
        # Compatibility/debug view; training never materializes per-sample bases.
        packed = self.candidates
        return [[BaselineStepOutput(
            packed.hidden[i, t], None, packed.ids[i, t, :n], packed.logits[i, t, :n],
            packed.basis.index_select(0, packed.basis_indices[i, t, :n]))
            for t, n in enumerate(row)] for i, row in enumerate(packed.lengths)]


class FrozenSFT(nn.Module):
    def __init__(self, model, prefix_tree):
        super().__init__()
        self.model = model.requires_grad_(False).eval()
        self.tree = prefix_tree
        self.hidden_dim = model.config.hidden_size
        head = model.get_output_embeddings()
        ids = torch.tensor(prefix_tree.token_ids, dtype=torch.long, device=head.weight.device)
        with gather_parameters([head.weight]):
            basis = head.weight.index_select(0, ids).detach().clone()
        self._sid_positions = {token: i for i, token in enumerate(prefix_tree.token_ids)}
        self._candidate_metadata = {}
        self.register_buffer('_sid_ids', ids, persistent=False)
        self.register_buffer('_sid_basis', basis, persistent=False)

    def train(self, mode=True):
        super().train(False)
        return self

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def hidden_states(self, input_ids, attention_mask):
        # Qwen's transformer only: avoid allocating [B,L,full-vocab] logits.
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        return self.model.base_model(input_ids=input_ids, attention_mask=attention_mask,
                                     position_ids=position_ids, use_cache=False,
                                     return_dict=True).last_hidden_state

    @torch.no_grad()
    def candidates(self, hidden, prefix, include_full_logits=False, vocabulary_logits=None):
        allowed = sorted(self.tree.get_allowed_tokens(tuple(prefix)))
        if not allowed:
            raise ValueError(f"Invalid or complete SID prefix: {prefix}")
        ids = torch.tensor(allowed, device=hidden.device, dtype=torch.long)
        head = self.model.get_output_embeddings()
        basis = self._sid_basis.index_select(0, torch.searchsorted(self._sid_ids, ids))
        # Preserve the actual frozen LM-head computation. Projecting only SID rows
        # can select different BF16 GEMM kernels and change tied beam rankings.
        full_logits = head(hidden).float() if vocabulary_logits is None else vocabulary_logits
        logits = full_logits if include_full_logits else None
        candidate_logits = full_logits.index_select(0, ids)
        return BaselineStepOutput(hidden, logits, ids, candidate_logits, basis)

    def pack_candidates(self, hidden, prefixes, vocabulary_logits, targets=None):
        """Build legal-candidate metadata on CPU, then transfer it in one batch.

        Hidden/logits may have any leading dimensions; prefixes are flattened in
        row-major order. Never expand the SID basis to [batch, stage, K, hidden].
        """
        metadata = []
        target_indices = []
        for i, prefix in enumerate(prefixes):
            prefix = tuple(prefix)
            if prefix not in self._candidate_metadata:
                allowed = sorted(self.tree.get_allowed_tokens(prefix))
                if not allowed:
                    raise ValueError(f'Invalid or complete SID prefix: {prefix}')
                self._candidate_metadata[prefix] = (
                    allowed, [self._sid_positions[token] for token in allowed],
                    {token: j for j, token in enumerate(allowed)})
            entry = self._candidate_metadata[prefix]
            metadata.append(entry)
            if targets is not None:
                if targets[i] not in entry[2]:
                    raise ValueError('Target SID is not a legal prefix-tree child')
                target_indices.append(entry[2][targets[i]])
        lengths = [len(entry[0]) for entry in metadata]
        width = max(lengths)
        shape = hidden.shape[:-1]
        device = hidden.device
        ids = torch.tensor([entry[0] + [0] * (width - n) for entry, n in zip(metadata, lengths)],
                           dtype=torch.long, device='cpu').to(device).view(*shape, width)
        indices = torch.tensor([entry[1] + [0] * (width - n) for entry, n in zip(metadata, lengths)],
                               dtype=torch.long, device='cpu').to(device).view(*shape, width)
        counts = torch.tensor(lengths, dtype=torch.long, device='cpu').to(device).view(shape)
        mask = torch.arange(width, device=device) < counts.unsqueeze(-1)
        logits = vocabulary_logits.gather(-1, ids).masked_fill(~mask, float('-inf'))
        target_index = None if targets is None else torch.tensor(
            target_indices, dtype=torch.long, device='cpu').to(device).view(shape)
        # lengths are CPU metadata, also used by the lazy ragged compatibility views.
        rows = [lengths[i:i + shape[-1]] for i in range(0, len(lengths), shape[-1])] if len(shape) == 2 else [lengths]
        return PackedCandidates(hidden, logits, ids, indices, mask, rows, self._sid_basis, target_index)

    @torch.no_grad()
    def step(self, prompt, prefix=()):
        ids = torch.tensor([list(prompt) + list(prefix)], device=self.device)
        hidden = self.hidden_states(ids, torch.ones_like(ids))[0, -1]
        return self.candidates(hidden, prefix, include_full_logits=True)

    @torch.no_grad()
    def prepare(self, batch):
        # Target is teacher-forced only after the decision position. Causal attention
        # keeps both decision and contextual history states independent of target.
        ids, mask = batch['input_ids'], batch['attention_mask']
        hidden = self.hidden_states(ids, mask)
        lengths = batch['prompt_length']
        rows = torch.arange(len(ids), device=ids.device)
        decision = hidden[rows, lengths - 1]
        positions = batch['history_positions']
        item_mask = batch['history_mask']
        valid_positions = positions[item_mask]
        prompt_limits = lengths[:, None, None].expand_as(positions)
        if ((positions >= prompt_limits) & item_mask[:, :, None]).any() or (valid_positions < 0).any():
            raise ValueError("History positions must refer exclusively to prompt tokens")
        history_tokens = hidden[rows[:, None, None], positions.clamp_min(0)]
        history = history_tokens.mean(dim=2) * item_mask[:, :, None]
        targets = ids[rows[:, None], lengths[:, None] + torch.arange(3, device=ids.device)]
        stage_hidden = hidden[rows[:, None], lengths[:, None] - 1 + torch.arange(3, device=ids.device)]
        stage_logits = self.model.get_output_embeddings()(stage_hidden).float()
        # One D2H copy rather than three GPU .tolist() synchronizations per sample.
        target_rows = targets.cpu().tolist()
        prefixes = [row[:t] for row in target_rows for t in range(3)]
        packed = self.pack_candidates(stage_hidden, prefixes, stage_logits,
                                      [token for row in target_rows for token in row])
        return FrozenFeatures(decision, history, item_mask, packed, targets)
