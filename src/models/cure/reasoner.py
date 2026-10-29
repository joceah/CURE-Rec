"""One decision-conditioned query over frozen, contextual history item states."""

from torch import nn


class BaselineAwareReasoner(nn.Module):
    def __init__(self, backbone_dim, hidden_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.query = nn.Linear(backbone_dim, hidden_dim)
        self.evidence = nn.Linear(backbone_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, decision, history, history_mask):
        if not history_mask.any(dim=1).all():
            raise ValueError("Reasoner needs at least one history item per sample")
        query = self.query(decision.to(self.query.weight.dtype)).unsqueeze(1)
        evidence = self.evidence(history.to(self.evidence.weight.dtype))
        state, _ = self.attention(query, evidence, evidence,
                                  key_padding_mask=~history_mask, need_weights=False)
        return self.norm(state[:, 0])
