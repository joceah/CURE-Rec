"""用单个可学习 query 从上下文化的历史商品表示中提取一个 latent。"""

import torch
from torch import nn


class LatentReasoner(nn.Module):
    """K=1 的轻量 cross-attention reasoner。

    输入每个商品的 3 个 SID hidden state 的均值；输出仍在 Qwen hidden 空间。
    """

    def __init__(self, hidden_size: int, hidden_dim: int = 256,
                 num_heads: int = 4, max_history_len: int = 20,
                 dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(hidden_size, hidden_dim)
        self.recency = nn.Embedding(max_history_len, hidden_dim)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, 4 * hidden_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(4 * hidden_dim, hidden_dim))
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_size)

    def forward(self, items: torch.Tensor, item_mask: torch.Tensor) -> torch.Tensor:
        """items: [B, N, H]；item_mask: [B, N]；返回 z: [B, H]。

        B 为有效推荐样本数，N 为填充后的历史商品数，H 为 Qwen hidden_size。
        mask=False 的商品不参与 attention。
        """
        if not item_mask.any(dim=1).all():
            raise ValueError("Every recommendation sample needs at least one history item")
        batch, count, _ = items.shape
        # 最近商品的 recency index 为 0；较早商品依次增大，填充项由 mask 排除。
        age = (item_mask.long().sum(1, keepdim=True) - 1 -
               item_mask.long().cumsum(1) + item_mask.long()).clamp(min=0)
        age = age.clamp(max=self.recency.num_embeddings - 1)
        # keys: [B, N, D]；query: [B, 1, D]，D 为 reasoner hidden_dim。
        keys = self.input_projection(items) + self.recency(age)
        query = self.query.expand(batch, -1, -1)
        attended, _ = self.attention(query, keys, keys,
                                     key_padding_mask=~item_mask.bool(), need_weights=False)
        # 将唯一 query 的输出 [B, 1, D] 投影回 [B, H]。
        state = self.norm1(query + attended)
        return self.output_projection(self.norm2(state + self.ffn(state)).squeeze(1))
