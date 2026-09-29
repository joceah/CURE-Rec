"""
RQ-VAE 模型：Residual Quantization VAE。

结构：
  x (input_dim)
    → Encoder MLP → z (latent_dim)
    → 第1层 VQ → 第2层 VQ → 第3层 VQ
    重建 z' = c1 + c2 + c3
    → Decoder MLP → x_recon (input_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


class VectorQuantizer(nn.Module):
    """单层向量量化。"""

    def __init__(self, codebook_size: int, latent_dim: int, commitment_weight: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.commitment_weight = commitment_weight

        self.codebook = nn.Embedding(codebook_size, latent_dim)
        nn.init.uniform_(self.codebook.weight, -1 / codebook_size, 1 / codebook_size)

    def forward(
        self,
        z: torch.Tensor,
        use_sinkhorn: bool = False,
        sk_epsilon: float = 0.05,
        sk_iters: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        codebook = self.codebook.weight
        dist = (
            z.pow(2).sum(dim=1, keepdim=True)
            + codebook.pow(2).sum(dim=1)
            - 2 * z @ codebook.T
        )

        if use_sinkhorn:
            # Center distances to [-1, 1] for numerical stability (matching MiniOneRec)
            d_max = dist.max()
            d_min = dist.min()
            middle = (d_max + d_min) / 2
            amplitude = d_max - middle + 1e-5
            d_centered = (dist - middle) / amplitude

            Q = torch.exp(-d_centered / sk_epsilon).double()
            B, K = Q.shape
            Q = Q / Q.sum()
            for _ in range(sk_iters):
                Q = Q / Q.sum(dim=1, keepdim=True)
                Q = Q / B
                Q = Q / Q.sum(dim=0, keepdim=True)
                Q = Q / K
            Q = Q * B
            indices = Q.argmax(dim=1)
        else:
            indices = dist.argmin(dim=1)

        z_q = self.codebook(indices)

        # Commitment loss
        codebook_loss = F.mse_loss(z.detach(), z_q)  # codebook → encoder
        commit_loss = F.mse_loss(z, z_q.detach())     # encoder → codebook
        loss = codebook_loss + self.commitment_weight * commit_loss

        # Straight-through
        z_q_st = z + (z_q - z).detach()
        return z_q_st, indices, loss

    @torch.no_grad()
    def init_codebook_with_kmeans(self, data: torch.Tensor, n_iters: int = 100) -> None:
        """用 sklearn KMeans 初始化码本。"""
        x = data.cpu().detach().numpy()
        kmeans = KMeans(n_clusters=self.codebook_size, max_iter=n_iters, n_init=1)
        kmeans.fit(x)
        centers = torch.from_numpy(kmeans.cluster_centers_).to(data.device)
        self.codebook.weight.data.copy_(centers)


class RQVAE(nn.Module):
    """Residual Quantization VAE with deep encoder/decoder."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_layers: int,
        codebook_size: int,
        encoder_layers: list[int] | None = None,
        commitment_loss_weight: float = 0.25,
        quant_loss_weight: float = 1.0,
        loss_type: str = "mse",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.quant_loss_weight = quant_loss_weight
        self.loss_type = loss_type

        if encoder_layers is None:
            encoder_layers = [256]

        # Encoder: input_dim → encoder_layers → latent_dim
        enc_blocks = []
        prev = input_dim
        for dim in encoder_layers:
            enc_blocks.append(nn.Linear(prev, dim))
            enc_blocks.append(nn.ReLU())
            prev = dim
        enc_blocks.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_blocks)

        # Decoder: latent_dim → reversed(encoder_layers) → input_dim
        dec_blocks = []
        prev = latent_dim
        for dim in reversed(encoder_layers):
            dec_blocks.append(nn.Linear(prev, dim))
            dec_blocks.append(nn.ReLU())
            prev = dim
        dec_blocks.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_blocks)

        # Per-layer VQ quantizers
        self.quantizers = nn.ModuleList([
            VectorQuantizer(codebook_size, latent_dim, commitment_loss_weight)
            for _ in range(num_layers)
        ])

        # Xavier init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(
        self,
        x: torch.Tensor,
        use_sinkhorn: bool = False,
        sk_epsilon: float = 0.05,
        sk_iters: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        z = self.encoder(x)
        z = F.normalize(z, p=2, dim=1)  # keep unit sphere for stable codebook
        residual = z
        z_q_total = torch.zeros_like(z)
        all_indices = []
        all_losses = []

        for quantizer in self.quantizers:
            z_q, indices, c_loss = quantizer(
                residual, use_sinkhorn=use_sinkhorn,
                sk_epsilon=sk_epsilon, sk_iters=sk_iters,
            )
            residual = residual - z_q.detach()
            z_q_total = z_q_total + z_q
            all_indices.append(indices)
            all_losses.append(c_loss)

        x_recon = self.decoder(z_q_total)
        recon_loss = F.mse_loss(x_recon, x) if self.loss_type == "mse" else F.l1_loss(x_recon, x)
        quant_loss = torch.stack(all_losses).mean()
        total_loss = recon_loss + self.quant_loss_weight * quant_loss

        return x_recon, total_loss, all_indices

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        z = self.encoder(x)
        z = F.normalize(z, p=2, dim=1)
        residual = z
        all_indices = []
        for quantizer in self.quantizers:
            dist = (
                residual.pow(2).sum(dim=1, keepdim=True)
                + quantizer.codebook.weight.pow(2).sum(dim=1)
                - 2 * residual @ quantizer.codebook.weight.T
            )
            indices = dist.argmin(dim=1)
            z_q = quantizer.codebook(indices)
            residual = residual - z_q
            all_indices.append(indices)
        return all_indices

    def init_codebooks(self, data: torch.Tensor) -> None:
        residual = self.encoder(data).detach()
        residual = F.normalize(residual, p=2, dim=1)
        for i, quantizer in enumerate(self.quantizers):
            quantizer.init_codebook_with_kmeans(residual)
            dist = (
                residual.pow(2).sum(dim=1, keepdim=True)
                + quantizer.codebook.weight.pow(2).sum(dim=1)
                - 2 * residual @ quantizer.codebook.weight.T
            )
            indices = dist.argmin(dim=1)
            z_q = quantizer.codebook(indices)
            residual = residual - z_q
