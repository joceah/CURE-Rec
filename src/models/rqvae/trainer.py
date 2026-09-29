"""
RQ-VAE 训练器：AdamW + 梯度裁剪 + warmup + 碰撞率 early stopping。
完全对齐 MiniOneRec rq/trainer.py 的实现。
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW

from models.rqvae.model import RQVAE

logger = logging.getLogger(__name__)

LAYER_PREFIXES = ["a", "b", "c", "d", "e"]


def train_rqvae(cfg: DictConfig) -> tuple[RQVAE, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg = cfg.model.rqvae
    tcfg = cfg.train

    # 加载 embeddings
    emb_path = cfg.model.text2emb.output_path
    embeddings = torch.from_numpy(np.load(emb_path)).float()
    logger.info(f"加载 embeddings: {embeddings.shape}")

    # 不做归一化，直接用原始 embedding（匹配 MiniOneRec）
    mean = embeddings.mean(dim=0)
    std = embeddings.std(dim=0).clamp(min=1e-6)

    loader = DataLoader(TensorDataset(embeddings), batch_size=tcfg.batch_size, shuffle=True)

    # 构建模型
    model = RQVAE(
        input_dim=mcfg.input_dim,
        encoder_layers=list(mcfg.encoder_layers),
        latent_dim=mcfg.latent_dim,
        num_layers=mcfg.num_layers,
        codebook_size=mcfg.codebook_size,
        commitment_loss_weight=mcfg.commitment_loss_weight,
        quant_loss_weight=mcfg.quant_loss_weight,
        loss_type=mcfg.loss_type,
    ).to(device)

    logger.info("K-Means 初始化码本...")
    model.init_codebooks(embeddings.to(device))
    logger.info("码本初始化完成")

    optimizer = AdamW(model.parameters(), lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay)

    # Warmup 50 epochs (matching MiniOneRec)
    warmup_steps = 50 * len(loader)
    total_steps = tcfg.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )

    use_wandb = False
    try:
        import wandb
        wandb.init(project="genrec", name="rqvae", config=dict(mcfg) | dict(tcfg))
        use_wandb = True
    except Exception:
        pass

    best_loss = float("inf")
    best_collision = float("inf")
    cur_eval_step = 0
    best_ckpt = Path(tcfg.checkpoint_path)
    best_ckpt.parent.mkdir(parents=True, exist_ok=True)

    eval_step = min(50, tcfg.epochs)  # matching their eval_step

    for epoch in range(1, tcfg.epochs + 1):
        model.train()
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            if scheduler.get_last_lr()[0] < tcfg.learning_rate:
                scheduler.step()

            out, rq_loss, indices = model(batch)
            recon_loss = torch.nn.functional.mse_loss(out, batch)
            loss = recon_loss + mcfg.quant_loss_weight * rq_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch)

        avg_loss = total_loss / len(embeddings)

        if epoch % 100 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch}/{tcfg.epochs}  loss={avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss

        if use_wandb:
            import wandb
            wandb.log({"loss": avg_loss, "epoch": epoch})

        # 碰撞率评估 + early stopping
        if epoch % eval_step == 0:
            collision_rate = _eval_collision(model, embeddings, device)
            logger.info(f"Epoch {epoch} collision_rate={collision_rate:.4%}")

            if collision_rate < best_collision:
                best_collision = collision_rate
                cur_eval_step = 0
                torch.save({"model": model.state_dict(), "mean": mean, "std": std}, best_ckpt)
                logger.info(f"✓ Saved best model (collision={best_collision:.4%})")
            else:
                cur_eval_step += 1
                if tcfg.early_stopping_patience > 0 and cur_eval_step >= tcfg.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}, collision={best_collision:.4%}")
                    break

    logger.info(f"训练完成，最佳碰撞率={best_collision:.4%}，checkpoint: {best_ckpt}")

    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt["mean"], ckpt["std"]


@torch.no_grad()
def _eval_collision(model: RQVAE, embeddings: torch.Tensor, device: torch.device) -> float:
    """计算当前模型的碰撞率（argmin）。"""
    model.eval()
    all_indices = model.encode(embeddings.to(device))
    sid_tuples = list(zip(*[idx.cpu().tolist() for idx in all_indices]))
    unique = len(set(sid_tuples))
    return (len(embeddings) - unique) / len(embeddings)


@torch.no_grad()
def generate_indices(
    model: RQVAE,
    embeddings: torch.Tensor,
    item_ids: list[int],
    num_layers: int,
    codebook_size: int,
    output_path: str,
    max_retries: int = 20,
) -> None:
    device = next(model.parameters()).device
    prefixes = LAYER_PREFIXES[:num_layers]

    sk_epsilon = 0.05
    sk_iters = 50
    for attempt in range(max_retries):
        model.eval()
        if sk_epsilon < 1e-8:
            all_indices = model.encode(embeddings.to(device))
        else:
            all_indices = _encode_with_sinkhorn(model, embeddings.to(device), sk_epsilon, sk_iters)

        sid_tuples = list(zip(*[idx.cpu().tolist() for idx in all_indices]))
        unique_sids = set(sid_tuples)
        collision_count = len(item_ids) - len(unique_sids)

        logger.info(
            f"尝试 {attempt + 1}: sk_epsilon={sk_epsilon:.4f}, sk_iters={sk_iters}, "
            f"碰撞数={collision_count}/{len(item_ids)} "
            f"({collision_count / len(item_ids) * 100:.2f}%)"
        )

        if collision_count == 0:
            break

        if sk_epsilon < 1e-8:
            sk_epsilon = 0.01
        else:
            sk_epsilon *= 1.5
        sk_iters = min(sk_iters + 10, 100)

    result = {}
    for iid, sid_tuple in zip(item_ids, sid_tuples):
        result[str(iid)] = [
            f"<{prefix}_{code}>"
            for prefix, code in zip(prefixes, sid_tuple)
        ]

    with open(output_path, "w") as f:
        json.dump(result, f)
    logger.info(f"item.index.json 已保存: {output_path}，共 {len(result)} 条，最终碰撞数={collision_count}")


@torch.no_grad()
def _encode_with_sinkhorn(model: RQVAE, x: torch.Tensor, sk_epsilon: float, sk_iters: int = 50) -> list[torch.Tensor]:
    _, _, all_indices = model(x, use_sinkhorn=True, sk_epsilon=sk_epsilon, sk_iters=sk_iters)
    return all_indices


def run_rqvae(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.train.seed)
    torch.cuda.manual_seed_all(cfg.train.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model, mean, std = train_rqvae(cfg)

    embeddings = torch.from_numpy(np.load(cfg.model.text2emb.output_path)).float()
    mean = mean.cpu() if isinstance(mean, torch.Tensor) else mean
    std = std.cpu() if isinstance(std, torch.Tensor) else std
    # 不做归一化，直接用原始 embedding（匹配训练）

    id_path = Path(cfg.model.text2emb.output_path).with_name("item_ids_order.json")
    with open(id_path) as f:
        item_ids = json.load(f)

    generate_indices(
        model=model,
        embeddings=embeddings,
        item_ids=item_ids,
        num_layers=cfg.model.rqvae.num_layers,
        codebook_size=cfg.model.rqvae.codebook_size,
        output_path=cfg.train.index_output_path,
    )
