"""CURE 损失函数接口；当前 Phase 1 仅启用推荐交叉熵。"""

import torch.nn.functional as F


def recommendation_loss(logits, labels):
    """自回归 CE：logits [B, T, V]，labels [B, T]，返回标量。

    位置 t 的 logits 预测位置 t+1 的 token；-100 表示不参与损失。
    """
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                           labels[:, 1:].reshape(-1), ignore_index=-100)


def cure_loss(logits, labels, variant="latent_ce", lambda_util=0.0,
              lambda_anchor=0.0):
    """统一方法入口；CURE-U/UA 的反事实损失留待 Phase 2 实现。"""
    if variant != "latent_ce" or lambda_util != 0 or lambda_anchor != 0:
        raise NotImplementedError("CURE-U and CURE-UA require Phase 2 counterfactual training")
    return recommendation_loss(logits, labels)
