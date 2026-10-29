"""Asymmetric objectives on the same three-stage constrained target score."""

import torch
from torch.nn import functional as F

VARIANTS = ('residual_ce', 'cure_g', 'cure_gp', 'cure_full')


def safe_kl(baseline, matched):
    if hasattr(baseline, 'padded_log_probs') and hasattr(matched, 'padded_log_probs'):
        p = baseline.padded_log_probs.detach()
        q = matched.padded_log_probs
        mask = baseline.candidate_mask
        # Never subtract -inf padding: it would create NaNs even with zero mass.
        difference = p.masked_fill(~mask, 0.) - q.masked_fill(~mask, 0.)
        return (p.exp() * difference).sum(-1).mean(-1)
    values = []
    for base_row, wrong_row in zip(baseline.stage_distributions, matched.stage_distributions):
        values.append(torch.stack([(p.detach().exp() * (p.detach() - q)).sum()
                                   for p, q in zip(base_row, wrong_row)]).mean())
    return torch.stack(values)


def cure_loss(outputs, variant='residual_ce', lambda_gain=1.0, lambda_personal=1.0,
              lambda_safe=1.0, margin_gain=0.05, margin_personal=0.05):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown CURE variant: {variant}")
    if min(lambda_gain, lambda_personal, lambda_safe, margin_gain, margin_personal) < 0:
        raise ValueError("Loss weights and margins must be nonnegative")
    base, correct = outputs['baseline'], outputs['correct']
    gain = correct.sequence_score - base.sequence_score.detach()
    rec = -correct.sequence_score.mean()
    zero = rec.new_zeros(())
    gain_loss = F.relu(margin_gain - gain).mean() if variant != 'residual_ce' else zero
    personal_loss, safe_loss = zero, zero
    if variant in ('cure_gp', 'cure_full'):
        matched = outputs['matched']
        personal_loss = F.relu(margin_personal - (correct.sequence_score - matched.sequence_score)).mean()
        if variant == 'cure_full':
            safe_loss = safe_kl(base, matched).mean()
    total = rec + lambda_gain * gain_loss + lambda_personal * personal_loss + lambda_safe * safe_loss
    return {'loss': total, 'rec': rec, 'gain': gain_loss, 'personal': personal_loss,
            'safe': safe_loss, 'mean_target_score_gain': gain.mean(),
            'baseline_rec': -base.sequence_score.detach().mean()}
