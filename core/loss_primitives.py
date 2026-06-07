"""Shared MoE/ReMoE loss primitives — masked token mean, ReMoE L1, router z-loss,
adaptive-L1 controller.

Canonical source = ``phase1/model_opcycle.py`` (these are defined there). ``phase1_5/train.py``
held verbatim copies; both now import from here. Behaviour is byte-identical.
"""

from __future__ import annotations

import torch


def masked_token_mean(per_token: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of a per-token scalar (B, T) over masked (active) tokens only."""
    m = mask.to(per_token.dtype)
    return (per_token * m).sum() / m.sum().clamp(min=1.0)


# Back-compat alias — the underscore-prefixed name is referenced internally in both
# packages (phase1/model_opcycle.py, phase1_5/train.py).
_masked_token_mean = masked_token_mean


def remoe_l1_loss(alpha: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """ReMoE adaptive-L1 penalty: mean ‖α_t‖₁ over masked tokens. Driving this down
    shrinks K_active; the coefficient λ_l1 is tuned (optionally adaptively) at train-time
    to hit a target sparsity. Returns the raw penalty term (coefficient applied by caller).
    """
    l1_per_token = alpha.abs().sum(dim=-1)                          # (B, T)
    return _masked_token_mean(l1_per_token, mask)


def router_z_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """ST-MoE (Zoph 2022) router z-loss: mean logsumexp(logits)² over masked tokens.
    Penalises large router logits, keeping the gate numerically stable.
    """
    lse = torch.logsumexp(logits, dim=-1)                          # (B, T)
    return _masked_token_mean(lse ** 2, mask)


def update_l1_lambda(
    lam: float,
    k_active_mean: float,
    k_target: float,
    *,
    factor: float = 1.2,
    lam_min: float = 1e-6,
    lam_max: float = 1.0,
) -> float:
    """ReMoE-style adaptive L1 coefficient controller (Wang et al. 2024).

    The L1 penalty `remoe_l1_loss` only shrinks the gates; what regulates the *level* of
    sparsity is the coefficient λ_l1. A fixed λ is brittle — slightly too large drives
    K_active→0 (recon collapse), slightly too small leaves the router dense. This nudges λ
    multiplicatively toward a target mean K_active each step: too dense → raise λ (more
    pressure), too sparse → lower λ. Clamped to [lam_min, lam_max].
    """
    if k_active_mean > k_target:
        lam = lam * factor
    elif k_active_mean < k_target:
        lam = lam / factor
    return float(min(max(lam, lam_min), lam_max))
