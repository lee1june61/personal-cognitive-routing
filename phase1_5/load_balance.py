"""Aux-loss-free load balancing for ReMoE (DeepSeek-V3 §2.2.2 / Wang et al. 2024).

Per-expert non-learnable bias ``b_k`` added to router logits before ReLU. The
bias is updated each training step by the simple sign rule

    b_k <- b_k - lr_bias * sign(f_k - target)

where ``f_k = fraction of Q-tokens that activate expert k`` (count-based) and
``target = K_active_target / K_routed``. Under-activated experts grow bias and
get selected next step; over-activated experts shrink it. Because the bias is
a buffer (not a parameter), the update is purely manual — the gradient path
of the routing decision itself is unchanged.

Why this matters for K=128 ReMoE:
    Dead-ReLU gates (``logit < 0`` → ``alpha = 0`` → no gradient) cannot recover
    via the adaptive-L1 controller alone — L1 only shifts the global sparsity
    pressure, never the individual gate's logit. The bias drift here lifts
    dead-expert logits back into the active half-plane independently of the
    autograd path, breaking the K_active=0 trap observed at K=128 cold-start.

References:
    - DeepSeek-AI. "DeepSeek-V3 Technical Report." 2024. §2.2.2.
    - Wang et al. "Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-
      Experts." arXiv:2408.15664. 2024.
"""

from __future__ import annotations

import torch
import torch.nn as nn


LB_OFF = "off"
LB_AUX_FREE = "aux_free"
LB_STRATEGIES = (LB_OFF, LB_AUX_FREE)


class AuxLossFreeLB(nn.Module):
    """Per-expert bias buffer + sign-rule update.

    Args:
        k: number of routed experts.
        target: per-expert target activation fraction (= ``K_active / K_routed``).
        lr_bias: update rate. DeepSeek-V3 default 1e-3.
    """

    def __init__(self, k: int, target: float, lr_bias: float = 1e-3):
        super().__init__()
        self.k = k
        self.target = float(target)
        self.lr_bias = float(lr_bias)
        self.register_buffer("bias", torch.zeros(k))

    @torch.no_grad()
    def step(self, alpha: torch.Tensor, mask: torch.Tensor) -> None:
        """Update ``bias`` from post-ReLU ``alpha`` over masked Q tokens.

        ``f_k = (sum_{b,t} 1[alpha_{btk} > 0] * mask_{bt}) / sum mask``. The
        sign rule is paper-faithful (DeepSeek-V3 §2.2.2) — proportional
        ``residual`` updates are also tractable but introduce a magnitude
        coupling between the bias drift and the gate-output scale.

        Numerics:
            - active/mask are upcast to fp32 inside step regardless of the
              autocast dtype of ``alpha``. Under fp16 AMP the residual at
              K=128 (target≈0.031) is on the order of 1e-3 — comparable to
              fp16 grid spacing — which would make ``torch.sign(residual)``
              oscillate near equilibrium. fp32 here is virtually free
              (the cast happens once on a (B,T,K) tensor outside the inner
              forward) and removes that oscillation.
            - An all-padded batch (``mask.sum() == 0``) is a no-op: f_k is
              undefined, and the old ``denom.clamp(min=1.0)`` would have
              interpreted it as 'every expert is dead', growing every bias.
        """
        active = (alpha > 0).float()                   # (B, T, K), fp32
        m = mask.float().unsqueeze(-1)                 # (B, T, 1), fp32
        total = m.sum()
        if total <= 0:
            return
        f_k = (active * m).sum(dim=(0, 1)) / total    # (K,)
        residual = f_k - self.target                    # (K,)
        self.bias.add_(-self.lr_bias * torch.sign(residual))


def make_lb(
    strategy: str,
    k_routed: int,
    k_active_target: float,
    *,
    lr_bias: float = 1e-3,
) -> AuxLossFreeLB | None:
    """Factory for an LB module from an ablation-row strategy string.

    Returns ``None`` for ``"off"`` (the Phase 1.5 Row F baseline); an
    ``AuxLossFreeLB`` for ``"aux_free"`` (Layer-1 default).

    Aux-weight variants ``aux_w_*`` are reserved for paper §7.4 Row F sweep
    and not implemented here — Layer 1 covers ``aux_free`` only.
    """
    if strategy == LB_OFF:
        return None
    if strategy == LB_AUX_FREE:
        return AuxLossFreeLB(
            k=k_routed,
            target=k_active_target / k_routed,
            lr_bias=lr_bias,
        )
    raise NotImplementedError(
        f"lb_strategy={strategy!r} not implemented; expected one of {LB_STRATEGIES}"
    )
