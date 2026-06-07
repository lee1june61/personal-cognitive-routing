"""Per-token operation-cycle model — Engine-A (ENGINE_A_DESIGN §2, 2026-05-27).

decode-side experts = generative operators (SMoE-VAE template). The frozen BGE
front-end emits per-token h (B, T, 1024); everything below is trainable:

    h_t (1024) ──SharedEncoderHead──▶ z_t (d_z=256)          # shared encoder
    z_t ──ReMoERouter (ReLU, independent gates)──▶ α_t (K)   # exact-zero, adaptive K
    ĥ_t = Σ_k α_{t,k} · DecoderExpert_k(z_t)                 # expert = generative operator

Contrast with the legacy pooled cycle (`model.py:Phase1MoE`): that routes a single
masked-mean fact_emb (B, 1024) through encode-side experts with a sparsegen simplex.
Here the operation signal lives per token (no pool), experts decode z back to h, and
routing is ReMoE (ReLU gate, no simplex normalisation, sparsity via adaptive L1).

Default = Engine-A start spec: d_z=256, K=16, d_hidden=512, d_model=1024, K_shared=0.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# SharedEncoderHead, ReMoERouter, and the loss primitives now live in core
# (extracted from this module — it is the canonical source). Imported here so the
# names remain available to phase1 callers / tests unchanged.
from core.shared_heads import SharedEncoderHead
from core.routers import ReMoERouter as _CoreReMoERouter
from core.loss_primitives import (
    _masked_token_mean,
    masked_token_mean,
    remoe_l1_loss,
    router_z_loss,
    update_l1_lambda,
)


class ReMoERouter(_CoreReMoERouter):
    """phase1 ReMoERouter — the core superset with phase1's original defaults.

    phase1's pre-extraction router was the SUBSET: ``(d_z=256, k=16)``, gate bias
    initialised to **zeros**, ``forward(z)`` returning ``(relu(logits), logits)``.
    The core superset defaults to ``bias_init=0.5`` (phase1_5). This subclass pins
    ``bias_init=0.0`` so phase1's behaviour (and ``test_opcycle_model.py``, which
    bare-constructs ``ReMoERouter(d_z, k)`` and asserts exact-zero gates at init) is
    unchanged. With ``external_bias=None`` + ``routing='relu_l1'`` + ``bias_init=0.0``
    the inherited forward is identical to the old phase1 forward.
    """

    def __init__(self, d_z: int = 256, k: int = 16, bias_init: float = 0.0, **kwargs):
        super().__init__(d_z=d_z, k=k, bias_init=bias_init, **kwargs)


# ----- Decoder expert (generative operator) ------------------------------------------

class DecoderExpert(nn.Module):
    """Single generative operator: z_t (d_z) → d_hidden → GELU → ĥ_t (d_model).

    Each expert reconstructs the per-token hidden state from the shared latent z; the
    mixture Σ_k α_k · expert_k(z) is the cycle output. Specialisation (which expert fires
    for which operation) is emergent from the unsupervised recon + ReMoE sparsity.
    """

    def __init__(self, d_z: int = 256, d_hidden: int = 512, d_model: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(d_z, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(z)))


# ----- Full operation-cycle MoE ------------------------------------------------------

class OpCycleMoE(nn.Module):
    """Per-token operation-cycle: SharedEncoderHead → ReMoERouter → decode-side experts.

    forward(h, mask=None) → dict:
        recon    (B, T, d_model)  — ĥ_t = Σ_k α_{t,k} · expert_k(z_t)
        alpha    (B, T, K)        — ReMoE gates (exact-zero, non-negative)
        z        (B, T, d_z)      — shared latent
        logits   (B, T, K)        — raw router logits (for z-loss / load-balance)
        k_active (B, T)           — count of non-zero gates per token

    `mask` is accepted for interface symmetry with the loss/eval path but does not change
    the forward (it gates which tokens count in the loss, not the per-token computation).
    """

    def __init__(
        self,
        d_model: int = 1024,
        d_z: int = 256,
        k: int = 16,
        d_hidden: int = 512,
        route_on_deviation: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_z = d_z
        self.k = k
        self.route_on_deviation = route_on_deviation
        self.encoder_head = SharedEncoderHead(d_model, d_z)
        # F3b lever: route on [z_t ‖ (z_t − seq-mean z)] so the gate reads within-sequence
        # local structure (where operations live) rather than the sequence-global content
        # (topic). The topic-collapse hypothesis is that pure-z routing keys on global
        # content; the deviation term makes a token's routing context-dependent. Experts
        # still decode plain z (deviation is router-input only), so d_z is unchanged.
        d_route = 2 * d_z if route_on_deviation else d_z
        # phase1 ReMoERouter subclass pins bias_init=0.0 (zeros bias) — reproduces the
        # pre-extraction behaviour; the core superset's default is +0.5 (phase1_5).
        self.router = ReMoERouter(d_route, k)
        self.experts = nn.ModuleList(
            DecoderExpert(d_z, d_hidden, d_model) for _ in range(k)
        )

    def _route_input(self, z: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Router input. Default = z. With route_on_deviation, [z ‖ (z − masked seq-mean z)]."""
        if not self.route_on_deviation:
            return z
        if mask is None:
            seq_mean = z.mean(dim=1, keepdim=True)                  # (B, 1, d_z)
        else:
            m = mask.to(z.dtype).unsqueeze(-1)                     # (B, T, 1)
            seq_mean = (z * m).sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True).clamp(min=1.0)
        return torch.cat([z, z - seq_mean], dim=-1)                # (B, T, 2·d_z)

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
        z = self.encoder_head(h)                                   # (B, T, d_z)
        alpha, logits = self.router(self._route_input(z, mask))    # (B, T, K) ×2
        # Accumulate ĥ = Σ_k α_k · expert_k(z) WITHOUT materialising the full
        # (B, T, K, d_model) expert stack — at B=256, T=128, K=16, d=1024 that stack
        # plus its α-product is ~4 GB of fp32 saved for backward. The running sum keeps
        # the peak at a couple of (B, T, d_model) tensors. (Mathematically identical.)
        recon = alpha.new_zeros(*h.shape[:-1], self.d_model)       # (B, T, d_model)
        for k, expert in enumerate(self.experts):
            recon = recon + alpha[..., k : k + 1] * expert(z)
        return {
            "recon": recon,
            "alpha": alpha,
            "z": z,
            "logits": logits,
            "k_active": (alpha > 0).sum(dim=-1),                   # (B, T)
        }


# ----- Per-token masked losses (ENGINE_A_DESIGN §3) ----------------------------------
# _masked_token_mean / remoe_l1_loss / router_z_loss / update_l1_lambda are imported
# from core.loss_primitives at the top of this module (canonical home).


def masked_recon_loss(
    h: torch.Tensor, recon: torch.Tensor, mask: torch.Tensor, cos: bool = True
) -> torch.Tensor:
    """Cycle reconstruction loss over masked tokens: mean (1 − cos(h_t, ĥ_t)) (or MSE).

    Padding tokens (mask == 0) are excluded so sequence-length differences don't bias the
    loss toward longer / shorter examples.
    """
    if cos:
        per_token = 1.0 - F.cosine_similarity(h, recon, dim=-1)     # (B, T)
    else:
        per_token = ((h - recon) ** 2).mean(dim=-1)                 # (B, T)
    return _masked_token_mean(per_token, mask)


def load_balance_loss(alpha: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Switch-style load-balance loss over the *actual ReMoE gates* (anti-collapse).

    L_lb = K · Σ_i f_i · P_i, with f_i = fraction of tokens that route to expert i
    (gate > 0) and P_i = mean per-token usage share of expert i (α normalised by its
    own per-token sum). Uniform usage minimises it.

    Form: importance/Herfindahl balance L = K · Σ_i P_i², where P_i is the mean per-token
    usage share of expert i (α normalised by its own per-token sum). P is a distribution
    over experts (Σ_i P_i = 1), so L ranges [1, K]: minimised (=1) at uniform usage,
    maximised (=K) when one expert carries all the mass. The classic Switch K·Σ f_i·P_i
    with f_i = top-1 dispatch fraction does NOT apply here — ReMoE routes each token to
    *multiple* experts, so there is no single top-1 assignment and that form degenerates
    (uniform and collapsed both score K).

    Crucially this reads the ReLU gates α — the distribution the model *actually* mixes
    with — not softmax(logits). softmax assigns non-zero mass to an expert whose logit ≤ 0,
    i.e. one that is dead under ReMoE (α=0, zero contribution to recon); a softmax-based
    balance loss would report a collapsed router as "balanced" and never penalise the
    K_active→0 failure mode (ENGINE_A_DESIGN §8). A fully-dead token contributes 0 to P.
    """
    K = alpha.shape[-1]
    m = mask.to(alpha.dtype).unsqueeze(-1)                         # (B, T, 1)
    n_tok = m.sum().clamp(min=1.0)
    usage = alpha / alpha.sum(dim=-1, keepdim=True).clamp(min=1e-9)  # per-token share, 0 if dead
    P = (usage * m).sum(dim=(0, 1)) / n_tok                        # (K,) mean usage share
    return K * (P ** 2).sum()


def opcycle_loss(
    out: dict,
    h: torch.Tensor,
    mask: torch.Tensor,
    lambda_recon: float = 1.0,
    lambda_l1: float = 1e-2,
    lambda_z: float = 1e-3,
    lambda_lb: float = 1e-2,
) -> tuple[torch.Tensor, dict]:
    """Combined operation-cycle objective. Returns (total, components) where components
    holds the raw (un-weighted, detached) term values for logging.

    Default λ: recon=1, l1=1e-2, z=1e-3, lb=1e-2 (ENGINE_A_DESIGN §3; raise λ_lb if
    K_active collapses toward 0).
    """
    recon = masked_recon_loss(h, out["recon"], mask)
    l1 = remoe_l1_loss(out["alpha"], mask)
    z = router_z_loss(out["logits"], mask)
    lb = load_balance_loss(out["alpha"], mask)
    total = lambda_recon * recon + lambda_l1 * l1 + lambda_z * z + lambda_lb * lb
    parts = {
        "recon": recon.detach().item(),
        "l1": l1.detach().item(),
        "z": z.detach().item(),
        "lb": lb.detach().item(),
        "k_active": out["k_active"].float().mean().item(),
    }
    return total, parts
