"""Phase 1 MoE — emergent relation-type expert specialization via unsupervised cycle.

Architecture (RESEARCH_PLAN_2026-05-19 §2.1 revision 4):

    fact_emb (B, d_model=1024)   ← frozen BGE-large-en encoder output
        │
        ▼
    Phase1MoE
      ├── fact_gate Linear(d_model → K_routed)
      ├── [Phase 1] global_lambda_log (scalar Parameter)
      ├── [Phase 2] user_logits Embedding(N_users, K_routed)
      ├── [Phase 2] lambda_mlp Linear(K_routed, 1)
      │
      ├── u = fact_gate(fact_emb)                              [Phase 1]
      ├── u = fact_gate(fact_emb) + user_logits(user_id)       [Phase 2]
      ├── λ = softplus(global_lambda_log)                       [Phase 1, scalar]
      ├── λ_u = softplus(lambda_mlp(user_logits(user_id)))      [Phase 2, per-user]
      │
      ├── routed_alpha = sparsegen(u, λ)        (B, K_routed) sparse simplex
      │
      ├── routed_experts × K_routed (each ExpertFFN: d_model → d_hidden → d_model)
      │     routed_outs = stack(...) → (B, K_routed, d_model)
      │     routed_mix = Σ_k routed_alpha[:,k] · routed_outs[:,k,:]
      │
      ├── shared_experts × K_shared (always-on, no gating)
      │     shared_outs = stack(...) → (B, K_shared, d_model)
      │     shared_mix = shared_outs.mean(dim=1)
      │
      └── sub_kg = routed_mix + shared_mix       (B, d_model)  ★ latent KG
          (distributional, emergent — expert k = emergent relation-type k)

Default = Stage 1 spec (Expanded #1): K_routed=16, K_shared=4, d_hidden=2048,
d_model=1024 (BGE-large-en). Stage 2 (Expanded #4): K_routed=64, K_shared=16,
d_hidden=3072 — same class via constructor args, no code change.

Why expert = emergent relation-type:
  - Cycle reconstruction loss + sparsegen routing → unsupervised specialization
  - Nikolic 2025 (arXiv 2509.10025): unsupervised optimal K may exceed num_classes,
    expert assignment > supervised by 8.3pp linear separability
  - SIMoE orthogonality reg (optional, λ_ortho > 0) for explicit specialization pressure
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----- Sparsegen (Laha et al. 2018, LD-MoLE Eq. 4-5) ---------------------------------

def sparsegen(u: torch.Tensor, lam: torch.Tensor | float) -> torch.Tensor:
    """Closed-form sparse projection onto probability simplex.

    Args:
        u   : (B, K) score vector.
        lam : (B, 1), (B,), or scalar — sparsity in (-∞, 1). Larger → sparser.

    Returns:
        p   : (B, K) — Σ p = 1, some entries exactly 0, K_active ∈ [1, K] adaptive.

    Properties (LD-MoLE Lemma 1): for λ < 1, the solution always has ≥ 1 nonzero —
    collapse impossible. Fully differentiable through gather + relu.
    """
    B, K = u.shape

    if not isinstance(lam, torch.Tensor):
        lam = torch.tensor(lam, device=u.device, dtype=u.dtype)
    if lam.dim() == 0:
        lam = lam.expand(B, 1)
    elif lam.dim() == 1:
        lam = lam.unsqueeze(-1)
    lam = lam.clamp(max=1.0 - 1e-3)
    one_minus_lam = 1.0 - lam

    sorted_u, _ = u.sort(dim=-1, descending=True)
    cumsum_u = sorted_u.cumsum(dim=-1)
    k_range = torch.arange(1, K + 1, device=u.device, dtype=u.dtype)

    condition = one_minus_lam + k_range * sorted_u > cumsum_u
    k_star = condition.sum(dim=-1, keepdim=True).clamp(min=1)

    U_kstar = cumsum_u.gather(dim=-1, index=k_star.long() - 1)
    tau = (U_kstar - 1.0 + lam) / k_star.to(u.dtype)

    p = F.relu((u - tau) / one_minus_lam)
    return p


# ----- Expert FFN --------------------------------------------------------------------

class ExpertFFN(nn.Module):
    """Single expert: Linear(d_model, d_hidden) → GELU → Linear(d_hidden, d_model).

    Stage 1 default: d_model=1024, d_hidden=2048 → ~3.1M params per expert.
    Stage 2: d_hidden=3072 → ~6.3M params per expert.
    """

    def __init__(self, d_model: int = 1024, d_hidden: int = 2048):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


# ----- Phase 1 / Phase 2 unified MoE -------------------------------------------------

class Phase1MoE(nn.Module):
    """Emergent relation-type MoE — Phase 1 (no user) / Phase 2 (+ user) via `use_user`.

    Phase 1: fact_gate(x) only, global_lambda_log scalar — user-agnostic routing
    Phase 2: fact_gate(x) + user_logits(user_id), per-user λ_u via lambda_mlp
    """

    def __init__(
        self,
        n_users: int,
        d_model: int = 1024,
        k_routed: int = 16,
        k_shared: int = 4,
        d_hidden: int = 2048,
        gate_temperature: float = 1.0,
        use_user: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.k_routed = k_routed
        self.k_shared = k_shared
        self.k_total = k_routed + k_shared
        self.gate_temperature = gate_temperature
        self.use_user = use_user

        self.routed_experts = nn.ModuleList(
            ExpertFFN(d_model, d_hidden) for _ in range(k_routed)
        )
        self.shared_experts = nn.ModuleList(
            ExpertFFN(d_model, d_hidden) for _ in range(k_shared)
        )

        self.fact_gate = nn.Linear(d_model, k_routed)
        nn.init.normal_(self.fact_gate.weight, std=0.01)
        nn.init.zeros_(self.fact_gate.bias)

        if use_user:
            # Phase 2: per-user bias + per-user sparsity
            self.user_logits = nn.Embedding(n_users, k_routed)
            nn.init.normal_(self.user_logits.weight, std=0.02)
            self.lambda_mlp = nn.Linear(k_routed, 1, bias=True)
            nn.init.normal_(self.lambda_mlp.weight, std=0.01)
            nn.init.zeros_(self.lambda_mlp.bias)
        else:
            # Phase 1: user-invariant scalar λ (log-space + softplus)
            # init=0 ⇒ softplus(0) ≈ 0.69 (matches Phase 2 lambda_u init range)
            self.global_lambda_log = nn.Parameter(torch.zeros(1, 1))

    def forward(
        self,
        fact_emb: torch.Tensor,
        user_id: torch.Tensor | None = None,
        return_expert_outs: bool = False,
    ):
        """fact_emb: (B, d_model). user_id: (B,) long (Phase 2) or None (Phase 1)."""
        B = fact_emb.size(0)

        if self.use_user:
            if user_id is None:
                raise ValueError("Phase1MoE(use_user=True) requires user_id")
            u_user = self.user_logits(user_id)
            u = (self.fact_gate(fact_emb) + u_user) / self.gate_temperature
            lam = F.softplus(self.lambda_mlp(u_user))
        else:
            u = self.fact_gate(fact_emb) / self.gate_temperature
            lam = F.softplus(self.global_lambda_log).expand(B, -1)            # (B, 1)

        routed_alpha = sparsegen(u, lam)                                     # (B, K_routed) sparse simplex

        routed_outs = torch.stack(
            [e(fact_emb) for e in self.routed_experts], dim=1
        )                                                                    # (B, K_routed, d_model)
        routed_mix = (routed_alpha.unsqueeze(-1) * routed_outs).sum(dim=1)   # (B, d_model)

        if self.k_shared > 0:
            shared_outs = torch.stack(
                [s(fact_emb) for s in self.shared_experts], dim=1
            )                                                                # (B, K_shared, d_model)
            shared_mix = shared_outs.mean(dim=1)                             # (B, d_model)
            sub_kg = routed_mix + shared_mix
        else:
            shared_outs = None
            sub_kg = routed_mix

        return {
            "sub_kg": sub_kg,                                                # (B, d_model)
            "routed_alpha": routed_alpha,                                    # (B, K_routed)
            "lam": lam,                                                      # (B, 1)
            "k_active": (routed_alpha > 1e-6).sum(dim=-1).float(),           # (B,)
            "routed_outs": routed_outs if return_expert_outs else None,
            "shared_outs": shared_outs if return_expert_outs else None,
        }

    def gu_table(self) -> torch.Tensor | None:
        """Snapshot of per-user G_u logits (Phase 2 only)."""
        if not self.use_user:
            return None
        return self.user_logits.weight.detach().clone()


# ----- Auxiliary losses --------------------------------------------------------------

def routing_load_balance(routed_alpha: torch.Tensor, lambda_lb: float = 0.1) -> torch.Tensor:
    """Switch Transformer / LD-MoLE Eq. 9 load-balance loss.

    L_lb = E · Σ_i F_i · P_i  where F_i = fraction picking expert i as top,
    P_i = mean router prob. Encourages uniform usage, prevents collapse.
    """
    K = routed_alpha.shape[-1]
    P = routed_alpha.mean(dim=0)
    F_onehot = F.one_hot(routed_alpha.argmax(dim=-1), num_classes=K).float()
    F_frac = F_onehot.mean(dim=0)
    return lambda_lb * K * (F_frac * P).sum()


def routing_orthogonality(routed_alpha: torch.Tensor, lambda_ortho: float = 0.0) -> torch.Tensor:
    """SIMoE (Chen 2025) expert pattern orthogonality loss.

    Promotes expert disentanglement at train-time:
        M  = column-normalized routed_alpha (B, K), each column unit norm
        L  = λ_ortho · ‖M^T M − I‖_F²

    Default off (0.0). Ablation candidate: 0.01-0.1.
    Direct lever for paradigm's *emergent relation-type expert specialization* claim.
    """
    if lambda_ortho <= 0.0:
        return torch.zeros((), device=routed_alpha.device, dtype=routed_alpha.dtype)
    K = routed_alpha.shape[-1]
    col_norm = routed_alpha.norm(dim=0, keepdim=True).clamp(min=1e-8)
    M = routed_alpha / col_norm
    sim = M.t() @ M
    eye = torch.eye(K, device=routed_alpha.device, dtype=routed_alpha.dtype)
    off_diag_sq = ((sim - eye) ** 2).sum() - ((sim.diag() - 1.0) ** 2).sum()
    return lambda_ortho * off_diag_sq


# ----- Reconstruction loss helper ----------------------------------------------------

def recon_loss(fact_emb: torch.Tensor, recon: torch.Tensor, cos_loss: bool = True) -> torch.Tensor:
    """Shared reconstruction loss — cosine (1 − cos) or MSE.

    Used by Phase1Cycle + GenericBaseline + StandardMoEBaseline so the metric stays
    identical across the three model variants.
    """
    if cos_loss:
        return (1 - F.cosine_similarity(fact_emb, recon, dim=-1)).mean()
    return F.mse_loss(recon, fact_emb)


# ----- Frozen encoder mixin ----------------------------------------------------------

class FrozenEncoderHost(nn.Module):
    """Mixin that pins `self.encoder` (if present) to eval mode regardless of train().

    Phase1Cycle, GenericBaseline, and StandardMoEBaseline all hold a frozen encoder;
    without this override calling `.train()` flips the encoder's LayerNorm/Dropout
    into train mode (parameters are still frozen, but forward behavior changes).
    """

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "encoder"):
            self.encoder.eval()
        return self


# ----- Cycle reconstruction decoder --------------------------------------------------

class FactDecoder(nn.Module):
    """Tight bottleneck decoder for cycle reconstruction.

    sub_kg (B, d_model) → tight bottleneck → fact_emb_recon (B, d_model)

    Design rationale (RESEARCH_PLAN §2.1):
    - Tight d_bottleneck (default 64) is critical — expressive decoder (e.g., 1024d hidden)
      lets the cycle solve via identity map (sub_kg ≈ fact_emb), destroying cycle signal.
    - The 64d bottleneck forces sub_kg to encode only fact-content information that can be
      re-expanded, providing implicit pressure for expert differentiation.
    """

    def __init__(self, d_model: int = 1024, d_bottleneck: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_bottleneck),
            nn.GELU(),
            nn.Linear(d_bottleneck, d_model),
        )

    def forward(self, sub_kg: torch.Tensor) -> torch.Tensor:
        return self.net(sub_kg)
