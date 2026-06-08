"""Phase 1.5 1a model — emergent operation router + cross-attention modulation + MC head.

Forward pass (paper §5.2 + RESEARCH_PLAN_2026-05-28_phase1_5.md §2):

    Q (B, T_q, d_emb) ──SharedEncoderHead──▶ z_q (B, T_q, d_z)
    z_q              ──ReMoERouter (K=128)──▶ alpha (B, T_q, K)
    kg_hidden        = Σ_k alpha[..., k:k+1] · OperationExpert_k(z_q)  → (B, T_q, d_z)
    P (B, T_p, d_emb) → side-channel only

    kg_hidden, P     ──CrossAttentionModulation (KG=Q, P=KV)──▶ kg_modulated (B, T_q, d_z)
    kg_summary       = masked_mean(kg_modulated, q_mask) → (B, d_z)
    cand_pooled      → trainable projection W_cand → cand_proj (B, 4, d_z)
    logits           = einsum("bd,bcd->bc", kg_summary, cand_proj) / √d_z  → (B, 4)

SharedEncoderHead / ReMoERouter / running-sum mixture are copy of
``phase1/model_opcycle.py`` (closed evidence — hardened, 120 tests passed).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cross_attention import CrossAttentionModulation
from .kg_hypernet import KGHypernetModulation
from .modulation import FiLMModulation

# SharedEncoderHead, ReMoERouter (superset originals lived here), and the LB factory now
# live in core. phase1_5 is the SUPERSET source for the head/router — the core
# copies are verbatim, so importing them here is behaviour-identical. Re-exported so
# ``phase1_5.model.{SharedEncoderHead,ReMoERouter}`` (used by tests) keep resolving.
from core.shared_heads import SharedEncoderHead
from core.routers import ReMoERouter
from core.load_balance import LB_OFF, make_lb


MOD_KG_HYPERNET = "kg_hypernet"
MOD_CROSS_ATTN = "cross_attn"
MOD_FILM = "film"
# kg_hypernet = 1a default (no-bypass, CONTEXT.md info-bottleneck #3, 2026-05-29).
# cross_attn / film retained as §7.4 Gap B ablation baselines — both empirically
# violate the no-bypass invariant (KG=0 still leaks passage), see ADR 0001.
MODULATION_TYPES = (MOD_KG_HYPERNET, MOD_CROSS_ATTN, MOD_FILM)


# ----- Operation expert (= DecoderExpert with output dim = d_z) ----------------------


class OperationExpert(nn.Module):
    """Per-expert operation primitive: z_t (d_z) → d_hidden → GELU → out (d_z).

    Phase 1.5 differs from phase1 ``DecoderExpert``: output stays in latent d_z
    (not d_emb). The mixture Σ_k α_k · expert_k(z) is the operation-KG hidden,
    not a reconstruction of h.
    """

    def __init__(self, d_z: int = 256, d_hidden: int = 512, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_z, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_z)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(z))))


# ----- MC head (paper §5.1 row 8 default = dot-product scoring) ----------------------


class MCHead(nn.Module):
    """Projected dot-product scorer between ``kg_summary`` (B, d_z) and 4 candidate
    embeddings (B, 4, d_emb). Candidates are projected to d_z, scaled by 1/√d_z, and
    scored via einsum. Used as the 1a default MC contrastive head (paper §5.1 row 8).
    """

    def __init__(self, d_z: int = 256, d_emb: int = 1024):
        super().__init__()
        self.cand_proj = nn.Linear(d_emb, d_z, bias=False)
        self.scale = 1.0 / math.sqrt(d_z)

    def forward(self, kg_summary: torch.Tensor, cand_pooled: torch.Tensor) -> torch.Tensor:
        cand_proj = self.cand_proj(cand_pooled)  # (B, 4, d_z)
        logits = torch.einsum("bd,bcd->bc", kg_summary, cand_proj) * self.scale
        return logits


# ----- Phase 1.5 1a model ------------------------------------------------------------


class Phase15MoE(nn.Module):
    """Phase 1.5 1a model: flat operation router + cross-attn modulation + MC head.

    Args:
        d_emb: frozen-encoder hidden dim (e5-large-v2 / BGE-large = 1024).
        d_z: operation-axis latent dim. Phase 1.5 default 256.
        k_routed: number of routed experts. Phase 1.5 default 128.
        d_hidden_expert: expert FFN hidden. Phase 1.5 default 512.
        modulation: ``"kg_hypernet"`` (1a default, no-bypass) / ``"cross_attn"`` /
            ``"film"``. cross_attn·film are §7.4 Gap B baselines that violate the
            no-bypass invariant (ADR 0001).
        cross_attn_heads: only used when ``modulation="cross_attn"``.
        lb_strategy: load-balancing strategy from ``load_balance.LB_STRATEGIES``.
            ``"off"`` = no LB (Row F baseline); ``"aux_free"`` = DeepSeek-V3
            per-expert bias (Phase 1.5 1a default, Layer-1 dead-router fix).
        lb_target_active: target K_active for the LB rule. Must match the
            train-time ``k_target`` (engine_1a forwards ``row.k_active_target``).
        lb_lr_bias: LB bias update rate. DeepSeek-V3 paper-faithful default 1e-3.
    """

    def __init__(
        self,
        d_emb: int = 1024,
        d_z: int = 256,
        k_routed: int = 128,
        d_hidden_expert: int = 512,
        modulation: str = MOD_KG_HYPERNET,
        cross_attn_heads: int = 4,
        lb_strategy: str = LB_OFF,
        lb_target_active: float = 4.0,
        lb_lr_bias: float = 1e-3,
        routing: str = "relu_l1",
        dropout: float = 0.0,
        chain_steps: int = 1,
    ):
        super().__init__()
        if modulation not in MODULATION_TYPES:
            raise ValueError(f"modulation must be in {MODULATION_TYPES}; got {modulation}")
        if chain_steps < 1:
            raise ValueError(f"chain_steps must be >= 1; got {chain_steps}")
        self.d_emb = d_emb
        self.d_z = d_z
        self.k_routed = k_routed
        self.modulation_type = modulation
        self.chain_steps = chain_steps

        self.encoder_head = SharedEncoderHead(d_emb, d_z, dropout=dropout)
        # routing="topk" → K_active ≡ round(lb_target_active) by construction
        # (Phase 3 diversity, ADR 0002); "relu_l1" → adaptive-L1 sparsity (default).
        self.router = ReMoERouter(
            d_z, k_routed, routing=routing, k_active=int(round(lb_target_active))
        )
        self.experts = nn.ModuleList(
            OperationExpert(d_z, d_hidden_expert, dropout=dropout) for _ in range(k_routed)
        )

        if modulation == MOD_KG_HYPERNET:
            self.modulation: nn.Module = KGHypernetModulation(
                d_z=d_z, d_emb=d_emb, n_heads=cross_attn_heads
            )
        elif modulation == MOD_CROSS_ATTN:
            self.modulation = CrossAttentionModulation(
                d_z=d_z, d_emb=d_emb, n_heads=cross_attn_heads
            )
        else:  # MOD_FILM
            self.modulation = FiLMModulation(d_z=d_z, d_emb=d_emb)

        # 1b chain-of-experts: one modulation block per chain step. Step 0 REUSES
        # ``self.modulation`` so ``forward_chain`` with chain_steps=1 is identical
        # to the flat ``forward``; steps 1..L-1 are fresh same-type blocks (each
        # reads P with its own KG-conditioned fact-selection → the next hop's fact).
        # Tied across steps would force one operation; per-step lets the program
        # compose distinct operations. ``forward`` itself never touches this list.
        extra_steps = [
            self._make_modulation(modulation, d_z, d_emb, cross_attn_heads)
            for _ in range(chain_steps - 1)
        ]
        self._chain_modulations = nn.ModuleList([self.modulation, *extra_steps])

        self.mc_head = MCHead(d_z=d_z, d_emb=d_emb)

        # LB module (Layer-1 dead-router fix). ``None`` when strategy="off".
        # When non-None, ``forward`` / ``compute_alpha`` pass ``lb.bias`` as
        # ``external_bias`` to the router; ``train_phase15`` calls ``lb.step``
        # after each optimiser step.
        self.lb = make_lb(
            lb_strategy,
            k_routed=k_routed,
            k_active_target=lb_target_active,
            lr_bias=lb_lr_bias,
        )

    @staticmethod
    def _make_modulation(modulation: str, d_z: int, d_emb: int, cross_attn_heads: int) -> nn.Module:
        """Construct a modulation block of the given type (shared by the flat
        modulation and the per-step chain modulations)."""
        if modulation == MOD_KG_HYPERNET:
            return KGHypernetModulation(d_z=d_z, d_emb=d_emb, n_heads=cross_attn_heads)
        if modulation == MOD_CROSS_ATTN:
            return CrossAttentionModulation(d_z=d_z, d_emb=d_emb, n_heads=cross_attn_heads)
        return FiLMModulation(d_z=d_z, d_emb=d_emb)

    def _route(self, z_q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Router call with the aux-free LB bias applied (``None`` when LB off).
        Shared by ``forward`` and ``compute_alpha`` so both route identically."""
        external_bias = self.lb.bias if self.lb is not None else None
        return self.router(z_q, external_bias=external_bias)

    def forward(self, batch: dict, alpha_override: torch.Tensor | None = None) -> dict:
        """batch keys (all tensors, B-leading):
        ``q_tokens`` (B, T_q, d_emb), ``q_mask`` (B, T_q),
        ``p_tokens`` (B, T_p, d_emb), ``p_mask`` (B, T_p),
        ``cand_pooled`` (B, 4, d_emb).

        Returns dict with: ``logits`` (B, 4), ``alpha`` (B, T_q, K),
        ``z_q`` (B, T_q, d_z), ``router_logits`` (B, T_q, K),
        ``kg_hidden`` (B, T_q, d_z), ``kg_modulated`` (B, T_q, d_z),
        ``kg_summary`` (B, d_z), ``k_active`` (B, T_q).
        """
        q_tokens = batch["q_tokens"]
        q_mask = batch["q_mask"]
        p_tokens = batch["p_tokens"]
        p_mask = batch["p_mask"]
        cand_pooled = batch["cand_pooled"]

        z_q = self.encoder_head(q_tokens)  # (B, T_q, d_z)
        alpha, router_logits = self._route(z_q)  # (B, T_q, K)
        # Intervention hook (intervention.py lesion/swap): replace the routed
        # alpha with a supplied pattern. router_logits keeps the true routing.
        if alpha_override is not None:
            alpha = alpha_override.to(z_q.dtype)

        # Running-sum mixture (copy of phase1 OpCycleMoE.forward; avoids the K-stack
        # OOM at K=128). **Accumulator forced to fp32** — under AMP / fp16 inputs,
        # summing K=128 expert outputs into an alpha-dtype accumulator can overflow
        # or shed precision. Cast back to the input dtype at the end.
        accum_dtype = torch.float32
        kg_hidden = torch.zeros(
            *z_q.shape[:-1], self.d_z, dtype=accum_dtype, device=z_q.device
        )  # (B, T_q, d_z) fp32
        for k, expert in enumerate(self.experts):
            kg_hidden = kg_hidden + (alpha[..., k : k + 1].to(accum_dtype) * expert(z_q).to(accum_dtype))
        kg_hidden = kg_hidden.to(z_q.dtype)

        kg_modulated = self.modulation(kg_hidden, p_tokens, p_mask)  # (B, T_q, d_z)

        # Masked mean pool over Q tokens.
        q_mask_f = q_mask.to(kg_modulated.dtype).unsqueeze(-1)  # (B, T_q, 1)
        kg_summary = (kg_modulated * q_mask_f).sum(dim=1) / q_mask_f.sum(dim=1).clamp(
            min=1.0
        )

        logits = self.mc_head(kg_summary, cand_pooled)  # (B, 4)

        return {
            "logits": logits,
            "alpha": alpha,
            "z_q": z_q,
            "router_logits": router_logits,
            "kg_hidden": kg_hidden,
            "kg_modulated": kg_modulated,
            "kg_summary": kg_summary,
            "k_active": (alpha > 0).sum(dim=-1),  # (B, T_q)
        }

    def forward_chain(self, batch: dict, alpha_override_steps: list | None = None) -> dict:
        """1b chain-of-experts forward (L = ``self.chain_steps``).

        ⚠ DEPRECATED (2026-06-08): the sequential-chain LAYOUT is a setup-failure
        (breadth {2,2,2} non-monotone; frozen-e5 ceiling still OPEN), superseded by
        direction-1 PARALLEL co-activation. This does NOT discard the operation-
        emergence hypothesis (lit-protected) nor parallel composition. Kept in-place
        because train.py / intervention.py / ablations.py call it (removal is invasive);
        the 1b orchestrator is archived at _archive/2026-06-08_seq_chain_1b/.

        Per step ℓ (state ``z^(0)=z_q``):
            ``route_in_ℓ = z_q + Σ_{j<ℓ} mod_j``     (router + experts see progress)
            ``alpha_ℓ    = route(route_in_ℓ)``
            ``kg_ℓ       = Σ_k alpha_ℓ_k · expert_k(route_in_ℓ)``
            ``mod_ℓ      = chain_modulation_ℓ(kg_ℓ, P)``

        **No-bypass (CONTEXT.md info-bottleneck #3):** the OUTPUT accumulator is
        ``kg_modulated = Σ_ℓ mod_ℓ`` — it excludes ``z_q``. So if every step's
        alpha is 0, every ``kg_ℓ=0`` ⇒ every ``mod_ℓ=0`` (kg_hypernet is
        bias-free) ⇒ ``kg_summary=0`` ⇒ uniform logits: neither P nor Q can reach
        the answer except through a KG-parameterised transform. The router input
        carries ``z_q`` (+progress) — routing is NOT the output path, so a step≥2
        can route non-trivially — but a residual into the *summary* would
        reintroduce a Q-bypass and is therefore deliberately absent.

        ``alpha_override_steps`` (list of (B,T_q,K) per step) lets the causal
        battery (``intervention.py``) lesion/swap the experts used at a given hop.
        For ``chain_steps=1`` this is numerically identical to flat ``forward``.
        """
        q_tokens, q_mask = batch["q_tokens"], batch["q_mask"]
        p_tokens, p_mask = batch["p_tokens"], batch["p_mask"]
        cand_pooled = batch["cand_pooled"]

        z_q = self.encoder_head(q_tokens)  # (B, T_q, d_z)
        accum_dtype = torch.float32
        kg_modulated = torch.zeros_like(z_q)  # Σ_ℓ mod_ℓ (output path — excludes z_q)
        progress = torch.zeros_like(z_q)  # Σ_{j<ℓ} mod_j (router/expert input only)

        alpha_steps: list[torch.Tensor] = []
        router_logits_steps: list[torch.Tensor] = []
        kg_steps: list[torch.Tensor] = []
        for ell in range(self.chain_steps):
            route_in = z_q + progress  # z^(ℓ-1)
            alpha, router_logits = self._route(route_in)
            if alpha_override_steps is not None:
                alpha = alpha_override_steps[ell].to(z_q.dtype)
            kg = torch.zeros(
                *z_q.shape[:-1], self.d_z, dtype=accum_dtype, device=z_q.device
            )
            for k, expert in enumerate(self.experts):
                kg = kg + (alpha[..., k : k + 1].to(accum_dtype) * expert(route_in).to(accum_dtype))
            kg = kg.to(z_q.dtype)
            mod = self._chain_modulations[ell](kg, p_tokens, p_mask)  # (B, T_q, d_z)
            kg_modulated = kg_modulated + mod
            progress = progress + mod
            alpha_steps.append(alpha)
            router_logits_steps.append(router_logits)
            kg_steps.append(kg)

        q_mask_f = q_mask.to(kg_modulated.dtype).unsqueeze(-1)
        kg_summary = (kg_modulated * q_mask_f).sum(dim=1) / q_mask_f.sum(dim=1).clamp(min=1.0)
        logits = self.mc_head(kg_summary, cand_pooled)

        return {
            "logits": logits,
            "alpha_steps": alpha_steps,
            "router_logits_steps": router_logits_steps,
            "kg_steps": kg_steps,
            "kg_modulated": kg_modulated,
            "kg_summary": kg_summary,
            "z_q": z_q,
            "k_active_steps": [(a > 0).sum(dim=-1) for a in alpha_steps],
        }

    def compute_alpha(self, q_tokens: torch.Tensor) -> dict:
        """Encoder-head + router only — returns ``alpha`` and ``router_logits`` without
        running the K-expert running-sum, the cross-attention block, or the MC head.

        Used by the selectivity probe (``engine_1a._compute_codes``) which reads only
        ``alpha``. At K=128 / T_q=128 / B=64 this skips ~99% of the eval forward FLOPs
        compared with the full ``forward`` (plus avoids the zero-``cand_pooled``
        fabrication coupling probe correctness to ``MCHead`` invariants).

        No ``@torch.no_grad()`` decorator — both current callers already wrap
        themselves in ``no_grad``, but future gradient-bearing uses (router-grad
        diagnostics, saliency probes) need the gradients available.
        """
        z_q = self.encoder_head(q_tokens)
        alpha, router_logits = self._route(z_q)
        return {
            "alpha": alpha,
            "router_logits": router_logits,
            "z_q": z_q,
            "k_active": (alpha > 0).sum(dim=-1),
        }
