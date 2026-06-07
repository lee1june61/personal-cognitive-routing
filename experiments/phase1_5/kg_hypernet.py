"""KG-hypernetwork modulation — no-bypass replacement for cross-attention (1a default).

Why this block exists (CONTEXT.md info-bottleneck #3, 2026-05-29):
    The cross-attention / FiLM modulations let the passage reach the answer head
    even when the operation-KG is absent — ``MHA(Q=0, K=V=P)`` degenerates to a
    uniform average of P (and FiLM's ``+β`` is a pure-P term), so the answer was
    being predicted by passage↔candidate matching with the router left vestigial
    (Phase 1.5 rev2: K_active=0 yet acc≈0.35). info-bottleneck #3 ("KG 빠지면
    변환 미정의 → bypass 차단") was empirically false.

Canonical invariant enforced here:
    (i) no-bypass: ``forward(kg_hidden = 0) == 0`` exactly. P reaches the output
        ONLY through a KG-parameterised transform. Enforced structurally: the
        only path to the output is ``U_proj(s_gen(kg) * V_proj(h_P))`` where
        ``s_gen`` and ``U_proj`` are bias-free, so ``kg = 0 → s_gen(kg) = 0 →
        output = 0``. No residual, no post-additive bias anywhere on the path.

Form (gated low-rank hypernetwork = attention fact-selection + KG-generated
transform):
    p_p   = p_proj(p_emb)                          # (B, T_p, d_z)
    h_P   = MHA(Q=kg_hidden, K=p_p, V=p_p)         # (B, T_q, d_z)  facts selected by KG
    basis = V_proj(h_P)                            # (B, T_q, r)
    gate  = s_gen(kg_hidden)                       # (B, T_q, r)    KG → low-rank gate
    out   = U_proj(gate * basis)                   # (B, T_q, d_z)

The KG both (1) chooses which facts to read (attention query) and (2) gates the
low-rank transform applied to them — so reasoning lives in the operation-KG
(C0 faithful), not in a KG-independent shared component.

I/O contract is a drop-in for ``CrossAttentionModulation`` /
``FiLMModulation``: ``forward(kg_hidden, p_emb, p_mask) -> (B, T_q, d_z)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attn_mask import safe_key_padding_mask


class KGHypernetModulation(nn.Module):
    """Gated low-rank hypernetwork modulation (Phase 1.5 1a default).

    Args:
        d_z: KG dim and module working dim. Phase 1.5 default 256.
        d_emb: P-side encoder dim. Phase 1.5 default 1024.
        n_heads: MHA head count (default 4 → head_dim=64 for d_z=256).
        rank: low-rank bottleneck of the KG-generated transform. Default
            ``d_z // 4`` (256 → 64).
        dropout: attn dropout. Default 0 (1a is small-scale).
    """

    def __init__(
        self,
        d_z: int = 256,
        d_emb: int = 1024,
        n_heads: int = 4,
        rank: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_z % n_heads != 0:
            raise ValueError(f"d_z={d_z} must be divisible by n_heads={n_heads}")
        self.d_z = d_z
        self.d_emb = d_emb
        self.n_heads = n_heads
        self.rank = rank if rank is not None else max(1, d_z // 4)

        self.p_proj = nn.Linear(d_emb, d_z)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_z,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Output path = U_proj(s_gen(kg) * V_proj(h_P)). s_gen / U_proj are
        # bias-free so that kg=0 → gate=0 → output=0 exactly (invariant (i)).
        # p_proj / attn internal biases are permitted: they are downstream of the
        # multiplicative gate, which zeroes them out when kg=0.
        self.V_proj = nn.Linear(d_z, self.rank, bias=False)
        self.s_gen = nn.Linear(d_z, self.rank, bias=False)
        self.U_proj = nn.Linear(self.rank, d_z, bias=False)

    def forward(
        self,
        kg_hidden: torch.Tensor,
        p_emb: torch.Tensor,
        p_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            kg_hidden: (B, T_q, d_z) — operation-KG hidden (query + gate source).
            p_emb: (B, T_p, d_emb) — frozen encoder of passage (key/value).
            p_mask: (B, T_p) — 1=real, 0=pad.

        Returns: (B, T_q, d_z). Exactly zero when ``kg_hidden`` is zero.
        """
        p_p = self.p_proj(p_emb)  # (B, T_p, d_z)
        key_pad = safe_key_padding_mask(p_mask)  # True = masked out; fully-padded guarded
        h_P, _ = self.attn(
            kg_hidden, p_p, p_p, key_padding_mask=key_pad, need_weights=False
        )  # (B, T_q, d_z)
        basis = self.V_proj(h_P)  # (B, T_q, r)
        gate = self.s_gen(kg_hidden)  # (B, T_q, r) — zero when kg_hidden is zero
        out = self.U_proj(gate * basis)  # (B, T_q, d_z)
        return out
