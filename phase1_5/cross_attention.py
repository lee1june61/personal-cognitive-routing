"""Cross-attention modulation block — paper §5.1 row 4 (default modulation).

KG-as-Q, P-as-KV, single-layer transformer block:

    p_proj      = W_kv(p_emb)                                # (B, T_p, d_z)
    attn_out, _ = MHA(Q=kg_hidden, K=p_proj, V=p_proj,
                      key_padding_mask=(p_mask == 0))         # (B, T_q, d_z)
    x           = LayerNorm(kg_hidden + attn_out)
    kg_modulated= LayerNorm(x + FFN(x))                       # (B, T_q, d_z)

Why P-side projection rather than KG up-projection: d_z is the operation-axis
dimension; keeping cross-attn in d_z preserves architectural homogeneity with the
emergent NMN target (1b chain) which reuses the same block.

Why ``concat`` is forbidden by IRON: a concat baseline lets the Z-as-bottleneck
factorisation break (Z and P become co-equal inputs to the answer head;
removing Z is recoverable from P alone). The modulation form enforces
"Z parameterises the P→A transform" (paper §6.1 row 2, §6.2).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attn_mask import safe_key_padding_mask


class CrossAttentionModulation(nn.Module):
    """Single transformer block: MHA(KG, P, P) + residual + FFN.

    Args:
        d_z: KG dim and module working dim. Phase 1.5 default 256.
        d_emb: P-side encoder dim. Phase 1.5 default 1024.
        n_heads: MHA head count (default 4 → head_dim=64 for d_z=256).
        ffn_mult: FFN inner dim = ``ffn_mult * d_z`` (default 4 → 1024).
        dropout: attn + FFN dropout. Default 0 (1a is small-scale, no overfit yet).
    """

    def __init__(
        self,
        d_z: int = 256,
        d_emb: int = 1024,
        n_heads: int = 4,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_z % n_heads != 0:
            raise ValueError(f"d_z={d_z} must be divisible by n_heads={n_heads}")
        self.d_z = d_z
        self.d_emb = d_emb
        self.n_heads = n_heads

        self.p_proj = nn.Linear(d_emb, d_z)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_z,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_z)
        self.ffn = nn.Sequential(
            nn.Linear(d_z, ffn_mult * d_z),
            nn.GELU(),
            nn.Linear(ffn_mult * d_z, d_z),
        )
        self.norm2 = nn.LayerNorm(d_z)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        kg_hidden: torch.Tensor,
        p_emb: torch.Tensor,
        p_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            kg_hidden: (B, T_q, d_z) — query side (KG hidden from operation router).
            p_emb: (B, T_p, d_emb) — key/value side (frozen encoder of passage).
            p_mask: (B, T_p) — attention mask (1=real, 0=pad).

        Returns: (B, T_q, d_z) modulated KG hidden.
        """
        p_proj = self.p_proj(p_emb)  # (B, T_p, d_z)
        key_pad = safe_key_padding_mask(p_mask)  # True = masked out; fully-padded guarded
        attn_out, _ = self.attn(
            kg_hidden, p_proj, p_proj, key_padding_mask=key_pad, need_weights=False
        )
        x = self.norm1(kg_hidden + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x
