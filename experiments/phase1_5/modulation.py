"""FiLM modulation block — Row B ablation (paper §7.4 Gap B fallback).

FiLM (Perez et al. 2017): masked-mean pool P → MLP → (γ, β) affine params, applied
element-wise to ``kg_hidden``. Drop-in shape contract with
``CrossAttentionModulation`` (same I/O).

Paper §7.4 Gap B: cross-attention vs FiLM is the modulation-form ablation that
isolates the modulation-primitive contribution. Concat is forbidden by IRON
(bottleneck violation).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FiLMModulation(nn.Module):
    """Element-wise affine modulation: γ, β = MLP(masked-mean(p_emb)), applied to
    each KG hidden vector.

    Args:
        d_z: KG dim. Phase 1.5 default 256.
        d_emb: P encoder dim. Phase 1.5 default 1024.
        d_hidden: FiLM MLP hidden. Default = ``2 * d_z``.
    """

    def __init__(self, d_z: int = 256, d_emb: int = 1024, d_hidden: int | None = None):
        super().__init__()
        self.d_z = d_z
        self.d_emb = d_emb
        d_hidden = d_hidden or (2 * d_z)
        self.film_gen = nn.Sequential(
            nn.Linear(d_emb, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 2 * d_z),
        )
        self.norm = nn.LayerNorm(d_z)

    def forward(
        self,
        kg_hidden: torch.Tensor,
        p_emb: torch.Tensor,
        p_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            kg_hidden: (B, T_q, d_z).
            p_emb: (B, T_p, d_emb).
            p_mask: (B, T_p).

        Returns: (B, T_q, d_z) modulated.
        """
        # Masked-mean pool P.
        m = p_mask.to(p_emb.dtype).unsqueeze(-1)  # (B, T_p, 1)
        p_pooled = (p_emb * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)  # (B, d_emb)
        film = self.film_gen(p_pooled)  # (B, 2*d_z)
        gamma, beta = film.split(self.d_z, dim=-1)  # (B, d_z) ×2
        # Broadcast over T_q.
        out = kg_hidden * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        return self.norm(out)
