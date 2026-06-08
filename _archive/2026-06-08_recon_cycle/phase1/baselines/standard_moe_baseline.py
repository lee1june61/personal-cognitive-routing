"""B1 baseline — Generic Encoder + Standard MoE (Switch Transformer style) + FactDecoder.

Revision 4 (2026-05-21): standard MoE without KG bottleneck + cycle reconstruction
of Phase 1. Tests *MoE inductive bias alone* — does KG-bottleneck-cycle add over plain
sparse routing?

Architecture:
    text → frozen BGE-large-en → fact_emb (B, 1024)
        → StandardMoE (top-1 hard routing, Switch style) → moe_out (B, 1024)
        → FactDecoder (tight 64d bottleneck) → fact_emb_recon
        → cosine loss

Differences from Phase1MoE:
  - Top-1 hard routing (Switch) instead of sparsegen sparse simplex (LD-MoLE)
  - No shared experts (Switch convention)
  - No emergent-relation-type framing — just classical MoE

**Capacity match** (paper §3.2): Phase 1 = 16 routed + 4 shared = 20 expert × 3.1M ≈ 62M.
B1 default `k_routed=20` (same expert count) + `d_hidden=2048` → ≈ 62M trainable. *Pure
architectural difference* (routing form: sparsegen vs Switch top-1 + shared vs no-shared).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..cycle import FrozenEncoder
from ..model import (
    ExpertFFN,
    FactDecoder,
    FrozenEncoderHost,
    recon_loss,
    routing_load_balance,
)


class StandardMoE(nn.Module):
    """Switch Transformer style top-1 hard routing MoE."""

    def __init__(
        self,
        d_model: int = 1024,
        k_routed: int = 16,
        d_hidden: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model
        self.k_routed = k_routed
        self.experts = nn.ModuleList(
            ExpertFFN(d_model, d_hidden) for _ in range(k_routed)
        )
        self.gate = nn.Linear(d_model, k_routed)
        nn.init.normal_(self.gate.weight, std=0.01)
        nn.init.zeros_(self.gate.bias)

    def forward(self, fact_emb: torch.Tensor) -> dict:
        logits = self.gate(fact_emb)
        gate_prob = F.softmax(logits, dim=-1)
        top_idx = gate_prob.argmax(dim=-1)
        top_w = gate_prob.gather(-1, top_idx.unsqueeze(-1)).squeeze(-1)

        out = torch.zeros_like(fact_emb)
        # Per-expert masked dispatch. No guard — PyTorch handles 0-row FFN forwards
        # without error, and unconditional dispatch avoids K device→host syncs/step.
        for k, expert in enumerate(self.experts):
            mask = top_idx == k
            out[mask] = expert(fact_emb[mask]) * top_w[mask].unsqueeze(-1)

        routed_alpha = F.one_hot(top_idx, num_classes=self.k_routed).float()
        return {
            "moe_out": out,
            "routed_alpha": routed_alpha,
            "gate_prob": gate_prob,
        }


class StandardMoEBaseline(FrozenEncoderHost):
    """Encoder (frozen) → StandardMoE → FactDecoder → cycle loss.

    `k_routed=20` matches Phase1MoE's 16 routed + 4 shared (capacity-matched).
    """

    def __init__(
        self,
        encoder_name: str = "BAAI/bge-large-en-v1.5",
        k_routed: int = 20,
        d_hidden: int = 2048,
        d_bottleneck: int = 64,
    ):
        super().__init__()
        self.encoder = FrozenEncoder(encoder_name)
        d_model = self.encoder.d_model
        self.moe = StandardMoE(d_model=d_model, k_routed=k_routed, d_hidden=d_hidden)
        self.decoder = FactDecoder(d_model=d_model, d_bottleneck=d_bottleneck)

    @torch.no_grad()
    def encode(self, texts: list[str], max_length: int = 256) -> torch.Tensor:
        return self.encoder(texts, max_length=max_length)

    def forward(self, fact_emb: torch.Tensor) -> dict:
        moe_out = self.moe(fact_emb)
        recon = self.decoder(moe_out["moe_out"])
        return {"sub_kg": moe_out["moe_out"], "recon": recon, **moe_out}

    def cycle_loss(
        self,
        fact_emb: torch.Tensor,
        cos_loss: bool = True,
        lambda_lb: float = 0.1,
    ) -> dict:
        out = self.forward(fact_emb)
        recon = recon_loss(fact_emb, out["recon"], cos_loss=cos_loss)
        lb_loss = routing_load_balance(out["routed_alpha"], lambda_lb=lambda_lb)
        return {
            "loss": recon + lb_loss,
            "loss_recon": recon.detach(),
            "loss_lb": lb_loss.detach(),
            "routed_alpha": out["routed_alpha"],
        }
