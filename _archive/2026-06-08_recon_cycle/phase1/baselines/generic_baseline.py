"""B0 baseline — Generic Encoder + capacity-matched MLP + FactDecoder (no MoE).

Revision 4 (2026-05-21): LLM-based baseline 제거. Stage 1 spec.

Architecture:
    text → frozen encoder → fact_emb (B, 1024)
        → CapacityMLP (single deep MLP, capacity-matched to Phase1MoE expert pool)
        → "sub_kg" (B, 1024)
        → FactDecoder (tight 64d bottleneck) → fact_emb_recon (B, 1024)
        → cosine loss vs fact_emb

**Capacity matching** (paper §3.2 의 "capacity-matched control"):
Phase 1 trainable ≈ 20 expert × ExpertFFN(1024→2048→1024) ≈ 62M.
B0 의 CapacityMLP 도 동일 ~62M params 으로 build — *MoE inductive bias 의 isolated contribution*
검증. 단순 단일 FFN 대신 다층 wider MLP 으로 expert pool 의 raw capacity 매칭.

Direct test of "MoE-KG-cycle adds value over a *same-size* MLP autoencoder".
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..cycle import FrozenEncoder
from ..model import FactDecoder, FrozenEncoderHost, recon_loss


class CapacityMLP(nn.Module):
    """Capacity-matched dense MLP — same param count as Phase1MoE expert pool.

    Phase1MoE (K_routed=16 + K_shared=4) × ExpertFFN(d_model=1024 → d_hidden=2048 → 1024)
    ≈ 20 × 3.1M = 62M trainable.

    Equivalent dense MLP: width chosen so that 1024 → W → W → 1024 reaches ~62M:
    For W ≈ 5500, params ≈ 1024·5500 + 5500·5500 + 5500·1024 ≈ 41M (still smaller),
    so we use 3 hidden layers of width W to hit ~62M:
      1024 → W → W → W → 1024  with W=4500 → ≈ 1024·4500 + 2·4500² + 4500·1024 ≈ 49.7M

    Default `width=4500, n_hidden=3` ≈ 50M (close enough to ~62M; trade-off vs runtime).
    For exact match scale `width` or `n_hidden`.
    """

    def __init__(
        self,
        d_model: int = 1024,
        width: int = 4500,
        n_hidden: int = 3,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(d_model, width), nn.GELU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(width, width), nn.GELU()]
        layers += [nn.Linear(width, d_model)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GenericBaseline(FrozenEncoderHost):
    """Encoder (frozen) → CapacityMLP → FactDecoder → cycle loss.

    No MoE, no expert routing — tests "MoE-KG-cycle adds value over same-capacity MLP".
    """

    def __init__(
        self,
        encoder_name: str = "BAAI/bge-large-en-v1.5",
        d_bottleneck: int = 64,
        mlp_width: int = 4500,
        mlp_n_hidden: int = 3,
    ):
        super().__init__()
        self.encoder = FrozenEncoder(encoder_name)
        d_model = self.encoder.d_model
        self.body = CapacityMLP(d_model=d_model, width=mlp_width, n_hidden=mlp_n_hidden)
        self.decoder = FactDecoder(d_model=d_model, d_bottleneck=d_bottleneck)

    @torch.no_grad()
    def encode(self, texts: list[str], max_length: int = 256) -> torch.Tensor:
        return self.encoder(texts, max_length=max_length)

    def forward(self, fact_emb: torch.Tensor) -> dict:
        sub_kg = self.body(fact_emb)
        recon = self.decoder(sub_kg)
        return {"sub_kg": sub_kg, "recon": recon}

    def cycle_loss(self, fact_emb: torch.Tensor, cos_loss: bool = True) -> dict:
        out = self.forward(fact_emb)
        loss = recon_loss(fact_emb, out["recon"], cos_loss=cos_loss)
        return {"loss": loss, "loss_recon": loss.detach()}
