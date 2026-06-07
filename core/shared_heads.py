"""Shared trainable encoder head — per-token d_emb → d_z (Linear → GELU → [Dropout] → Linear).

Superset of the two copies (phase1/model_opcycle.py ``SharedEncoderHead``,
phase1_5/model.py ``SharedEncoderHead``). The phase1_5 version added a ``dropout`` param;
with ``dropout=0.0`` the dropout slot is ``nn.Identity`` → identical to phase1's net.

Both original keyword spellings of the first arg are accepted:
  - phase1 test calls ``SharedEncoderHead(d_model=1024, d_z=256)``
  - phase1_5 test calls ``SharedEncoderHead(d_emb=1024, d_z=256)``
``d_emb`` is the canonical name; ``d_model`` is accepted as an alias so neither caller
breaks. Positional first-arg (in_dim) also works for both.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SharedEncoderHead(nn.Module):
    """Per-token d_emb → d_z. Linear → GELU → [Dropout] → Linear.

    Args:
        d_emb: input (frozen-encoder hidden) dim. Accepts ``d_model=`` as a back-compat
            keyword alias (phase1 caller spelling).
        d_z: latent dim.
        dropout: dropout prob between GELU and the second Linear. ``0.0`` → ``nn.Identity``
            (identical to phase1's dropout-free net).
    """

    def __init__(self, d_emb: int = 1024, d_z: int = 256, dropout: float = 0.0, *, d_model: int | None = None):
        super().__init__()
        if d_model is not None:
            d_emb = d_model
        self.net = nn.Sequential(
            nn.Linear(d_emb, d_z),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(d_z, d_z),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)
