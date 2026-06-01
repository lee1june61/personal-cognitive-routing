"""Shared attention-mask helper for modulation blocks (cross_attention, kg_hypernet)."""

from __future__ import annotations

import torch


def safe_key_padding_mask(p_mask: torch.Tensor) -> torch.Tensor:
    """Build an MHA ``key_padding_mask`` (True = masked out) from ``p_mask``
    (1=real, 0=pad), guarding against fully-padded rows.

    An all-True key_padding row makes ``softmax(-inf, ..., -inf)`` → NaN attention
    weights → NaN output. This occurs when a row has an empty passage (HF
    schema-drift) or when ``t_cap_p`` is shorter than every P. When a row is
    entirely padded, keep position 0 visible so attention falls back to that
    single token.
    """
    key_pad = p_mask <= 0  # (B, T_p) bool
    fully_padded = key_pad.all(dim=-1)  # (B,) bool
    if fully_padded.any():
        key_pad = key_pad.clone()
        key_pad[fully_padded, 0] = False
    return key_pad
