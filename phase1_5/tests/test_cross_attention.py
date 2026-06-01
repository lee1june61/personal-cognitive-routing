"""Tests for `phase1_5.cross_attention`."""

from __future__ import annotations

import pytest
import torch

from research.demo.phase1_5.cross_attention import CrossAttentionModulation


def test_cross_attention_output_shape():
    block = CrossAttentionModulation(d_z=16, d_emb=32, n_heads=4)
    kg = torch.randn(2, 5, 16)
    p = torch.randn(2, 7, 32)
    p_mask = torch.ones(2, 7)
    out = block(kg, p, p_mask)
    assert out.shape == (2, 5, 16)


def test_cross_attention_d_z_must_divide_by_n_heads():
    with pytest.raises(ValueError, match="divisible"):
        CrossAttentionModulation(d_z=17, d_emb=32, n_heads=4)


def test_cross_attention_padded_p_does_not_attend():
    """Where p_mask=0 the attention should not put weight on those positions.
    We check via gradient: zeroing pads should not change the output relative
    to leaving them with random values."""
    block = CrossAttentionModulation(d_z=16, d_emb=32, n_heads=4).eval()
    kg = torch.randn(1, 3, 16)
    p1 = torch.randn(1, 5, 32)
    p2 = p1.clone()
    p2[0, 3:] = torch.randn(2, 32) * 100  # pad positions get garbage
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
    with torch.no_grad():
        out1 = block(kg, p1, mask)
        out2 = block(kg, p2, mask)
    assert torch.allclose(out1, out2, atol=1e-4)


def test_cross_attention_backward():
    block = CrossAttentionModulation(d_z=8, d_emb=16, n_heads=2)
    kg = torch.randn(2, 3, 8, requires_grad=True)
    p = torch.randn(2, 4, 16, requires_grad=True)
    mask = torch.ones(2, 4)
    out = block(kg, p, mask)
    out.sum().backward()
    assert kg.grad is not None
    assert p.grad is not None
    assert block.p_proj.weight.grad is not None


def test_cross_attention_default_dims():
    """Phase 1.5 1a default — d_z=256, d_emb=1024, n_heads=4."""
    block = CrossAttentionModulation()
    assert block.d_z == 256
    assert block.d_emb == 1024
    assert block.n_heads == 4
    assert block.p_proj.in_features == 1024
    assert block.p_proj.out_features == 256


# ---- FiLM modulation (Row B drop-in) ------------------------------------------------


def test_film_modulation_output_shape():
    from research.demo.phase1_5.modulation import FiLMModulation

    film = FiLMModulation(d_z=16, d_emb=32)
    kg = torch.randn(2, 5, 16)
    p = torch.randn(2, 7, 32)
    mask = torch.ones(2, 7)
    out = film(kg, p, mask)
    assert out.shape == (2, 5, 16)


def test_film_modulation_io_contract_matches_cross_attn():
    """Both modulation blocks must accept ``(kg_hidden, p_emb, p_mask)`` and return
    ``(B, T_q, d_z)`` — drop-in swap is required for ablation Row B."""
    from research.demo.phase1_5.modulation import FiLMModulation

    xattn = CrossAttentionModulation(d_z=8, d_emb=16, n_heads=2)
    film = FiLMModulation(d_z=8, d_emb=16)

    kg = torch.randn(2, 3, 8)
    p = torch.randn(2, 4, 16)
    mask = torch.ones(2, 4)
    out_xattn = xattn(kg, p, mask)
    out_film = film(kg, p, mask)
    assert out_xattn.shape == out_film.shape


def test_film_modulation_p_mask_affects_pool():
    """Masked-mean of P excludes padded positions."""
    from research.demo.phase1_5.modulation import FiLMModulation

    film = FiLMModulation(d_z=8, d_emb=16).eval()
    kg = torch.randn(1, 2, 8)
    p_all = torch.randn(1, 4, 16)
    mask_all = torch.ones(1, 4)
    mask_partial = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    with torch.no_grad():
        out_all = film(kg, p_all, mask_all)
        out_partial = film(kg, p_all, mask_partial)
    # Different masks → different pooled P → different (γ, β) → different output.
    assert not torch.allclose(out_all, out_partial, atol=1e-4)
