"""Tests for `phase1_5.kg_hypernet` — KGHypernetModulation (no-bypass modulation).

Invariant under test (info-bottleneck #3, CONTEXT.md): with the operation-KG
removed (``kg_hidden = 0``) the modulation output must be exactly zero, so the
passage cannot reach the answer head except through a KG-parameterised transform.
"""

from __future__ import annotations

import pytest
import torch

from experiments.phase1_5.kg_hypernet import KGHypernetModulation


def test_kg_hypernet_output_shape():
    block = KGHypernetModulation(d_z=16, d_emb=32, n_heads=4, rank=8)
    kg = torch.randn(2, 5, 16)
    p = torch.randn(2, 7, 32)
    p_mask = torch.ones(2, 7)
    out = block(kg, p, p_mask)
    assert out.shape == (2, 5, 16)


def test_kg_hypernet_zero_kg_gives_exact_zero_output():
    """★ info-bottleneck #3 (i): kg_hidden=0 ⟹ output exactly 0.

    The passage must not reach the answer head when the operation-KG is absent.
    Uses non-trivial P (random, fully attended) to prove it is the *gate*, not a
    degenerate P, that zeroes the output. Eval mode to disable dropout.
    """
    block = KGHypernetModulation(d_z=16, d_emb=32, n_heads=4, rank=8).eval()
    kg = torch.zeros(2, 5, 16)
    p = torch.randn(2, 7, 32)
    p_mask = torch.ones(2, 7)
    out = block(kg, p, p_mask)
    assert torch.all(out == 0)


def test_kg_hypernet_no_additive_bias_on_output_path():
    """The gate and output projections must be bias-free, else a KG-independent
    constant leaks through when kg=0 (re-opening the bypass)."""
    block = KGHypernetModulation(d_z=16, d_emb=32, rank=8)
    assert block.s_gen.bias is None
    assert block.U_proj.bias is None
    assert block.V_proj.bias is None


def test_kg_hypernet_nonzero_kg_gives_nonzero_output():
    block = KGHypernetModulation(d_z=16, d_emb=32, n_heads=4, rank=8).eval()
    kg = torch.randn(2, 5, 16)
    p = torch.randn(2, 7, 32)
    out = block(kg, p, torch.ones(2, 7))
    assert not torch.all(out == 0)


def test_kg_hypernet_io_contract_matches_cross_attn():
    """Drop-in swap: same (kg, p, mask) → (B, T_q, d_z) as CrossAttentionModulation."""
    from experiments.phase1_5.cross_attention import CrossAttentionModulation

    kg = torch.randn(2, 3, 8)
    p = torch.randn(2, 4, 16)
    mask = torch.ones(2, 4)
    out_hyper = KGHypernetModulation(d_z=8, d_emb=16, n_heads=2, rank=4)(kg, p, mask)
    out_xattn = CrossAttentionModulation(d_z=8, d_emb=16, n_heads=2)(kg, p, mask)
    assert out_hyper.shape == out_xattn.shape == (2, 3, 8)


def test_kg_hypernet_padded_p_does_not_attend():
    """Garbage in padded P positions must not change the output (eval, kg≠0)."""
    block = KGHypernetModulation(d_z=16, d_emb=32, n_heads=4, rank=8).eval()
    kg = torch.randn(1, 3, 16)
    p1 = torch.randn(1, 5, 32)
    p2 = p1.clone()
    p2[0, 3:] = torch.randn(2, 32) * 100  # pads get garbage
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
    with torch.no_grad():
        out1 = block(kg, p1, mask)
        out2 = block(kg, p2, mask)
    assert torch.allclose(out1, out2, atol=1e-4)


def test_kg_hypernet_backward():
    block = KGHypernetModulation(d_z=8, d_emb=16, n_heads=2, rank=4)
    kg = torch.randn(2, 3, 8, requires_grad=True)
    p = torch.randn(2, 4, 16, requires_grad=True)
    out = block(kg, p, torch.ones(2, 4))
    out.sum().backward()
    assert kg.grad is not None
    assert p.grad is not None
    assert block.s_gen.weight.grad is not None
    assert block.U_proj.weight.grad is not None
    assert block.p_proj.weight.grad is not None


def test_kg_hypernet_default_dims():
    """Phase 1.5 1a default — d_z=256, d_emb=1024, n_heads=4, rank=d_z//4=64."""
    block = KGHypernetModulation()
    assert block.d_z == 256
    assert block.d_emb == 1024
    assert block.n_heads == 4
    assert block.rank == 64


def test_kg_hypernet_d_z_must_divide_by_n_heads():
    with pytest.raises(ValueError, match="divisible"):
        KGHypernetModulation(d_z=17, d_emb=32, n_heads=4)
