"""Tests for the per-token masked operation-cycle loss (ENGINE_A_DESIGN §3).

    L = λ_recon·(1 − cos(h_t, ĥ_t))      # masked tokens only, mean
      + λ_l1·‖α‖₁                          # ReMoE adaptive L1, K_active self-regulates
      + λ_z·z_loss                         # ST-MoE router z-loss (stability)
      + λ_lb·load_balance                  # expert-utilisation balance (anti-collapse)

All terms are masked (padding tokens excluded) and tiny-dim CPU.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from phase1.model_opcycle import (
    OpCycleMoE,
    masked_recon_loss,
    remoe_l1_loss,
    router_z_loss,
    load_balance_loss,
    opcycle_loss,
)

D_MODEL = 16
D_Z = 8
D_HIDDEN = 12
K = 4
BATCH = 3
T = 5


def _mask() -> torch.Tensor:
    # rows of varying length: 5, 3, 1 active tokens
    m = torch.zeros(BATCH, T, dtype=torch.long)
    m[0, :5] = 1
    m[1, :3] = 1
    m[2, :1] = 1
    return m


def _rand(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


# ---------------------------------------------------------------------------
# masked_recon_loss — padded tokens must not affect the loss
# ---------------------------------------------------------------------------


def test_masked_recon_loss_ignores_padding():
    h = _rand(BATCH, T, D_MODEL, seed=1)
    recon = _rand(BATCH, T, D_MODEL, seed=2)
    mask = _mask()
    base = masked_recon_loss(h, recon, mask)
    # perturb only padded positions → loss unchanged
    recon2 = recon.clone()
    recon2[mask == 0] += 99.0
    assert torch.allclose(base, masked_recon_loss(h, recon2, mask))


def test_masked_recon_loss_zero_when_identical():
    h = _rand(BATCH, T, D_MODEL, seed=3)
    mask = _mask()
    loss = masked_recon_loss(h, h.clone(), mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# remoe_l1_loss — masked, grows with larger gates
# ---------------------------------------------------------------------------


def test_remoe_l1_increases_with_gate_magnitude():
    mask = _mask()
    small = torch.full((BATCH, T, K), 0.1)
    large = torch.full((BATCH, T, K), 1.0)
    assert remoe_l1_loss(large, mask) > remoe_l1_loss(small, mask)


def test_remoe_l1_ignores_padding():
    mask = _mask()
    alpha = torch.rand(BATCH, T, K).abs()
    base = remoe_l1_loss(alpha, mask)
    alpha2 = alpha.clone()
    alpha2[mask == 0] += 50.0
    assert torch.allclose(base, remoe_l1_loss(alpha2, mask))


# ---------------------------------------------------------------------------
# router_z_loss / load_balance_loss — finite, masked, non-negative
# ---------------------------------------------------------------------------


def test_router_z_loss_finite_and_nonneg():
    logits = _rand(BATCH, T, K, seed=4)
    z = router_z_loss(logits, _mask())
    assert torch.isfinite(z) and z.item() >= 0.0


def test_load_balance_loss_finite_and_nonneg():
    alpha = _rand(BATCH, T, K, seed=5).relu()          # gates are ReLU output, ≥ 0
    lb = load_balance_loss(alpha, _mask())
    assert torch.isfinite(lb) and lb.item() >= 0.0


def test_load_balance_penalizes_collapse_over_uniform():
    # The whole point of the ReMoE-gate fix: a router that sends every token to ONE expert
    # must score a HIGHER load-balance loss than one that spreads uniformly. A softmax-based
    # loss would miss this because dead experts still get softmax mass.
    mask = _mask()
    uniform = torch.full((BATCH, T, K), 0.5)           # all experts equally active
    collapsed = torch.zeros(BATCH, T, K)
    collapsed[..., 0] = 1.0                            # only expert 0 ever fires
    assert load_balance_loss(collapsed, mask) > load_balance_loss(uniform, mask)


def test_update_l1_lambda_controller():
    from phase1.model_opcycle import update_l1_lambda
    # too dense (K_active above target) → raise λ; too sparse → lower λ
    assert update_l1_lambda(0.01, k_active_mean=8.0, k_target=4.0) > 0.01
    assert update_l1_lambda(0.01, k_active_mean=1.0, k_target=4.0) < 0.01
    # clamps to [lam_min, lam_max]
    assert update_l1_lambda(1e9, 8.0, 4.0, lam_max=1.0) == 1.0
    assert update_l1_lambda(1e-9, 1.0, 4.0, lam_min=1e-6) == 1e-6


# ---------------------------------------------------------------------------
# opcycle_loss — combined total + components, backward produces grads
# ---------------------------------------------------------------------------


def test_opcycle_loss_returns_total_and_components():
    model = OpCycleMoE(d_model=D_MODEL, d_z=D_Z, k=K, d_hidden=D_HIDDEN)
    h = _rand(BATCH, T, D_MODEL, seed=6)
    mask = _mask()
    out = model(h, mask)
    total, parts = opcycle_loss(out, h, mask)
    assert torch.isfinite(total)
    for key in ("recon", "l1", "z", "lb"):
        assert key in parts and torch.isfinite(torch.as_tensor(parts[key]))


def test_opcycle_loss_backward_grads_all_trainable():
    model = OpCycleMoE(d_model=D_MODEL, d_z=D_Z, k=K, d_hidden=D_HIDDEN)
    h = _rand(BATCH, T, D_MODEL, seed=7)
    mask = _mask()
    out = model(h, mask)
    total, _ = opcycle_loss(out, h, mask)
    total.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_opcycle_loss_total_is_weighted_sum():
    # Guards against a loss term being silently dropped from the total: total must equal
    # the explicit λ-weighted sum of the four component terms.
    model = OpCycleMoE(d_model=D_MODEL, d_z=D_Z, k=K, d_hidden=D_HIDDEN)
    h = _rand(BATCH, T, D_MODEL, seed=8)
    mask = _mask()
    out = model(h, mask)
    lr, ll1, lz, llb = 1.0, 1e-2, 1e-3, 1e-2
    total, parts = opcycle_loss(
        out, h, mask, lambda_recon=lr, lambda_l1=ll1, lambda_z=lz, lambda_lb=llb
    )
    expected = lr * parts["recon"] + ll1 * parts["l1"] + lz * parts["z"] + llb * parts["lb"]
    assert total.item() == pytest.approx(expected, rel=1e-5)
