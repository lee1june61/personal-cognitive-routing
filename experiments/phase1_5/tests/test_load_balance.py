"""Tests for `phase1_5.load_balance` — Aux-loss-free LB (DeepSeek-V3).

CPU, tiny dims. Verifies the sign-rule update direction, factory dispatch,
and integration shape with the router's ``external_bias`` path.
"""

from __future__ import annotations

import pytest
import torch

from experiments.phase1_5.load_balance import (
    LB_AUX_FREE,
    LB_OFF,
    AuxLossFreeLB,
    make_lb,
)
from experiments.phase1_5.model import Phase15MoE, ReMoERouter


# ---- AuxLossFreeLB.step --------------------------------------------------------


def test_aux_lb_bias_init_zeros():
    lb = AuxLossFreeLB(k=8, target=0.25)
    assert lb.bias.shape == (8,)
    assert torch.all(lb.bias == 0)


def test_aux_lb_step_all_dead_grows_bias_uniformly():
    """All-dead alpha (every expert f_k = 0): sign(f_k - target) = -1 → bias += lr."""
    lb = AuxLossFreeLB(k=4, target=0.25, lr_bias=1e-2)
    alpha = torch.zeros(2, 3, 4)
    mask = torch.ones(2, 3)
    lb.step(alpha, mask)
    expected = torch.full((4,), 1e-2)
    assert torch.allclose(lb.bias, expected)


def test_aux_lb_step_all_active_shrinks_bias_uniformly():
    """All-active alpha (f_k = 1 > target): sign = +1 → bias -= lr."""
    lb = AuxLossFreeLB(k=4, target=0.25, lr_bias=1e-2)
    alpha = torch.ones(2, 3, 4)
    mask = torch.ones(2, 3)
    lb.step(alpha, mask)
    expected = torch.full((4,), -1e-2)
    assert torch.allclose(lb.bias, expected)


def test_aux_lb_step_at_target_no_drift():
    """f_k == target exactly → residual = 0 → sign(0) = 0 → no bias drift.

    Half of token positions activate each expert → f_k = 0.5 for all = target.
    """
    lb = AuxLossFreeLB(k=4, target=0.5, lr_bias=1e-2)
    alpha = torch.zeros(2, 4, 4)
    # First half of tokens activate every expert; second half activate none.
    alpha[:, :2, :] = 1.0
    mask = torch.ones(2, 4)
    lb.step(alpha, mask)
    assert torch.allclose(lb.bias, torch.zeros(4))


def test_aux_lb_step_empty_mask_is_noop():
    """An all-padded batch (mask.sum() == 0) must not drift the bias.

    Otherwise f_k = 0/clamp(0)=0 → residual = -target < 0 → bias drifts up
    uniformly, interpreting 'no data' as 'all experts dead'.
    """
    lb = AuxLossFreeLB(k=4, target=0.25, lr_bias=1e-2)
    alpha = torch.zeros(2, 3, 4)
    mask = torch.zeros(2, 3)
    lb.step(alpha, mask)
    assert torch.allclose(lb.bias, torch.zeros(4))


def test_aux_lb_step_respects_mask():
    """Padded positions are excluded from the f_k denominator."""
    lb = AuxLossFreeLB(k=2, target=0.5, lr_bias=1e-2)
    alpha = torch.zeros(1, 4, 2)
    alpha[0, :2, 0] = 1.0  # expert 0 active on tokens 0..1 only
    # Mask out tokens 2,3 — those zero-alpha tokens must not count toward f_0.
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    lb.step(alpha, mask)
    # f_0 = 2/2 = 1.0 > target → bias[0] -= lr; f_1 = 0/2 = 0 < target → bias[1] += lr.
    assert torch.allclose(lb.bias[0], torch.tensor(-1e-2))
    assert torch.allclose(lb.bias[1], torch.tensor(1e-2))


# ---- make_lb factory -----------------------------------------------------------


def test_make_lb_off_returns_none():
    assert make_lb(LB_OFF, k_routed=8, k_active_target=4.0) is None


def test_make_lb_aux_free_target_ratio():
    lb = make_lb(LB_AUX_FREE, k_routed=128, k_active_target=4.0)
    assert isinstance(lb, AuxLossFreeLB)
    assert lb.k == 128
    assert lb.target == pytest.approx(4.0 / 128)


def test_make_lb_unknown_raises():
    with pytest.raises(NotImplementedError):
        make_lb("aux_w_001", k_routed=8, k_active_target=2.0)


# ---- Router external_bias integration ------------------------------------------


def test_router_external_bias_shifts_logits():
    """external_bias added pre-ReLU: alpha = ReLU(gate(z) + bias)."""
    router = ReMoERouter(d_z=8, k=4, bias_init=0.0)
    with torch.no_grad():
        router.gate.weight.zero_()
        router.gate.bias.fill_(-1.0)
    z = torch.randn(2, 3, 8)
    alpha_off, logits_off = router(z)
    # With bias=-1 and weights=0, all logits = -1 → all alpha = 0.
    assert torch.all(alpha_off == 0)
    # External bias of +2 lifts every logit to +1 → all alpha = 1.
    ext = torch.full((4,), 2.0)
    alpha_on, logits_on = router(z, external_bias=ext)
    assert torch.allclose(logits_on, torch.ones_like(logits_on))
    assert torch.allclose(alpha_on, torch.ones_like(alpha_on))


# ---- Phase15MoE integration ----------------------------------------------------


def test_phase15_lb_off_no_lb_module():
    model = Phase15MoE(d_emb=16, d_z=8, k_routed=4, lb_strategy=LB_OFF)
    assert model.lb is None


def test_phase15_lb_aux_free_has_module():
    model = Phase15MoE(
        d_emb=16, d_z=8, k_routed=4, lb_strategy=LB_AUX_FREE, lb_target_active=1.0
    )
    assert isinstance(model.lb, AuxLossFreeLB)
    assert model.lb.target == pytest.approx(0.25)
    # bias buffer registered, on the same device as model parameters
    assert "lb.bias" in dict(model.named_buffers())


def test_phase15_forward_with_lb_smoke():
    model = Phase15MoE(
        d_emb=16,
        d_z=8,
        k_routed=4,
        d_hidden_expert=16,
        lb_strategy=LB_AUX_FREE,
        lb_target_active=1.0,
    )
    batch = {
        "q_tokens": torch.randn(2, 5, 16),
        "q_mask": torch.ones(2, 5),
        "p_tokens": torch.randn(2, 7, 16),
        "p_mask": torch.ones(2, 7),
        "cand_pooled": torch.randn(2, 4, 16),
    }
    out = model(batch)
    assert out["logits"].shape == (2, 4)
    assert out["alpha"].shape == (2, 5, 4)
