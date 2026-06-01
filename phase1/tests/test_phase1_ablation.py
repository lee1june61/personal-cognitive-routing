"""Phase 1 / Phase 2 ablation unit tests (revision 4, embedding-only).

Verifies the `use_user: bool` flag isolation in `phase1.model.Phase1MoE`:
  - use_user=False → fact_gate only + global_lambda_log scalar (no user_logits, no lambda_mlp)
  - use_user=True  → fact_gate + user_logits, per-user λ_u via lambda_mlp
  - Phase 1 → Phase 2 weight transfer via Phase1Cycle.load_phase1_weights

Plus FactDecoder + cycle loss + sparsegen shape sanity. All on tiny dims (no encoder
download) — real encoder integration check is in slow tests.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from phase1.model import (
    Phase1MoE,
    FactDecoder,
    sparsegen,
    routing_load_balance,
    routing_orthogonality,
)


# Tiny test dims (no encoder load, fast CPU)
D_MODEL = 32
K_ROUTED = 4
K_SHARED = 2
N_USERS = 16
BATCH = 8


def _fact_batch(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH, D_MODEL, generator=g)


def _user_batch(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed + 1)
    return torch.randint(0, N_USERS, (BATCH,), generator=g)


# ---------------------------------------------------------------------------
# Phase1MoE — use_user ablation
# ---------------------------------------------------------------------------


def test_phase1_moe_has_no_user_params():
    moe = Phase1MoE(
        n_users=N_USERS, d_model=D_MODEL, k_routed=K_ROUTED, k_shared=K_SHARED,
        d_hidden=64, use_user=False,
    )
    assert moe.use_user is False
    assert not hasattr(moe, "user_logits")
    assert not hasattr(moe, "lambda_mlp")
    assert hasattr(moe, "global_lambda_log")
    assert moe.global_lambda_log.shape == (1, 1)


def test_phase2_moe_has_user_params():
    moe = Phase1MoE(
        n_users=N_USERS, d_model=D_MODEL, k_routed=K_ROUTED, k_shared=K_SHARED,
        d_hidden=64, use_user=True,
    )
    assert moe.use_user is True
    assert moe.user_logits.weight.shape == (N_USERS, K_ROUTED)
    assert moe.lambda_mlp.in_features == K_ROUTED
    assert not hasattr(moe, "global_lambda_log")


def test_phase1_forward_shapes_and_user_id_none_ok():
    moe = Phase1MoE(
        n_users=N_USERS, d_model=D_MODEL, k_routed=K_ROUTED, k_shared=K_SHARED,
        d_hidden=64, use_user=False,
    )
    out = moe(_fact_batch(), user_id=None)
    assert out["sub_kg"].shape == (BATCH, D_MODEL)
    assert out["routed_alpha"].shape == (BATCH, K_ROUTED)
    assert out["lam"].shape == (BATCH, 1)
    assert torch.allclose(out["routed_alpha"].sum(dim=-1), torch.ones(BATCH), atol=1e-5)


def test_phase1_forward_ignores_user_id():
    moe = Phase1MoE(
        n_users=N_USERS, d_model=D_MODEL, k_routed=K_ROUTED, k_shared=K_SHARED,
        d_hidden=64, use_user=False,
    )
    moe.eval()
    fact = _fact_batch()
    a = moe(fact, user_id=torch.zeros(BATCH, dtype=torch.long))["sub_kg"]
    b = moe(fact, user_id=torch.full((BATCH,), N_USERS - 1, dtype=torch.long))["sub_kg"]
    # Phase 1 = user-invariant routing → identical output regardless of user_id
    assert torch.allclose(a, b)


def test_phase2_forward_requires_user_id():
    moe = Phase1MoE(
        n_users=N_USERS, d_model=D_MODEL, k_routed=K_ROUTED, k_shared=K_SHARED,
        d_hidden=64, use_user=True,
    )
    with pytest.raises(ValueError, match="requires user_id"):
        moe(_fact_batch(), user_id=None)


def test_phase1_backward_reaches_global_lambda_and_experts():
    moe = Phase1MoE(
        n_users=N_USERS, d_model=D_MODEL, k_routed=K_ROUTED, k_shared=K_SHARED,
        d_hidden=64, use_user=False,
    )
    out = moe(_fact_batch(), user_id=None)
    out["sub_kg"].sum().backward()

    assert moe.global_lambda_log.grad is not None
    assert torch.isfinite(moe.global_lambda_log.grad).all()
    assert moe.fact_gate.weight.grad is not None
    expert_grads = [e.fc1.weight.grad for e in moe.routed_experts]
    assert any(g is not None and g.abs().sum() > 0 for g in expert_grads)


def test_phase1_gu_table_is_none():
    moe = Phase1MoE(
        n_users=N_USERS, d_model=D_MODEL, k_routed=K_ROUTED, k_shared=K_SHARED,
        d_hidden=64, use_user=False,
    )
    assert moe.gu_table() is None


def test_phase2_gu_table_shape():
    moe = Phase1MoE(
        n_users=N_USERS, d_model=D_MODEL, k_routed=K_ROUTED, k_shared=K_SHARED,
        d_hidden=64, use_user=True,
    )
    gu = moe.gu_table()
    assert gu is not None
    assert gu.shape == (N_USERS, K_ROUTED)


# ---------------------------------------------------------------------------
# FactDecoder + auxiliary losses
# ---------------------------------------------------------------------------


def test_fact_decoder_shape_and_bottleneck():
    decoder = FactDecoder(d_model=D_MODEL, d_bottleneck=4)
    x = torch.randn(BATCH, D_MODEL)
    out = decoder(x)
    assert out.shape == (BATCH, D_MODEL)
    # Bottleneck enforced — first linear maps to d_bottleneck=4
    assert decoder.net[0].out_features == 4


def test_routing_load_balance_finite():
    ra = torch.rand(BATCH, K_ROUTED)
    ra = ra / ra.sum(dim=-1, keepdim=True)
    loss = routing_load_balance(ra, lambda_lb=0.1)
    assert torch.isfinite(loss)
    assert loss >= 0


def test_routing_orthogonality_off_returns_zero():
    ra = torch.rand(BATCH, K_ROUTED)
    loss = routing_orthogonality(ra, lambda_ortho=0.0)
    assert float(loss) == 0.0


def test_routing_orthogonality_on_positive():
    ra = torch.rand(BATCH, K_ROUTED)
    loss = routing_orthogonality(ra, lambda_ortho=0.1)
    assert torch.isfinite(loss)
    assert loss >= 0


def test_sparsegen_simplex_property():
    """Output must be on probability simplex (sum to 1)."""
    u = torch.randn(BATCH, K_ROUTED)
    lam = torch.full((BATCH, 1), 0.5)
    p = sparsegen(u, lam)
    assert torch.allclose(p.sum(dim=-1), torch.ones(BATCH), atol=1e-5)
    assert (p >= 0).all()


# ---------------------------------------------------------------------------
# Phase1Cycle wrapper + load_phase1_weights helper
# ---------------------------------------------------------------------------


def _make_phase1_cycle(use_user: bool):
    """Construct a real Phase1Cycle without loading the BGE encoder.

    Production `Phase1Cycle.__init__` calls `FrozenEncoder(encoder_name)` which downloads
    ~334 MB on first run. Tests bypass this by allocating the module via __new__ and
    wiring just the MoE + decoder. The `load_phase1_weights` helper under test is the
    real method, so a bug there is caught.
    """
    from phase1.cycle import Phase1Cycle, CycleConfig

    obj = Phase1Cycle.__new__(Phase1Cycle)
    torch.nn.Module.__init__(obj)
    obj.config = CycleConfig(d_bottleneck=8)
    obj.use_user = use_user
    obj.encoder = torch.nn.Identity()
    obj.encoder.d_model = D_MODEL
    obj.moe = Phase1MoE(
        n_users=N_USERS, d_model=D_MODEL, k_routed=K_ROUTED, k_shared=K_SHARED,
        d_hidden=64, use_user=use_user,
    )
    obj.decoder = FactDecoder(d_model=D_MODEL, d_bottleneck=8)
    return obj


def test_load_phase1_into_phase2_accepts_expected_key_delta():
    phase1 = _make_phase1_cycle(use_user=False)
    phase2 = _make_phase1_cycle(use_user=True)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "phase1.pt"
        torch.save(phase1.state_dict(), ckpt)
        report = phase2.load_phase1_weights(str(ckpt))
    assert set(report["missing"]) == {
        "moe.user_logits.weight",
        "moe.lambda_mlp.weight",
        "moe.lambda_mlp.bias",
    }
    assert set(report["unexpected"]) == {"moe.global_lambda_log"}
    p1_w = phase1.moe.routed_experts[0].fc1.weight
    p2_w = phase2.moe.routed_experts[0].fc1.weight
    assert torch.allclose(p1_w, p2_w)


def test_load_phase1_into_phase1_rejected():
    p1_target = _make_phase1_cycle(use_user=False)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "phase1.pt"
        torch.save(p1_target.state_dict(), ckpt)
        with pytest.raises(RuntimeError, match="Phase 2"):
            p1_target.load_phase1_weights(str(ckpt))
