"""End-to-end Engine-A runner test (ENGINE_A_DESIGN §5, runner = train→probe→verdict).

The runner takes *pre-computed* per-token embeddings (the frozen-BGE encode is the
boundary, covered elsewhere) so the whole go/no-go pipeline runs on tiny synthetic
tensors on CPU: train the operation-cycle a few steps, extract sequence codes on the
probe set, and emit a selectivity verdict. We assert the pipeline's *shape and contract*
— finite training, correct code shape, a {PASS,FAIL} verdict — not that a 5-step toy
model passes the gate (that is the real Colab experiment's job).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phase1.engine_a import train_opcycle, compute_codes, run_engine_a
from phase1.model_opcycle import OpCycleMoE, opcycle_loss
from phase1.eval_opcycle import sequence_code

D_MODEL = 16
D_Z = 8
D_HIDDEN = 12
K = 4
N = 30
T = 6


def _tokens(n=N, seed=0):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randn(n, T, D_MODEL, generator=g)
    mask = torch.ones(n, T, dtype=torch.long)
    mask[:, T - 1] = 0  # last token padded everywhere
    return tokens, mask


def _model(seed=0):
    torch.manual_seed(seed)
    return OpCycleMoE(d_model=D_MODEL, d_z=D_Z, k=K, d_hidden=D_HIDDEN)


# ---------------------------------------------------------------------------
# train_opcycle — finite history, parameters actually move
# ---------------------------------------------------------------------------


def test_train_opcycle_runs_and_updates_params():
    model = _model()
    tokens, mask = _tokens()
    before = [p.detach().clone() for p in model.parameters()]
    history = train_opcycle(model, tokens, mask, epochs=5, batch_size=10, lr=1e-2)
    assert len(history) == 5
    assert all(np.isfinite(h["total"]) and np.isfinite(h["recon"]) for h in history)
    after = list(model.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


# ---------------------------------------------------------------------------
# compute_codes — (N, K) sequence codes, consistent with sequence_code
# ---------------------------------------------------------------------------


def test_compute_codes_matches_sequence_code():
    model = _model()
    tokens, mask = _tokens()
    codes = compute_codes(model, tokens, mask, batch_size=10)
    assert codes.shape == (N, K)
    # recompute reference in one shot and compare
    with torch.no_grad():
        ref = sequence_code(model(tokens, mask)["alpha"], mask)
    np.testing.assert_allclose(codes, ref, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# run_engine_a — full pipeline returns history + codes + verdict
# ---------------------------------------------------------------------------


def test_run_engine_a_end_to_end_returns_verdict():
    train_tokens, train_mask = _tokens(seed=1)
    probe_tokens, probe_mask = _tokens(n=24, seed=2)
    rng = np.random.default_rng(0)
    operation_labels = rng.integers(0, 3, size=24)
    topic = rng.integers(0, 3, size=24)

    result = run_engine_a(
        train_tokens, train_mask,
        probe_tokens, probe_mask, operation_labels,
        control_label_sets={"topic": topic},
        d_z=D_Z, k=K, d_hidden=D_HIDDEN,
        epochs=3, batch_size=10, lr=1e-2, seed=0,
    )
    assert len(result["history"]) == 3
    assert result["codes"].shape == (24, K)
    assert result["report"]["verdict"] in ("PASS", "FAIL")
    assert "acc_operation" in result["report"]


def test_run_engine_a_empty_probe_raises_clear_error():
    train_tokens, train_mask = _tokens(seed=1)
    empty_tokens = torch.empty(0, T, D_MODEL)
    empty_mask = torch.empty(0, T, dtype=torch.long)
    with pytest.raises(ValueError, match="empty probe"):
        run_engine_a(
            train_tokens, train_mask,
            empty_tokens, empty_mask, np.array([], dtype=int),
            d_z=D_Z, k=K, d_hidden=D_HIDDEN, epochs=1, batch_size=10,
        )


def test_run_engine_a_empty_train_raises_clear_error():
    # Empty train set (e.g. Super-NI load returned an empty corpus) would otherwise skip
    # every batch loop → cryptic KeyError on epoch['recon'] (log_every/return_best/k_target).
    empty_tokens = torch.empty(0, T, D_MODEL)
    empty_mask = torch.empty(0, T, dtype=torch.long)
    probe_tokens, probe_mask = _tokens(n=24, seed=2)
    operation_labels = np.random.default_rng(0).integers(0, 3, size=24)
    with pytest.raises(ValueError, match="empty train"):
        run_engine_a(
            empty_tokens, empty_mask,
            probe_tokens, probe_mask, operation_labels,
            d_z=D_Z, k=K, d_hidden=D_HIDDEN, epochs=1, batch_size=10, log_every=1,
        )


def test_compute_codes_empty_probe_returns_shaped_empty():
    model = _model()
    empty_tokens = torch.empty(0, T, D_MODEL)
    empty_mask = torch.empty(0, T, dtype=torch.long)
    codes = compute_codes(model, empty_tokens, empty_mask, batch_size=10)
    assert codes.shape == (0, K)
    codes_mm = compute_codes(model, empty_tokens, empty_mask, batch_size=10, agg="meanmax")
    assert codes_mm.shape == (0, 2 * K)


def _recon_of(model, tokens, mask):
    with torch.no_grad():
        out = model(tokens, mask)
        _, parts = opcycle_loss(out, tokens, mask)
    return parts["recon"]


def test_train_opcycle_return_best_default_unchanged():
    # default (return_best=False) keeps the existing contract: a list of history dicts.
    model = _model()
    tokens, mask = _tokens()
    history = train_opcycle(model, tokens, mask, epochs=3, batch_size=10, lr=1e-2)
    assert isinstance(history, list) and len(history) == 3


def test_train_opcycle_return_best_gives_checkpoint_no_worse_than_final():
    # With return_best=True we get (history, best_state); the best-recon checkpoint must
    # reconstruct at least as well as the final weights (the whole point: avoid an
    # over-sparsified final epoch with degraded recon).
    model = _model()
    tokens, mask = _tokens()
    history, best_state = train_opcycle(
        model, tokens, mask, epochs=8, batch_size=10, lr=1e-2,
        k_target=0.5, return_best=True,   # aggressive sparsity → recon likely degrades late
    )
    assert isinstance(best_state, dict) and len(best_state) > 0
    final_recon = _recon_of(model, tokens, mask)
    best_model = _model()
    best_model.load_state_dict(best_state)
    best_recon = _recon_of(best_model, tokens, mask)
    assert best_recon <= final_recon + 1e-5


def test_run_engine_a_use_best_recon_returns_shaped_codes_and_verdict():
    train_tokens, train_mask = _tokens(seed=1)
    probe_tokens, probe_mask = _tokens(n=24, seed=2)
    rng = np.random.default_rng(0)
    operation_labels = rng.integers(0, 3, size=24)
    result = run_engine_a(
        train_tokens, train_mask,
        probe_tokens, probe_mask, operation_labels,
        control_label_sets={"topic": rng.integers(0, 3, size=24)},
        d_z=D_Z, k=K, d_hidden=D_HIDDEN,
        epochs=5, batch_size=10, lr=1e-2, seed=0,
        k_target=0.5, use_best_recon=True,
    )
    assert result["codes"].shape == (24, K)
    assert result["report"]["verdict"] in ("PASS", "FAIL")


def test_run_engine_a_forwards_route_on_deviation_to_model():
    train_tokens, train_mask = _tokens(seed=1)
    probe_tokens, probe_mask = _tokens(n=24, seed=2)
    rng = np.random.default_rng(0)
    result = run_engine_a(
        train_tokens, train_mask,
        probe_tokens, probe_mask, rng.integers(0, 3, size=24),
        control_label_sets={"topic": rng.integers(0, 3, size=24)},
        d_z=D_Z, k=K, d_hidden=D_HIDDEN,
        epochs=2, batch_size=10, lr=1e-2, seed=0,
        route_on_deviation=True,
    )
    assert result["model"].route_on_deviation is True
    assert result["model"].router.gate.in_features == 2 * D_Z
    assert result["codes"].shape == (24, K)


def test_train_opcycle_adaptive_l1_raises_lambda_when_too_dense():
    # k_target far below the realised K_active → controller must push λ_l1 UP every epoch.
    model = _model()
    tokens, mask = _tokens()
    history = train_opcycle(model, tokens, mask, epochs=4, batch_size=10, lr=1e-2, k_target=0.0)
    lams = [h["lam_l1"] for h in history]
    assert lams[-1] > lams[0]
    assert all(b >= a for a, b in zip(lams, lams[1:]))  # monotonically non-decreasing
