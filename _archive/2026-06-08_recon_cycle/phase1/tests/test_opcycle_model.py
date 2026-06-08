"""Tests for the per-token operation-cycle model (ENGINE_A_DESIGN §2).

decode-side experts (SMoE-VAE template): frozen BGE → per-token h (B,T,1024) →
SharedEncoderHead → z (B,T,d_z) → ReMoERouter (ReLU gate, exact-zero, adaptive K) →
Σ_k α_k · DecoderExpert_k(z) = recon ĥ (B,T,1024).

All tiny-dim CPU (no encoder, no network). The frozen BGE front-end is exercised
separately (test_token_encode.py); here we test only the trainable cycle.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from phase1.model_opcycle import (
    SharedEncoderHead,
    ReMoERouter,
    DecoderExpert,
    OpCycleMoE,
)


# Tiny test dims (no encoder load, fast CPU)
D_MODEL = 16
D_Z = 8
D_HIDDEN = 12
K = 4
BATCH = 3
T = 5


def _hidden_batch(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH, T, D_MODEL, generator=g)


# ---------------------------------------------------------------------------
# SharedEncoderHead — per-token 1024 → d_z
# ---------------------------------------------------------------------------


def test_shared_encoder_head_maps_to_d_z():
    head = SharedEncoderHead(d_model=D_MODEL, d_z=D_Z)
    z = head(_hidden_batch())
    assert z.shape == (BATCH, T, D_Z)


# ---------------------------------------------------------------------------
# ReMoERouter — ReLU gate, exact-zero sparsity, adaptive K, exposes logits
# ---------------------------------------------------------------------------


def test_remoe_router_returns_alpha_and_logits_shapes():
    router = ReMoERouter(d_z=D_Z, k=K)
    g = torch.Generator().manual_seed(1)
    z = torch.randn(BATCH, T, D_Z, generator=g)
    alpha, logits = router(z)
    assert alpha.shape == (BATCH, T, K)
    assert logits.shape == (BATCH, T, K)


def test_remoe_router_gates_are_nonneg_with_exact_zeros():
    # ReLU gate ⇒ all gates ≥ 0 and at least some are *exactly* zero (sparsity).
    router = ReMoERouter(d_z=D_Z, k=K)
    g = torch.Generator().manual_seed(2)
    z = torch.randn(BATCH, T, D_Z, generator=g)
    alpha, _ = router(z)
    assert (alpha >= 0).all()
    assert (alpha == 0).any()  # exact zeros, not just small


def test_remoe_router_k_active_counts_nonzero_gates():
    router = ReMoERouter(d_z=D_Z, k=K)
    g = torch.Generator().manual_seed(3)
    z = torch.randn(BATCH, T, D_Z, generator=g)
    alpha, _ = router(z)
    k_active = (alpha > 0).sum(dim=-1)
    assert k_active.shape == (BATCH, T)
    assert (k_active <= K).all()


# ---------------------------------------------------------------------------
# DecoderExpert — generative operator z_t → ĥ_t (d_model)
# ---------------------------------------------------------------------------


def test_decoder_expert_maps_z_to_d_model():
    expert = DecoderExpert(d_z=D_Z, d_hidden=D_HIDDEN, d_model=D_MODEL)
    g = torch.Generator().manual_seed(4)
    z = torch.randn(BATCH, T, D_Z, generator=g)
    out = expert(z)
    assert out.shape == (BATCH, T, D_MODEL)


# ---------------------------------------------------------------------------
# OpCycleMoE — full forward: dict of {recon, alpha, z, logits, k_active}
# ---------------------------------------------------------------------------


def _model() -> OpCycleMoE:
    return OpCycleMoE(d_model=D_MODEL, d_z=D_Z, k=K, d_hidden=D_HIDDEN)


def test_opcycle_forward_returns_expected_shapes():
    model = _model()
    out = model(_hidden_batch())
    assert out["recon"].shape == (BATCH, T, D_MODEL)
    assert out["alpha"].shape == (BATCH, T, K)
    assert out["z"].shape == (BATCH, T, D_Z)
    assert out["logits"].shape == (BATCH, T, K)
    assert out["k_active"].shape == (BATCH, T)


def test_opcycle_recon_is_mixture_of_experts():
    # With all gates forced to zero, recon must be exactly zero (Σ 0·expert = 0).
    model = _model()
    h = _hidden_batch()
    # monkeypatch the router to emit zero gates
    orig = model.router.forward

    def zero_gates(z):
        alpha, logits = orig(z)
        return torch.zeros_like(alpha), logits

    model.router.forward = zero_gates
    out = model(h)
    assert torch.allclose(out["recon"], torch.zeros_like(out["recon"]))


def test_opcycle_recon_matches_manual_mixture():
    # Real coverage of the mixture formula ĥ = Σ_k α_k · expert_k(z) (not just the
    # all-zero-gate identity): recompute it by hand from z and compare.
    model = _model()
    h = _hidden_batch()
    out = model(h)
    z = out["z"]
    alpha = out["alpha"]
    manual = sum(alpha[..., k : k + 1] * model.experts[k](z) for k in range(K))
    assert torch.allclose(out["recon"], manual, atol=1e-5)


def test_opcycle_grad_flows_to_trainable_params():
    model = _model()
    out = model(_hidden_batch())
    out["recon"].sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


# ---------------------------------------------------------------------------
# route_on_deviation (F3b) — route on within-sequence local structure, not global content
# ---------------------------------------------------------------------------


def test_route_on_deviation_doubles_router_input_and_keeps_output_shapes():
    model = OpCycleMoE(d_model=D_MODEL, d_z=D_Z, k=K, d_hidden=D_HIDDEN,
                       route_on_deviation=True)
    # router consumes [z ‖ (z − seq-mean z)] → gate in-features = 2·d_z
    assert model.router.gate.in_features == 2 * D_Z
    out = model(_hidden_batch())
    assert out["recon"].shape == (BATCH, T, D_MODEL)
    assert out["alpha"].shape == (BATCH, T, K)
    assert out["z"].shape == (BATCH, T, D_Z)  # latent z stays d_z (deviation is router-only)


def test_route_on_deviation_makes_routing_depend_on_sequence_context():
    # Same token at position 0, two sequences with different remaining tokens (→ different
    # sequence means). Deviation routing → position-0 gates differ across the two sequences;
    # default routing (z_t only) → identical position-0 gates.
    g = torch.Generator().manual_seed(7)
    shared = torch.randn(1, D_MODEL, generator=g)
    seq_a = torch.cat([shared, torch.randn(T - 1, D_MODEL, generator=g)], dim=0)
    seq_b = torch.cat([shared, torch.randn(T - 1, D_MODEL, generator=g) + 5.0], dim=0)
    batch = torch.stack([seq_a, seq_b], dim=0)  # (2, T, D_MODEL), position 0 identical

    off = OpCycleMoE(d_model=D_MODEL, d_z=D_Z, k=K, d_hidden=D_HIDDEN, route_on_deviation=False)
    a_off = off(batch)["alpha"]
    assert torch.allclose(a_off[0, 0], a_off[1, 0], atol=1e-6)  # context-independent

    on = OpCycleMoE(d_model=D_MODEL, d_z=D_Z, k=K, d_hidden=D_HIDDEN, route_on_deviation=True)
    a_on = on(batch)["alpha"]
    assert not torch.allclose(a_on[0, 0], a_on[1, 0], atol=1e-4)  # context-dependent


def test_route_on_deviation_uses_mask_for_sequence_mean():
    # Padded tokens must not pollute the sequence mean used for the deviation.
    model = OpCycleMoE(d_model=D_MODEL, d_z=D_Z, k=K, d_hidden=D_HIDDEN, route_on_deviation=True)
    h = _hidden_batch()
    mask = torch.ones(BATCH, T, dtype=torch.long)
    mask[:, T - 1] = 0
    # Changing the value of the PADDED token must not change any active-token routing.
    h2 = h.clone()
    h2[:, T - 1] = h2[:, T - 1] + 100.0
    a1 = model(h, mask)["alpha"][:, : T - 1]
    a2 = model(h2, mask)["alpha"][:, : T - 1]
    assert torch.allclose(a1, a2, atol=1e-5)
