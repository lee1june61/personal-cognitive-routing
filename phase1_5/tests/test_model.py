"""Tests for `phase1_5.model` — Phase15MoE + supporting blocks.

All CPU, tiny dims. K=128 instantiation test confirms the running-sum mixture
path scales to the paper's default K without materialising the K-stack.
"""

from __future__ import annotations

import pytest
import torch

from research.demo.phase1_5.model import (
    MOD_CROSS_ATTN,
    MOD_FILM,
    MCHead,
    OperationExpert,
    Phase15MoE,
    ReMoERouter,
    SharedEncoderHead,
)


# ---- shared encoder head -------------------------------------------------------


def test_shared_encoder_head_shape():
    head = SharedEncoderHead(d_emb=32, d_z=16)
    h = torch.randn(2, 5, 32)
    z = head(h)
    assert z.shape == (2, 5, 16)


def test_shared_encoder_head_backprop():
    head = SharedEncoderHead(d_emb=32, d_z=16)
    h = torch.randn(2, 5, 32, requires_grad=True)
    head(h).sum().backward()
    assert h.grad is not None


# ---- ReMoE router --------------------------------------------------------------


def test_remoe_router_alpha_exact_zeros_when_logits_negative():
    router = ReMoERouter(d_z=8, k=4)
    # Force gate to have a very negative bias so alpha ~ 0 always.
    with torch.no_grad():
        router.gate.bias.fill_(-100.0)
    z = torch.randn(3, 6, 8)
    alpha, logits = router(z)
    assert torch.all(alpha == 0)


def test_remoe_router_alpha_non_negative():
    router = ReMoERouter(d_z=8, k=4)
    z = torch.randn(3, 6, 8)
    alpha, _ = router(z)
    assert torch.all(alpha >= 0)


def test_remoe_router_logits_match_gate_linear():
    router = ReMoERouter(d_z=8, k=4)
    z = torch.randn(2, 3, 8)
    alpha, logits = router(z)
    expected = router.gate(z)
    assert torch.allclose(logits, expected, atol=1e-6)


# ---- operation expert ----------------------------------------------------------


def test_operation_expert_output_dim_is_d_z():
    expert = OperationExpert(d_z=16, d_hidden=32)
    z = torch.randn(2, 4, 16)
    out = expert(z)
    assert out.shape == (2, 4, 16)  # output stays in d_z (not d_emb)


# ---- MC head -------------------------------------------------------------------


def test_mc_head_logits_shape():
    mc = MCHead(d_z=16, d_emb=32)
    kg = torch.randn(3, 16)
    cands = torch.randn(3, 4, 32)
    logits = mc(kg, cands)
    assert logits.shape == (3, 4)


def test_mc_head_logits_match_manual_dot():
    mc = MCHead(d_z=8, d_emb=8)
    # Identity-like projection
    with torch.no_grad():
        mc.cand_proj.weight.copy_(torch.eye(8))
    kg = torch.randn(2, 8)
    cands = torch.randn(2, 4, 8)
    out = mc(kg, cands)
    expected = (kg.unsqueeze(1) * cands).sum(dim=-1) / (8**0.5)
    assert torch.allclose(out, expected, atol=1e-5)


# ---- Phase15MoE end-to-end -----------------------------------------------------


def _tiny_batch(b=2, t_q=4, t_p=6, d_emb=32, n_cand=4):
    return {
        "q_tokens": torch.randn(b, t_q, d_emb),
        "q_mask": torch.ones(b, t_q),
        "p_tokens": torch.randn(b, t_p, d_emb),
        "p_mask": torch.ones(b, t_p),
        "cand_pooled": torch.randn(b, n_cand, d_emb),
    }


def test_phase15_moe_forward_returns_required_keys():
    model = Phase15MoE(d_emb=32, d_z=16, k_routed=4, d_hidden_expert=24)
    out = model(_tiny_batch())
    for k in (
        "logits",
        "alpha",
        "z_q",
        "router_logits",
        "kg_hidden",
        "kg_modulated",
        "kg_summary",
        "k_active",
    ):
        assert k in out


def test_phase15_moe_logits_shape():
    model = Phase15MoE(d_emb=32, d_z=16, k_routed=4, d_hidden_expert=24)
    out = model(_tiny_batch(b=3))
    assert out["logits"].shape == (3, 4)


# ---- 1b chain-of-experts forward_chain -------------------------------------------


def test_chain_L1_reduces_to_flat():
    """chain_steps=1 forward_chain must equal the flat forward (step-0 modulation
    IS self.modulation, same route + same expert mixture)."""
    torch.manual_seed(0)
    model = Phase15MoE(d_emb=32, d_z=16, k_routed=4, d_hidden_expert=24, chain_steps=1).eval()
    batch = _tiny_batch()
    flat = model(batch)
    chain = model.forward_chain(batch)
    assert torch.allclose(flat["logits"], chain["logits"], atol=1e-5)


def test_forward_chain_no_bypass():
    """All-steps alpha=0 ⇒ kg_summary=0 ⇒ uniform logits: P and Q cannot reach the
    answer except through a KG-parameterised transform (output excludes z_q)."""
    model = Phase15MoE(d_emb=32, d_z=16, k_routed=4, chain_steps=3).eval()
    with torch.no_grad():
        model.router.gate.bias.fill_(-100.0)  # ReLU kills every gate every step
    out = model.forward_chain(_tiny_batch(b=2))
    assert torch.allclose(out["kg_summary"], torch.zeros_like(out["kg_summary"]), atol=1e-6)
    logits = out["logits"]
    assert torch.allclose(logits, logits[:, :1].expand_as(logits), atol=1e-6)  # uniform


def test_forward_chain_step_outputs_length_L():
    model = Phase15MoE(d_emb=32, d_z=16, k_routed=8, chain_steps=3)
    out = model.forward_chain(_tiny_batch(b=2, t_q=5))
    assert len(out["alpha_steps"]) == 3
    assert len(out["kg_steps"]) == 3
    assert out["alpha_steps"][0].shape == (2, 5, 8)
    assert out["logits"].shape == (2, 4)


def test_forward_chain_alpha_override_steps_honored():
    """Per-step alpha override (intervention hook) replaces the routed alpha at
    each step; overriding all to 0 zeroes the output."""
    model = Phase15MoE(d_emb=32, d_z=16, k_routed=4, chain_steps=2).eval()
    zeros = [torch.zeros(2, 4, 4), torch.zeros(2, 4, 4)]
    out = model.forward_chain(_tiny_batch(b=2, t_q=4), alpha_override_steps=zeros)
    assert torch.all(out["alpha_steps"][0] == 0)
    assert torch.allclose(out["kg_summary"], torch.zeros_like(out["kg_summary"]), atol=1e-6)


def test_phase15_moe_alpha_shape_and_nonneg():
    model = Phase15MoE(d_emb=32, d_z=16, k_routed=8, d_hidden_expert=24)
    out = model(_tiny_batch(b=2, t_q=5))
    assert out["alpha"].shape == (2, 5, 8)
    assert torch.all(out["alpha"] >= 0)


def test_phase15_moe_kg_hidden_d_z_shape():
    model = Phase15MoE(d_emb=32, d_z=16, k_routed=4)
    out = model(_tiny_batch(t_q=5))
    assert out["kg_hidden"].shape == (2, 5, 16)


def test_phase15_moe_kg_summary_pooled_shape():
    model = Phase15MoE(d_emb=32, d_z=16, k_routed=4)
    out = model(_tiny_batch(b=2))
    assert out["kg_summary"].shape == (2, 16)


def test_phase15_moe_backward_flows_to_all_modules():
    model = Phase15MoE(d_emb=32, d_z=16, k_routed=4, d_hidden_expert=24)
    out = model(_tiny_batch())
    out["logits"].sum().backward()
    # Gradient must flow through encoder head, router, at least one expert, modulation,
    # MC head.
    assert model.encoder_head.net[0].weight.grad is not None
    assert model.router.gate.weight.grad is not None
    assert any(e.fc1.weight.grad is not None for e in model.experts)
    assert model.mc_head.cand_proj.weight.grad is not None


def test_phase15_moe_running_sum_matches_explicit_stack():
    """Running-sum mixture must equal the explicit (B, T_q, K, d_z) stack form."""
    model = Phase15MoE(d_emb=16, d_z=8, k_routed=4, d_hidden_expert=12).eval()
    batch = _tiny_batch(b=2, t_q=3, t_p=4, d_emb=16)
    with torch.no_grad():
        out = model(batch)
        # Recompute kg_hidden via explicit stack.
        z = model.encoder_head(batch["q_tokens"])
        alpha, _ = model.router(z)
        stacked = torch.stack([e(z) for e in model.experts], dim=-2)  # (B, T_q, K, d_z)
        expected = (alpha.unsqueeze(-1) * stacked).sum(dim=-2)
    assert torch.allclose(out["kg_hidden"], expected, atol=1e-5)


def test_phase15_moe_k_routed_128_constructs_and_forwards():
    """K=128 default must instantiate and forward on tiny B/T_q without OOM."""
    model = Phase15MoE(d_emb=16, d_z=16, k_routed=128, d_hidden_expert=16)
    out = model(_tiny_batch(b=1, t_q=2, t_p=2, d_emb=16))
    assert out["alpha"].shape == (1, 2, 128)


def test_phase15_moe_modulation_film_swap():
    model = Phase15MoE(d_emb=32, d_z=16, k_routed=4, modulation=MOD_FILM)
    out = model(_tiny_batch())
    assert out["logits"].shape == (2, 4)
    # Modulation module is FiLM.
    from research.demo.phase1_5.modulation import FiLMModulation

    assert isinstance(model.modulation, FiLMModulation)


def test_phase15_moe_modulation_default_is_kg_hypernet():
    model = Phase15MoE(d_emb=32, d_z=16, k_routed=4)
    from research.demo.phase1_5.kg_hypernet import KGHypernetModulation

    assert isinstance(model.modulation, KGHypernetModulation)


def test_phase15_moe_dead_router_gives_zero_kg_modulated():
    """End-to-end no-bypass invariant (info-bottleneck #3 (i)): if the router is
    forced dead (alpha=0 ⟹ kg_hidden=0), the modulation output must be exactly
    zero — the passage cannot reach the answer head without the operation-KG.
    Guards against regressing to a leaky modulation (cross_attn/FiLM)."""
    model = Phase15MoE(d_emb=32, d_z=16, k_routed=4).eval()
    with torch.no_grad():
        model.router.gate.bias.fill_(-100.0)  # all logits negative → alpha=0
        out = model(_tiny_batch())
    assert torch.all(out["kg_hidden"] == 0)
    assert torch.all(out["kg_modulated"] == 0)


def test_phase15_moe_invalid_modulation_raises():
    with pytest.raises(ValueError, match="modulation"):
        Phase15MoE(d_emb=32, d_z=16, k_routed=4, modulation="bogus")


def test_compute_alpha_returns_expected_keys():
    model = Phase15MoE(d_emb=32, d_z=16, k_routed=4)
    batch = _tiny_batch(b=2, t_q=3, d_emb=32)
    out = model.compute_alpha(batch["q_tokens"])
    assert set(out.keys()) >= {"alpha", "router_logits", "z_q", "k_active"}
    assert out["alpha"].shape == (2, 3, 4)


def test_compute_alpha_matches_full_forward_alpha():
    """Probe-correctness invariant: compute_alpha must yield the same alpha as forward().
    If they drift (e.g., compute_alpha builds its own router), the selectivity probe
    measures something different from what training optimised."""
    model = Phase15MoE(d_emb=32, d_z=16, k_routed=4).eval()
    batch = _tiny_batch(b=2, t_q=4, d_emb=32)
    with torch.no_grad():
        full = model(batch)
        partial = model.compute_alpha(batch["q_tokens"])
    assert torch.allclose(full["alpha"], partial["alpha"], atol=1e-6)
    assert torch.allclose(full["router_logits"], partial["router_logits"], atol=1e-6)


def test_compute_alpha_preserves_gradients_when_outer_grad_enabled():
    """No no_grad decorator — gradients must flow through compute_alpha so future
    router-grad diagnostics work."""
    model = Phase15MoE(d_emb=16, d_z=8, k_routed=4)
    q = torch.randn(2, 3, 16, requires_grad=True)
    out = model.compute_alpha(q)
    out["alpha"].sum().backward()
    assert q.grad is not None
    assert model.router.gate.weight.grad is not None


def test_phase15_moe_q_mask_zero_positions_excluded_from_summary():
    """When a Q position is masked out, its kg_modulated contribution is dropped."""
    model = Phase15MoE(d_emb=16, d_z=8, k_routed=4)
    batch = _tiny_batch(b=1, t_q=4, t_p=2, d_emb=16)
    # Mask out positions 2 and 3.
    batch["q_mask"] = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    out = model(batch)
    # Summary = mean of kg_modulated over positions 0,1.
    expected = out["kg_modulated"][0, :2].mean(dim=0)
    assert torch.allclose(out["kg_summary"][0], expected, atol=1e-5)


# ---- ReMoERouter top-k routing (Phase 3 diversity) ----------------------------


def test_remoe_router_topk_activates_exactly_k():
    router = ReMoERouter(d_z=8, k=16, routing="topk", k_active=4)
    z = torch.randn(2, 5, 8)
    alpha, logits = router(z)
    assert alpha.shape == (2, 5, 16)
    assert torch.all((alpha > 0).sum(-1) == 4)            # exactly k active
    assert torch.allclose(alpha.sum(-1), torch.ones(2, 5), atol=1e-5)  # softmax over top-k


def test_remoe_router_topk_selection_uses_external_bias():
    """aux-free LB bias steers *selection*: a large positive bias on a dead expert
    forces it into the top-k."""
    router = ReMoERouter(d_z=8, k=16, routing="topk", k_active=4)
    z = torch.randn(1, 1, 8)
    bias = torch.zeros(16)
    bias[15] = 1e3  # expert 15 forced selected
    alpha, _ = router(z, external_bias=bias)
    assert alpha[0, 0, 15] > 0


def test_remoe_router_default_routing_is_relu_l1_unchanged():
    """Default path unchanged: alpha = relu(gate(z)) (no top-k)."""
    router = ReMoERouter(d_z=8, k=4)
    z = torch.randn(2, 3, 8)
    alpha, logits = router(z)
    assert torch.allclose(alpha, torch.relu(logits), atol=1e-6)


def test_phase15_moe_topk_routing_activates_k_active():
    """Phase15MoE(routing='topk') → router selects exactly k_active experts/token
    (= lb_target_active), giving a stable sparse-but-diverse K regime."""
    model = Phase15MoE(d_emb=32, d_z=16, k_routed=16, routing="topk", lb_target_active=4.0)
    out = model(_tiny_batch(b=2, t_q=5))
    assert torch.all(out["k_active"] == 4)


# ---- forward alpha_override hook (intervention battery) ------------------------


def test_forward_alpha_override_zero_gives_zero_kg():
    """alpha_override replaces routed alpha in the mixture; zeros → no-bypass (kg=0)."""
    model = Phase15MoE(d_emb=32, d_z=16, k_routed=4).eval()
    batch = _tiny_batch(b=2, t_q=3)
    with torch.no_grad():
        out = model(batch, alpha_override=torch.zeros(2, 3, 4))
    assert torch.all(out["alpha"] == 0)
    assert torch.all(out["kg_hidden"] == 0)
    assert torch.all(out["kg_modulated"] == 0)


def test_forward_alpha_override_changes_output_and_default_unchanged():
    model = Phase15MoE(d_emb=32, d_z=16, k_routed=4).eval()
    batch = _tiny_batch(b=2, t_q=3)
    ov = torch.zeros(2, 3, 4)
    ov[..., 0] = 1.0  # force expert 0 only
    with torch.no_grad():
        out_def = model(batch)                       # default: routed alpha
        out_ov = model(batch, alpha_override=ov)
        out_def2 = model(batch)                       # default still identical
    assert not torch.allclose(out_def["logits"], out_ov["logits"])
    assert torch.allclose(out_def["logits"], out_def2["logits"])


# ---- dropout (Phase 1 regularization) -----------------------------------------


def test_operation_expert_dropout_active_in_train_off_in_eval():
    torch.manual_seed(0)
    e = OperationExpert(d_z=16, d_hidden=32, dropout=0.5)
    z = torch.randn(2, 4, 16)
    e.train()
    assert not torch.allclose(e(z), e(z))      # stochastic in train
    e.eval()
    assert torch.allclose(e(z), e(z))           # deterministic in eval
    assert e(z).shape == (2, 4, 16)


def test_phase15_moe_dropout_plumbs_through():
    torch.manual_seed(0)
    m = Phase15MoE(d_emb=32, d_z=16, k_routed=4, dropout=0.5).train()
    b = _tiny_batch()
    assert not torch.allclose(m(b)["logits"], m(b)["logits"])   # dropout active
    m.eval()
    assert torch.allclose(m(b)["logits"], m(b)["logits"])
