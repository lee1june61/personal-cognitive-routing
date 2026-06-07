"""Tests for `phase1_5.train` — loss primitives + tiny end-to-end."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.phase1_5.data import MCDataset
from experiments.phase1_5.model import Phase15MoE
from experiments.phase1_5.train import (
    TrainConfig,
    _masked_token_mean,
    mc_ce_loss,
    remoe_l1_loss,
    router_z_loss,
    train_phase15,
    update_l1_lambda,
)


# ---- loss primitives ----------------------------------------------------------


def test_masked_token_mean_excludes_padded_positions():
    per_tok = torch.tensor([[1.0, 2.0, 99.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    out = _masked_token_mean(per_tok, mask)
    assert torch.isclose(out, torch.tensor(1.5))


def test_mc_ce_loss_correctness():
    logits = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
    answer_idx = torch.tensor([0, 1])
    out = mc_ce_loss(logits, answer_idx)
    expected = torch.nn.functional.cross_entropy(logits, answer_idx)
    assert torch.isclose(out, expected, atol=1e-5)


def test_remoe_l1_loss_zero_for_zero_alpha():
    alpha = torch.zeros(2, 3, 4)
    mask = torch.ones(2, 3)
    out = remoe_l1_loss(alpha, mask)
    assert float(out) == 0.0


def test_remoe_l1_loss_increases_with_active_experts():
    alpha_dense = torch.ones(1, 1, 4)
    alpha_sparse = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    mask = torch.ones(1, 1)
    assert float(remoe_l1_loss(alpha_dense, mask)) > float(remoe_l1_loss(alpha_sparse, mask))


def test_router_z_loss_nonneg():
    logits = torch.randn(2, 3, 8)
    mask = torch.ones(2, 3)
    out = router_z_loss(logits, mask)
    assert float(out) >= 0.0


# ---- adaptive L1 controller --------------------------------------------------


def test_update_l1_lambda_raises_when_too_dense():
    lam_new = update_l1_lambda(lam=1e-2, k_active_mean=10.0, k_target=4.0)
    assert lam_new > 1e-2


def test_update_l1_lambda_lowers_when_too_sparse():
    lam_new = update_l1_lambda(lam=1e-2, k_active_mean=1.0, k_target=4.0)
    assert lam_new < 1e-2


def test_update_l1_lambda_clamps_to_max():
    lam = 0.5
    for _ in range(20):
        lam = update_l1_lambda(lam, k_active_mean=100.0, k_target=1.0, lam_max=1.0)
    assert lam <= 1.0


def test_update_l1_lambda_clamps_to_min():
    lam = 1e-5
    for _ in range(20):
        lam = update_l1_lambda(lam, k_active_mean=0.0, k_target=10.0, lam_min=1e-6)
    assert lam >= 1e-6


def test_update_l1_lambda_unchanged_at_target():
    lam_new = update_l1_lambda(lam=1e-2, k_active_mean=4.0, k_target=4.0)
    assert lam_new == 1e-2


# ---- tiny end-to-end ---------------------------------------------------------


def _tiny_loader(n: int = 16, t_q: int = 4, t_p: int = 6, d_emb: int = 16, n_cand: int = 4):
    rng = np.random.default_rng(0)
    ds = MCDataset(
        q_tokens=rng.standard_normal((n, t_q, d_emb)).astype(np.float16),
        q_mask=np.ones((n, t_q), dtype=np.int8),
        p_tokens=rng.standard_normal((n, t_p, d_emb)).astype(np.float16),
        p_mask=np.ones((n, t_p), dtype=np.int8),
        cand_pooled=rng.standard_normal((n, n_cand, d_emb)).astype(np.float32),
        answer_idx=rng.integers(0, n_cand, size=n).astype(np.int64),
    )
    return DataLoader(ds, batch_size=4, shuffle=True)


def test_train_phase15_tiny_end_to_end_no_nan():
    model = Phase15MoE(d_emb=16, d_z=8, k_routed=4, d_hidden_expert=12)
    loader = _tiny_loader(n=16, d_emb=16)
    result = train_phase15(
        model,
        loader,
        cfg=TrainConfig(epochs=2, lr=1e-3, log_every=0, use_best_val=False, seed=0),
        device="cpu",
    )
    assert len(result["history"]) == 2
    for h in result["history"]:
        assert not np.isnan(h["loss"])
        assert not np.isnan(h["ce"])


def test_train_phase15_best_metric_acc_selects_ckpt():
    """best_metric='acc' selects a checkpoint by val accuracy (robust to the
    overfit-inflated val CE) and still returns a usable model."""
    model = Phase15MoE(d_emb=16, d_z=8, k_routed=4, d_hidden_expert=12)
    result = train_phase15(
        model, _tiny_loader(n=16, d_emb=16), val_loader=_tiny_loader(n=8, d_emb=16),
        cfg=TrainConfig(epochs=3, best_metric="acc", log_every=0, seed=0),
        device="cpu",
    )
    assert result["model"] is not None
    assert all("val_mc_acc" in h for h in result["history"])


def test_train_phase15_chain_steps_end_to_end_no_nan():
    """1b: a chain_steps>1 model trains end-to-end via forward_chain (the
    _forward_normalized adapter routes train/eval to the chain path)."""
    model = Phase15MoE(d_emb=16, d_z=8, k_routed=4, d_hidden_expert=12, chain_steps=3)
    loader = _tiny_loader(n=16, d_emb=16)
    result = train_phase15(
        model,
        loader,
        val_loader=_tiny_loader(n=8, d_emb=16),
        cfg=TrainConfig(epochs=2, lr=1e-3, log_every=0, use_best_val=False, seed=0),
        device="cpu",
    )
    assert len(result["history"]) == 2
    for h in result["history"]:
        assert not np.isnan(h["loss"]) and not np.isnan(h["ce"])


def test_train_phase15_with_val_records_best_state():
    model = Phase15MoE(d_emb=16, d_z=8, k_routed=4, d_hidden_expert=12)
    train = _tiny_loader(n=16, d_emb=16)
    val = _tiny_loader(n=8, d_emb=16)
    result = train_phase15(
        model,
        train,
        val_loader=val,
        cfg=TrainConfig(epochs=3, lr=1e-3, log_every=0, use_best_val=True, seed=0),
        device="cpu",
    )
    assert result["best_state"] is not None
    assert result["best_val_loss"] is not None
    assert "val_loss" in result["history"][0]
    assert "val_mc_acc" in result["history"][0]


def test_train_phase15_returns_model_with_trainable_weights_updated():
    """After training, at least one expert weight should differ from init."""
    model = Phase15MoE(d_emb=16, d_z=8, k_routed=4, d_hidden_expert=12)
    init_weight = model.experts[0].fc1.weight.detach().clone()
    loader = _tiny_loader(n=16, d_emb=16)
    result = train_phase15(
        model,
        loader,
        cfg=TrainConfig(epochs=2, lr=1e-2, log_every=0, use_best_val=False, seed=0),
        device="cpu",
    )
    trained_weight = result["model"].experts[0].fc1.weight
    assert not torch.allclose(init_weight, trained_weight)


def test_train_phase15_history_has_required_keys():
    model = Phase15MoE(d_emb=16, d_z=8, k_routed=4)
    loader = _tiny_loader(n=12, d_emb=16)
    result = train_phase15(
        model,
        loader,
        cfg=TrainConfig(epochs=1, lr=1e-3, log_every=0, use_best_val=False, seed=0),
        device="cpu",
    )
    h = result["history"][0]
    for k in ("loss", "ce", "l1", "z_loss", "k_active_mean", "mc_acc", "lam_l1", "lr"):
        assert k in h
