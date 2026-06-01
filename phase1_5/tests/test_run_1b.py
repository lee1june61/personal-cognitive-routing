"""Tests for `phase1_5.run_1b` — 1b PASS-bar verdict logic (CPU; the full GPU
pipeline ``run_1b_experiment`` is exercised on Colab)."""

from __future__ import annotations

import numpy as np
import torch

from research.demo.phase1_5.model import Phase15MoE
from research.demo.phase1_5.run_1b import (
    _balance_train_by_hop,
    evaluate_1b,
    hop_depth_selective,
    step_breadth_by_depth,
)


def _imbalanced_data():
    rt, split = [], []
    for c, k in [("2hop", 10), ("3hop", 4), ("4hop", 2)]:
        rt += [c] * k
        split += ["train"] * k
    rt += ["2hop", "3hop", "4hop"]
    split += ["val", "val", "test"]
    n = len(split)
    return {"split": np.array(split), "reasoning_type": np.array(rt), "answer_idx": np.arange(n)}


def test_balance_train_by_hop_min_count_when_no_cap():
    """cap=None → full equalise to the minority count; val/test untouched."""
    import collections

    out = _balance_train_by_hop(_imbalanced_data(), seed=0)
    cnt = collections.Counter(out["reasoning_type"][out["split"] == "train"].tolist())
    assert set(cnt.values()) == {2}  # min=2
    assert (out["split"] == "val").sum() == 2 and (out["split"] == "test").sum() == 1


def test_balance_train_by_hop_cap_keeps_small_classes_whole():
    """cap=5 → dominant 2hop trimmed to 5, smaller 3hop(4)/4hop(2) kept whole
    (capping discards far less data than full equalise)."""
    import collections

    out = _balance_train_by_hop(_imbalanced_data(), seed=0, cap=5)
    cnt = collections.Counter(out["reasoning_type"][out["split"] == "train"].tolist())
    assert cnt["2hop"] == 5 and cnt["3hop"] == 4 and cnt["4hop"] == 2


def test_hop_depth_selective_monotone_true():
    """Last-step self-drop increasing in hop depth ⇒ depth-selective PASS(a)."""
    lesion = {
        "n_steps": 3,
        "drop_by_step": {
            0: {}, 1: {},
            2: {  # last step: self-drop 2hop < 3hop < 4hop
                "2hop": {"2hop": 0.01, "3hop": 0.0, "4hop": 0.0},
                "3hop": {"2hop": 0.0, "3hop": 0.10, "4hop": 0.0},
                "4hop": {"2hop": 0.0, "3hop": 0.0, "4hop": 0.30},
            },
        },
    }
    out = hop_depth_selective(lesion)
    assert out["monotone_in_depth"] is True
    assert out["hops"] == ["2hop", "3hop", "4hop"]


def test_hop_depth_selective_uniform_false():
    """Uniform last-step drop ⇒ non-compositional ⇒ PASS(a) False."""
    lesion = {
        "n_steps": 2,
        "drop_by_step": {
            0: {},
            1: {
                "2hop": {"2hop": 0.1, "3hop": 0.1, "4hop": 0.1},
                "3hop": {"2hop": 0.1, "3hop": 0.1, "4hop": 0.1},
                "4hop": {"2hop": 0.1, "3hop": 0.1, "4hop": 0.1},
            },
        },
    }
    assert hop_depth_selective(lesion)["monotone_in_depth"] is False


def _tiny_data(d_emb=16, t_q=4, t_p=5, n_cand=4):
    rng = np.random.default_rng(0)
    n = 18
    return {
        "q_tokens": rng.standard_normal((n, t_q, d_emb)).astype(np.float16),
        "q_mask": np.ones((n, t_q), dtype=np.int8),
        "p_tokens": rng.standard_normal((n, t_p, d_emb)).astype(np.float16),
        "p_mask": np.ones((n, t_p), dtype=np.int8),
        "cand_pooled": rng.standard_normal((n, n_cand, d_emb)).astype(np.float32),
        "answer_idx": rng.integers(0, n_cand, size=n).astype(np.int64),
        "reasoning_type": np.array((["2hop", "3hop", "4hop"] * 6)),
        "split": np.array(["train"] * 9 + ["test"] * 9),
    }


def test_step_breadth_by_depth_increasing_true():
    """# steps that matter grows with hop depth (2hop:1, 3hop:2, 4hop:3) ⇒ PASS."""
    lesion = {
        "n_steps": 3,
        "drop_by_step": {
            0: {"2hop": {"2hop": 0.00}, "3hop": {"3hop": 0.10}, "4hop": {"4hop": 0.10}},
            1: {"2hop": {"2hop": 0.00}, "3hop": {"3hop": 0.10}, "4hop": {"4hop": 0.10}},
            2: {"2hop": {"2hop": 0.10}, "3hop": {"3hop": 0.10}, "4hop": {"4hop": 0.10}},
        },
    }
    out = step_breadth_by_depth(lesion, threshold=0.05)
    assert out["breadth"] == {"2hop": 1, "3hop": 3, "4hop": 3}
    assert out["monotone"] is True


def test_step_breadth_by_depth_flat_false():
    """Same breadth for all hops ⇒ no depth-scaling ⇒ FALSE."""
    lesion = {
        "n_steps": 2,
        "drop_by_step": {
            0: {"2hop": {"2hop": 0.1}, "4hop": {"4hop": 0.1}},
            1: {"2hop": {"2hop": 0.0}, "4hop": {"4hop": 0.0}},
        },
    }
    assert step_breadth_by_depth(lesion)["monotone"] is False


def test_evaluate_1b_runs_and_reports_both_criteria():
    model = Phase15MoE(d_emb=16, d_z=8, k_routed=8, routing="topk",
                       lb_target_active=2.0, chain_steps=3).eval()
    out = evaluate_1b(model, _tiny_data(), k_top=2, device="cpu")
    assert out["n_probe"] == 9
    assert isinstance(out["passed"], bool)
    # both the original (transparency) and the re-specified primary criterion present
    assert out["pass_a_laststep"]["hops"] == ["2hop", "3hop", "4hop"]
    assert "breadth" in out["pass_a_breadth"] and "monotone" in out["pass_a_breadth"]
    assert "op_purity" in out["s1_motif"]
    assert out["lesion"]["n_steps"] == 3
