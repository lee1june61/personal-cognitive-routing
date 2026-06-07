"""Tests for `phase1_5.eval`."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.phase1_5.eval import (
    _adjusted_accuracy,
    chance_rate,
    geometry_baseline,
    make_phase15_controls,
    probe_accuracy,
    raw_embedding_report,
    selectivity_gate_phase15,
    selectivity_report,
    sequence_code,
)


# ---- sequence_code ------------------------------------------------------------


def test_sequence_code_mean_shape():
    alpha = np.random.RandomState(0).standard_normal((5, 4, 3)).astype(np.float32)
    mask = np.ones((5, 4), dtype=np.int8)
    out = sequence_code(alpha, mask, agg="mean")
    assert out.shape == (5, 3)


def test_sequence_code_max_shape():
    alpha = np.random.RandomState(0).standard_normal((5, 4, 3)).astype(np.float32)
    mask = np.ones((5, 4), dtype=np.int8)
    out = sequence_code(alpha, mask, agg="max")
    assert out.shape == (5, 3)


def test_sequence_code_meanmax_shape_doubles_k():
    alpha = np.random.RandomState(0).standard_normal((5, 4, 3)).astype(np.float32)
    mask = np.ones((5, 4), dtype=np.int8)
    out = sequence_code(alpha, mask, agg="meanmax")
    assert out.shape == (5, 6)


def test_sequence_code_mask_excludes_padded_positions():
    alpha = np.array([[[1.0, 0.0], [99.0, 0.0]]], dtype=np.float32)
    mask = np.array([[1, 0]], dtype=np.int8)
    out = sequence_code(alpha, mask, agg="mean")
    assert np.isclose(out[0, 0], 1.0)


def test_sequence_code_invalid_agg_raises():
    with pytest.raises(ValueError, match="unknown sequence_code"):
        sequence_code(np.zeros((1, 1, 1)), np.zeros((1, 1)), agg="bogus")


# ---- chance_rate / adjusted accuracy ------------------------------------------


def test_chance_rate_majority():
    labels = np.array([0, 0, 0, 1])
    assert chance_rate(labels) == 0.75


def test_chance_rate_empty():
    assert chance_rate(np.array([])) == 0.0


def test_adjusted_accuracy_zero_at_chance():
    labels = np.array([0, 0, 1, 1])
    assert _adjusted_accuracy(0.5, labels) == 0.0


def test_adjusted_accuracy_one_at_perfect():
    labels = np.array([0, 0, 1, 1])
    assert _adjusted_accuracy(1.0, labels) == 1.0


# ---- geometry_baseline --------------------------------------------------------


def test_geometry_shuffle_preserves_marginals():
    codes = np.random.RandomState(0).standard_normal((20, 5))
    shuffled = geometry_baseline(codes, mode="shuffle", seed=0)
    assert shuffled.shape == codes.shape
    # column statistics preserved
    assert np.allclose(np.sort(codes, axis=0), np.sort(shuffled, axis=0))


def test_geometry_rotation_preserves_norm_distribution():
    codes = np.random.RandomState(0).standard_normal((20, 5))
    rotated = geometry_baseline(codes, mode="rotation", seed=0)
    assert rotated.shape == codes.shape
    assert np.allclose(np.linalg.norm(codes, axis=1).sum(), np.linalg.norm(rotated, axis=1).sum())


def test_geometry_invalid_mode_raises():
    with pytest.raises(ValueError, match="unknown geometry baseline"):
        geometry_baseline(np.zeros((4, 2)), mode="bogus")


# ---- probe_accuracy -----------------------------------------------------------


def test_probe_accuracy_perfect_when_signal_planted():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, size=200)
    eye = np.eye(3)
    codes = eye[labels] + rng.standard_normal((200, 3)) * 0.05
    acc = probe_accuracy(codes, labels, seed=0)
    assert acc > 0.95


def test_probe_accuracy_chance_when_random_labels():
    rng = np.random.default_rng(0)
    codes = rng.standard_normal((200, 8))
    labels = rng.integers(0, 4, size=200)
    acc = probe_accuracy(codes, labels, seed=0)
    assert 0.10 < acc < 0.40  # near chance ≈ 0.25 with some noise


def test_probe_accuracy_degenerate_raises():
    codes = np.random.RandomState(0).standard_normal((10, 4))
    labels = np.zeros(10, dtype=int)
    with pytest.raises(ValueError, match="≥2 classes"):
        probe_accuracy(codes, labels)


# ---- selectivity_report -------------------------------------------------------


def _planted_setup(n=200, k=3, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, size=n)
    eye = np.eye(k)
    codes = eye[labels] + rng.standard_normal((n, k)) * 0.1
    return codes, labels


def test_selectivity_report_pass_with_planted_signal():
    codes, labels = _planted_setup()
    rep = selectivity_report(codes, labels, control_label_sets={"topic": np.zeros(len(labels), dtype=int) + np.arange(len(labels)) % 5})
    assert rep["verdict"] == "PASS"
    assert rep["adj_operation"] > 0.8


def test_selectivity_report_fail_when_no_signal():
    rng = np.random.default_rng(0)
    codes = rng.standard_normal((200, 8))
    labels = rng.integers(0, 4, size=200)
    topic = rng.integers(0, 4, size=200)
    rep = selectivity_report(codes, labels, control_label_sets={"topic": topic})
    assert rep["verdict"] == "FAIL"


def test_selectivity_report_degenerate_control_skipped_with_warning():
    codes, labels = _planted_setup()
    rep = selectivity_report(
        codes, labels, control_label_sets={"topic": np.arange(len(labels)) % 5, "bogus": np.zeros(len(labels))}
    )
    assert any("bogus" in w for w in rep["warnings"])


def test_raw_embedding_report_runs_on_tokens_directly():
    rng = np.random.default_rng(0)
    n, t, d = 60, 4, 8
    labels = rng.integers(0, 3, size=n)
    eye = np.eye(d)[:3]
    tokens = eye[labels][:, None, :].repeat(t, axis=1).astype(np.float32) + rng.standard_normal(
        (n, t, d)
    ).astype(np.float32) * 0.05
    mask = np.ones((n, t), dtype=np.int8)
    rep = raw_embedding_report(tokens, mask, labels, agg="mean")
    assert rep["adj_operation"] > 0.5


# ---- phase1_5 adapters --------------------------------------------------------


def test_make_phase15_controls_returns_topic_and_token_arrays():
    qs = [f"is the answer {i}?" for i in range(20)]
    ps = [f"some passage with topic {i % 4}" for i in range(20)]
    out = make_phase15_controls(qs, ps, topic_n_clusters=4, seed=0)
    assert "topic" in out and "token" in out
    assert out["topic"].shape == (20,)
    assert out["token"].shape == (20,)


def test_make_phase15_controls_topic_has_multiple_clusters():
    rng = np.random.default_rng(0)
    qs = [f"question {rng.choice(['a', 'b', 'c', 'd'])} {i}" for i in range(30)]
    ps = [f"passage discussing {rng.choice(['alpha', 'beta', 'gamma'])}" for _ in range(30)]
    out = make_phase15_controls(qs, ps, topic_n_clusters=3, seed=0)
    assert len(np.unique(out["topic"])) >= 2


def test_selectivity_gate_phase15_pass_with_planted_signal():
    codes, labels = _planted_setup(n=300, k=4, seed=0)
    controls = {"topic": np.arange(len(labels)) % 5, "token": np.array(["N"] * len(labels))}
    rep = selectivity_gate_phase15(codes, labels, controls, n_boot=50, seed=0)
    assert "sigma_adj_operation" in rep
    assert "threshold" in rep
    assert rep["passes_sigma_gate"]
    assert rep["verdict"] == "PASS"


def test_selectivity_gate_phase15_fail_when_no_signal():
    rng = np.random.default_rng(0)
    codes = rng.standard_normal((200, 8))
    labels = rng.integers(0, 4, size=200)
    controls = {"topic": rng.integers(0, 4, size=200), "token": np.array(["N"] * 200)}
    rep = selectivity_gate_phase15(codes, labels, controls, n_boot=50, seed=0)
    assert rep["verdict"] == "FAIL"


def test_selectivity_gate_phase15_sigma_records():
    codes, labels = _planted_setup(n=120, k=3, seed=0)
    controls = {"topic": np.arange(len(labels)) % 4, "token": np.array(["N"] * len(labels))}
    rep = selectivity_gate_phase15(codes, labels, controls, n_boot=30, seed=0)
    assert rep["sigma_adj_operation"] >= 0
    assert rep["n_boot"] == 30


# ---- build_operation_labels (regex axis, drop other/rare) ---------------------


def _op_questions():
    """Stems with known infer_reasoning_type: 3 weaken, 3 inference, 1 assumption, 2 other."""
    return [
        "Which one, if true, most weakens the argument?",
        "Which most weakens the conclusion above?",
        "What weakens the reasoning?",
        "Which one must be true?",
        "Which must be true given the statements?",
        "Which one must be true on the basis above?",
        "The argument assumes which of the following?",
        "Hello world, nothing to classify here.",
        "Another unclassifiable filler sentence.",
    ]


def test_build_operation_labels_drops_other_and_rare():
    """Now takes a PRECOMPUTED per-row label array (source-agnostic): drops the
    ``drop_labels`` sentinels (``other``/``unknown``) and classes < min_count."""
    from experiments.phase1_5.data import infer_reasoning_type
    from experiments.phase1_5.eval import build_operation_labels

    raw = [infer_reasoning_type(q) for q in _op_questions()]
    labels, keep = build_operation_labels(raw, min_count=2)
    # 'other' (2) dropped; 'assumption' (1 < min_count) dropped; keep weaken(3)+inference(3).
    assert keep.dtype == bool and keep.shape == (9,)
    assert int(keep.sum()) == 6
    assert labels.shape == (6,)
    assert set(labels.tolist()) == {"weaken", "inference"}


def test_build_operation_labels_from_hop_axis():
    """MuSiQue hop axis: precomputed labels, 'unknown' sentinel + rare dropped."""
    from experiments.phase1_5.eval import build_operation_labels

    raw = ["2hop"] * 10 + ["3hop"] * 10 + ["4hop"] * 3 + ["unknown"] * 5
    labels, keep = build_operation_labels(raw, min_count=5)
    assert set(labels.tolist()) == {"2hop", "3hop"}  # 4hop(3<5) + unknown dropped
    assert int(keep.sum()) == 20


# ---- regate_operation_selectivity (offline re-gate on diverse op-axis) --------


def test_regate_operation_selectivity_runs_on_kept_subset():
    from experiments.phase1_5.eval import regate_operation_selectivity

    rng = np.random.RandomState(0)
    N, two_k = 40, 16
    codes = rng.standard_normal((N, two_k)).astype(np.float64)
    questions = (
        ["Which one, if true, most weakens the argument?"] * 15
        + ["Which one must be true?"] * 15
        + ["unclassifiable filler sentence"] * 10
    )
    passages = [f"passage about cluster {i % 3}" for i in range(N)]
    out = regate_operation_selectivity(
        codes, questions, passages, min_count=5, n_boot=20, seed=0
    )
    assert {"verdict", "adj_operation", "adj_controls"} <= set(out)
    # 'other' (10) dropped; weaken(15)+inference(15) kept.
    assert out["n_operation_examples"] == 30
    assert set(out["operation_classes"]) == {"weaken", "inference"}
    # granularity-robust consistency reported alongside the linear-probe gate.
    assert "consistency" in out and "op_purity" in out["consistency"]


def test_regate_operation_selectivity_with_precomputed_hop_labels():
    """MuSiQue path: caller supplies the hop axis via op_labels (not regex)."""
    from experiments.phase1_5.eval import regate_operation_selectivity

    rng = np.random.RandomState(0)
    N, two_k = 40, 16
    codes = rng.standard_normal((N, two_k)).astype(np.float64)
    questions = [f"some multi-hop question {i}" for i in range(N)]
    passages = [f"passage about cluster {i % 3}" for i in range(N)]
    hop = ["2hop"] * 15 + ["3hop"] * 15 + ["unknown"] * 10
    out = regate_operation_selectivity(
        codes, questions, passages, op_labels=hop, min_count=5, n_boot=20, seed=0
    )
    assert out["n_operation_examples"] == 30  # unknown dropped
    assert set(out["operation_classes"]) == {"2hop", "3hop"}


# ---- operation_consistency (granularity-robust kNN purity, topic-controlled) --


def test_operation_consistency_detects_operation_clustered_codes():
    from experiments.phase1_5.eval import operation_consistency

    rng = np.random.RandomState(0)
    ops = np.array(["a", "b", "c"] * 30)            # 90 items, op = i % 3
    topics = np.array(["t0", "t1"] * 45)             # topic = i % 2 → crosscuts op (no aliasing)
    centroids = {"a": [3, 0, 0], "b": [0, 3, 0], "c": [0, 0, 3]}
    codes = np.array([centroids[o] for o in ops], float) + rng.standard_normal((90, 3)) * 0.3
    out = operation_consistency(codes, ops, topics, k=8)
    # codes cluster by OPERATION → high op purity, above chance, beats topic.
    assert out["op_purity"] > out["op_chance"] + 0.2
    assert out["op_above_chance"] is True
    assert out["op_beats_topic"] is True


def test_operation_consistency_topic_clustered_codes_do_not_fake_operation():
    """If codes cluster by TOPIC (op crosscuts), op_purity must stay ~chance and
    topic must beat op — guards against crediting operation to content structure."""
    from experiments.phase1_5.eval import operation_consistency

    rng = np.random.RandomState(1)
    ops = np.array(["a", "b", "c"] * 30)             # op = i % 3 (crosscuts topic)
    topics = np.array(["t0", "t1"] * 45)             # topic = i % 2
    tc = {"t0": [4, 0], "t1": [0, 4]}
    codes = np.array([tc[t] for t in topics], float) + rng.standard_normal((90, 2)) * 0.3
    out = operation_consistency(codes, ops, topics, k=8)
    assert out["topic_purity"] > out["op_purity"]
    assert out["op_purity"] < out["op_chance"] + 0.15  # op ~ chance, no real op structure


def test_make_phase15_controls_stem_is_question_only():
    """'stem' control clusters on the QUESTION only → invariant to the passage
    (vs 'topic' which uses question+passage). Lets the gate test whether routing
    beats stem-surface phrasing, not just q+p topic."""
    qs = [f"which one most weakens type {i % 4} argument here" for i in range(40)]
    ps_a = [f"passage A about {i}" for i in range(40)]
    ps_b = [f"completely unrelated passage B {i} blah" for i in range(40)]
    out_a = make_phase15_controls(qs, ps_a, seed=0)
    out_b = make_phase15_controls(qs, ps_b, seed=0)
    assert "stem" in out_a and out_a["stem"].shape == (40,)
    # stem depends only on questions → identical across different passage sets.
    assert np.array_equal(out_a["stem"], out_b["stem"])
