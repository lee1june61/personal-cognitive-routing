"""Tests for Engine-A selectivity eval (ENGINE_A_DESIGN §5, Hewitt-Liang control task).

The go/no-go gate: a linear probe on the sequence code (mean_t α) must decode the
*operation* (reasoning-type) label better than four controls — random labels, topic,
token-type, and a geometry-destroyed baseline. These tests use a *planted* synthetic
code where the operation signal is linearly decodable by construction, so we can assert
acc(operation) > acc(control) without any real model/training.
"""

from __future__ import annotations

import numpy as np
import pytest

from phase1.eval_opcycle import (
    sequence_code,
    probe_accuracy,
    geometry_baseline,
    selectivity_report,
    raw_embedding_report,
    chance_rate,
    _adjusted_accuracy,
)

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# sequence_code — masked mean of per-token gates → (N, K)
# ---------------------------------------------------------------------------


def test_sequence_code_is_masked_mean_over_tokens():
    # 1 example, 3 tokens, K=2; only first 2 tokens active.
    alpha = torch.tensor([[[1.0, 0.0], [3.0, 2.0], [9.0, 9.0]]])  # (1, 3, 2)
    mask = torch.tensor([[1, 1, 0]])
    code = sequence_code(alpha, mask)
    assert code.shape == (1, 2)
    # mean over active tokens only: ([1,0]+[3,2])/2 = [2,1]; padded token ignored
    np.testing.assert_allclose(code[0], [2.0, 1.0], rtol=1e-5)


# ---------------------------------------------------------------------------
# planted synthetic code: operation linearly decodable, controls are not
# ---------------------------------------------------------------------------

N_CLASS = 3
PER_CLASS = 40
K_CODE = 6


def _planted_codes(seed: int = 0):
    """codes whose first 3 dims one-hot the class (+ noise) → linearly separable;
    operation labels = class id; topic labels = an *independent* random partition."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(N_CLASS), PER_CLASS)
    codes = 0.05 * rng.standard_normal((N_CLASS * PER_CLASS, K_CODE)).astype(np.float32)
    for i, c in enumerate(labels):
        codes[i, c] += 1.0  # plant class signal in dims 0..2
    topic = rng.integers(0, N_CLASS, size=labels.shape)  # independent of code
    return codes, labels, topic


def test_probe_accuracy_high_when_signal_planted():
    codes, labels, _ = _planted_codes()
    acc = probe_accuracy(codes, labels, seed=0)
    assert acc > 0.9


def test_probe_accuracy_chance_on_random_labels():
    codes, labels, _ = _planted_codes()
    rng = np.random.default_rng(1)
    random_labels = rng.permutation(labels)  # destroy code↔label alignment
    acc = probe_accuracy(codes, random_labels, seed=0)
    assert acc < 0.6  # near 1/3 chance, well below the planted-signal accuracy


# ---------------------------------------------------------------------------
# geometry_baseline — destroying code structure collapses accuracy to chance
# ---------------------------------------------------------------------------


def test_geometry_baseline_destroys_decodability():
    codes, labels, _ = _planted_codes()
    shuffled = geometry_baseline(codes, mode="shuffle", seed=2)
    assert shuffled.shape == codes.shape
    acc = probe_accuracy(shuffled, labels, seed=0)
    assert acc < 0.6


# ---------------------------------------------------------------------------
# selectivity_report — operation beats all controls → PASS verdict
# ---------------------------------------------------------------------------


def test_selectivity_report_pass_when_operation_dominates():
    codes, labels, topic = _planted_codes()
    rep = selectivity_report(
        codes, labels, control_label_sets={"topic": topic}, seed=0
    )
    assert rep["acc_operation"] > 0.9
    # every control accuracy is below operation, so every selectivity gap is positive
    assert all(gap > 0 for gap in rep["selectivity"].values())
    assert set(rep["controls"]) >= {"random", "topic", "geometry"}
    assert rep["verdict"] == "PASS"


# ---------------------------------------------------------------------------
# raw_embedding_report — ceiling check: probe raw pooled embeddings, no MoE
# ---------------------------------------------------------------------------


def test_raw_embedding_report_decodes_planted_operation_from_pooled_tokens():
    # Per-token embeddings whose masked-mean one-hots the operation class → a linear probe
    # on the POOLED raw embedding (no MoE) recovers the operation. This is the Stage-0
    # ceiling: "is operation even linearly present in the encoder output?".
    N, T, D = 90, 5, 6
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(3), N // 3)
    tokens = 0.05 * rng.standard_normal((N, T, D)).astype(np.float32)
    for i, c in enumerate(labels):
        tokens[i, :, c] += 1.0  # every active token carries the class signal
    mask = np.ones((N, T), dtype=np.int8)
    topic = rng.integers(0, 3, size=N)  # independent of the planted signal

    rep = raw_embedding_report(tokens, mask, labels, control_label_sets={"topic": topic}, seed=0)
    assert rep["acc_operation"] > 0.9
    assert set(rep["controls"]) >= {"random", "topic", "geometry"}
    assert rep["verdict"] in ("PASS", "FAIL")


def test_raw_embedding_report_at_chance_when_operation_absent_from_embeddings():
    # Pooled embedding carries NO operation signal → ceiling is chance → adj_operation ≈ 0.
    N, T, D = 90, 5, 6
    rng = np.random.default_rng(1)
    labels = np.repeat(np.arange(3), N // 3)
    tokens = rng.standard_normal((N, T, D)).astype(np.float32)  # pure noise, no class signal
    mask = np.ones((N, T), dtype=np.int8)
    rep = raw_embedding_report(tokens, mask, labels, seed=0)
    assert rep["adj_operation"] < 0.3  # no genuine operation structure recoverable


# ---------------------------------------------------------------------------
# A2 — probe_accuracy is robust to singleton classes / degenerate labels
# ---------------------------------------------------------------------------


def test_probe_accuracy_singleton_class_does_not_crash():
    codes, labels, _ = _planted_codes()
    labels = labels.copy()
    labels[0] = 99  # one example forms a brand-new singleton class
    acc = probe_accuracy(codes, labels, seed=0)  # must fall back to unstratified split
    assert 0.0 <= acc <= 1.0


def test_probe_accuracy_raises_on_single_class():
    codes, _, _ = _planted_codes()
    with pytest.raises(ValueError):
        probe_accuracy(codes, np.zeros(codes.shape[0], dtype=int), seed=0)


# ---------------------------------------------------------------------------
# B3 — sequence_code agg modes; max recovers a few-token operation signal
# ---------------------------------------------------------------------------


def test_sequence_code_max_and_meanmax_shapes():
    alpha = torch.rand(4, 7, 3)
    mask = torch.ones(4, 7, dtype=torch.long)
    assert sequence_code(alpha, mask, agg="max").shape == (4, 3)
    assert sequence_code(alpha, mask, agg="meanmax").shape == (4, 6)


def test_sequence_code_max_recovers_sparse_signal_that_mean_dilutes():
    # One salient token (t=0) carries the operation spike on expert 1; the other 19 tokens
    # fire expert 0 (generic). The mean dilutes the spike ~1/20; the max keeps it.
    N, T, K = 30, 20, 2
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1], N // 2)
    alpha = np.zeros((N, T, K), dtype=np.float32)
    alpha[:, 1:, 0] = 1.0                         # generic expert on all non-pivot tokens
    alpha[labels == 1, 0, 1] = 1.0                # pivot spike only for class 1
    mask = np.ones((N, T), dtype=np.int8)
    mean_acc = probe_accuracy(sequence_code(alpha, mask, agg="mean"), labels, seed=0)
    max_acc = probe_accuracy(sequence_code(alpha, mask, agg="max"), labels, seed=0)
    assert max_acc >= mean_acc  # max preserves the operation signal at least as well


# ---------------------------------------------------------------------------
# B4 — chance normalization, degenerate-control skip, topic-gated PASS
# ---------------------------------------------------------------------------


def test_adjusted_accuracy_normalizes_for_cardinality():
    # raw 0.50 on a 50-class control (chance .02) is FAR more selective than raw 0.60 on a
    # 2-class control (chance .50): adjusted ordering must reflect that.
    many = np.repeat(np.arange(50), 4)   # 50 classes, chance ~0.02
    two = np.repeat([0, 1], 100)         # 2 classes, chance ~0.50
    assert _adjusted_accuracy(0.50, many) > _adjusted_accuracy(0.60, two)
    assert chance_rate(two) == pytest.approx(0.5)


def test_selectivity_skips_degenerate_control_with_warning():
    codes, labels, _ = _planted_codes()
    degenerate = np.zeros(codes.shape[0], dtype=int)  # 1-class token-type bucket
    rep = selectivity_report(
        codes, labels,
        control_label_sets={"topic": np.random.default_rng(0).integers(0, 3, codes.shape[0]),
                            "token_type": degenerate},
        seed=0,
    )
    assert "token_type" not in rep["controls"]        # skipped, not scored as trivial 1.0
    assert any("token_type" in w for w in rep["warnings"])


def test_selectivity_withholds_pass_when_topic_control_absent():
    # Strong operation codes, but the only supplied real control is degenerate → no usable
    # topic control → PASS must be withheld (can't green-light without topic).
    codes, labels, _ = _planted_codes()
    rep = selectivity_report(
        codes, labels,
        control_label_sets={"token_type": np.zeros(codes.shape[0], dtype=int)},
        seed=0,
    )
    assert rep["verdict"] == "FAIL"
    assert any("topic" in w for w in rep["warnings"])
