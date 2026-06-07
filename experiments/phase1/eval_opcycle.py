"""Engine-A selectivity eval — the go/no-go gate (ENGINE_A_DESIGN §5).

Hewitt-Liang (2019) control-task logic: a representation is *selective* for a property
only if a linear probe decodes that property far better than it decodes a control. Here
the property is the operation (reasoning-type) and the sequence code is `mean_t α`
(per-token ReMoE gates, masked-averaged). We compare:

    acc(operation)   — probe code → reasoning-type label
  vs four controls
    ① random   — probe code → shuffled labels        (Hewitt-Liang control task)
    ② topic    — probe code → passage/topic id        (is it just topic?)
    ③ token    — probe code → token-type/POS label    (is it just surface form?)
    ④ geometry — probe (structure-destroyed code) → operation  (artifact baseline)

    selectivity_c = acc(operation) − acc(control_c)

PASS (go) iff acc(operation) exceeds every control (esp. topic & geometry). FAIL → the
experts are not isolating operations and the direction is reconsidered.

LLM-free: works on cached codes + labels only.
"""

from __future__ import annotations

import numpy as np

try:  # torch is optional here — sequence_code accepts tensors or arrays
    import torch
except Exception:  # pragma: no cover
    torch = None


# ----- sequence code: masked mean of per-token gates ---------------------------------

def sequence_code(alpha, mask, agg: str = "mean") -> np.ndarray:
    """Sequence-level operation code from per-token gates α, over active tokens.

    alpha: (N, T, K). mask: (N, T) 1=active. agg:
      "mean"    → mean_t α  (N, K)        — default; can wash out a few-token signal
      "max"     → max_t α   (N, K)        — preserves a salient-token operation spike
      "meanmax" → [mean ‖ max]  (N, 2K)   — both; recommended for the gate

    Why "max"/"meanmax": the per-token pivot exists because pooling collapses operations
    into topic. A plain mean re-pools and can dilute an operation expressed in a handful of
    tokens (3/60) under topic-driven mass; the per-token max keeps that spike legible
    (ENGINE_A_DESIGN §4). Returns np.float32.
    """
    if torch is not None and isinstance(alpha, torch.Tensor):
        alpha = alpha.detach().cpu().numpy()
    if torch is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    alpha = np.asarray(alpha, dtype=np.float32)
    m = np.asarray(mask, dtype=np.float32)[..., None]          # (N, T, 1)
    mean = (alpha * m).sum(axis=1) / m.sum(axis=1).clip(min=1.0)   # (N, K)
    if agg == "mean":
        return mean
    # masked max: gates are ≥0, so set padded positions to -inf before max, then any
    # all-padded row (max == -inf) falls back to 0.
    masked = np.where(m.astype(bool), alpha, -np.inf)
    mx = masked.max(axis=1)                                    # (N, K)
    mx = np.where(np.isfinite(mx), mx, 0.0).astype(np.float32)
    if agg == "max":
        return mx
    if agg == "meanmax":
        return np.concatenate([mean, mx], axis=1)             # (N, 2K)
    raise ValueError(f"unknown sequence_code agg: {agg!r}")


# ----- linear probe accuracy (Hewitt-Liang probe) ------------------------------------

def chance_rate(labels) -> float:
    """Majority-class baseline accuracy (a probe that always predicts the top class)."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    return float(counts.max() / counts.sum())


def probe_accuracy(codes, labels, seed: int = 0, test_size: float = 0.3) -> float:
    """Held-out accuracy of a linear (logistic-regression) probe code → label.

    A single train/test split keeps it deterministic and cheap; the probe is deliberately
    linear so accuracy reflects *linear* decodability of the property, per Hewitt-Liang.
    Features are standardised so scale doesn't dominate.

    Stratification is used only when every class has ≥2 members (otherwise sklearn raises
    on the realistic case of a singleton reasoning-type / passage id); it falls back to an
    unstratified split there. Raises ValueError if fewer than 2 classes are present (a
    degenerate label set the caller must handle, not silently score as trivially 1.0).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    codes = np.asarray(codes, dtype=np.float64)
    labels = np.asarray(labels)
    uniq, counts = np.unique(labels, return_counts=True)
    if uniq.size < 2:
        raise ValueError(
            f"probe_accuracy needs ≥2 classes, got {uniq.size} (degenerate label set)"
        )
    strat = labels if counts.min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        codes, labels, test_size=test_size, random_state=seed, stratify=strat
    )
    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(scaler.transform(X_tr), y_tr)
    return float(clf.score(scaler.transform(X_te), y_te))


# ----- geometry baseline (artifact control) ------------------------------------------

def geometry_baseline(codes, mode: str = "shuffle", seed: int = 0) -> np.ndarray:
    """Destroy the code↔example structure while preserving per-feature statistics.

    mode="shuffle": permute each feature column independently across examples, so a
    sample's code becomes a Frankenstein of unrelated feature values and the joint
    feature↔label structure is gone → a linear probe drops to chance. This isolates
    whether operation accuracy is genuine structure vs. a geometry/dimensionality
    artifact. mode="rotation" applies a random orthogonal rotation (note: a linear probe
    is rotation-invariant, so this is the *weak* control — kept for completeness).
    """
    codes = np.asarray(codes, dtype=np.float64).copy()
    rng = np.random.default_rng(seed)
    if mode == "shuffle":
        for j in range(codes.shape[1]):
            codes[:, j] = codes[rng.permutation(codes.shape[0]), j]
        return codes
    if mode == "rotation":
        d = codes.shape[1]
        q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        return codes @ q
    raise ValueError(f"unknown geometry baseline mode: {mode!r}")


# ----- selectivity report (the verdict) ----------------------------------------------

def _adjusted_accuracy(acc: float, labels) -> float:
    """Chance-normalised accuracy (acc − chance)/(1 − chance), clamped ≥ 0.

    Needed because controls can have very different #classes than the operation: a 50-way
    topic control (chance ≈ 2%) and a 4-way operation (chance ≈ 25%) are not comparable by
    raw accuracy — raw acc(op) can exceed acc(topic) purely from the higher chance floor
    even when the code is *more* topic-selective. Normalising to "fraction of the headroom
    above chance" makes the selectivity gap meaningful across cardinalities.
    """
    chance = chance_rate(labels)
    denom = 1.0 - chance
    return max(0.0, (acc - chance) / denom) if denom > 1e-9 else 0.0


def raw_embedding_report(
    tokens,
    mask,
    operation_labels,
    control_label_sets: dict | None = None,
    agg: str = "meanmax",
    seed: int = 0,
    margin: float = 0.0,
) -> dict:
    """Stage-0 ceiling check: run the selectivity gate directly on the RAW frozen-encoder
    per-token embeddings (no MoE), to test whether the operation is even linearly present
    in the encoder output at all.

    tokens (N, T, d), mask (N, T) → pooled to a sequence code via `sequence_code(agg)` (same
    masked mean/max used for MoE gates, here applied to the raw embedding), then handed to
    `selectivity_report`. If operation isn't decodable here, no downstream cycle can isolate
    it (the failure is the encoder/target, not the routing). The resulting acc(operation) is
    also the *ceiling* the MoE codes are expected to approach. Returns the same dict shape as
    `selectivity_report`.
    """
    pooled = sequence_code(tokens, mask, agg=agg)
    return selectivity_report(
        pooled, operation_labels, control_label_sets=control_label_sets,
        seed=seed, margin=margin,
    )


def selectivity_report(
    codes,
    operation_labels,
    control_label_sets: dict | None = None,
    seed: int = 0,
    margin: float = 0.0,
) -> dict:
    """Compute chance-normalised acc(operation), the control accuracies, selectivity gaps,
    and the go/no-go verdict.

    control_label_sets: optional {name: labels} for real-attribute controls (e.g.
    {"topic": passage_ids, "token": pos_majority}). The `random` and `geometry` controls
    are constructed automatically. A control whose label set is degenerate (<2 classes —
    e.g. token-length buckets that all collapse to the truncation cap) is skipped and noted
    in `warnings` rather than scored as a trivially-perfect control that forces a false FAIL.

    Verdict is PASS iff the chance-adjusted operation accuracy exceeds every (non-skipped)
    control by > margin AND both core controls (topic, geometry) were actually evaluated —
    so omitting the topic control can't manufacture a PASS.
    """
    codes = np.asarray(codes, dtype=np.float64)
    operation_labels = np.asarray(operation_labels)
    rng = np.random.default_rng(seed)
    warnings: list[str] = []

    acc_operation = probe_accuracy(codes, operation_labels, seed=seed)
    adj_operation = _adjusted_accuracy(acc_operation, operation_labels)

    controls: dict[str, float] = {}
    adj_controls: dict[str, float] = {}

    def _add(name: str, probe_codes, probe_labels):
        if np.unique(np.asarray(probe_labels)).size < 2:
            warnings.append(f"control '{name}' skipped: <2 classes (degenerate)")
            return
        acc = probe_accuracy(probe_codes, probe_labels, seed=seed)
        controls[name] = acc
        adj_controls[name] = _adjusted_accuracy(acc, probe_labels)

    # ① random-type — Hewitt-Liang control task (shuffled labels)
    _add("random", codes, rng.permutation(operation_labels))
    # ② / ③ caller-supplied real attributes (topic, token-type, …)
    for name, labels in (control_label_sets or {}).items():
        _add(name, codes, np.asarray(labels))
    # ④ geometry — operation labels on a structure-destroyed code
    _add("geometry", geometry_baseline(codes, mode="shuffle", seed=seed), operation_labels)

    selectivity = {name: adj_operation - adj for name, adj in adj_controls.items()}
    beats_all = all(gap > margin for gap in selectivity.values()) and len(selectivity) > 0
    # core controls that must be present for a trustworthy PASS
    core_present = "geometry" in adj_controls and (
        "topic" in adj_controls or not control_label_sets
    )
    if control_label_sets and "topic" not in adj_controls:
        warnings.append("no usable 'topic' control → PASS withheld")
    verdict = "PASS" if (beats_all and core_present) else "FAIL"

    return {
        "acc_operation": acc_operation,
        "adj_operation": adj_operation,
        "controls": controls,
        "adj_controls": adj_controls,
        "selectivity": selectivity,
        "warnings": warnings,
        "verdict": verdict,
    }
