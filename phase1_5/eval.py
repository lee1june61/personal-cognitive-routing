"""Phase 1.5 evaluation — 4-control selectivity probe + 1σ bootstrap gate.

The primitives ``sequence_code``, ``chance_rate``, ``probe_accuracy``,
``geometry_baseline``, ``_adjusted_accuracy``, ``raw_embedding_report``, and
``selectivity_report`` are **verbatim copies** of ``phase1/eval_opcycle.py``
(120 passed tests under phase1 — see ``phase1/tests/test_opcycle_eval.py``).
Phase 1.5 adds two adapters:

- ``make_phase15_controls`` — builds the topic (TF-IDF KMeans 9-cluster) and
  token (POS-majority via NLTK) labels per paper §8.3 pre-flight Gap D commit.
- ``selectivity_gate_phase15`` — applies a 1σ-bootstrap (n_boot=200) margin to
  ``selectivity_report`` per paper §7.4 / §8.3 1a PASS rule.

Both adapters are read-only over the probe data; they do not modify the underlying
hardened primitives.
"""

from __future__ import annotations

import numpy as np

try:  # torch is optional for primitives — accepts tensors or arrays
    import torch
except Exception:  # pragma: no cover
    torch = None


# =====================================================================================
# === phase1 hardened primitives (verbatim copy from phase1/eval_opcycle.py) ==========
# =====================================================================================


def sequence_code(alpha, mask, agg: str = "mean") -> np.ndarray:
    """Sequence-level operation code from per-token gates α, over active tokens.

    alpha: (N, T, K). mask: (N, T) 1=active. agg:
      "mean"    → mean_t α  (N, K)        — default; can wash out a few-token signal
      "max"     → max_t α   (N, K)        — preserves a salient-token operation spike
      "meanmax" → [mean ‖ max]  (N, 2K)   — both; recommended for the gate
    """
    if torch is not None and isinstance(alpha, torch.Tensor):
        alpha = alpha.detach().cpu().numpy()
    if torch is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    alpha = np.asarray(alpha, dtype=np.float32)
    m = np.asarray(mask, dtype=np.float32)[..., None]  # (N, T, 1)
    mean = (alpha * m).sum(axis=1) / m.sum(axis=1).clip(min=1.0)
    if agg == "mean":
        return mean
    masked = np.where(m.astype(bool), alpha, -np.inf)
    mx = masked.max(axis=1)
    mx = np.where(np.isfinite(mx), mx, 0.0).astype(np.float32)
    if agg == "max":
        return mx
    if agg == "meanmax":
        return np.concatenate([mean, mx], axis=1)
    raise ValueError(f"unknown sequence_code agg: {agg!r}")


def chance_rate(labels) -> float:
    """Majority-class baseline accuracy."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    return float(counts.max() / counts.sum())


def probe_accuracy(codes, labels, seed: int = 0, test_size: float = 0.3) -> float:
    """Held-out accuracy of a linear (logistic-regression) probe code → label."""
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


def geometry_baseline(codes, mode: str = "shuffle", seed: int = 0) -> np.ndarray:
    """Destroy code↔example structure while preserving per-feature statistics."""
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


def _adjusted_accuracy(acc: float, labels) -> float:
    """Chance-normalised accuracy (acc − chance)/(1 − chance), clamped ≥ 0."""
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
    """Stage-0 ceiling check — run the gate directly on raw frozen-encoder tokens."""
    pooled = sequence_code(tokens, mask, agg=agg)
    return selectivity_report(
        pooled,
        operation_labels,
        control_label_sets=control_label_sets,
        seed=seed,
        margin=margin,
    )


def selectivity_report(
    codes,
    operation_labels,
    control_label_sets: dict | None = None,
    seed: int = 0,
    margin: float = 0.0,
) -> dict:
    """Chance-adjusted operation vs control verdict (Hewitt-Liang)."""
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

    _add("random", codes, rng.permutation(operation_labels))
    for name, labels in (control_label_sets or {}).items():
        _add(name, codes, np.asarray(labels))
    _add("geometry", geometry_baseline(codes, mode="shuffle", seed=seed), operation_labels)

    selectivity = {name: adj_operation - adj for name, adj in adj_controls.items()}
    beats_all = all(gap > margin for gap in selectivity.values()) and len(selectivity) > 0
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


# =====================================================================================
# === phase1_5 adapters ===============================================================
# =====================================================================================


DROP_LABELS = ("other", "unknown")


def build_operation_labels(
    labels_raw, *, min_count: int = 20, drop_labels: tuple[str, ...] = DROP_LABELS
) -> tuple[np.ndarray, np.ndarray]:
    """Filter a PRECOMPUTED per-row operation-label array to the *classifiable*
    multi-class subset (source-agnostic — LSAT regex axis for LogiQA/ReClor, hop
    axis for MuSiQue, etc.). Two filters:

      - drop ``drop_labels`` sentinels (``"other"`` = LSAT regex unclassified,
        ``"unknown"`` = MuSiQue hop unparsed — not operation classes), and
      - drop classes with fewer than ``min_count`` members (too sparse to probe).

    Returns ``(labels, keep_mask)`` where ``keep_mask`` is a boolean over the input
    rows selecting the retained ones (caller subsets ``codes`` / controls with it)
    and ``labels`` are the operation labels for those rows.

    Raises ``ValueError`` if fewer than 2 classes survive — selectivity is then
    undefined (the caller should treat it as "axis unmeasurable", not a negative).
    """
    from collections import Counter

    raw = np.asarray(labels_raw, dtype=object)
    drop = set(drop_labels)
    counts = Counter(r for r in raw.tolist() if r not in drop)
    keep_classes = {c for c, n in counts.items() if n >= min_count}
    keep_mask = np.array([r in keep_classes for r in raw], dtype=bool)
    labels = raw[keep_mask]
    if np.unique(labels).size < 2:
        raise ValueError(
            f"build_operation_labels: <2 classes survive (min_count={min_count}); "
            f"got {sorted(set(labels.tolist()))} — operation axis unmeasurable on this probe"
        )
    return labels, keep_mask


def operation_consistency(
    codes,
    operation_labels,
    topic_labels=None,
    *,
    k: int = 10,
    seed: int = 0,
) -> dict:
    """Granularity-robust routing-consistency: kNN label purity in code space.

    Complements ``selectivity_report`` (a *linear-separability / alignment* test
    that can read FAIL when operations emerge on an axis finer than — or
    orthogonal to — the coarse label). This asks the weaker, granularity-robust
    question: are an item's nearest neighbours in code space *consistently* the
    same operation, more than chance, and more than the same purity for topic?

    ``op_purity`` = mean fraction of each item's k nearest neighbours (cosine,
    self excluded) sharing its operation label. ``op_chance`` = Σ(n_c/N)² (the
    purity expected from random neighbours). ``op_beats_topic`` controls for
    content: operation routing is only credible if it clusters *tighter* than
    topic. ``seed`` is accepted for interface symmetry (kNN is deterministic).
    """
    from sklearn.neighbors import NearestNeighbors

    codes = np.asarray(codes, dtype=np.float64)
    op = np.asarray(operation_labels)
    n = op.shape[0]
    k_eff = max(1, min(k, n - 1))
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine").fit(codes)
    _, idx = nn.kneighbors(codes)
    idx = idx[:, 1:]  # drop self (nearest is the point itself)

    def _purity(labels) -> tuple[float, float]:
        labels = np.asarray(labels)
        hits = float((labels[idx] == labels[:, None]).mean())
        _, counts = np.unique(labels, return_counts=True)
        chance = float(((counts / counts.sum()) ** 2).sum())
        return hits, chance

    op_purity, op_chance = _purity(op)
    out = {
        "op_purity": op_purity,
        "op_chance": op_chance,
        "op_above_chance": op_purity > op_chance,
        "k": k_eff,
    }
    if topic_labels is not None:
        topic_purity, topic_chance = _purity(topic_labels)
        out.update(
            {
                "topic_purity": topic_purity,
                "topic_chance": topic_chance,
                "op_beats_topic": op_purity > topic_purity,
            }
        )
    return out


def regate_operation_selectivity(
    codes,
    questions: list[str],
    passages: list[str],
    *,
    op_labels=None,
    min_count: int = 20,
    margin_sigma: float = 1.0,
    n_boot: int = 200,
    seed: int = 0,
) -> dict:
    """Run the operation-selectivity gate on the *operation-classifiable* subset.

    Decouples expensive routing (the trained model's α → ``codes``) from the cheap
    operation-label axis, so the axis can be re-defined offline without retraining.

    ``op_labels`` is the precomputed per-row operation axis. When ``None`` it is
    derived from the questions via the LSAT-stem regex (``infer_reasoning_type`` —
    the LogiQA/ReClor control-arm default); MuSiQue callers pass the hop axis
    (``corpus["reasoning_type"]``). Steps: ``build_operation_labels`` (drop
    sentinels/rare) → subset ``codes`` + rebuild topic/token controls on the kept
    rows → ``selectivity_gate_phase15``.

    ``codes`` is the already-pooled sequence code (N, 2K) from ``sequence_code``.
    Returns the gate dict augmented with ``n_operation_examples`` and
    ``operation_classes``. Propagates ``build_operation_labels``'s ValueError when
    <2 classes survive (axis unmeasurable — not a negative result).
    """
    codes = np.asarray(codes, dtype=np.float64)
    if op_labels is None:
        from .data import infer_reasoning_type

        op_labels = [infer_reasoning_type(q) for q in questions]
    labels, keep = build_operation_labels(op_labels, min_count=min_count)
    codes_k = codes[keep]
    q_k = [q for q, k in zip(questions, keep) if k]
    p_k = [p for p, k in zip(passages, keep) if k]
    controls = make_phase15_controls(q_k, p_k, seed=seed)
    gate = selectivity_gate_phase15(
        codes_k, labels, controls, margin_sigma=margin_sigma, n_boot=n_boot, seed=seed
    )
    gate["n_operation_examples"] = int(keep.sum())
    gate["operation_classes"] = sorted(set(labels.tolist()))
    # Granularity-robust companion to the linear-probe gate (catches operation
    # structure finer than / orthogonal to the linear-separability test).
    gate["consistency"] = operation_consistency(
        codes_k, labels, controls.get("topic"), seed=seed
    )
    return gate


def _tfidf_kmeans_labels(texts: list[str], n_clusters: int, seed: int) -> np.ndarray:
    """TF-IDF → KMeans cluster labels with a single-cluster fallback on degenerate
    input. Shared by the ``topic`` (q+p) and ``stem`` (q-only) controls."""
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    n = len(texts)
    labels = np.zeros(n, dtype=np.int32)
    vec = TfidfVectorizer(max_features=2048, stop_words="english")
    try:
        tfidf = vec.fit_transform(texts)
        # n_clusters must satisfy 2 ≤ k ≤ n_samples (KMeans constraint).
        k = min(n_clusters, max(2, n // 4))
        if k >= 2 and tfidf.shape[1] > 0 and n >= k:
            km = KMeans(n_clusters=k, random_state=seed, n_init=4)
            labels = km.fit_predict(tfidf).astype(np.int32)
    except (ValueError, MemoryError) as e:
        # ValueError = degenerate corpus (n=1, all-stopword). MemoryError = very
        # large vocab. Both fall back to a single cluster (gate logs it degenerate).
        print(f"[controls] TF-IDF/KMeans failed ({type(e).__name__}); single-cluster fallback")
    return labels


def make_phase15_controls(
    probe_questions: list[str],
    probe_passages: list[str],
    *,
    topic_n_clusters: int = 9,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Build the topic + token control label sets per paper §8.3 Gap D commit.

    - ``topic`` = KMeans cluster over TF-IDF features of ``question + " " + passage``,
      ``n_clusters = topic_n_clusters`` (paper §8.3: 9-cluster).
    - ``token`` = POS-majority bucket over the question (NLTK
      ``averaged_perceptron_tagger``). Bucket = first letter of the majority POS tag
      (e.g. "N" for noun-dominated, "V" for verb-dominated). NLTK falls back to all-N
      if tagger unavailable.

    The ``random`` and ``geometry`` controls are auto-constructed by
    ``selectivity_report``; this returns ``topic`` (q+p), ``stem`` (q-only), and
    ``token``.
    """
    qp_texts = [f"{q} {p}" for q, p in zip(probe_questions, probe_passages)]
    topic = _tfidf_kmeans_labels(qp_texts, topic_n_clusters, seed)
    # stem = clustering on the QUESTION only → invariant to the passage. Lets the
    # gate test whether routing beats surface question-phrasing, not just q+p
    # topic. NOTE: bounds (does not fully resolve) the stem-lexical confound — the
    # regex operation label is itself stem-derived, so adj_stem and adj_operation
    # are correlated by construction.
    stem = _tfidf_kmeans_labels(list(probe_questions), topic_n_clusters, seed)
    token = _pos_majority(probe_questions)

    return {"topic": topic, "token": token, "stem": stem}


def _pos_majority(texts: list[str]) -> np.ndarray:
    """Per-text POS-majority bucket. Returns first letter of the majority POS tag."""
    try:
        import nltk

        try:
            nltk.data.find("taggers/averaged_perceptron_tagger_eng")
            tagger_id = "averaged_perceptron_tagger_eng"
        except LookupError:
            try:
                nltk.data.find("taggers/averaged_perceptron_tagger")
                tagger_id = "averaged_perceptron_tagger"
            except LookupError:
                nltk.download("averaged_perceptron_tagger_eng", quiet=True)
                tagger_id = "averaged_perceptron_tagger_eng"
        _ = tagger_id  # silence unused-var if both tags resolved
    except Exception:
        return np.array(["N"] * len(texts), dtype=object)

    out: list[str] = []
    from collections import Counter

    for t in texts:
        try:
            tokens = nltk.word_tokenize(t)
            tags = nltk.pos_tag(tokens)
            if not tags:
                out.append("N")
                continue
            tag_letters = [tag[:1] for _, tag in tags if tag]
            if not tag_letters:
                out.append("N")
                continue
            top = Counter(tag_letters).most_common(1)[0][0]
            out.append(top)
        except Exception:
            out.append("N")
    return np.array(out, dtype=object)


def selectivity_gate_phase15(
    codes: np.ndarray,
    operation_labels: np.ndarray,
    controls: dict[str, np.ndarray],
    *,
    margin_sigma: float = 1.0,
    n_boot: int = 200,
    seed: int = 0,
) -> dict:
    """Apply the 1a PASS gate: ``adj_op > max(adj_controls) + margin_sigma · σ_bootstrap``.

    σ_bootstrap is the std of ``adj_operation`` across ``n_boot`` probe-set resamples
    (rows of ``codes`` + ``operation_labels`` sampled with replacement). The base
    ``selectivity_report`` provides the point estimates; this adapter adds the
    σ-margin layer on top.

    Returns the ``selectivity_report`` dict augmented with:
      ``sigma_adj_operation``, ``margin``, ``passes_sigma_gate``, ``threshold``.
    """
    base = selectivity_report(
        codes,
        operation_labels,
        control_label_sets=controls,
        seed=seed,
        margin=0.0,  # we apply our own σ-margin
    )

    rng = np.random.default_rng(seed)
    n = codes.shape[0]
    boot_adj: list[float] = []
    n_dropped = 0
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            acc_b = probe_accuracy(codes[idx], operation_labels[idx], seed=seed + b + 1)
        except ValueError:
            # resample dropped a class → skip
            n_dropped += 1
            continue
        boot_adj.append(_adjusted_accuracy(acc_b, operation_labels[idx]))
    sigma = float(np.std(boot_adj)) if boot_adj else 0.0

    adj_controls = base["adj_controls"]
    max_adj_control = max(adj_controls.values()) if adj_controls else 0.0
    threshold = max_adj_control + margin_sigma * sigma
    passes_sigma_gate = base["adj_operation"] > threshold

    base.update(
        {
            "sigma_adj_operation": sigma,
            "margin": margin_sigma * sigma,
            "threshold": threshold,
            "passes_sigma_gate": passes_sigma_gate,
            "n_boot": n_boot,
            "n_boot_dropped": n_dropped,
            "controls_applied_to_threshold": sorted(adj_controls.keys()),
        }
    )

    # Warn when the σ-margin effectively disappears (empty bootstrap or all
    # resamples dropped a class) — the gate then degrades to a raw "beat max
    # control" check with no statistical margin.
    if not boot_adj:
        base["warnings"].append(
            f"bootstrap dropped all {n_boot} resamples (degenerate label set); "
            f"σ=0 effectively disables the {margin_sigma}σ margin"
        )
    elif n_dropped > n_boot // 2:
        base["warnings"].append(
            f"bootstrap dropped {n_dropped}/{n_boot} resamples; σ may underestimate "
            f"variance and the {margin_sigma}σ margin may be too tight"
        )

    # Warn when controls were silently skipped — the threshold then doesn't
    # include them, so the gate's PASS strength is overstated.
    expected_controls = set(controls.keys()) | {"random", "geometry"}
    missing = expected_controls - set(adj_controls.keys())
    if missing:
        base["warnings"].append(
            f"controls absent from σ-threshold: {sorted(missing)} (degenerate "
            f"label set or <2 classes); PASS would not have been tested against them"
        )

    # Verdict resolution:
    # - PASS in the inner report + σ-gate PASS → keep PASS.
    # - PASS in the inner report + σ-gate FAIL → downgrade to FAIL.
    # - FAIL in the inner report → keep FAIL regardless of σ-gate (the inner
    #   verdict already accounts for degenerate-control bookkeeping and missing
    #   core controls; we don't upgrade FAIL→PASS via a numerically-favourable σ).
    if base["verdict"] == "PASS" and not passes_sigma_gate:
        base["verdict"] = "FAIL"
        base["warnings"].append(
            f"adj_operation={base['adj_operation']:.4f} did not exceed "
            f"threshold={threshold:.4f} (margin_sigma={margin_sigma}, σ={sigma:.4f})"
        )
    elif base["verdict"] == "FAIL" and passes_sigma_gate:
        # Don't upgrade — but make the mismatch explicit so summary CSVs can flag.
        base["warnings"].append(
            f"σ-gate PASSed (adj_op={base['adj_operation']:.4f} > threshold="
            f"{threshold:.4f}) but inner verdict=FAIL (degenerate controls or "
            f"missing core 'topic'/'geometry'); verdict kept FAIL"
        )
    return base
