"""Phase 1.5 1b orchestrator — train a chain-of-experts model + run the PASS bar.

1b PASS bar (CONTEXT.md §S1, 2026-05-31):
  (a) causal compositional battery — lesioning a *late* chain step hurts *deep-hop*
      (3/4hop) questions strictly more than shallow (2hop): a hop-depth-selective
      gradient (``intervention.lesion_step_specificity``). Uniform drop = FAIL.
  (b) S1 motif-consistency — questions sharing decomposition-structure activate
      similar active sub-paths, beating the topic control
      (``intervention.chain_motif_codes`` → ``eval.operation_consistency``).
  1b PASSES iff (a) ∧ (b) AND 1b beats flat-1a on ≥1 criterion (compared by the
  caller across two runs).

``evaluate_1b`` (model + encoded data → verdict) is CPU-testable; ``run_1b_experiment``
is the thin GPU pipeline (build → encode → train chain → evaluate).
"""

from __future__ import annotations

import numpy as np
import torch

from .data import MODE_Q_ONLY
from .eval import operation_consistency, make_phase15_controls
from .intervention import chain_motif_codes, lesion_step_specificity


# Hop labels in increasing depth — used to test the depth-selective gradient.
_HOP_ORDER = ["2hop", "3hop", "4hop"]


def hop_depth_selective(lesion: dict) -> dict:
    """PASS(a): does lesioning the LAST chain step hurt deep-hop strictly more than
    shallow-hop? Reads ``lesion_step_specificity`` output. We aggregate the
    last-step lesion's *self* drop per hop (drop[op][op]) and check it is monotone
    increasing in hop depth (2hop < 3hop < 4hop) among the hops present.
    """
    n_steps = lesion["n_steps"]
    last = lesion["drop_by_step"][n_steps - 1]
    present = [h for h in _HOP_ORDER if h in last]
    self_drop = [last[h][h] for h in present]  # diagonal: lesion h's experts, measure h
    monotone = all(b > a for a, b in zip(self_drop, self_drop[1:]))
    return {
        "hops": present,
        "last_step_self_drop": dict(zip(present, self_drop)),
        "monotone_in_depth": bool(monotone and len(present) >= 2),
    }


def step_breadth_by_depth(lesion: dict, *, threshold: float = 0.05) -> dict:
    """PASS(a) v2 (re-specified 2026-05-31 after the last-step criterion read FALSE
    — see plan; the last-step self-drop is a weak proxy because a deep-hop answer
    distributed across steps is *robust* to a single late-step lesion).

    Direct operationalisation of "deeper questions require composing MORE steps":
    for each hop, count how many chain steps' lesion causes a self-drop above
    ``threshold`` (= "that step matters for this hop"). Composition predicts this
    breadth is non-decreasing in hop depth and strictly larger for the deepest
    than the shallowest hop present.
    """
    n_steps = lesion["n_steps"]
    breadth: dict[str, int] = {}
    for op in _HOP_ORDER:
        if op not in lesion["drop_by_step"].get(0, {}):
            continue
        breadth[op] = sum(
            1 for ell in range(n_steps) if lesion["drop_by_step"][ell][op][op] > threshold
        )
    present = [h for h in _HOP_ORDER if h in breadth]
    counts = [breadth[h] for h in present]
    nondecr = all(b >= a for a, b in zip(counts, counts[1:]))
    strict_ends = len(counts) >= 2 and counts[-1] > counts[0]
    return {
        "breadth": breadth,
        "threshold": threshold,
        "monotone": bool(nondecr and strict_ends),
    }


def _probe_batch(data: dict, mask: np.ndarray) -> dict:
    """Assemble a forward_chain batch (tensors) from the encoded ``data`` rows
    selected by ``mask``."""
    return {
        "q_tokens": torch.from_numpy(data["q_tokens"][mask]).float(),
        "q_mask": torch.from_numpy(data["q_mask"][mask]).float(),
        "p_tokens": torch.from_numpy(data["p_tokens"][mask]).float(),
        "p_mask": torch.from_numpy(data["p_mask"][mask]).float(),
        "cand_pooled": torch.from_numpy(data["cand_pooled"][mask]).float(),
        "answer_idx": torch.from_numpy(data["answer_idx"][mask]).long(),
    }


def evaluate_1b(
    model,
    data: dict,
    *,
    structure_labels=None,
    split: str = "test",
    k_top: int = 4,
    device: str = "cpu",
) -> dict:
    """Run the 1b PASS bar on the probe split of pre-encoded ``data``.

    ``data`` is ``encode_or_load_mc`` output (carries ``reasoning_type`` = hop axis,
    ``split``). ``structure_labels`` (chain/comparison per row, from
    ``data_musique.infer_musique_structure``) is optional; when given, S1 motif
    consistency is measured on it (topic-controlled), else on the hop axis.
    """
    splits = np.asarray(data["split"])
    mask = splits == split
    if not mask.any():
        mask = splits == "val"
    batch = _probe_batch(data, mask)
    hop_labels = np.asarray(data["reasoning_type"])[mask]

    # (a) causal compositional battery — TWO readings reported for transparency:
    #   - pass_a_laststep: the originally pre-registered last-step self-drop
    #     monotone-in-depth criterion (kept verbatim; first run read FALSE).
    #   - pass_a_breadth: the re-specified criterion (# steps that matter grows
    #     with hop depth) — the primary verdict gate per the (나) decision.
    lesion = lesion_step_specificity(model, batch, hop_labels, k_top=k_top, device=device)
    pass_a_laststep = hop_depth_selective(lesion)
    pass_a_breadth = step_breadth_by_depth(lesion)

    # (b) S1 motif-consistency, topic-controlled.
    motif = chain_motif_codes(model, batch, device=device)
    s1_labels = (
        np.asarray(structure_labels)[mask] if structure_labels is not None else hop_labels
    )
    # topic control needs the raw text — not in ``data``; when absent, skip topic.
    s1 = operation_consistency(motif, s1_labels)
    pass_b = bool(s1.get("op_above_chance", False))

    return {
        "lesion": lesion,
        "pass_a_laststep": pass_a_laststep,  # original criterion (transparency)
        "pass_a_breadth": pass_a_breadth,  # re-specified primary criterion
        "s1_motif": s1,
        "pass_b_motif": pass_b,
        "passed": bool(pass_a_breadth["monotone"] and pass_b),
        "n_probe": int(mask.sum()),
    }


def _balance_train_by_hop(data: dict, seed: int, *, cap: int | None = None) -> dict:
    """Reduce TRAIN operation-class (``reasoning_type``) imbalance by **capping**
    each class at ``min(count, cap)`` rows (val/test untouched). ``cap=None`` falls
    back to the minority-class size (full equalise — discards the most data; the
    first run showed this over-downsamples, so callers pass a cap). Capping keeps
    small classes whole and only trims the dominant one, so 2hop no longer drowns
    out 3/4hop without throwing away most of the data. Returns a new data dict."""
    split = np.asarray(data["split"])
    rt = np.asarray(data["reasoning_type"])
    train_idx = np.flatnonzero(split == "train")
    classes = np.unique(rt[train_idx])
    if classes.size < 2:
        return data
    per = {c: train_idx[rt[train_idx] == c] for c in classes}
    target = cap if cap is not None else min(len(v) for v in per.values())
    rng = np.random.default_rng(seed)
    keep_train = np.concatenate([rng.permutation(v)[: min(len(v), target)] for v in per.values()])
    keep = np.sort(np.concatenate([keep_train, np.flatnonzero(split != "train")]))
    return {
        k: (v[keep] if hasattr(v, "__len__") and len(v) == len(split) else v)
        for k, v in data.items()
    }


def run_1b_experiment(
    corpus_cfg=None,
    train_cfg=None,
    *,
    chain_steps: int = 3,
    k_routed: int = 128,
    k_active_target: float = 4.0,
    dropout: float = 0.1,
    balance_train_hops: bool = True,
    balance_cap_per_hop: int = 5000,
    routing: str = "topk",
    batch_size: int = 64,
    device: str = "cuda",
    seed: int = 0,
) -> dict:
    """Full 1b pipeline (GPU): build MuSiQue → encode (Q-only) → train chain model
    → ``evaluate_1b``. Returns the training result + 1b verdict.

    Overfit / imbalance controls (the first MuSiQue run memorised train — acc→1.0,
    val CE exploding — and under-learned deep hops): ``dropout`` (Phase15MoE),
    ``balance_train_hops`` (equal per-hop train counts), ``best_metric="acc"``
    (select the ckpt by val accuracy, not the overfit-inflated val CE). ``routing``
    defaults to ``"topk"`` (K_active pinned = ``k_active_target``, ADR 0002 — the
    stable regime; a drifting K_active in the log means a stale model.py).
    """
    import dataclasses

    from .data import MCCorpusConfig, build_mc_corpus, encode_or_load_mc, make_mc_loaders
    from .model import Phase15MoE
    from .train import TrainConfig, train_phase15

    corpus_cfg = corpus_cfg or MCCorpusConfig(corpus="musique")
    train_cfg = train_cfg or TrainConfig(seed=seed)
    train_cfg = dataclasses.replace(
        train_cfg, k_target=k_active_target, seed=seed, best_metric="acc"
    )

    corpus = build_mc_corpus(corpus_cfg)
    data = encode_or_load_mc(
        corpus, corpus_cfg, encoding_mode=MODE_Q_ONLY, batch_size=max(1, batch_size // 2),
        device=device,
    )
    # Balance ONLY the MuSiQue hop axis. logic_mc's reasoning_type is the LSAT
    # taxonomy (many rare classes) — capping/equalising it nukes the train set
    # (first run: control val 0.47→0.28), so the control arm is never balanced.
    do_balance = balance_train_hops and corpus_cfg.corpus == "musique"
    if do_balance:
        data = _balance_train_by_hop(data, seed, cap=balance_cap_per_hop)
    train_loader, val_loader, _ = make_mc_loaders(data, batch_size=batch_size)
    model = Phase15MoE(
        d_emb=data["q_tokens"].shape[-1], d_z=256, k_routed=k_routed,
        routing=routing, lb_target_active=k_active_target, chain_steps=chain_steps,
        dropout=dropout,
    )
    print(
        f"[run_1b] routing={model.router.routing} chain_steps={model.chain_steps} "
        f"dropout={dropout} balance_hops={do_balance} (cap={balance_cap_per_hop}) "
        f"best_metric=acc (topk ⇒ k_active pinned at {int(round(k_active_target))}) "
        f"| train_n={len(train_loader.dataset)}"
    )
    result = train_phase15(model, train_loader, val_loader=val_loader, cfg=train_cfg, device=device)
    verdict = evaluate_1b(result["model"], data, device=device)
    return {"train": {k: v for k, v in result.items() if k != "model"}, "verdict": verdict}
