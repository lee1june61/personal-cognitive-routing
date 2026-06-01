"""Phase 1.5 1a orchestrator — Stage 0 ceiling check + training + 4-control gate.

Mirrors the shape of ``phase1/engine_a.py:run_engine_a`` but adapted for the
Phase 1.5 forward (Q/P-separated encoding + cross-attention modulation + MC head).

Returns a dict suitable for direct JSON dump under
``out/phase1_5/ablations/row_{row_id}_seed{N}.json`` via ``ablations.run_ablation_row``.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (
    MCCorpusConfig,
    MODE_Q_ONLY,
    build_mc_corpus,
    encode_or_load_mc,
    infer_reasoning_type,
    make_mc_loaders,
)
from .eval import (
    make_phase15_controls,
    raw_embedding_report,
    regate_operation_selectivity,
    selectivity_gate_phase15,
    sequence_code,
)
from .model import Phase15MoE
from .train import TrainConfig, _masked_token_mean, resolve_device, train_phase15

if TYPE_CHECKING:
    from .ablations import AblationRow


def run_engine_1a(
    *,
    ablation_row: "AblationRow",
    corpus_cfg: MCCorpusConfig | None = None,
    train_cfg: TrainConfig | None = None,
    batch_size: int = 64,
    device: str = "cuda",
    seed: int = 0,
    run_stage0: bool = True,
    progress: bool = False,
    out_dir: str | Path | None = None,
) -> dict:
    """End-to-end 1a run for one ablation row.

    Steps:
        1. Build corpus + encode (Q-mode per row).
        2. Stage 0: raw-encoder ceiling on a *Q-only* re-encode of the probe split
           (row-comparable: every row's Stage 0 reports the same Q-only ceiling,
           not the row-specific encoding's ceiling — which for A.3 would include
           passage content and inflate the number).
        3. Train Phase15MoE on the train split.
        4. Compute sequence codes on the probe split.
        5. Build 4 controls (random/topic/token/geometry are auto + computed).
        6. Apply 1σ-bootstrap selectivity gate.
        7. Collect engineering stats (k_active mean / dead-expert frac).

    Returns a dict with all metrics (JSON-serialisable on top level).
    """
    corpus_cfg = corpus_cfg or MCCorpusConfig()
    # AblationRow.corpus (when set) pins the row to a corpus; None defers to the
    # passed corpus_cfg.corpus (1b pivot — MuSiQue primary, logic_mc control arm).
    if getattr(ablation_row, "corpus", None) is not None:
        corpus_cfg = dataclasses.replace(corpus_cfg, corpus=ablation_row.corpus)
    train_cfg = train_cfg or TrainConfig(seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    timing: dict[str, float] = {}
    warnings: list[str] = []
    t0 = time.time()

    # ----- corpus + encoding (per ablation row) -----
    corpus = build_mc_corpus(corpus_cfg)
    encode_batch_size = max(1, batch_size // 2)
    data = encode_or_load_mc(
        corpus,
        corpus_cfg,
        encoder_override=ablation_row.encoder_name,
        encoding_mode=ablation_row.encoding_mode,
        batch_size=encode_batch_size,
        device=device,
    )
    timing["encode_s"] = time.time() - t0

    # Choose probe split: prefer test, fall back to val (LogiQA test has labels).
    has_test = bool((data["split"] == "test").any())
    probe_split = "test" if has_test else "val"
    if not has_test:
        warnings.append(
            "probe_split='val' (no test split available) — model was selected on "
            "val_loss, so σ-gate suffers selection bias; treat PASS as tentative"
        )
    probe_mask = data["split"] == probe_split
    probe_q_tokens = data["q_tokens"][probe_mask]
    probe_q_mask = data["q_mask"][probe_mask]
    probe_labels = np.asarray(data["reasoning_type"])[probe_mask]
    probe_corpus = corpus[corpus["split"] == probe_split].reset_index(drop=True)

    # Production-path mirror of notebook cell-9: if the cached ``reasoning_type``
    # column is degenerate (single class — happens when the parquet was built
    # before the data.py heuristic landed), re-infer from question text. Without
    # this, ``probe_accuracy`` raises ValueError("≥2 classes") on the σ-gate.
    if len(set(probe_labels.tolist())) < 2 and len(probe_corpus):
        warnings.append(
            f"reasoning_type degenerate ({sorted(set(probe_labels.tolist()))}); "
            f"re-inferring from question text via infer_reasoning_type"
        )
        probe_labels = np.array(
            [infer_reasoning_type(q) for q in probe_corpus["question"].tolist()]
        )

    # ``encode_or_load_mc`` preserves ``corpus`` iteration order. The probe slice
    # of ``data`` is by boolean mask, and ``probe_corpus`` is the corresponding
    # ``split == probe_split`` slice of ``corpus``. Assert the lengths match to
    # catch any future reordering inside ``encode_or_load_mc`` (length-bucketing
    # etc.) before the gate silently misaligns controls with codes.
    assert probe_corpus.shape[0] == probe_q_tokens.shape[0], (
        f"probe_corpus / probe_q_tokens row-order mismatch: "
        f"{probe_corpus.shape[0]} vs {probe_q_tokens.shape[0]}"
    )

    # Controls are derived from (question, passage) and are independent of the
    # encoding mode — compute once and reuse for both Stage 0 and the final gate.
    controls = make_phase15_controls(
        probe_corpus["question"].tolist(),
        probe_corpus["passage"].tolist(),
        seed=seed,
    )

    # ----- Stage 0 ceiling -----
    # Force Q-only encoding for the ceiling so the number is row-comparable
    # across A.1/A.2/A.3. If the active row already uses Q_only, reuse the cached
    # arrays; otherwise re-encode *just the probe rows* under Q_only — passing
    # the full corpus would re-encode p_tokens and cand_pooled too (cf. paper
    # §7.4 row C BGE-swap minutes cost) for outputs Stage 0 never reads.
    stage0: dict | None = None
    if run_stage0:
        if ablation_row.encoding_mode == MODE_Q_ONLY:
            stage0_q_tokens = probe_q_tokens
            stage0_q_mask = probe_q_mask
        else:
            data_q_only = encode_or_load_mc(
                probe_corpus,
                corpus_cfg,
                encoder_override=ablation_row.encoder_name,
                encoding_mode=MODE_Q_ONLY,
                batch_size=encode_batch_size,
                device=device,
            )
            stage0_q_tokens = data_q_only["q_tokens"]
            stage0_q_mask = data_q_only["q_mask"]
        try:
            stage0 = raw_embedding_report(
                stage0_q_tokens.astype(np.float32),
                stage0_q_mask.astype(np.float32),
                probe_labels,
                control_label_sets={"topic": controls["topic"], "token": controls["token"]},
                agg="meanmax",
                seed=seed,
            )
        except Exception as e:
            stage0 = {"error": f"{type(e).__name__}: {e}"}

    # ----- train -----
    t_train = time.time()
    train_loader, val_loader, test_loader = make_mc_loaders(
        data, batch_size=batch_size, num_workers=0
    )
    model = Phase15MoE(
        d_emb=data["q_tokens"].shape[-1],
        d_z=256,
        k_routed=ablation_row.k_routed,
        modulation=ablation_row.modulation,
        lb_strategy=ablation_row.lb_strategy,
        lb_target_active=ablation_row.k_active_target,
        routing=ablation_row.routing,
        chain_steps=getattr(ablation_row, "chain_steps", 1),
        dropout=getattr(ablation_row, "dropout", 0.0),
    )
    # Use dataclasses.replace so future TrainConfig fields are forwarded
    # automatically — manual field-by-field copy would silently drop them.
    train_cfg = dataclasses.replace(
        train_cfg,
        k_target=ablation_row.k_active_target,
        seed=seed,
    )
    result = train_phase15(
        model,
        train_loader,
        val_loader=val_loader,
        cfg=train_cfg,
        device=device,
        progress=progress,
    )
    timing["train_s"] = time.time() - t_train

    # ----- probe codes -----
    t_eval = time.time()
    codes = _compute_codes(
        result["model"],
        probe_q_tokens,
        probe_q_mask,
        device=device,
        batch_size=batch_size,
    )

    try:
        gate = selectivity_gate_phase15(
            codes,
            probe_labels,
            {"topic": controls["topic"], "token": controls["token"]},
            margin_sigma=1.0,
            n_boot=200,
            seed=seed,
        )
    except Exception as e:
        gate = {"error": f"{type(e).__name__}: {e}", "verdict": "ERROR"}

    # Operation-selectivity on the regex (drop-"other"/rare) axis. The LogiQA
    # type-dict ``reasoning_type`` is degenerate (≈95% one class — adj_op
    # unmeasurable); the regex axis gives a multi-class operation label on the
    # classifiable subset. ValueError = <2 classes survive = axis unmeasurable
    # (NOT a negative result), surfaced as verdict="UNMEASURABLE".
    # Operation axis: MuSiQue questions are multi-hop (regex → all-"other"), so the
    # precomputed hop label (``reasoning_type``) is the axis; LogiQA/ReClor keep the
    # regex default (op_labels=None).
    op_axis = probe_labels if corpus_cfg.corpus == "musique" else None
    operation_gate = _safe_regate(
        codes,
        probe_corpus["question"].tolist(),
        probe_corpus["passage"].tolist(),
        seed=seed,
        op_labels=op_axis,
    )

    # Raw-Q operation ceiling: operation-selectivity of the *frozen-encoder* Q code
    # (no router) on the same regex axis. If routing's adj_op ≈ this ceiling, the
    # operation signal is already trivially in the encoder (stem) and the router
    # adds nothing; if routing > ceiling, the router organises operation further.
    operation_ceiling_raw = None
    if run_stage0:
        raw_codes = sequence_code(
            stage0_q_tokens.astype(np.float32),
            stage0_q_mask.astype(np.float32),
            agg="meanmax",
        )
        operation_ceiling_raw = _safe_regate(
            raw_codes,
            probe_corpus["question"].tolist(),
            probe_corpus["passage"].tolist(),
            seed=seed,
            op_labels=op_axis,
        )

    # Cross-dataset probe: operation-selectivity on the ReClor subset (val; ReClor
    # has no labelled test split). ReClor's regex axis is more diverse than
    # LogiQA's, so this tests whether operation routing generalises beyond the
    # LogiQA-only primary probe. ``data`` arrays are in ``corpus`` row order.
    operation_gate_reclor = None
    reclor_mask = (np.asarray(data["split"]) == "val") & (np.asarray(data["source"]) == "reclor")
    if reclor_mask.any():
        rc_corpus = corpus[
            (corpus["split"] == "val") & (corpus["source"] == "reclor")
        ].reset_index(drop=True)
        rc_codes = _compute_codes(
            result["model"],
            data["q_tokens"][reclor_mask],
            data["q_mask"][reclor_mask],
            device=device,
            batch_size=batch_size,
        )
        operation_gate_reclor = _safe_regate(
            rc_codes,
            rc_corpus["question"].tolist(),
            rc_corpus["passage"].tolist(),
            seed=seed,
        )
        if out_dir is not None:
            _save_probe_artifacts(
                out_dir, f"{ablation_row.row_id}_reclor", seed, rc_codes, rc_corpus
            )

    # Persist probe codes + meta so the operation axis can be re-defined and
    # re-gated offline (``regate_operation_selectivity``) without retraining.
    probe_codes_path = (
        _save_probe_artifacts(out_dir, ablation_row.row_id, seed, codes, probe_corpus)
        if out_dir is not None
        else None
    )
    timing["eval_s"] = time.time() - t_eval

    # ----- engineering report -----
    # Use the final epoch's history entry for k_active mean (already accumulated
    # during training) rather than re-running a full-train forward. Per-expert
    # load needs a separate accumulation pass — done in a single epoch on the
    # val loader (smaller than train) so it stays cheap.
    engineering = _engineering_stats(
        result["model"],
        val_loader or train_loader,
        device=device,
        history=result["history"],
    )

    return {
        "row_id": ablation_row.row_id,
        "row_config": ablation_row.to_dict(),
        "seed": seed,
        "timing": timing,
        "stage0_ceiling": stage0,
        "stage0_q_only_forced": run_stage0,
        "gate": gate,
        "operation_gate": operation_gate,
        "operation_gate_reclor": operation_gate_reclor,
        "operation_ceiling_raw": operation_ceiling_raw,
        "probe_codes_path": probe_codes_path,
        "engineering": engineering,
        "history": result["history"],
        "best_val_loss": result["best_val_loss"],
        "probe_split": probe_split,
        "probe_size": int(probe_mask.sum()),
        "warnings": warnings,
    }


def _safe_regate(
    codes, questions: list[str], passages: list[str], *, seed: int, op_labels=None
) -> dict:
    """``regate_operation_selectivity`` with the <2-surviving-class case surfaced as
    ``verdict='UNMEASURABLE'`` (operation axis degenerate — NOT a negative result),
    rather than raising. Shared by the main / raw-Q-ceiling / ReClor probes.

    ``op_labels`` is the precomputed operation axis: ``None`` → LSAT-stem regex on
    the questions (LogiQA/ReClor control arm); for MuSiQue the caller passes the
    hop axis (questions are multi-hop, so regex would yield all-"other")."""
    try:
        return regate_operation_selectivity(
            codes, questions, passages, op_labels=op_labels, seed=seed
        )
    except ValueError as e:
        return {"error": f"{type(e).__name__}: {e}", "verdict": "UNMEASURABLE"}


def _save_probe_artifacts(
    out_dir: str | Path,
    row_id: str,
    seed: int,
    codes: np.ndarray,
    probe_corpus,
) -> str:
    """Persist probe sequence-codes (.npy) + probe meta (.parquet) next to the run.

    Enables ``regate_operation_selectivity`` to re-measure operation-selectivity
    offline under a re-defined label axis without re-training. Returns the codes
    path (str) for the result dict.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    codes_path = out_dir / f"probe_codes_{row_id}_seed{seed}.npy"
    meta_path = out_dir / f"probe_meta_{row_id}_seed{seed}.parquet"
    np.save(codes_path, np.asarray(codes))
    cols = [
        c for c in ("question", "passage", "reasoning_type", "source")
        if c in probe_corpus.columns
    ]
    probe_corpus[cols].reset_index(drop=True).to_parquet(meta_path)
    return str(codes_path)


@torch.no_grad()
def _compute_codes(
    model: Phase15MoE,
    q_tokens: np.ndarray,
    q_mask: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Run the encoder-head + router on the probe split and aggregate alpha
    → meanmax sequence code. Uses ``Phase15MoE.compute_alpha`` (which skips the
    K-expert running-sum, cross-attn, and MC head) — ~99% cheaper than the full
    forward, and avoids the zero-cand_pooled fabrication that previously
    coupled probe correctness to ``MCHead`` invariants.
    """
    torch_device = resolve_device(device)
    model.eval().to(torch_device)
    n = q_tokens.shape[0]
    all_alpha: list[np.ndarray] = []
    all_mask: list[np.ndarray] = []
    for i in range(0, n, batch_size):
        q_batch = torch.from_numpy(q_tokens[i : i + batch_size]).float().to(torch_device)
        m_batch = torch.from_numpy(q_mask[i : i + batch_size]).float().to(torch_device)
        out = model.compute_alpha(q_batch)
        all_alpha.append(out["alpha"].cpu().numpy())
        all_mask.append(m_batch.cpu().numpy())
    alpha_arr = np.concatenate(all_alpha, axis=0)
    mask_arr = np.concatenate(all_mask, axis=0)
    return sequence_code(alpha_arr, mask_arr, agg="meanmax")


@torch.no_grad()
def _engineering_stats(
    model: Phase15MoE,
    loader: DataLoader,
    *,
    device: str,
    history: list[dict],
) -> dict:
    """Per-expert load distribution + dead-expert fraction.

    ``k_active_mean_*`` is reported in two columns: ``train_last_epoch`` (from
    history, sample-weighted across batches as accumulated during training) and
    ``eval`` (token-weighted across the supplied loader, with soft-mask values
    preserved). The load-distribution pass uses ``compute_alpha`` to avoid the
    expensive expert running-sum + cross-attn for a statistic that only needs
    alpha.

    ``sigma_collapse`` flags the failure mode where total load is effectively
    zero (all gates ReLU-dead) — disambiguates the (a) sigma-collapse vs
    (b) load-imbalance interpretation of a high ``dead_expert_frac``.
    """
    torch_device = resolve_device(device)
    model.eval().to(torch_device)
    k_routed = model.k_routed
    total_load = np.zeros(k_routed, dtype=np.float64)
    total_tokens = 0.0
    k_active_sum = 0.0

    for batch in loader:
        q_tokens = batch["q_tokens"].to(torch_device)
        q_mask = batch["q_mask"].to(torch_device)
        out = model.compute_alpha(q_tokens)
        m = q_mask.to(out["alpha"].dtype).unsqueeze(-1)
        per_expert = (out["alpha"] * m).sum(dim=(0, 1)).cpu().numpy()
        total_load += per_expert
        # Token-weighted sums (raw, not pre-meaned) so the final divide gives a
        # true masked mean across the entire loader. Keep ``q_mask`` as float so
        # soft masks (future label-smoothing / attention-weight masks) are not
        # silently truncated to {0, 1} by an int64 cast.
        m_f = q_mask.float()
        total_tokens += float(m_f.sum())
        k_active_sum += float((out["k_active"].float() * m_f).sum())

    total_load_sum = float(total_load.sum())
    sigma_collapse = total_load_sum < 1e-6
    p_k = total_load / max(total_load_sum, 1e-9)
    dead_thresh = 0.1 / k_routed
    dead_expert_frac = float((p_k < dead_thresh).sum() / k_routed)
    last_history = history[-1] if history else {}
    return {
        "k_active_mean_train_last_epoch": float(last_history.get("k_active_mean", 0.0)),
        "k_active_mean_eval": (
            float(k_active_sum / max(total_tokens, 1.0)) if total_tokens > 0 else None
        ),
        # Back-compat alias for the summary CSV column promoted by
        # ``ablations._row_summary``.
        "k_active_mean_token_weighted": float(last_history.get("k_active_mean", 0.0)),
        "dead_expert_frac": dead_expert_frac,
        "sigma_collapse": sigma_collapse,
        "expert_load_distribution": p_k.tolist(),
        "k_routed": k_routed,
    }
