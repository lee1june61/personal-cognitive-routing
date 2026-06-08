"""Phase 1.5 1a ablation dispatcher — paper §7.4 Table 1 rows A.1 / A.2 / A.3.

Each ``AblationRow`` is a frozen dataclass with the *single config delta* that
distinguishes one Table 1 row from the next. The row → run mapping is 1-to-1:
swapping a row swaps exactly one architectural / data commitment, preserving
attribution per the forced-design IRON (paper §8).

CLI usage::

    python -m experiments.phase1_5.ablations \\
        --row A.1 --seed 0 --device cuda --out_dir out/phase1_5/ablations

Or programmatically::

    from experiments.phase1_5.ablations import PHASE1_5_INITIAL_ROWS, run_all_rows
    df = run_all_rows(PHASE1_5_INITIAL_ROWS, epochs=40, device='cuda')
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .data import MCCorpusConfig, MODE_Q_FULL, MODE_Q_ONLY, MODE_Q_PMASK
from .engine_1a import run_engine_1a
from .model import MOD_KG_HYPERNET
from .train import TrainConfig


def _cfg_hash(
    corpus_cfg: MCCorpusConfig | None,
    train_cfg: TrainConfig | None,
    row: "AblationRow | None" = None,
) -> str:
    """SHA1 hash of (corpus_cfg + train_cfg + row) for cache invalidation.

    The cache file ``row_{id}_seed{seed}_cfg{...}.json`` alone would silently
    return a stale result after a config change. The hash must include:

    - ``corpus_cfg``: changing sample caps / cache_root invalidates encodings.
    - ``train_cfg``: epochs / lr / λ_* affect the training trajectory.
    - ``row`` fields: ``row_id`` alone is NOT enough — the *contents* of a row
      (k_routed, lb_strategy, modulation, encoder_name, …) can change between
      runs while the row_id is reused (e.g. Layer 2-C kept row_id='A.1' but
      flipped k_routed 128→64). Without hashing row contents, a stale K=128
      result would silently load when re-running the K=64 row.

    Backward-compatible: callers that pass only ``(corpus_cfg, train_cfg)``
    still get a stable hash for those two fields (``row=None`` is excluded
    from the payload), at the cost of not catching row-contents staleness."""
    payload: dict = {}
    if corpus_cfg is not None:
        payload["corpus"] = asdict(corpus_cfg)
    if train_cfg is not None:
        payload["train"] = asdict(train_cfg)
    if row is not None:
        payload["row"] = row.to_dict()
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:8]


# ----- AblationRow ------------------------------------------------------------------


@dataclass(frozen=True)
class AblationRow:
    """One row of the paper §7.4 Table 1 ablation matrix.

    Phase 1.5 1a initial scope (A.1/A.2/A.3) only varies ``encoding_mode``.
    Later-round rows (B, C, F, F.1) vary ``modulation`` / ``encoder_name`` /
    ``lb_strategy`` / ``k_routed`` respectively.
    """

    row_id: str
    name: str
    encoding_mode: str = MODE_Q_ONLY
    encoder_name: str = "intfloat/e5-large-v2"
    # 2026-05-29: default flipped cross_attn → kg_hypernet (no-bypass, ADR 0001).
    # cross_attn / film remain selectable for the §7.4 Gap B modulation ablation.
    modulation: str = MOD_KG_HYPERNET
    k_routed: int = 128
    k_active_target: float = 4.0
    # Strategies supported by ``phase1_5.load_balance.make_lb`` (currently
    # ``"off"`` and ``"aux_free"``). Aux-weight variants are reserved for the
    # paper §7.4 Row F sweep and not implemented yet — using a reserved name
    # here would crash inside ``Phase15MoE.__init__`` after the corpus is
    # already encoded, so keep this aligned with ``LB_STRATEGIES``.
    lb_strategy: str = "off"
    # "relu_l1" (default, adaptive-L1 sparsity) or "topk" (K_active fixed via top-k,
    # Phase 3 diversity, ADR 0002 — sidesteps the ReLU+L1+LB collapse pathology).
    routing: str = "relu_l1"
    # Corpus override (None → defer to the passed corpus_cfg.corpus). Set to
    # "musique" / "logic_mc" / "both" to pin a row to a corpus (1b pivot).
    corpus: str | None = None
    # 1b chain-of-experts depth (1 = flat 1a; 2-4 = chain). Threaded into
    # Phase15MoE(chain_steps=...); engine routes train/eval via forward_chain.
    # ⚠ DEPRECATED chain (2026-06-08): seq layout = setup-failure → direction-1
    # parallel. chain_steps > 1 is legacy (1b orchestrator archived).
    chain_steps: int = 1
    # Dropout in the encoder head + experts (overfit control; MuSiQue memorises
    # at 0.0). Threaded into Phase15MoE(dropout=...).
    dropout: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


PHASE1_5_INITIAL_ROWS: list[AblationRow] = [
    # Layer 2-C: K_routed override 128 → 64 for the 1a initial scope.
    # Phase 1.5 Layer-1 (K=128) Colab run showed an ep3 cliff that all three
    # of (Layer 2-A λ_z=0, Layer 2-B lam_l1_init↓, Layer 2-C K↓) are intended
    # to fight in parallel. K=64 matches DeepSeek-V2 (K=64,K_active=6) /
    # OLMoE (K=64,K_active=8) precedent and is the first entry of the paper
    # §7.4 Row F.1 K_routed sweep (64 / 128 / 256) — keeping AblationRow.k_routed
    # default at 128 so a future Row F.1 row can target 128/256 by explicit
    # field override without conflicting with the 1a baseline.
    #
    # Sparsity ratio note: K_active=4 at K=64 → 6.25% per-token activation
    # rate. This sits between DeepSeek-V3 (3.1%) and V2 (9.4%) / OLMoE (12.5%);
    # it is intentionally retained from paper §5.1 row 8 (K_active=4 commit)
    # rather than rescaled to a constant ratio, so that across the future
    # Row F.1 K sweep the *absolute* K_active stays comparable.
    AblationRow(
        row_id="A.1",
        name="Q-only default",
        encoding_mode=MODE_Q_ONLY,
        k_routed=64,
        lb_strategy="aux_free",
        notes="Default — Q-only encoding under e5-large-v2 + cross-attn modulation + aux-free LB + K=64.",
    ),
    AblationRow(
        row_id="A.2",
        name="Q + position-masked-P strict control",
        encoding_mode=MODE_Q_PMASK,
        k_routed=64,
        lb_strategy="aux_free",
        notes="Partial-Q-leak control per Reviewer 3 / paper §7.1. P positions preserved, content erased. K=64.",
    ),
    AblationRow(
        row_id="A.3",
        name="Q + full-P-cross-encoded ceiling",
        encoding_mode=MODE_Q_FULL,
        k_routed=64,
        lb_strategy="aux_free",
        notes="Bottleneck violation upper bound: full P content in encoder input. K=64.",
    ),
]


# ----- Run a single row -------------------------------------------------------------


def run_ablation_row(
    row: AblationRow,
    *,
    corpus_cfg: MCCorpusConfig | None = None,
    train_cfg: TrainConfig | None = None,
    out_dir: Path | str = "out/phase1_5/ablations",
    device: str = "cuda",
    seed: int = 0,
    batch_size: int = 64,
    skip_if_exists: bool = True,
    progress: bool = False,
) -> dict:
    """Run one ablation row and persist the result as JSON."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_tag = _cfg_hash(corpus_cfg, train_cfg, row=row)
    out_path = out_dir / f"row_{row.row_id}_seed{seed}_cfg{cfg_tag}.json"

    if skip_if_exists and out_path.exists():
        print(f"[ablation] {row.row_id} seed={seed} cfg={cfg_tag} cached → {out_path}")
        with open(out_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"[ablation] ▶ {row.row_id} '{row.name}' (seed={seed}, device={device})")
    result = run_engine_1a(
        ablation_row=row,
        corpus_cfg=corpus_cfg,
        train_cfg=train_cfg,
        batch_size=batch_size,
        device=device,
        seed=seed,
        progress=progress,
        out_dir=out_dir,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=_json_default)
    print(f"[ablation] ✓ {row.row_id} → {out_path}")
    return result


def _json_default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON-serialisable: {type(o).__name__}")


# ----- Run multiple rows -----------------------------------------------------------


def run_all_rows(
    rows: list[AblationRow] | None = None,
    *,
    corpus_cfg: MCCorpusConfig | None = None,
    train_cfg: TrainConfig | None = None,
    out_dir: Path | str = "out/phase1_5/ablations",
    device: str = "cuda",
    seed: int = 0,
    batch_size: int = 64,
    skip_if_exists: bool = True,
) -> pd.DataFrame:
    """Run all rows, return a comparison DataFrame indexed by row_id."""
    rows = rows or PHASE1_5_INITIAL_ROWS
    summaries: list[dict] = []
    for row in rows:
        result = run_ablation_row(
            row,
            corpus_cfg=corpus_cfg,
            train_cfg=train_cfg,
            out_dir=out_dir,
            device=device,
            seed=seed,
            batch_size=batch_size,
            skip_if_exists=skip_if_exists,
        )
        summaries.append(_row_summary(result))
    df = pd.DataFrame(summaries).set_index("row_id")
    csv_path = Path(out_dir) / f"summary_seed{seed}.csv"
    df.to_csv(csv_path)
    print(f"[ablation] summary → {csv_path}")
    return df


def _row_summary(result: dict) -> dict:
    gate = result.get("gate") or {}
    op_gate = result.get("operation_gate") or {}
    eng = result.get("engineering") or {}
    last_hist = (result.get("history") or [{}])[-1]
    return {
        "row_id": result["row_id"],
        "verdict": gate.get("verdict", "ERROR"),
        "op_verdict": op_gate.get("verdict"),
        "op_adj_operation": op_gate.get("adj_operation"),
        "op_threshold": op_gate.get("threshold"),
        "op_n_examples": op_gate.get("n_operation_examples"),
        "op_classes": op_gate.get("operation_classes"),
        "op_purity": (op_gate.get("consistency") or {}).get("op_purity"),
        "op_purity_beats_topic": (op_gate.get("consistency") or {}).get("op_beats_topic"),
        "op_reclor_verdict": (result.get("operation_gate_reclor") or {}).get("verdict"),
        "op_reclor_adj": (result.get("operation_gate_reclor") or {}).get("adj_operation"),
        "op_ceiling_raw_adj": (result.get("operation_ceiling_raw") or {}).get("adj_operation"),
        "adj_operation": gate.get("adj_operation"),
        "sigma_adj_operation": gate.get("sigma_adj_operation"),
        "threshold": gate.get("threshold"),
        "passes_sigma_gate": gate.get("passes_sigma_gate"),
        "adj_random": (gate.get("adj_controls") or {}).get("random"),
        "adj_topic": (gate.get("adj_controls") or {}).get("topic"),
        "adj_token": (gate.get("adj_controls") or {}).get("token"),
        "adj_geometry": (gate.get("adj_controls") or {}).get("geometry"),
        "k_active_mean": eng.get("k_active_mean_token_weighted"),
        "dead_expert_frac": eng.get("dead_expert_frac"),
        "val_mc_acc_final": last_hist.get("val_mc_acc"),
        "val_loss_final": last_hist.get("val_loss"),
        "train_mc_acc_final": last_hist.get("mc_acc"),
        "seed": result.get("seed"),
    }


# ----- CLI ------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Phase 1.5 1a ablation runner")
    parser.add_argument(
        "--row",
        type=str,
        default="all",
        help="row_id (A.1/A.2/A.3) or 'all'",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_dir", type=str, default="out/phase1_5/ablations")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_train", type=int, default=20_000)
    parser.add_argument(
        "--no_skip_cached", action="store_true", help="Re-run even if JSON exists"
    )
    args = parser.parse_args()

    corpus_cfg = MCCorpusConfig(max_train_samples=args.max_train)
    train_cfg = TrainConfig(epochs=args.epochs, lr=args.lr, seed=args.seed)

    rows = PHASE1_5_INITIAL_ROWS
    if args.row != "all":
        rows = [r for r in rows if r.row_id == args.row]
        if not rows:
            raise SystemExit(
                f"row_id={args.row!r} not in {[r.row_id for r in PHASE1_5_INITIAL_ROWS]}"
            )

    if len(rows) == 1:
        run_ablation_row(
            rows[0],
            corpus_cfg=corpus_cfg,
            train_cfg=train_cfg,
            out_dir=args.out_dir,
            device=args.device,
            seed=args.seed,
            batch_size=args.batch,
            skip_if_exists=not args.no_skip_cached,
        )
    else:
        df = run_all_rows(
            rows,
            corpus_cfg=corpus_cfg,
            train_cfg=train_cfg,
            out_dir=args.out_dir,
            device=args.device,
            seed=args.seed,
            batch_size=args.batch,
            skip_if_exists=not args.no_skip_cached,
        )
        print(df.to_string())


if __name__ == "__main__":
    main()
