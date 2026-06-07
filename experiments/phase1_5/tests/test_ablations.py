"""Tests for `phase1_5.ablations` — row dataclass + tiny end-to-end."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.phase1_5.ablations import (
    PHASE1_5_INITIAL_ROWS,
    AblationRow,
    _row_summary,
)
from experiments.phase1_5.data import MODE_Q_FULL, MODE_Q_ONLY, MODE_Q_PMASK


# ---- AblationRow dataclass -----------------------------------------------------


def test_initial_rows_have_correct_ids():
    ids = [r.row_id for r in PHASE1_5_INITIAL_ROWS]
    assert ids == ["A.1", "A.2", "A.3"]


def test_initial_rows_use_distinct_encoding_modes():
    modes = [r.encoding_mode for r in PHASE1_5_INITIAL_ROWS]
    assert set(modes) == {MODE_Q_ONLY, MODE_Q_PMASK, MODE_Q_FULL}


def test_ablation_row_is_frozen():
    row = PHASE1_5_INITIAL_ROWS[0]
    with pytest.raises(Exception):  # FrozenInstanceError
        row.row_id = "B"  # type: ignore[misc]


def test_ablation_row_to_dict_roundtrips():
    row = AblationRow(row_id="X", name="x")
    d = row.to_dict()
    assert d["row_id"] == "X"
    assert d["encoder_name"] == "intfloat/e5-large-v2"
    assert d["k_routed"] == 128


def test_initial_rows_share_encoder_modulation_k_routed():
    """A.1/A.2/A.3 must only differ on encoding_mode (per the plan: 1-to-1 attribution)."""
    a1, a2, a3 = PHASE1_5_INITIAL_ROWS
    assert a1.encoder_name == a2.encoder_name == a3.encoder_name
    assert a1.modulation == a2.modulation == a3.modulation
    assert a1.k_routed == a2.k_routed == a3.k_routed
    assert a1.k_active_target == a2.k_active_target == a3.k_active_target


# ---- _row_summary serialiser ---------------------------------------------------


def test_row_summary_extracts_top_level_keys():
    fake_result = {
        "row_id": "A.1",
        "seed": 0,
        "gate": {
            "verdict": "PASS",
            "adj_operation": 0.45,
            "sigma_adj_operation": 0.02,
            "threshold": 0.40,
            "passes_sigma_gate": True,
            "adj_controls": {"random": 0.01, "topic": 0.10, "token": 0.08, "geometry": 0.0},
        },
        "engineering": {
            "k_active_mean_token_weighted": 4.2,
            "dead_expert_frac": 0.05,
        },
        "history": [{"mc_acc": 0.30, "val_mc_acc": 0.35, "val_loss": 1.30}],
    }
    s = _row_summary(fake_result)
    assert s["row_id"] == "A.1"
    assert s["verdict"] == "PASS"
    assert s["adj_operation"] == 0.45
    assert s["adj_topic"] == 0.10
    assert s["k_active_mean"] == 4.2
    assert s["val_mc_acc_final"] == 0.35


def test_row_summary_handles_missing_gate():
    fake_result = {"row_id": "A.1", "seed": 0, "history": []}
    s = _row_summary(fake_result)
    assert s["verdict"] == "ERROR"
    assert s["adj_operation"] is None


# ---- engine_1a tiny smoke test (slow — runs a 1-epoch train on tiny synthetic) ----


@pytest.mark.slow
def test_run_engine_1a_tiny_smoke(tmp_path: Path):
    """Run engine_1a end-to-end on a tiny synthetic MC corpus (CPU, 1 epoch).

    Stage 0 ceiling, train, eval, gate — all should succeed; verdict is whatever
    the random init produces (we don't assert PASS here).
    """
    # Build a tiny corpus parquet manually so we skip HF download.
    import pandas as pd

    from experiments.phase1_5.data import MCCorpusConfig, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL
    from experiments.phase1_5.engine_1a import run_engine_1a
    from experiments.phase1_5.train import TrainConfig

    rng = np.random.default_rng(0)
    n_each = 20
    rows = []
    for split in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST):
        for i in range(n_each):
            rt = ["why", "what", "how"][i % 3]
            rows.append(
                {
                    "passage": f"This is passage {i} about cluster {i % 3}.",
                    "question": f"{rt} question about {i % 5}?",
                    "options": [f"option_{i}_{k}" for k in range(4)],
                    "answer_idx": int(rng.integers(0, 4)),
                    "reasoning_type": rt,
                    "source": "synthetic",
                    "split": split,
                }
            )
    df = pd.DataFrame(rows)
    cfg = MCCorpusConfig(
        max_train_samples=n_each,
        max_val_samples=n_each,
        max_test_samples=n_each,
        t_cap_q=32,
        t_cap_p=48,
        t_cap_cand=16,
        cache_root=str(tmp_path / "cache"),
    )
    cache_path = Path(cfg.cache_root) / f"corpus_{cfg.cache_key()}.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)

    train_cfg = TrainConfig(epochs=1, lr=1e-3, log_every=0, use_best_val=False, seed=0)
    result = run_engine_1a(
        ablation_row=PHASE1_5_INITIAL_ROWS[0],
        corpus_cfg=cfg,
        train_cfg=train_cfg,
        batch_size=4,
        device="cpu",
        seed=0,
    )
    assert "gate" in result
    assert "engineering" in result
    assert "history" in result


def test_save_probe_artifacts_writes_codes_and_meta(tmp_path):
    """_save_probe_artifacts persists probe codes (.npy) + meta (.parquet) for
    offline re-gating — no encoder/model needed."""
    import numpy as np
    import pandas as pd

    from experiments.phase1_5.engine_1a import _save_probe_artifacts

    codes = np.random.RandomState(0).standard_normal((5, 8)).astype(np.float32)
    probe = pd.DataFrame(
        {
            "question": [f"q{i}" for i in range(5)],
            "passage": [f"p{i}" for i in range(5)],
            "reasoning_type": ["weaken"] * 5,
            "source": ["logiqa2"] * 5,
        }
    )
    path = _save_probe_artifacts(tmp_path, "X.1", 0, codes, probe)
    assert Path(path).exists()
    loaded = np.load(path)
    assert loaded.shape == (5, 8)
    meta = pd.read_parquet(Path(path).parent / "probe_meta_X.1_seed0.parquet")
    assert list(meta.columns) == ["question", "passage", "reasoning_type", "source"]
    assert len(meta) == 5
