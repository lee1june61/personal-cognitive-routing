"""Tests for `phase1_5.data`.

Most tests are CPU-only and stub the HF datasets / encoder via fake objects.
HF-live tests are marked ``slow``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.phase1_5.data import (
    ENCODING_MODES,
    MCCorpusConfig,
    MCDataset,
    MODE_Q_FULL,
    MODE_Q_ONLY,
    MODE_Q_PMASK,
    SOURCE_LOGIQA2,
    SOURCE_RECLOR,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    _build_q_text,
    _first_present_in,
    _normalize_answer_idx,
    _options_to_list,
    build_mc_corpus,
    make_mc_loaders,
)


# ---- fast: config + cache key ----------------------------------------------------


def test_cache_key_stable_for_same_config():
    cfg_a = MCCorpusConfig(seed=42)
    cfg_b = MCCorpusConfig(seed=42)
    assert cfg_a.cache_key() == cfg_b.cache_key()


def test_cache_key_changes_with_encoder():
    cfg_e5 = MCCorpusConfig(encoder_name="intfloat/e5-large-v2")
    cfg_bge = MCCorpusConfig(encoder_name="BAAI/bge-large-en-v1.5")
    assert cfg_e5.cache_key() != cfg_bge.cache_key()


def test_cache_key_short_length():
    assert len(MCCorpusConfig().cache_key()) == 10


# ---- fast: helpers ----------------------------------------------------------------


def test_first_present_in_returns_first_match():
    assert _first_present_in(("a", "b", "c"), ("b", "d")) == "b"
    assert _first_present_in(("a",), ("b", "c")) is None


def test_normalize_answer_idx_int_zero_based():
    assert _normalize_answer_idx(0, 4) == 0
    assert _normalize_answer_idx(3, 4) == 3


def test_normalize_answer_idx_int_one_based():
    assert _normalize_answer_idx(4, 4) == 3  # 1..4 → 0..3


def test_normalize_answer_idx_str_digit():
    assert _normalize_answer_idx("2", 4) == 2


def test_normalize_answer_idx_letter():
    assert _normalize_answer_idx("a", 4) == 0
    assert _normalize_answer_idx("D", 4) == 3


def test_normalize_answer_idx_invalid_returns_none():
    assert _normalize_answer_idx(None, 4) is None
    assert _normalize_answer_idx("xyz", 4) is None
    assert _normalize_answer_idx(10, 4) is None


def test_options_to_list_from_list():
    out = _options_to_list(["a", "b", "c", "d"], 4)
    assert out == ["a", "b", "c", "d"]


def test_options_to_list_from_ndarray():
    out = _options_to_list(np.array(["a", "b", "c", "d"]), 4)
    assert out == ["a", "b", "c", "d"]


def test_options_to_list_from_dict():
    out = _options_to_list({"a": "first", "b": "second", "c": "third", "d": "fourth"}, 4)
    assert out == ["first", "second", "third", "fourth"]


def test_options_to_list_wrong_count_returns_none():
    assert _options_to_list(["a", "b"], 4) is None


# ---- fast: encoding mode q-text construction --------------------------------------


def test_build_q_text_q_only_drops_passage():
    out = _build_q_text("question", "passage content here", "[SEP]", MODE_Q_ONLY)
    assert "passage" not in out
    assert out == "question"


def test_build_q_text_q_full_includes_passage():
    out = _build_q_text("Q1", "Long passage content.", "[SEP]", MODE_Q_FULL)
    assert "Long passage content." in out
    assert "Q1" in out
    assert "[SEP]" in out


def test_build_q_text_q_pmask_preserves_positions_no_content():
    passage = "this is a five word passage"  # 6 words
    out = _build_q_text("Q1", passage, "[SEP]", MODE_Q_PMASK, pmask_placeholder="<pad>")
    assert "this" not in out  # content erased
    assert "Q1" in out
    # 6 placeholder tokens
    assert out.count("<pad>") == len(passage.split())


def test_build_q_text_q_pmask_uses_supplied_placeholder():
    out = _build_q_text(
        "Q1", "two words", "[SEP]", MODE_Q_PMASK, pmask_placeholder="[PAD]"
    )
    assert "[PAD]" in out
    assert "<pad>" not in out


# ---- infer_reasoning_type heuristic ----------------------------------------------


def test_infer_reasoning_type_strengthen_beats_inference_on_supports_argument():
    """LSAT canonical: 'supports the argument' = strengthen, not inference.
    Priority fix verified — the inference bucket no longer shadows strengthen."""
    from experiments.phase1_5.data import infer_reasoning_type

    stem = "Which one of the following, if true, most strongly supports the speaker's argument?"
    assert infer_reasoning_type(stem) == "strengthen"


def test_infer_reasoning_type_supported_by_passage_is_inference():
    """LSAT canonical: 'supported by the passage' (passive) = inference."""
    from experiments.phase1_5.data import infer_reasoning_type

    stem = "Which of the following is most strongly supported by the passage?"
    assert infer_reasoning_type(stem) == "inference"


def test_infer_reasoning_type_can_be_inferred_is_inference():
    from experiments.phase1_5.data import infer_reasoning_type

    stem = "Which one of the following can be properly inferred from the statements above?"
    assert infer_reasoning_type(stem) == "inference"


def test_infer_reasoning_type_weakens_is_weaken():
    from experiments.phase1_5.data import infer_reasoning_type

    stem = "Which of the following, if true, most weakens the conclusion?"
    assert infer_reasoning_type(stem) == "weaken"


def test_infer_reasoning_type_assumption_word_boundary():
    """'depend on' = assumption, but 'independent' must NOT trigger."""
    from experiments.phase1_5.data import infer_reasoning_type

    assert infer_reasoning_type("The argument depends on which assumption?") == "assumption"
    assert infer_reasoning_type("The independent variable was X.") == "other"


def test_infer_reasoning_type_flaw_word_boundary():
    """'flaw' = flaw, but 'flawless' must NOT trigger (substring footgun fix)."""
    from experiments.phase1_5.data import infer_reasoning_type

    assert infer_reasoning_type("The flaw in this reasoning is that") == "flaw"
    assert infer_reasoning_type("the author's flawless argument") == "other"


def test_infer_reasoning_type_empty_returns_other():
    from experiments.phase1_5.data import infer_reasoning_type

    assert infer_reasoning_type("") == "other"
    assert infer_reasoning_type("   ") == "other"
    assert infer_reasoning_type(None) == "other"  # type: ignore[arg-type]


def test_infer_reasoning_type_unmatched_returns_other_not_wh_word():
    """Unmatched stems must NOT fall through to a wh-word bucket — that would
    reintroduce Phase 1's F3 format-artifact confound at the label layer."""
    from experiments.phase1_5.data import infer_reasoning_type

    # Plain wh-word question with no LSAT-style stem.
    assert infer_reasoning_type("Which color is the sky?") == "other"
    assert infer_reasoning_type("What time is it?") == "other"


# ---- LogiQA 2.0 GitHub type-dict → snake_case label ------------------------------


def test_logiqa2_type_to_label_sufficient_priority_when_multiple_true():
    """Priority order: Sufficient > Necessary > Disjunctive > Conjunctive > Categorical.
    Multiple True keys → first in priority list wins."""
    from experiments.phase1_5.data import _logiqa2_type_to_label

    type_dict = {
        "Sufficient Conditional Reasoning": True,
        "Conjunctive Reasoning": True,
        "Categorical Reasoning": False,
        "Necessary Conditional Reasoning": False,
        "Disjunctive Reasoning": False,
    }
    assert _logiqa2_type_to_label(type_dict) == "sufficient_conditional"


def test_logiqa2_type_to_label_each_canonical_value():
    from experiments.phase1_5.data import _logiqa2_type_to_label

    cases = [
        ("Sufficient Conditional Reasoning", "sufficient_conditional"),
        ("Necessary Conditional Reasoning", "necessary_conditional"),
        ("Disjunctive Reasoning", "disjunctive"),
        ("Conjunctive Reasoning", "conjunctive"),
        ("Categorical Reasoning", "categorical"),
    ]
    for key, expected in cases:
        td = {k: (k == key) for k, _ in cases}
        assert _logiqa2_type_to_label(td) == expected


def test_logiqa2_type_to_label_all_false_returns_none():
    """All False → None → caller falls back to infer_reasoning_type."""
    from experiments.phase1_5.data import _logiqa2_type_to_label

    type_dict = {
        "Sufficient Conditional Reasoning": False,
        "Necessary Conditional Reasoning": False,
        "Disjunctive Reasoning": False,
        "Conjunctive Reasoning": False,
        "Categorical Reasoning": False,
    }
    assert _logiqa2_type_to_label(type_dict) is None


def test_logiqa2_type_to_label_non_dict_returns_none():
    from experiments.phase1_5.data import _logiqa2_type_to_label

    assert _logiqa2_type_to_label(None) is None
    assert _logiqa2_type_to_label("Sufficient") is None
    assert _logiqa2_type_to_label(["a", "b"]) is None
    assert _logiqa2_type_to_label({}) is None


def test_logiqa2_type_to_label_priority_keys_not_in_dict_returns_none():
    """Foreign keys (not in priority list) → None even if True."""
    from experiments.phase1_5.data import _logiqa2_type_to_label

    type_dict = {"Modus Ponens": True, "Modus Tollens": True}
    assert _logiqa2_type_to_label(type_dict) is None


# ---- LogiQA 2.0 GitHub row → schema record ----------------------------------------


def test_logiqa2_github_row_to_record_full_round_trip():
    from experiments.phase1_5.data import (
        SOURCE_LOGIQA2,
        SPLIT_TRAIN,
        _logiqa2_github_row_to_record,
    )

    row = {
        "id": 42,
        "answer": 2,
        "text": "All swans are white.",
        "question": "Which assumption is needed?",
        "options": ["A", "B", "C", "D"],
        "type": {
            "Necessary Conditional Reasoning": True,
            "Sufficient Conditional Reasoning": False,
            "Disjunctive Reasoning": False,
            "Conjunctive Reasoning": False,
            "Categorical Reasoning": False,
        },
    }
    rec = _logiqa2_github_row_to_record(row, SPLIT_TRAIN, 4)
    assert rec is not None
    assert rec["passage"] == "All swans are white."
    assert rec["answer_idx"] == 2
    assert rec["reasoning_type"] == "necessary_conditional"
    assert rec["source"] == SOURCE_LOGIQA2
    assert rec["split"] == SPLIT_TRAIN
    assert rec["options"] == ["A", "B", "C", "D"]


def test_logiqa2_github_row_to_record_all_false_type_falls_back_to_heuristic():
    """type dict all False → heuristic infer_reasoning_type fires on question text."""
    from experiments.phase1_5.data import (
        SPLIT_TRAIN,
        _logiqa2_github_row_to_record,
    )

    row = {
        "answer": 0,
        "text": "passage",
        "question": "Which one of the following is an assumption required by the argument?",
        "options": ["A", "B", "C", "D"],
        "type": {
            "Sufficient Conditional Reasoning": False,
            "Necessary Conditional Reasoning": False,
            "Disjunctive Reasoning": False,
            "Conjunctive Reasoning": False,
            "Categorical Reasoning": False,
        },
    }
    rec = _logiqa2_github_row_to_record(row, SPLIT_TRAIN, 4)
    assert rec is not None
    assert rec["reasoning_type"] == "assumption"  # heuristic match


def test_logiqa2_github_row_to_record_invalid_returns_none():
    from experiments.phase1_5.data import SPLIT_TRAIN, _logiqa2_github_row_to_record

    # Wrong option count
    bad_opts = {
        "answer": 0, "text": "p", "question": "q",
        "options": ["A", "B", "C"], "type": {},
    }
    assert _logiqa2_github_row_to_record(bad_opts, SPLIT_TRAIN, 4) is None

    # Missing passage
    no_passage = {
        "answer": 0, "text": None, "question": "q",
        "options": ["A", "B", "C", "D"], "type": {},
    }
    assert _logiqa2_github_row_to_record(no_passage, SPLIT_TRAIN, 4) is None

    # Out-of-range answer (n_candidates=4 means 0..3 or 1..4)
    bad_answer = {
        "answer": 99, "text": "p", "question": "q",
        "options": ["A", "B", "C", "D"], "type": {},
    }
    assert _logiqa2_github_row_to_record(bad_answer, SPLIT_TRAIN, 4) is None

    # Non-dict input
    assert _logiqa2_github_row_to_record("not a dict", SPLIT_TRAIN, 4) is None  # type: ignore[arg-type]


# ---- GitHub raw integration (slow) ----------------------------------------------


@pytest.mark.slow
def test_load_logiqa2_from_github_one_split_shape(tmp_path: Path):
    """End-to-end: GitHub raw dev.txt download → DataFrame schema check.

    Requires network. Slow-marked so default ``pytest`` skip skips it.
    """
    from experiments.phase1_5.data import MCCorpusConfig, _load_logiqa2_from_github

    cfg = MCCorpusConfig(cache_root=str(tmp_path))
    df = _load_logiqa2_from_github(cfg)
    assert len(df) > 0
    assert set(df.columns) >= {
        "passage", "question", "options", "answer_idx", "reasoning_type", "source", "split",
    }
    assert df["options"].apply(len).eq(4).all()
    assert df["answer_idx"].between(0, 3).all()
    assert df["split"].isin({"train", "val", "test"}).all()


def test_infer_reasoning_type_private_alias_still_works():
    """Back-compat: ``_infer_reasoning_type`` (underscored) is an alias to
    the public ``infer_reasoning_type`` — cell-9 of the notebook imports the
    underscored name historically."""
    from experiments.phase1_5.data import _infer_reasoning_type, infer_reasoning_type

    assert _infer_reasoning_type is infer_reasoning_type


def test_normalize_answer_idx_rejects_bool():
    """``isinstance(True, int)`` is True in Python — must short-circuit."""
    assert _normalize_answer_idx(True, 4) is None
    assert _normalize_answer_idx(False, 4) is None


def test_build_q_text_invalid_mode_raises():
    with pytest.raises(ValueError):
        _build_q_text("Q", "P", "[SEP]", "invalid_mode")


def test_encoding_modes_constant_complete():
    assert set(ENCODING_MODES) == {MODE_Q_ONLY, MODE_Q_PMASK, MODE_Q_FULL}


# ---- fast: MCDataset ---------------------------------------------------------------


def _tiny_data_dict(n: int = 6, t_q: int = 8, t_p: int = 12, d: int = 16, n_cand: int = 4):
    rng = np.random.default_rng(0)
    return {
        "q_tokens": rng.standard_normal((n, t_q, d)).astype(np.float16),
        "q_mask": np.ones((n, t_q), dtype=np.int8),
        "p_tokens": rng.standard_normal((n, t_p, d)).astype(np.float16),
        "p_mask": np.ones((n, t_p), dtype=np.int8),
        "cand_pooled": rng.standard_normal((n, n_cand, d)).astype(np.float32),
        "answer_idx": rng.integers(0, n_cand, size=n).astype(np.int64),
    }


def test_mc_dataset_len_and_getitem_shapes():
    d = _tiny_data_dict(n=5)
    ds = MCDataset(**d)
    assert len(ds) == 5
    item = ds[2]
    assert item["q_tokens"].shape == (8, 16)
    assert item["p_tokens"].shape == (12, 16)
    assert item["cand_pooled"].shape == (4, 16)
    assert item["answer_idx"].dtype.is_floating_point is False
    assert int(item["answer_idx"]) in range(4)


def test_mc_dataset_dtypes_are_float32_after_getitem():
    d = _tiny_data_dict()
    ds = MCDataset(**d)
    item = ds[0]
    assert item["q_tokens"].dtype.is_floating_point
    assert item["q_mask"].dtype.is_floating_point  # mask returned as float for matmul
    assert item["cand_pooled"].dtype.is_floating_point


def test_make_mc_loaders_split_distribution():
    d = _tiny_data_dict(n=12)
    # Fake split column: 8 train, 2 val, 2 test
    split = np.array([SPLIT_TRAIN] * 8 + [SPLIT_VAL] * 2 + [SPLIT_TEST] * 2)
    d_full = {**d, "split": split, "reasoning_type": np.array(["x"] * 12), "source": np.array(["s"] * 12)}
    train, val, test = make_mc_loaders(d_full, batch_size=2, num_workers=0)
    assert train is not None and val is not None and test is not None
    assert len(train.dataset) == 8
    assert len(val.dataset) == 2
    assert len(test.dataset) == 2


def test_make_mc_loaders_returns_none_for_empty_split():
    d = _tiny_data_dict(n=4)
    split = np.array([SPLIT_TRAIN] * 4)
    d_full = {**d, "split": split, "reasoning_type": np.array(["x"] * 4), "source": np.array(["s"] * 4)}
    train, val, test = make_mc_loaders(d_full, batch_size=2, num_workers=0)
    assert train is not None
    assert val is None
    assert test is None


# ---- slow: HF live load ------------------------------------------------------------


@pytest.mark.slow
def test_load_logiqa2_returns_mc_schema(tmp_path: Path):
    cfg = MCCorpusConfig(
        max_train_samples=200,
        max_val_samples=50,
        max_test_samples=50,
        cache_root=str(tmp_path),
    )
    from experiments.phase1_5.data import load_logiqa2

    df = load_logiqa2(cfg)
    assert "passage" in df.columns
    assert "question" in df.columns
    assert "options" in df.columns
    assert "answer_idx" in df.columns
    assert "reasoning_type" in df.columns
    if len(df):
        assert (df["source"] == SOURCE_LOGIQA2).all()
        assert df["options"].apply(len).eq(4).all()
        assert df["answer_idx"].between(0, 3).all()


@pytest.mark.slow
def test_load_reclor_returns_mc_schema(tmp_path: Path):
    cfg = MCCorpusConfig(cache_root=str(tmp_path))
    from experiments.phase1_5.data import load_reclor

    df = load_reclor(cfg)
    if len(df):
        assert (df["source"] == SOURCE_RECLOR).all()
        assert df["options"].apply(len).eq(4).all()
        assert df["answer_idx"].between(0, 3).all()


@pytest.mark.slow
def test_build_mc_corpus_caches(tmp_path: Path):
    cfg = MCCorpusConfig(
        max_train_samples=500,
        max_val_samples=100,
        max_test_samples=100,
        cache_root=str(tmp_path),
    )
    df_1 = build_mc_corpus(cfg)
    assert len(df_1) > 0
    cache_file = Path(tmp_path) / f"corpus_{cfg.cache_key()}.parquet"
    assert cache_file.exists()
    # Second call hits cache
    df_2 = build_mc_corpus(cfg)
    assert len(df_2) == len(df_1)


@pytest.mark.slow
def test_encode_or_load_mc_q_only_shapes(tmp_path: Path):
    cfg = MCCorpusConfig(
        max_train_samples=20,
        max_val_samples=10,
        max_test_samples=10,
        t_cap_q=64,
        t_cap_p=128,
        cache_root=str(tmp_path),
    )
    from experiments.phase1_5.data import encode_or_load_mc

    corpus = build_mc_corpus(cfg)
    out = encode_or_load_mc(corpus, cfg, encoding_mode=MODE_Q_ONLY, batch_size=8)
    n = len(corpus)
    assert out["q_tokens"].shape == (n, 64, 1024)
    assert out["q_mask"].shape == (n, 64)
    assert out["p_tokens"].shape == (n, 128, 1024)
    assert out["cand_pooled"].shape == (n, 4, 1024)
    assert out["answer_idx"].shape == (n,)


@pytest.mark.slow
def test_encode_or_load_mc_pmask_differs_from_qonly(tmp_path: Path):
    """A.2 (pmask) must produce different q_tokens than A.1 (q_only) for same rows."""
    cfg = MCCorpusConfig(
        max_train_samples=10,
        max_val_samples=2,
        max_test_samples=2,
        t_cap_q=64,
        t_cap_p=64,
        cache_root=str(tmp_path),
    )
    from experiments.phase1_5.data import encode_or_load_mc

    corpus = build_mc_corpus(cfg)
    out_qonly = encode_or_load_mc(corpus, cfg, encoding_mode=MODE_Q_ONLY, batch_size=8)
    out_pmask = encode_or_load_mc(corpus, cfg, encoding_mode=MODE_Q_PMASK, batch_size=8)
    # The pmask mode encodes a longer Q text (Q + [SEP] + placeholders), so the
    # token sequences differ.
    assert not np.allclose(
        out_qonly["q_tokens"].astype(np.float32),
        out_pmask["q_tokens"].astype(np.float32),
        atol=1e-3,
    )


@pytest.mark.slow
def test_encode_or_load_mc_full_uses_more_q_tokens_than_qonly(tmp_path: Path):
    """A.3 (Q_full) should fill more positions than A.1 (Q_only) on average."""
    cfg = MCCorpusConfig(
        max_train_samples=10,
        max_val_samples=2,
        max_test_samples=2,
        t_cap_q=128,
        t_cap_p=128,
        cache_root=str(tmp_path),
    )
    from experiments.phase1_5.data import encode_or_load_mc

    corpus = build_mc_corpus(cfg)
    out_qonly = encode_or_load_mc(corpus, cfg, encoding_mode=MODE_Q_ONLY, batch_size=8)
    out_qfull = encode_or_load_mc(corpus, cfg, encoding_mode=MODE_Q_FULL, batch_size=8)
    assert out_qfull["q_mask"].sum() > out_qonly["q_mask"].sum()


def test_encode_or_load_mc_invalid_mode_raises():
    cfg = MCCorpusConfig()
    from experiments.phase1_5.data import encode_or_load_mc

    df = pd.DataFrame(
        {
            "passage": ["p"],
            "question": ["q"],
            "options": [["a", "b", "c", "d"]],
            "answer_idx": [0],
            "reasoning_type": ["unknown"],
            "source": ["s"],
            "split": [SPLIT_TRAIN],
        }
    )
    with pytest.raises(ValueError, match="encoding_mode"):
        encode_or_load_mc(df, cfg, encoding_mode="bogus")
