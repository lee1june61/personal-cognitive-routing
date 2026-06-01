"""Tests for `phase1_5.data_musique` — MuSiQue multi-hop → 4-way MC conversion.

CPU-only; HF `load_dataset` is faked. HF-live tests marked ``slow``.

MuSiQue row schema (HF dgslibisey/MuSiQue):
  id: "2hop__.." / "3hop__.." / "4hop__.."
  paragraphs: list[{idx, title, paragraph_text}]
  question: str (multi-hop)
  question_decomposition: list[{id, question, answer, paragraph_support_idx}]
                          (last hop answer == final gold; earlier = intermediates)
  answer: str, answer_aliases: list[str], answerable: bool
"""

from __future__ import annotations

import numpy as np
import pytest

from research.demo.phase1_5.data import MCCorpusConfig


def _musique_row(
    *,
    hops: int = 4,
    gold: str = "1960",
    intermediates: list[str] | None = None,
    aliases: list[str] | None = None,
    answerable: bool = True,
    question: str = "When was the owner of The Collegian founded?",
):
    """Build a fake MuSiQue row. ``intermediates`` are the non-final hop answers;
    the final decomposition answer is set to ``gold``."""
    if intermediates is None:
        intermediates = [f"Entity {i}" for i in range(hops - 1)]
    assert len(intermediates) == hops - 1
    decomp = [
        {"id": 100 + i, "question": f"sub-q {i}", "answer": ans, "paragraph_support_idx": i}
        for i, ans in enumerate(intermediates)
    ]
    decomp.append(
        {"id": 200, "question": f"final #{hops-1}", "answer": gold, "paragraph_support_idx": hops - 1}
    )
    paragraphs = [
        {"idx": i, "title": f"Title{i}", "paragraph_text": f"Paragraph text number {i}."}
        for i in range(hops + 2)
    ]
    return {
        "id": f"{hops}hop__abc_{gold}",
        "paragraphs": paragraphs,
        "question": question,
        "question_decomposition": decomp,
        "answer": gold,
        "answer_aliases": aliases or [],
        "answerable": answerable,
    }


# ---- tracer bullet: core 4-hop conversion (no backfill needed) --------------------


def test_row_to_record_4hop_uses_intermediates_as_distractors():
    """A 4-hop row has 3 intermediate answers → exactly fills the 3 distractor
    slots with no backfill. The record is valid 4-way MC with answer_idx→gold."""
    from research.demo.phase1_5.data_musique import _musique_row_to_record

    row = _musique_row(hops=4, gold="1960", intermediates=["Alice", "Bob", "Carol"])
    cfg = MCCorpusConfig()
    rec = _musique_row_to_record(row, "train", cfg, answer_pool=None, rng_seed=0)

    assert rec is not None
    assert len(rec["options"]) == 4
    # gold is present and answer_idx points to it
    assert rec["options"][rec["answer_idx"]] == "1960"
    # the three intermediates are the distractors
    assert set(rec["options"]) == {"1960", "Alice", "Bob", "Carol"}
    assert rec["source"] == "musique"


def test_unanswerable_row_dropped():
    from research.demo.phase1_5.data_musique import _musique_row_to_record

    row = _musique_row(hops=4, answerable=False)
    rec = _musique_row_to_record(row, "train", MCCorpusConfig(), answer_pool=None, rng_seed=0)
    assert rec is None


def test_2hop_without_pool_dropped():
    """2-hop has only 1 intermediate → needs 2 backfill; with no pool the row
    cannot reach 4 unique options and is dropped (never padded with duplicates)."""
    from research.demo.phase1_5.data_musique import _musique_row_to_record

    row = _musique_row(hops=2, gold="1960", intermediates=["Alice"])
    rec = _musique_row_to_record(row, "train", MCCorpusConfig(), answer_pool=None, rng_seed=0)
    assert rec is None


def test_backfill_fills_to_four_from_same_answer_type():
    """2-hop (1 intermediate) back-fills 2 distractors from the same-answer-type
    pool. A DATE gold draws DATE distractors, not entity-type ones."""
    from research.demo.phase1_5.data_musique import _build_answer_pool, _musique_row_to_record

    pool_rows = [
        _musique_row(hops=2, gold="1885", intermediates=["Xavier"]),
        _musique_row(hops=2, gold="1992", intermediates=["Yolanda"]),
        _musique_row(hops=2, gold="2001", intermediates=["Zachary"]),
        _musique_row(hops=2, gold="1733", intermediates=["Wallace"]),
    ]
    pool = _build_answer_pool(pool_rows)
    row = _musique_row(hops=2, gold="1960", intermediates=["Alice"])
    rec = _musique_row_to_record(row, "train", MCCorpusConfig(), answer_pool=pool, rng_seed=1)

    assert rec is not None
    assert len(rec["options"]) == 4
    assert rec["options"][rec["answer_idx"]] == "1960"
    assert "Alice" in rec["options"]  # intermediate kept
    backfill = set(rec["options"]) - {"1960", "Alice"}
    assert len(backfill) == 2
    # same-type (DATE/year) only — entity-type intermediates must not be drawn
    assert backfill <= {"1885", "1992", "2001", "1733"}


def test_leak_guard_excludes_intermediate_equal_to_alias():
    """An intermediate answer that equals a gold ALIAS is a hidden-correct
    distractor → must be dropped (and back-filled), never appear as an option."""
    from research.demo.phase1_5.data_musique import _build_answer_pool, _musique_row_to_record

    pool = _build_answer_pool(
        [_musique_row(hops=2, gold=g, intermediates=["e"]) for g in ("Paris", "Berlin", "Rome", "Madrid")]
    )
    row = _musique_row(
        hops=4,
        gold="London",
        intermediates=["Alice", "Greater London", "Bob"],
        aliases=["Greater London"],  # one intermediate equals this alias
    )
    rec = _musique_row_to_record(row, "train", MCCorpusConfig(), answer_pool=pool, rng_seed=2)
    assert rec is not None
    assert len(rec["options"]) == 4
    assert "Greater London" not in rec["options"]  # leak excluded
    assert rec["options"][rec["answer_idx"]] == "London"


# ---- operation labels (eval/S1 only) ----------------------------------------------


def test_infer_musique_op_label_hop_count():
    from research.demo.phase1_5.data_musique import infer_musique_op_label

    assert infer_musique_op_label({"id": "2hop__a_b"}) == "2hop"
    assert infer_musique_op_label({"id": "3hop__x"}) == "3hop"
    assert infer_musique_op_label({"id": "4hop__y"}) == "4hop"
    assert infer_musique_op_label({"id": "weird"}) == "unknown"


def test_infer_musique_structure_chain_vs_comparison():
    """Chain/bridge: a later sub-question references an earlier answer via '#k'.
    Comparison: independent sub-questions (no '#k' reference)."""
    from research.demo.phase1_5.data_musique import infer_musique_structure

    chain = {
        "question_decomposition": [
            {"question": "The Collegian >> owned by", "answer": "HBU"},
            {"question": "When was #1 founded?", "answer": "1960"},
        ]
    }
    comparison = {
        "question_decomposition": [
            {"question": "When was Alice born?", "answer": "1900"},
            {"question": "When was Bob born?", "answer": "1910"},
        ]
    }
    assert infer_musique_structure(chain) == "chain"
    assert infer_musique_structure(comparison) == "comparison"


# ---- passage assembly --------------------------------------------------------------


def test_assemble_passage_supporting_first_and_budget():
    """Supporting paragraphs (those a hop relies on) lead, in hop order; the rest
    fill until the word budget (t_cap_p) is exhausted."""
    from research.demo.phase1_5.data_musique import _assemble_passage

    row = {
        "paragraphs": [
            {"idx": 0, "title": "P0", "paragraph_text": "alpha alpha alpha"},
            {"idx": 1, "title": "P1", "paragraph_text": "bravo bravo bravo"},
            {"idx": 2, "title": "P2", "paragraph_text": "charlie charlie charlie"},
            {"idx": 3, "title": "P3", "paragraph_text": "delta delta delta"},
        ],
        "question_decomposition": [
            {"answer": "x", "paragraph_support_idx": 2},
            {"answer": "y", "paragraph_support_idx": 3},
        ],
    }
    p = _assemble_passage(row, MCCorpusConfig(t_cap_p=12))
    # supporting paragraphs present and in hop order (2 before 3)
    assert "charlie" in p and "delta" in p
    assert p.index("charlie") < p.index("delta")
    # budget excludes at least one non-supporting paragraph
    assert ("alpha" not in p) or ("bravo" not in p)


# ---- distractor leak-check (M1 GATE) ----------------------------------------------


def _mc_df(rows):
    import pandas as pd

    return pd.DataFrame(rows)


def test_distractor_leak_check_flags_question_leak():
    """A planted question→option leak (gold option = the question's words) must
    push question_to_option_acc ≈ 1.0 — this is the GATE's tripwire."""
    from research.demo.phase1_5.data_musique import musique_distractor_leak_check

    rows = []
    for i in range(20):
        q = f"unique{i} marker{i} token{i}"
        opts = [f"unrelated alpha {j}" for j in range(3)] + [q]  # gold idx 3 == question
        rows.append(
            {"passage": "filler words here", "question": q, "options": opts,
             "answer_idx": 3, "reasoning_type": "2hop", "source": "musique", "split": "test"}
        )
    res = musique_distractor_leak_check(_mc_df(rows))
    assert res["question_to_option_acc"] > 0.9
    assert res["n"] == 20


def test_distractor_leak_check_near_chance_when_unrelated():
    """Options lexically unrelated to Q/P → both lexical baselines near chance."""
    from research.demo.phase1_5.data_musique import musique_distractor_leak_check

    rng = np.random.default_rng(0)
    vocab = [f"w{k}" for k in range(200)]
    rows = []
    for _ in range(80):
        q = " ".join(rng.choice(vocab, 3))
        opts = [" ".join(rng.choice(vocab, 2)) for _ in range(4)]
        rows.append(
            {"passage": " ".join(rng.choice(vocab, 10)), "question": q, "options": opts,
             "answer_idx": int(rng.integers(0, 4)), "reasoning_type": "2hop",
             "source": "musique", "split": "test"}
        )
    res = musique_distractor_leak_check(_mc_df(rows))
    assert res["question_to_option_acc"] <= 0.45  # near chance 0.25


# ---- load_musique (fake HF) -------------------------------------------------------


def test_load_musique_emits_mc_schema_with_holdout_test(monkeypatch):
    """load_musique → generic 7-col schema; validation→val; a deterministic
    holdout is carved from train into test (MuSiQue has no labelled public test)."""
    from research.demo.phase1_5 import data_musique

    train = [
        _musique_row(hops=4, gold=str(1900 + i), intermediates=[f"A{i}", f"B{i}", f"C{i}"])
        for i in range(20)
    ]
    val = [
        _musique_row(hops=4, gold=str(1800 + i), intermediates=[f"D{i}", f"E{i}", f"F{i}"])
        for i in range(8)
    ]
    monkeypatch.setattr(
        data_musique, "_hf_load_musique", lambda cfg: {"train": train, "validation": val}
    )
    cfg = MCCorpusConfig(corpus="musique", musique_holdout_test_frac=0.2)
    df = data_musique.load_musique(cfg)

    assert set(df.columns) >= {
        "passage", "question", "options", "answer_idx", "reasoning_type", "source", "split",
    }
    assert (df["source"] == "musique").all()
    assert df["options"].apply(len).eq(4).all()
    assert df["answer_idx"].between(0, 3).all()
    assert set(df["split"]) <= {"train", "val", "test"}
    assert (df["split"] == "test").any()  # holdout carved from train
    assert (df["split"] == "val").sum() == 8


def test_build_mc_corpus_dispatches_to_musique(monkeypatch, tmp_path):
    """build_mc_corpus(corpus='musique') routes to load_musique (not LogiQA)."""
    import pandas as pd

    from research.demo.phase1_5.data import build_mc_corpus

    fake = pd.DataFrame(
        [
            {"passage": "p", "question": "q", "options": ["a", "b", "c", "d"],
             "answer_idx": 0, "reasoning_type": "2hop", "source": "musique",
             "split": s}
            for s in (["train"] * 6 + ["val"] * 2 + ["test"] * 2)
        ]
    )
    monkeypatch.setattr(
        "research.demo.phase1_5.data_musique.load_musique", lambda cfg: fake
    )
    cfg = MCCorpusConfig(corpus="musique", cache_root=str(tmp_path))
    out = build_mc_corpus(cfg)
    assert (out["source"] == "musique").all()
    assert len(out) == 10
