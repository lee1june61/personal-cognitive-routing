"""Tests for the generation-anchor corpus loaders added to data.py (2026-05-26).

The new loaders (news / arXiv / sentiment140 / Amazon Reviews) call HuggingFace
`datasets.load_dataset`. Those tests monkeypatch `load_dataset` (+ `concatenate_datasets`)
with tiny in-memory fakes — a legitimate boundary mock (HF download), not an internal
collaborator. We exercise the public `_load_*_part(cfg)` behaviour: text construction
from source fields, char filter + cap, and graceful `_empty_part()` on failure.

CPU-only, no network. Mirrors test_phase1_ablation.py conventions (module helpers, no
fixtures).
"""

from __future__ import annotations

import sys
import types

import pytest

pd = pytest.importorskip("pandas")

from phase1 import data


# ---------------------------------------------------------------------------
# Fake HuggingFace datasets surface (tiny, in-memory)
# ---------------------------------------------------------------------------


class _FakeDS:
    """Minimal Dataset: column dict + the access methods our loaders use."""

    def __init__(self, cols: dict[str, list]):
        self._cols = cols

    @property
    def column_names(self) -> list[str]:
        return list(self._cols.keys())

    def __getitem__(self, key):
        return self._cols[key]

    def select_columns(self, cols):
        return _FakeDS({c: self._cols[c] for c in cols})

    def to_pandas(self):
        return pd.DataFrame(self._cols)

    def __len__(self) -> int:
        return len(next(iter(self._cols.values()))) if self._cols else 0


class _FakeDD(dict):
    """Minimal DatasetDict: split → _FakeDS (keys()/__getitem__ via dict)."""


class _FakeStream:
    """Minimal streaming dataset: only `.take(n)` is used by streaming loaders."""

    def __init__(self, rows: list[dict]):
        self._rows = rows

    def take(self, n: int):
        return self._rows[:n]


def _fake_concatenate(ds_list):
    merged: dict[str, list] = {}
    for ds in ds_list:
        for c in ds.column_names:
            merged.setdefault(c, []).extend(ds[c])
    return _FakeDS(merged)


def _fake_datasets_module(load_fn) -> types.ModuleType:
    """A stand-in `datasets` module — local env has no `datasets` installed, so the
    loaders' `from datasets import load_dataset, concatenate_datasets` resolves here."""
    mod = types.ModuleType("datasets")
    mod.load_dataset = load_fn
    mod.concatenate_datasets = _fake_concatenate
    return mod


def _install(monkeypatch, load_return):
    mod = _fake_datasets_module(lambda *a, **k: load_return)
    monkeypatch.setitem(sys.modules, "datasets", mod)


def _install_raises(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("HF mirror down")

    monkeypatch.setitem(sys.modules, "datasets", _fake_datasets_module(_boom))


# ---------------------------------------------------------------------------
# News loader
# ---------------------------------------------------------------------------


def test_load_news_builds_text_and_filters_short(monkeypatch):
    dd = _FakeDD(train=_FakeDS({
        "headline": ["Big news today about markets", "Tiny", "Third headline reported here"],
        "short_description": ["A full description sentence follows.", "x", "Another full description here."],
    }))
    _install(monkeypatch, dd)
    cfg = data.CorpusConfig()
    out = data._load_news_part(cfg)
    assert set(out.columns) >= {"text", "source", "author"}
    assert (out["source"] == data.SOURCE_NEWS).all()
    # headline + short_description joined; the "Tiny / x" row is below min_chars → dropped
    assert len(out) == 2
    assert out["text"].iloc[0].startswith("Big news today")


def test_load_news_graceful_on_failure(monkeypatch):
    _install_raises(monkeypatch)
    out = data._load_news_part(data.CorpusConfig())
    assert len(out) == 0
    assert list(out.columns) == list(data._EMPTY_PART_COLS)


# ---------------------------------------------------------------------------
# arXiv loader
# ---------------------------------------------------------------------------


def test_load_arxiv_builds_title_abstract(monkeypatch):
    dd = _FakeDD(train=_FakeDS({
        "title": ["Deep nets for vision tasks", "Short"],
        "abstract": ["We study deep networks and report strong results on vision.", "y"],
    }))
    _install(monkeypatch, dd)
    out = data._load_arxiv_part(data.CorpusConfig())
    assert (out["source"] == data.SOURCE_ARXIV).all()
    assert len(out) == 1  # second row below gen_min_chars
    assert "Deep nets for vision" in out["text"].iloc[0]


def test_load_arxiv_graceful_on_failure(monkeypatch):
    _install_raises(monkeypatch)
    assert len(data._load_arxiv_part(data.CorpusConfig())) == 0


# ---------------------------------------------------------------------------
# Tweet loader (cardiffnlp/tweet_eval)
# ---------------------------------------------------------------------------


def test_load_tweet_text_and_min_chars(monkeypatch):
    dd = _FakeDD(train=_FakeDS({
        "text": ["Loving the new phone its great", "hi", "Another decent tweet here"],
        "label": [2, 1, 0],
    }))
    _install(monkeypatch, dd)
    out = data._load_tweet_part(data.CorpusConfig())
    assert (out["source"] == data.SOURCE_TWEET).all()
    assert len(out) == 2  # "hi" below tweet_min_chars


def test_load_tweet_graceful_on_failure(monkeypatch):
    _install_raises(monkeypatch)
    assert len(data._load_tweet_part(data.CorpusConfig())) == 0


# ---------------------------------------------------------------------------
# Amazon Reviews (streaming) loader
# ---------------------------------------------------------------------------


def test_load_amazon_streams_title_body(monkeypatch):
    # fancyzhx/amazon_polarity rows: label / title / content
    rows = [
        {"label": 1, "title": "Great book", "content": "I really enjoyed reading this novel a lot."},
        {"label": 0, "title": "x", "content": "y"},
        {"label": 1, "title": "Decent", "content": "It was a reasonably good purchase overall here."},
    ]
    _install(monkeypatch, _FakeStream(rows))
    out = data._load_amazon_part(data.CorpusConfig())
    assert (out["source"] == data.SOURCE_AMAZON).all()
    assert len(out) == 2  # "x y" below gen_min_chars


def test_load_amazon_graceful_on_failure(monkeypatch):
    _install_raises(monkeypatch)
    assert len(data._load_amazon_part(data.CorpusConfig())) == 0


# ---------------------------------------------------------------------------
# Super-NaturalInstructions loader (operation-axis training corpus, 2026-05-27)
# ---------------------------------------------------------------------------


def test_load_superni_builds_text_and_groups_author_by_task(monkeypatch):
    # Muennighoff/natural-instructions schema: definition / inputs / task_name (+targets)
    dd = _FakeDD(train=_FakeDS({
        "definition": [
            "Given a sentence, classify its sentiment.",
            "Given a sentence, classify its sentiment.",
            "Translate the English sentence to French.",
        ],
        "inputs": [
            "The movie was absolutely wonderful and moving.",
            "x",  # too short → dropped after join with definition? definition is long, so kept
            "Hello, how are you doing today my friend?",
        ],
        "task_name": ["task1_sentiment", "task1_sentiment", "task2_translate"],
    }))
    _install(monkeypatch, dd)
    out = data._load_superni_part(data.CorpusConfig())
    assert set(out.columns) >= {"text", "source", "author"}
    assert (out["source"] == data.SOURCE_SUPERNI).all()
    # author groups by task_name (operation) so user_id factorize = per-operation grouping
    assert set(out["author"].unique()) == {"sni_task1_sentiment", "sni_task2_translate"}
    # definition is prepended so the operation type is encoded in the text
    assert "classify its sentiment" in out["text"].iloc[0]


def test_load_superni_caps_per_task(monkeypatch):
    dd = _FakeDD(train=_FakeDS({
        "definition": ["Do the operation precisely as described here."] * 10,
        "inputs": [f"instance input number {i} with sufficient length here" for i in range(10)],
        "task_name": ["onlytask"] * 10,
    }))
    _install(monkeypatch, dd)
    cfg = data.CorpusConfig(superni_max_per_task=3)
    out = data._load_superni_part(cfg)
    assert len(out) == 3  # capped per task


def test_load_superni_keeps_integer_task_id_zero(monkeypatch):
    # Mirror without task_name but with an integer `task_id` (candidate fallback).
    # task_id == 0 must NOT be falsy-dropped, and the author key must be a string.
    dd = _FakeDD(train=_FakeDS({
        "definition": ["Perform the described operation precisely on the input."] * 2,
        "inputs": ["first instance input here", "second instance input here"],
        "task_id": [0, 1],
    }))
    _install(monkeypatch, dd)
    out = data._load_superni_part(data.CorpusConfig())
    assert len(out) == 2  # task_id 0 survives
    assert "sni_0" in set(out["author"])


def test_load_superni_graceful_on_failure(monkeypatch):
    _install_raises(monkeypatch)
    out = data._load_superni_part(data.CorpusConfig())
    assert len(out) == 0
    assert list(out.columns) == list(data._EMPTY_PART_COLS)


# ---------------------------------------------------------------------------
# QuAIL probe loader (held-out reasoning-type = operation labels, 2026-05-27)
# ---------------------------------------------------------------------------


def test_load_quail_probe_builds_text_and_operation_label(monkeypatch):
    dd = _FakeDD(
        validation=_FakeDS({
            "context": ["The bridge collapsed after the heavy storm overnight."],
            "question": ["Why did the bridge collapse?"],
            "answers": [["Because of the storm", "It was old", "Vandalism", "Unknown"]],
            "correct_answer_id": [0],
            "question_type": ["Causality"],
        }),
    )
    _install(monkeypatch, dd)
    out = data.load_quail_probe(data.CorpusConfig())
    assert set(out.columns) >= {"text", "label"}
    # label = question_type (the operation axis)
    assert out["label"].iloc[0] == "Causality"
    # text = context + question + correct answer (correct_answer_id picks from answers)
    t = out["text"].iloc[0]
    assert "bridge collapsed" in t and "Why did the bridge" in t and "Because of the storm" in t


def test_load_quail_probe_question_first_reorders_text(monkeypatch):
    # question_first=True puts question + answer BEFORE context, so the operation-bearing
    # question survives right-truncation at small T (the Engine-A probe truncation fix).
    dd = _FakeDD(validation=_FakeDS({
        "context": ["The bridge collapsed after the heavy storm overnight."],
        "question": ["Why did the bridge collapse?"],
        "answers": [["Because of the storm", "It was old", "Vandalism", "Unknown"]],
        "correct_answer_id": [0],
        "question_type": ["Causality"],
    }))
    _install(monkeypatch, dd)
    t = data.load_quail_probe(data.CorpusConfig(), question_first=True)["text"].iloc[0]
    # question + answer precede the context
    assert t.index("Why did the bridge") < t.index("bridge collapsed")
    assert t.index("Because of the storm") < t.index("bridge collapsed")
    # default keeps context first (backward-compat)
    _install(monkeypatch, dd)
    t0 = data.load_quail_probe(data.CorpusConfig())["text"].iloc[0]
    assert t0.index("bridge collapsed") < t0.index("Why did the bridge")


def test_load_quail_probe_caps_samples(monkeypatch):
    n = 20
    dd = _FakeDD(validation=_FakeDS({
        "context": [f"Context sentence number {i} here." for i in range(n)],
        "question": [f"Question {i}?" for i in range(n)],
        "answers": [["a0", "a1", "a2", "a3"] for _ in range(n)],
        "correct_answer_id": [1] * n,
        "question_type": ["Causality"] * n,
    }))
    _install(monkeypatch, dd)
    out = data.load_quail_probe(data.CorpusConfig(quail_max_samples=5))
    assert len(out) == 5


def test_load_quail_probe_drops_string_answers_instead_of_char_indexing(monkeypatch):
    # If a mirror types `answers` as a JSON string, answers[int(cid)] would index a single
    # char silently — the loader must drop such rows, not corrupt the probe text.
    dd = _FakeDD(validation=_FakeDS({
        "context": ["Valid context one here.", "Valid context two here."],
        "question": ["Q one?", "Q two?"],
        "answers": [["good answer", "b", "c", "d"], "['x','y','z','w']"],  # 2nd is a string
        "correct_answer_id": [0, 0],
        "question_type": ["Causality", "Causality"],
    }))
    _install(monkeypatch, dd)
    out = data.load_quail_probe(data.CorpusConfig())
    assert len(out) == 1  # string-answers row dropped
    assert "good answer" in out["text"].iloc[0]
    assert "[" not in out["text"].iloc[0]  # no raw JSON-string fragment leaked in


def test_load_quail_probe_graceful_on_failure(monkeypatch):
    _install_raises(monkeypatch)
    out = data.load_quail_probe(data.CorpusConfig())
    assert len(out) == 0
    assert set(out.columns) >= {"text", "label"}


# ---------------------------------------------------------------------------
# build_corpus — wiring + split + user_id factorize
# ---------------------------------------------------------------------------


def _stub_part(src: str, n: int) -> "pd.DataFrame":
    return pd.DataFrame({
        "text": [f"{src} document number {i} with enough characters" for i in range(n)],
        "source": src,
        "author": [f"{src}_{i}" for i in range(n)],
    })


def test_build_corpus_wires_new_sources_splits_and_user_id(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "_load_news_part", lambda cfg: _stub_part(data.SOURCE_NEWS, 12))
    monkeypatch.setattr(data, "_load_arxiv_part", lambda cfg: _stub_part(data.SOURCE_ARXIV, 8))
    cfg = data.CorpusConfig(
        include_news=True, include_arxiv=True,
        include_tweet=False, include_amazon=False, include_reddit=False,
    )
    corpus = data.build_corpus(cfg, cache_path=tmp_path / "corpus.parquet")

    assert set(corpus["source"].unique()) == {data.SOURCE_NEWS, data.SOURCE_ARXIV}
    assert len(corpus) == 20
    assert set(corpus["split"].unique()) <= set(data.SPLITS)
    assert not corpus["split"].isna().any()
    # user_id is a dense factorize of distinct authors (20 distinct here)
    assert corpus["user_id"].nunique() == 20
    assert corpus["user_id"].notna().all()


def test_build_corpus_wires_superni(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "_load_superni_part", lambda cfg: _stub_part(data.SOURCE_SUPERNI, 15))
    cfg = data.CorpusConfig(
        include_news=False, include_arxiv=False, include_tweet=False,
        include_amazon=False, include_reddit=False, include_superni=True,
    )
    corpus = data.build_corpus(cfg, cache_path=tmp_path / "corpus.parquet")
    assert set(corpus["source"].unique()) == {data.SOURCE_SUPERNI}
    assert len(corpus) == 15
    assert not corpus["split"].isna().any()


def test_default_config_is_generation_anchor_set():
    cfg = data.CorpusConfig()
    assert (cfg.include_news, cfg.include_arxiv, cfg.include_tweet,
            cfg.include_amazon, cfg.include_reddit) == (True, True, True, True, True)
    assert not any([
        cfg.include_pennebaker, cfg.include_pandora, cfg.include_personachat,
        cfg.include_rocstories, cfg.include_anli, cfg.include_socialiqa,
    ])


def test_legacy_v6_corpus_config_restores_old_sources():
    cfg = data.legacy_v6_corpus_config()
    assert (cfg.include_pennebaker, cfg.include_reddit, cfg.include_pandora,
            cfg.include_personachat) == (True, True, True, True)
    assert not any([cfg.include_news, cfg.include_arxiv, cfg.include_tweet, cfg.include_amazon])
    # cache_key differs from the new default → no silent cache collision across corpora
    assert cfg.cache_key() != data.CorpusConfig().cache_key()


def test_build_corpus_raises_when_all_sources_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "_load_news_part", lambda cfg: data._empty_part())
    cfg = data.CorpusConfig(
        include_news=True, include_arxiv=False, include_tweet=False,
        include_amazon=False, include_reddit=False,
    )
    with pytest.raises(ValueError, match="disables every source"):
        data.build_corpus(cfg, cache_path=tmp_path / "corpus.parquet")
