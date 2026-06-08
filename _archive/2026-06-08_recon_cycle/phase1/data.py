"""Phase 1 corpus loader — Pennebaker + Reddit, sample-level split, BGE-large fact_emb.

Revision 4 (2026-05-21): LLM tokenize 제거, embedding-only path. fact_emb 는 BGE-large-en
masked-mean pool (1024d) — cache 로 한 번만 encode 후 재사용.

Phase 1 의 user-agnostic 정신:
- user_id 정보 = sample-level split 의 *index 만* (user-level holdout 아님)
- Phase 2 진입 시 user_id 가 routing 의 conditional 변수
- 같은 corpus, 같은 cache, 같은 split 으로 두 Phase 평가

Output (after build + encode):
- fact_emb: (N, 1024) — BGE-large-en cached
- user_id: (N,) — author 기반 (Pennebaker = unique per essay, Reddit = grouped)
- split: train/val/test labels
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ----- Constants ----------------------------------------------------------------------

# Reddit author values to skip (bot / deleted)
SKIP_AUTHORS = frozenset({"[deleted]", "AutoModerator"})

# Source labels for corpus rows
SOURCE_PENNEBAKER = "pennebaker"
SOURCE_REDDIT = "reddit"
SOURCE_PANDORA = "pandora"
SOURCE_PERSONACHAT = "personachat"
# Diversity expansion (2026-05-22, Path A — paradigm corpus diversification test)
SOURCE_ROCSTORIES = "rocstories"      # narrative / storytelling
SOURCE_ANLI = "anli"                  # abductive reasoning (cause→effect inference)
SOURCE_SOCIALIQA = "socialiqa"        # theory of mind / social commonsense
# Generation-anchor rebuild (2026-05-26) — corpus = union of generation eval domains
# (LaMP-4/5/7 + LongLaMP). real-user / consistent format / domain-aligned. See
# research/DATASET_SPEC_2026-05-26_generation_anchor.md.
SOURCE_NEWS = "news"                  # news reporting — heegyu/news-category-dataset (→ LaMP-4)
SOURCE_ARXIV = "arxiv"                # scholarly — CShorten/ML-ArXiv-Papers (→ LaMP-5, LongLaMP abstract)
SOURCE_TWEET = "tweet"                # short colloquial — cardiffnlp/tweet_eval (→ LaMP-7)
SOURCE_AMAZON = "amazon"             # evaluative review — fancyzhx/amazon_polarity (→ LongLaMP review)
# Operation-cycle pivot (2026-05-27) — operation-axis training corpus + probe.
# See research/demo/phase1/ENGINE_A_DESIGN.md §1. Super-NI = cycle training (operation
# variance, not topic), QuAIL = held-out probe with reasoning-type (operation) labels.
SOURCE_SUPERNI = "superni"           # Super-NaturalInstructions — Muennighoff/natural-instructions
# (sentiment140 / McAuley Amazon-Reviews-2023 use HF loading scripts, rejected by
#  datasets>=4.x; switched to parquet mirrors above, verified 2026-05-26.)

# Defensive column-name candidates (HF mirror schema drift) for the new loaders.
NEWS_HEADLINE_COL_CANDIDATES = ("headline", "title")
NEWS_BODY_COL_CANDIDATES = ("short_description", "description", "text", "content")
ARXIV_TITLE_COL_CANDIDATES = ("title",)
ARXIV_ABSTRACT_COL_CANDIDATES = ("abstract", "summary")
TWEET_TEXT_COL_CANDIDATES = ("text", "tweet", "content")
AMAZON_TITLE_COL_CANDIDATES = ("title",)
AMAZON_BODY_COL_CANDIDATES = ("text", "review_body", "content")

# PANDORA HF mirror schema drift defensives — fallback column-name candidates
PANDORA_TEXT_COL_CANDIDATES = ("body", "text", "comment")
PANDORA_AUTHOR_COL_CANDIDATES = ("author", "user", "user_id", "username")

# Super-NI (Muennighoff/natural-instructions) schema drift defensives.
SUPERNI_DEFINITION_COL_CANDIDATES = ("definition", "instruction")
SUPERNI_INPUT_COL_CANDIDATES = ("inputs", "input", "source")
SUPERNI_TASK_COL_CANDIDATES = ("task_name", "task", "task_id")
# QuAIL (textmachinelab/quail) — MC QA with reasoning-type labels (the operation axis).
QUAIL_CONTEXT_COL_CANDIDATES = ("context",)
QUAIL_QUESTION_COL_CANDIDATES = ("question",)
QUAIL_ANSWERS_COL_CANDIDATES = ("answers",)
# Only `correct_answer_id` (0-based index into `answers`). A `label` fallback was rejected:
# its index base/semantics differ across mirrors, so it would silently mis-index the
# correct answer (in-range → no error), poisoning the probe.
QUAIL_CORRECT_ID_COL_CANDIDATES = ("correct_answer_id",)
QUAIL_QTYPE_COL_CANDIDATES = ("question_type", "metadata_question_type")

# Shared empty fallback for loader skip paths
_EMPTY_PART_COLS = ("text", "source", "author")


def _empty_part() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_EMPTY_PART_COLS))


def _first_present_in(available: tuple[str, ...], candidates: tuple[str, ...]) -> str | None:
    return next((c for c in candidates if c in available), None)


def _cap_rows(rows: list[dict], cap: int, seed: int) -> list[dict]:
    """Uniformly down-sample a list of row dicts to `cap` (seeded). No-op if already small."""
    if len(rows) <= cap:
        return rows
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(rows), size=cap, replace=False)
    return [rows[i] for i in keep]


def _cap_and_filter_texts(
    texts: list[str],
    max_samples: int,
    min_chars: int,
    seed: int,
    source: str,
    prefix: str,
) -> pd.DataFrame:
    """Common tail for new dataset loaders: char filter + sample cap + DataFrame build."""
    texts = [t for t in texts if len(t) >= min_chars]
    if len(texts) > max_samples:
        rng = np.random.default_rng(seed)
        texts = [texts[i] for i in rng.choice(len(texts), size=max_samples, replace=False)]
    if not texts:
        return _empty_part()
    return pd.DataFrame({
        "text": texts,
        "source": source,
        "author": [f"{prefix}_{i}" for i in range(len(texts))],
    })

# Train/val/test split labels
SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"
SPLITS = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)


# ----- Config --------------------------------------------------------------------------

@dataclass
class CorpusConfig:
    """Multi-source corpus config (revision 5, 2026-05-22 — Path A diversification).

    Cycle pretrain corpus:
    - Original (revision 4): Pennebaker + Reddit + PANDORA + PersonaChat (~118k).
      Opinion/advice/factual/self-expression 위주.
    - Diversity expansion: ROCStories (narrative) + αNLI (abductive reasoning) +
      SocialIQA (theory of mind / social commonsense) (~70k).

    Total ~190k samples covering broader cognitive operation space — paradigm 의
    expert specialization 의 emergence 조건 검증 (Path A test).
    """
    # Reddit — multi-subreddit for cognitive context diversity
    reddit_splits: tuple[str, ...] = (
        "programming", "politics", "cooking", "science",
        "relationships", "personalfinance", "AskHistorians", "philosophy",
    )
    reddit_max_rows_per_split: int = 80_000
    reddit_min_comments: int = 5
    reddit_max_comments: int = 25
    min_comment_chars: int = 80

    # PANDORA (jingjietan/pandora-big5) — Reddit users with OCEAN labels, multi-comment.
    # Falls back to no-op if HF mirror strips the `author` column (silently skipped).
    pandora_max_comments_per_user: int = 30
    pandora_min_comments_per_user: int = 5

    # PersonaChat (bavard/personachat_truecased) — persona-grounded dialogue.
    # `persona` (synthetic) treated as user_id. Useful for cycle context diversity,
    # but `use_user=True` Phase 2 should weight real-user sources (Reddit/PANDORA) higher.
    personachat_max_responses_per_persona: int = 20

    # Diversity expansion (Path A) — paradigm corpus-diversification test.
    # 우리 corpus 가 opinion/advice/factual/self-expression 위주 → narrative/abductive/
    # social reasoning 명시적 부재. expert specialization 의 emergence signal 부족 가설.
    rocstories_max_samples: int = 30_000
    anli_max_samples: int = 20_000
    socialiqa_max_samples: int = 20_000
    # Lower char threshold for structured short text (ROCStories ~150-400, αNLI ~200,
    # SocialIQA ~150). Reddit/PANDORA 의 min_comment_chars=80 와 별도.
    structured_min_chars: int = 40

    # Generation-anchor rebuild (2026-05-26). News/arXiv/Amazon = long-form prose;
    # tweets = short. Caps keep Colab RAM bounded (Amazon 2023 = 571M rows → stream).
    news_max_samples: int = 40_000
    arxiv_max_samples: int = 40_000
    tweet_max_samples: int = 40_000
    tweet_config: str = "sentiment"              # cardiffnlp/tweet_eval requires a config
    amazon_max_samples: int = 40_000
    amazon_scan_multiplier: int = 3              # over-scan: take N*mult rows, keep N after min-char filter
    gen_min_chars: int = 40                      # news / arXiv / Amazon prose floor
    tweet_min_chars: int = 15                    # tweets are short

    # Operation-cycle pivot (2026-05-27, ENGINE_A_DESIGN §1). Super-NI = training corpus
    # along the operation axis (many tasks, few instances each → operation variance
    # dominates topic). QuAIL = held-out probe (reasoning-type = operation label).
    superni_max_samples: int = 40_000
    superni_max_per_task: int = 50               # cap per task → broad operation coverage
    superni_min_chars: int = 40
    quail_max_samples: int = 8_000

    # Source toggles. Defaults = generation-anchor set (2026-05-26): news + arXiv +
    # sentiment140 + Amazon + Reddit. Synthetic/structured sources (PersonaChat,
    # ROCStories, αNLI, SocialIQA) + PANDORA(personality) disabled — Phase 1 selects
    # corpus by cognitive variance + format consistency, not personality/per-user.
    include_news: bool = True
    include_arxiv: bool = True
    include_tweet: bool = True
    include_amazon: bool = True
    include_reddit: bool = True
    include_pennebaker: bool = False
    include_pandora: bool = False
    include_personachat: bool = False
    include_rocstories: bool = False
    include_anli: bool = False
    include_socialiqa: bool = False
    # Operation-cycle (2026-05-27): off by default. NOTE cache_key() hashes asdict(self),
    # so adding these fields *does* change the key of every config (incl. the bare default)
    # → the next build_corpus() / encode_or_load() rebuilds the corpus parquet (re-filters
    # the HF sources). The fact_emb .npy is text-hash-keyed, so it still hits if the rebuilt
    # text rows are identical. The Engine-A runner builds a Super-NI-only corpus explicitly.
    include_superni: bool = False

    # Encoder
    encoder_name: str = "BAAI/bge-large-en-v1.5"
    encoder_max_length: int = 256
    seed: int = 42

    def cache_key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha1(payload).hexdigest()[:10]


# ----- Legacy config escape hatch ------------------------------------------------------

def legacy_v6_corpus_config(**overrides) -> CorpusConfig:
    """The pre-2026-05-26 (v3–v6 ablation) 7-source corpus as an explicit config.

    ⚠️ BLAST RADIUS of the 2026-05-26 default flip: `CorpusConfig()` now builds the
    *generation-anchor* corpus (news+arXiv+tweet+amazon+reddit), NOT the corpus the
    recorded v3–v6 runs were trained on. Any code that constructs a bare `CorpusConfig()`
    (e.g. eval.py, train.py, cluster_analysis.py, notebooks) now gets the NEW corpus and a
    new cache_key. To reproduce / evaluate an OLD run, pass this config explicitly.
    """
    base = dict(
        include_news=False, include_arxiv=False, include_tweet=False, include_amazon=False,
        include_reddit=True, include_pennebaker=True, include_pandora=True,
        include_personachat=True, include_rocstories=True, include_anli=True,
        include_socialiqa=True,
    )
    base.update(overrides)
    return CorpusConfig(**base)


# ----- Corpus build --------------------------------------------------------------------

def _load_pennebaker_part():
    """Pennebaker Essays — 1 essay/user, Big Five labelled (~2.5k)."""
    from datasets import load_dataset, concatenate_datasets
    ds = load_dataset("jingjietan/essays-big5")
    merged = concatenate_datasets([ds[s] for s in ds.keys()])
    texts = merged["text"]
    table = pd.DataFrame({
        "text": texts,
        "source": SOURCE_PENNEBAKER,
        "author": [f"pen_{i}" for i in range(len(texts))],
    })
    print(f"[corpus] Pennebaker: {len(table)} essays")
    return table


def _load_reddit_part(cfg: "CorpusConfig"):
    """Reddit multi-subreddit — broader cognitive context coverage."""
    from datasets import load_dataset
    from tqdm.auto import tqdm

    # Cross-subreddit author dedup: same Reddit user posting in multiple subreddits
    # contributes to a *single* user_id — Phase 2 의 user_logits 가 *user 의 다양
    # cognitive context activity* 를 single distribution 으로 학습 (paradigm 의 진짜 intent).
    author_to_bodies: dict[str, list[str]] = {}
    for split in cfg.reddit_splits:
        try:
            stream = load_dataset(
                "HuggingFaceGECLM/REDDIT_comments", split=split, streaming=True,
            )
        except Exception as e:
            print(f"[corpus] r/{split} load failed ({type(e).__name__}); skipping")
            continue
        scanned = 0
        for row in tqdm(
            stream.take(cfg.reddit_max_rows_per_split),
            total=cfg.reddit_max_rows_per_split,
            desc=f"scan r/{split}", unit="rows",
        ):
            scanned += 1
            a = row.get("author")
            if not a or a in SKIP_AUTHORS:
                continue
            body = row.get("body") or ""
            if len(body) < cfg.min_comment_chars:
                continue
            bodies = author_to_bodies.setdefault(a, [])
            if len(bodies) < cfg.reddit_max_comments:
                bodies.append(body)
        print(f"[corpus] r/{split}: scanned {scanned}")

    active = {a: bs for a, bs in author_to_bodies.items() if len(bs) >= cfg.reddit_min_comments}
    rows = [
        {"text": body, "source": SOURCE_REDDIT, "author": f"rdt_{a}"}
        for a, bs in active.items() for body in bs
    ]
    if not rows:
        return _empty_part()
    table = pd.DataFrame(rows)
    print(f"[corpus] Reddit (8 subreddits, dedup): {len(table)} comments / {len(active)} unique users")
    return table


def _load_pandora_part(cfg: "CorpusConfig"):
    """PANDORA Big Five labelled (jingjietan/pandora-big5) — Reddit users × OCEAN.

    Silently returns empty if HF mirror strips the author column (observed historically).
    """
    from datasets import load_dataset, concatenate_datasets
    try:
        ds = load_dataset("jingjietan/pandora-big5")
    except Exception as e:
        print(f"[corpus] PANDORA load failed ({type(e).__name__}); skipping")
        return _empty_part()

    # Peek at the first split's columns to find the text/author keys before deserializing.
    sample_split = next(iter(ds.keys()))
    sample_cols = tuple(ds[sample_split].column_names)
    text_col = _first_present_in(sample_cols, PANDORA_TEXT_COL_CANDIDATES)
    author_col = _first_present_in(sample_cols, PANDORA_AUTHOR_COL_CANDIDATES)
    if text_col is None or author_col is None:
        print(f"[corpus] PANDORA: missing text/author col (cols={list(sample_cols)}); skipping")
        return _empty_part()

    # Select only the columns we need before to_pandas — cuts pandas peak memory ~3-5x
    # on the multi-MB-per-row PANDORA mirror.
    merged = concatenate_datasets([ds[s] for s in ds.keys()]).select_columns([text_col, author_col])
    df = merged.to_pandas()
    df = df.dropna(subset=[text_col, author_col])
    df = df[df[text_col].str.len() >= cfg.min_comment_chars]

    # Vectorized: shuffle once + groupby.head(N) replaces filter→groupby.apply(sample).
    # Then filter active users by count.
    df = (df.sample(frac=1.0, random_state=cfg.seed)
            .groupby(author_col, group_keys=False, sort=False)
            .head(cfg.pandora_max_comments_per_user))
    counts = df.groupby(author_col).size()
    active_authors = counts[counts >= cfg.pandora_min_comments_per_user].index
    df = df[df[author_col].isin(active_authors)].reset_index(drop=True)

    table = pd.DataFrame({
        "text": df[text_col].astype(str).values,
        "source": SOURCE_PANDORA,
        "author": [f"pan_{a}" for a in df[author_col].values],
    })
    print(f"[corpus] PANDORA: {len(table)} comments / {len(active_authors)} labelled users")
    return table


def _load_personachat_part(cfg: "CorpusConfig"):
    """PersonaChat (bavard/personachat_truecased) — persona-grounded dialogue responses.

    Treats each persona as a (synthetic) user. Phase 2 routing should weight real-user
    sources (Reddit/PANDORA) higher — PersonaChat contributes context diversity, not
    real-user signal.

    Schema assumption: each row has `personality` (list of persona sentences) and
    `utterances` (list of {`history`, `candidates`}) with the gold response at
    `candidates[-1]`. Verifies columns at runtime; returns empty + diagnostic on schema
    drift or load failure.
    """
    from datasets import load_dataset, concatenate_datasets
    try:
        ds = load_dataset("bavard/personachat_truecased")
    except Exception as e:
        print(f"[corpus] PersonaChat load failed ({type(e).__name__}); skipping")
        return _empty_part()

    sample_split = next(iter(ds.keys()))
    sample_cols = ds[sample_split].column_names
    if "personality" not in sample_cols or "utterances" not in sample_cols:
        print(f"[corpus] PersonaChat schema mismatch (cols={sample_cols}); skipping")
        return _empty_part()

    # Drop the ~6 unused cols (history/candidates/etc. at row level) before to_pandas.
    df = concatenate_datasets([ds[s] for s in ds.keys()]).select_columns(["personality", "utterances"]).to_pandas()

    rows = []
    persona_seen: dict[tuple, int] = {}
    max_resp = cfg.personachat_max_responses_per_persona
    for personality, utterances in df[["personality", "utterances"]].itertuples(index=False):
        persona = tuple(personality or ())
        if not persona:
            continue
        pid = persona_seen.setdefault(persona, len(persona_seen))
        n_resp = 0
        for u in (utterances or ()):
            cands = u.get("candidates") if isinstance(u, dict) else None
            if not cands:
                continue
            resp = cands[-1]                                    # gold response (PersonaChat convention)
            if isinstance(resp, str) and len(resp) >= cfg.min_comment_chars:
                rows.append({"text": resp, "source": SOURCE_PERSONACHAT, "author": f"per_{pid}"})
                n_resp += 1
            if n_resp >= max_resp:
                break

    if not rows:
        print(f"[corpus] PersonaChat: 0 responses extracted (schema may have drifted)")
        return _empty_part()
    table = pd.DataFrame(rows)
    print(f"[corpus] PersonaChat: {len(table)} responses / {len(persona_seen)} personas")
    return table


def _load_rocstories_part(cfg: "CorpusConfig"):
    """ROCStories — 5-sentence commonsense narrative (~98k stories).

    Cognitive operation: narrative / storytelling / temporal event sequence.
    우리 baseline corpus 에 명시적으로 부재. text = 5 sentences joined.

    Schema (mintujupally/ROCStories): single-column `text` 또는 sentence1..5 column.
    HF mirror schema drift 시 silently skip.
    """
    from datasets import load_dataset, concatenate_datasets
    try:
        ds = load_dataset("mintujupally/ROCStories")
    except Exception as e:
        print(f"[corpus] ROCStories load failed ({type(e).__name__}); skipping")
        return _empty_part()

    sample_split = next(iter(ds.keys()))
    cols = tuple(ds[sample_split].column_names)
    merged = concatenate_datasets([ds[s] for s in ds.keys()])

    if "text" in cols:
        texts = [t for t in merged["text"] if isinstance(t, str)]
    elif all(f"sentence{i}" in cols for i in range(1, 6)):
        texts = [
            " ".join(str(s) for s in row if s) for row in zip(*[merged[f"sentence{i}"] for i in range(1, 6)])
        ]
    elif "story" in cols:
        texts = [t for t in merged["story"] if isinstance(t, str)]
    else:
        print(f"[corpus] ROCStories schema mismatch (cols={list(cols)}); skipping")
        return _empty_part()

    table = _cap_and_filter_texts(
        texts, cfg.rocstories_max_samples, cfg.structured_min_chars,
        cfg.seed, SOURCE_ROCSTORIES, "roc",
    )
    if len(table):
        print(f"[corpus] ROCStories (narrative): {len(table)} stories")
    return table


def _load_anli_part(cfg: "CorpusConfig"):
    """αNLI (allenai/art) — Abductive NLI (~170k).

    Cognitive operation: abductive reasoning (between two observations, infer the
    intermediate cause). text = "obs1. obs2. Hypothesis: correct_hyp" 형식.

    Schema: observation_1, observation_2, hypothesis_1, hypothesis_2, label (1 or 2).
    """
    from datasets import load_dataset, concatenate_datasets
    try:
        ds = load_dataset("allenai/art")
    except Exception as e:
        print(f"[corpus] αNLI load failed ({type(e).__name__}); skipping")
        return _empty_part()

    sample_split = next(iter(ds.keys()))
    cols = tuple(ds[sample_split].column_names)
    required = ("observation_1", "observation_2", "hypothesis_1", "hypothesis_2", "label")
    if not all(c in cols for c in required):
        print(f"[corpus] αNLI schema mismatch (cols={list(cols)}); skipping")
        return _empty_part()

    df = concatenate_datasets([ds[s] for s in ds.keys()]).select_columns(list(required)).to_pandas()
    # label = 1 or 2 — coerce, drop NaN, restrict to valid label set (silent garbage 방지)
    df["label_int"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label_int"])
    df["label_int"] = df["label_int"].astype(int)
    df = df[df["label_int"].isin({1, 2})]
    correct_hyp = np.where(df["label_int"].values == 1, df["hypothesis_1"].values, df["hypothesis_2"].values)
    texts = [
        f"{o1} {o2} Hypothesis: {h}"
        for o1, o2, h in zip(df["observation_1"], df["observation_2"], correct_hyp)
        if isinstance(o1, str) and isinstance(o2, str) and isinstance(h, str)
    ]
    table = _cap_and_filter_texts(
        texts, cfg.anli_max_samples, cfg.structured_min_chars,
        cfg.seed, SOURCE_ANLI, "anli",
    )
    if len(table):
        print(f"[corpus] αNLI (abductive reasoning): {len(table)} samples")
    return table


def _load_socialiqa_part(cfg: "CorpusConfig"):
    """SocialIQA (allenai/social_i_qa) — social commonsense (~38k).

    Cognitive operation: theory of mind / social commonsense (predict motivation,
    emotion, reaction). text = "context question correct_answer" 형식.

    Schema: context, question, answerA/B/C, label ("1"/"2"/"3" string).
    """
    from datasets import load_dataset, concatenate_datasets
    try:
        ds = load_dataset("allenai/social_i_qa", trust_remote_code=True)
    except Exception as e:
        print(f"[corpus] SocialIQA load failed ({type(e).__name__}); skipping")
        return _empty_part()

    sample_split = next(iter(ds.keys()))
    cols = tuple(ds[sample_split].column_names)
    required = ("context", "question", "answerA", "answerB", "answerC", "label")
    if not all(c in cols for c in required):
        print(f"[corpus] SocialIQA schema mismatch (cols={list(cols)}); skipping")
        return _empty_part()

    df = concatenate_datasets([ds[s] for s in ds.keys()]).select_columns(list(required)).to_pandas()
    df["label_int"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label_int"])
    df["label_int"] = df["label_int"].astype(int)
    df = df[df["label_int"].isin({1, 2, 3})]
    answers = np.where(df["label_int"].values == 1, df["answerA"].values,
              np.where(df["label_int"].values == 2, df["answerB"].values, df["answerC"].values))
    texts = [
        f"{c} {q} {a}"
        for c, q, a in zip(df["context"], df["question"], answers)
        if isinstance(c, str) and isinstance(q, str) and isinstance(a, str)
    ]
    table = _cap_and_filter_texts(
        texts, cfg.socialiqa_max_samples, cfg.structured_min_chars,
        cfg.seed, SOURCE_SOCIALIQA, "siq",
    )
    if len(table):
        print(f"[corpus] SocialIQA (theory of mind): {len(table)} samples")
    return table


def _join_text_columns(merged, head_col, body_col) -> list[str]:
    """Join an optional headline/title column with a body column into prose lines.

    Either column may be absent (None) or carry non-str cells; both are coerced to "".
    Empty joins are dropped (length filtering happens later in _cap_and_filter_texts).
    """
    n = len(merged)
    heads = merged[head_col] if head_col else [""] * n
    bodies = merged[body_col] if body_col else [""] * n
    texts = []
    for h, b in zip(heads, bodies):
        h = h if isinstance(h, str) else ""
        b = b if isinstance(b, str) else ""
        t = f"{h} {b}".strip()
        if t:
            texts.append(t)
    return texts


def _load_news_part(cfg: "CorpusConfig"):
    """News reporting — HuffPost (heegyu/news-category-dataset). → LaMP-4 headline domain.

    text = headline + short_description. Schema drift / mirror failure → empty (skip).
    """
    from datasets import load_dataset, concatenate_datasets
    try:
        ds = load_dataset("heegyu/news-category-dataset")
    except Exception as e:
        print(f"[corpus] News load failed ({type(e).__name__}); skipping")
        return _empty_part()

    sample_cols = tuple(ds[next(iter(ds.keys()))].column_names)
    head_col = _first_present_in(sample_cols, NEWS_HEADLINE_COL_CANDIDATES)
    body_col = _first_present_in(sample_cols, NEWS_BODY_COL_CANDIDATES)
    if head_col is None and body_col is None:
        print(f"[corpus] News schema mismatch (cols={list(sample_cols)}); skipping")
        return _empty_part()

    merged = concatenate_datasets([ds[s] for s in ds.keys()])
    texts = _join_text_columns(merged, head_col, body_col)
    table = _cap_and_filter_texts(
        texts, cfg.news_max_samples, cfg.gen_min_chars, cfg.seed, SOURCE_NEWS, "news",
    )
    if len(table):
        print(f"[corpus] News (HuffPost): {len(table)} articles")
    return table


def _load_arxiv_part(cfg: "CorpusConfig"):
    """Scholarly abstraction — CShorten/ML-ArXiv-Papers (title + abstract). → LaMP-5.

    Schema drift / mirror failure → empty (skip).
    """
    from datasets import load_dataset, concatenate_datasets
    try:
        ds = load_dataset("CShorten/ML-ArXiv-Papers")
    except Exception as e:
        print(f"[corpus] arXiv load failed ({type(e).__name__}); skipping")
        return _empty_part()

    sample_cols = tuple(ds[next(iter(ds.keys()))].column_names)
    title_col = _first_present_in(sample_cols, ARXIV_TITLE_COL_CANDIDATES)
    abstract_col = _first_present_in(sample_cols, ARXIV_ABSTRACT_COL_CANDIDATES)
    if title_col is None and abstract_col is None:
        print(f"[corpus] arXiv schema mismatch (cols={list(sample_cols)}); skipping")
        return _empty_part()

    merged = concatenate_datasets([ds[s] for s in ds.keys()])
    texts = _join_text_columns(merged, title_col, abstract_col)
    table = _cap_and_filter_texts(
        texts, cfg.arxiv_max_samples, cfg.gen_min_chars, cfg.seed, SOURCE_ARXIV, "arx",
    )
    if len(table):
        print(f"[corpus] arXiv (scholarly): {len(table)} abstracts")
    return table


def _load_tweet_part(cfg: "CorpusConfig"):
    """Short colloquial — cardiffnlp/tweet_eval (tweet text). → LaMP-7.

    Parquet mirror (sentiment140 uses a loading script, rejected by datasets>=4.x).
    Requires the `sentiment` config. Single text column; lower char floor.
    """
    from datasets import load_dataset, concatenate_datasets
    try:
        ds = load_dataset("cardiffnlp/tweet_eval", cfg.tweet_config)
    except Exception as e:
        print(f"[corpus] tweet_eval load failed ({type(e).__name__}); skipping")
        return _empty_part()

    sample_cols = tuple(ds[next(iter(ds.keys()))].column_names)
    text_col = _first_present_in(sample_cols, TWEET_TEXT_COL_CANDIDATES)
    if text_col is None:
        print(f"[corpus] tweet_eval schema mismatch (cols={list(sample_cols)}); skipping")
        return _empty_part()

    merged = concatenate_datasets([ds[s] for s in ds.keys()])
    texts = [t for t in merged[text_col] if isinstance(t, str)]
    table = _cap_and_filter_texts(
        texts, cfg.tweet_max_samples, cfg.tweet_min_chars, cfg.seed, SOURCE_TWEET, "twt",
    )
    if len(table):
        print(f"[corpus] tweet_eval (tweets): {len(table)} tweets")
    return table


def _load_amazon_part(cfg: "CorpusConfig"):
    """Evaluative review — fancyzhx/amazon_polarity (title + content). → LongLaMP review.

    Parquet mirror (McAuley Amazon-Reviews-2023 uses a loading script, rejected by
    datasets>=4.x). ~3.6M rows → streaming + scan budget. Streamed rows are plain dicts.
    """
    from datasets import load_dataset
    try:
        stream = load_dataset(
            "fancyzhx/amazon_polarity", split="train", streaming=True,
        )
    except Exception as e:
        print(f"[corpus] Amazon load failed ({type(e).__name__}); skipping")
        return _empty_part()

    texts: list[str] = []
    try:
        # Over-scan: short reviews get filtered out, so scan more than the keep cap to
        # avoid silently under-filling Amazon relative to the other sources.
        for row in stream.take(cfg.amazon_max_samples * cfg.amazon_scan_multiplier):
            title = _first_present_in(tuple(row.keys()), AMAZON_TITLE_COL_CANDIDATES)
            body = _first_present_in(tuple(row.keys()), AMAZON_BODY_COL_CANDIDATES)
            h = row.get(title) if title else ""
            b = row.get(body) if body else ""
            h = h if isinstance(h, str) else ""
            b = b if isinstance(b, str) else ""
            t = f"{h} {b}".strip()
            if t:
                texts.append(t)
    except Exception as e:
        print(f"[corpus] Amazon stream error ({type(e).__name__}); using {len(texts)} scanned")

    table = _cap_and_filter_texts(
        texts, cfg.amazon_max_samples, cfg.gen_min_chars, cfg.seed, SOURCE_AMAZON, "amz",
    )
    if len(table):
        print(f"[corpus] Amazon reviews (amazon_polarity): {len(table)} reviews")
    return table


def _load_superni_part(cfg: "CorpusConfig"):
    """Super-NaturalInstructions (Muennighoff/natural-instructions) — operation-axis corpus.

    Operation-cycle pivot (2026-05-27, ENGINE_A_DESIGN §1): the cycle trains along the
    *operation* axis, not topic. Each row carries a `definition` (the operation spec) and
    an `inputs` instance; `text = definition + inputs` so the operation type is encoded in
    the text. `author = task_name` so the user_id factorize groups *by operation* (harmless
    for Phase 1 which ignores user_id, useful for post-hoc operation-cluster analysis).

    Per-task cap keeps operation coverage broad (many tasks) rather than a few large tasks
    dominating — the ST-MoE "token-type specialization" failure mode we are avoiding.
    Schema drift / mirror failure → empty (skip).
    """
    from datasets import load_dataset, concatenate_datasets
    try:
        # verification_mode="no_checks": the Muennighoff/natural-instructions metadata
        # declares a `validation` split that the current mirror no longer materialises
        # (only train + test download), so the default post-download split-count check
        # raises ExpectedMoreSplitsError even though the data we need is fully present.
        ds = load_dataset("Muennighoff/natural-instructions", verification_mode="no_checks")
    except Exception as e:
        print(f"[corpus] Super-NI load failed ({type(e).__name__}); skipping")
        return _empty_part()

    sample_cols = tuple(ds[next(iter(ds.keys()))].column_names)
    def_col = _first_present_in(sample_cols, SUPERNI_DEFINITION_COL_CANDIDATES)
    in_col = _first_present_in(sample_cols, SUPERNI_INPUT_COL_CANDIDATES)
    task_col = _first_present_in(sample_cols, SUPERNI_TASK_COL_CANDIDATES)
    if in_col is None or task_col is None:
        print(f"[corpus] Super-NI schema mismatch (cols={list(sample_cols)}); skipping")
        return _empty_part()

    # Select only the needed columns before to_pandas — Super-NI is ~1.6M rows with long
    # `definition` strings; decoding every column to Python lists would OOM Colab before a
    # single row is filtered (matches the αNLI/SocialIQA loaders' Arrow-columnar pattern).
    needed = [c for c in (def_col, in_col, task_col) if c]
    df = concatenate_datasets([ds[s] for s in ds.keys()]).select_columns(needed).to_pandas()
    rows: list[dict] = []
    per_task: dict[str, int] = {}
    for row in df.itertuples(index=False):
        rec = row._asdict()
        d = rec.get(def_col) if def_col else ""
        d = d if isinstance(d, str) else ""
        inp = rec.get(in_col)
        inp = inp if isinstance(inp, str) else ""
        # task_col may be int (task_id) — coerce to str so task 0 is not falsy-dropped and
        # author keys stay homogeneous strings.
        task_raw = rec.get(task_col)
        task = str(task_raw) if task_raw is not None and task_raw != "" else ""
        text = f"{d} {inp}".strip()
        if len(text) < cfg.superni_min_chars or not task:
            continue
        if per_task.get(task, 0) >= cfg.superni_max_per_task:
            continue
        per_task[task] = per_task.get(task, 0) + 1
        rows.append({"text": text, "source": SOURCE_SUPERNI, "author": f"sni_{task}"})

    if not rows:
        print(f"[corpus] Super-NI: 0 instances extracted (schema may have drifted)")
        return _empty_part()
    # Global cap (operation-balanced: per-task cap already applied, so a uniform sample
    # preserves operation diversity) via the shared seed.
    rows = _cap_rows(rows, cfg.superni_max_samples, cfg.seed)
    table = pd.DataFrame(rows)
    print(f"[corpus] Super-NI (operation-axis): {len(table)} instances / {len(per_task)} tasks")
    return table


_EMPTY_PROBE_COLS = ("text", "label")


def _empty_probe() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_EMPTY_PROBE_COLS))


def load_quail_probe(
    cfg: "CorpusConfig | None" = None, question_first: bool = False
) -> pd.DataFrame:
    """QuAIL (textmachinelab/quail) — held-out operation probe (ENGINE_A_DESIGN §5).

    Returns a `(text, label)` DataFrame where `label = question_type` (the reasoning-type =
    *operation* axis the Engine-A go/no-go gate tests for). `text = context + question +
    correct answer` (correct_answer_id indexes the answers list). This is a *probe*, not
    training corpus — it carries labels and does NOT flow through `build_corpus`.

    `question_first=True` swaps the order to `question + answer + context`. The operation
    signal lives in the question (reasoning-type), but the default context-first order puts
    it last, so at a small token cap (T=128) QuAIL's long context fills the window and
    right-truncation drops the question entirely — probing the model on topic-only input.
    Question-first keeps the operation-bearing text inside the truncation window (the
    Engine-A probe-truncation fix; see RESEARCH_PLAN go/no-go diagnosis).

    Schema drift / mirror failure → empty (skip).
    """
    cfg = cfg or CorpusConfig()
    from datasets import load_dataset, concatenate_datasets
    try:
        ds = load_dataset("textmachinelab/quail")
    except Exception as e:
        print(f"[probe] QuAIL load failed ({type(e).__name__}); skipping")
        return _empty_probe()

    sample_cols = tuple(ds[next(iter(ds.keys()))].column_names)
    ctx_col = _first_present_in(sample_cols, QUAIL_CONTEXT_COL_CANDIDATES)
    q_col = _first_present_in(sample_cols, QUAIL_QUESTION_COL_CANDIDATES)
    ans_col = _first_present_in(sample_cols, QUAIL_ANSWERS_COL_CANDIDATES)
    cid_col = _first_present_in(sample_cols, QUAIL_CORRECT_ID_COL_CANDIDATES)
    qt_col = _first_present_in(sample_cols, QUAIL_QTYPE_COL_CANDIDATES)
    if any(c is None for c in (ctx_col, q_col, ans_col, cid_col, qt_col)):
        print(f"[probe] QuAIL schema mismatch (cols={list(sample_cols)}); skipping")
        return _empty_probe()

    merged = concatenate_datasets([ds[s] for s in ds.keys()])
    rows: list[dict] = []
    for ctx, q, answers, cid, qt in zip(
        merged[ctx_col], merged[q_col], merged[ans_col], merged[cid_col], merged[qt_col]
    ):
        if not isinstance(ctx, str) or not isinstance(q, str) or not isinstance(qt, str):
            continue
        # `answers` must be a real sequence of options. A JSON-string mirror would let
        # answers[int(cid)] index a single CHARACTER with no error → silent corruption.
        if not isinstance(answers, (list, tuple, np.ndarray)):
            continue
        try:
            ans = answers[int(cid)]
        except (TypeError, ValueError, IndexError):
            continue
        ans = ans if isinstance(ans, str) else ""
        text = f"{q} {ans} {ctx}" if question_first else f"{ctx} {q} {ans}"
        rows.append({"text": text.strip(), "label": qt})

    if not rows:
        print(f"[probe] QuAIL: 0 items extracted (schema may have drifted)")
        return _empty_probe()
    rows = _cap_rows(rows, cfg.quail_max_samples, cfg.seed)
    table = pd.DataFrame(rows)
    print(f"[probe] QuAIL: {len(table)} items / {table['label'].nunique()} reasoning types")
    return table


def build_corpus(
    cfg: CorpusConfig | None = None,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Multi-source combined corpus (revision 5, Path A — diversified cognitive ops).

    Columns: 'text', 'source' (7 source 중 하나), 'author', 'user_id', 'split'.

    Sources load independently and degrade gracefully (skip on HF mirror failure).
    """
    cfg = cfg or CorpusConfig()
    if cache_path is None:
        cache_path = Path("out/phase1/cache") / f"corpus_{cfg.cache_key()}.parquet"
    cache_path = Path(cache_path)

    if cache_path.exists():
        print(f"[corpus] reusing cache: {cache_path}")
        return pd.read_parquet(cache_path)

    parts: list[pd.DataFrame] = []
    # Generation-anchor sources (2026-05-26 defaults)
    if cfg.include_news:
        parts.append(_load_news_part(cfg))
    if cfg.include_arxiv:
        parts.append(_load_arxiv_part(cfg))
    if cfg.include_tweet:
        parts.append(_load_tweet_part(cfg))
    if cfg.include_amazon:
        parts.append(_load_amazon_part(cfg))
    # Legacy sources (default off; loaders retained for ablation / reproducibility)
    if cfg.include_pennebaker:
        parts.append(_load_pennebaker_part())
    if cfg.include_reddit:
        parts.append(_load_reddit_part(cfg))
    if cfg.include_pandora:
        parts.append(_load_pandora_part(cfg))
    if cfg.include_personachat:
        parts.append(_load_personachat_part(cfg))
    if cfg.include_rocstories:
        parts.append(_load_rocstories_part(cfg))
    if cfg.include_anli:
        parts.append(_load_anli_part(cfg))
    if cfg.include_socialiqa:
        parts.append(_load_socialiqa_part(cfg))
    # Operation-cycle pivot (2026-05-27)
    if cfg.include_superni:
        parts.append(_load_superni_part(cfg))
    parts = [p for p in parts if len(p) > 0]

    if not parts:
        raise ValueError("CorpusConfig disables every source (or all loaders failed).")

    corpus = pd.concat(parts, ignore_index=True).reset_index(drop=True)

    # Author → user_id mapping via pd.factorize (vectorized C path, ~3-5x faster than
    # sorted-dict on 100k+ authors). Phase 1 ignores user_id; Phase 2 uses it.
    codes, _ = pd.factorize(corpus["author"], sort=True)
    corpus["user_id"] = codes.astype(np.int64)

    # Sample-level split (Phase 1 user-agnostic 정신).
    n = len(corpus)
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n)
    train_idx = perm[: int(0.8 * n)]
    val_idx = perm[int(0.8 * n) : int(0.9 * n)]
    test_idx = perm[int(0.9 * n) :]
    split = np.empty(n, dtype=object)
    split[train_idx] = SPLIT_TRAIN
    split[val_idx] = SPLIT_VAL
    split[test_idx] = SPLIT_TEST
    corpus["split"] = split
    assert not (corpus["split"].isna().any()), "split assignment incomplete"
    n_users = corpus["user_id"].nunique()
    print(f"[corpus] unified: {len(corpus)} samples, {n_users} users, "
          f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(cache_path)
    print(f"[corpus] cached → {cache_path}")
    return corpus


# ----- fact_emb cache + dataset ------------------------------------------------------

def _text_cache_tag(corpus: pd.DataFrame, encoder_name: str) -> tuple[str, str]:
    """Returns (enc_tag, text_hash) for cache filenames.

    Chunked SHA1 over text — avoids materializing the full corpus (~hundreds of MB at 200k
    samples) as one Python string just to hash it. Colab RAM-safe.
    """
    h = hashlib.sha1()
    for t in corpus["text"]:
        h.update(t.encode("utf-8", errors="ignore"))
        h.update(b"\n")
    enc_tag = encoder_name.replace("/", "_").replace(":", "_")
    return enc_tag, h.hexdigest()[:10]


def encode_or_load(
    corpus: pd.DataFrame,
    encoder_name: str = "BAAI/bge-large-en-v1.5",
    encoder_max_length: int = 256,
    cache_dir: str | Path = "out/phase1/cache",
    batch_size: int = 32,
) -> np.ndarray:
    """Encode corpus text → fact_emb (N, d_model) with caching.

    Cache key = encoder_name + text hash. Re-running same encoder + same corpus reuses.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    enc_tag, text_hash = _text_cache_tag(corpus, encoder_name)
    cache_file = cache_dir / f"fact_emb_{enc_tag}_L{encoder_max_length}_{text_hash}.npy"

    if cache_file.exists():
        print(f"[encode] reusing cached fact_emb: {cache_file}")
        return np.load(cache_file)

    print(f"[encode] encoding {len(corpus)} samples with {encoder_name}...")
    # Reuse FrozenEncoder so the masked-mean-pool formula lives in exactly one place.
    from .cycle import FrozenEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = FrozenEncoder(encoder_name).to(device)
    fact_emb = encoder.encode_batched(
        corpus["text"].tolist(),
        batch_size=batch_size,
        max_length=encoder_max_length,
    )

    np.save(cache_file, fact_emb)
    print(f"[encode] cached → {cache_file}  shape={fact_emb.shape}")
    return fact_emb


def encode_or_load_tokens(
    corpus: pd.DataFrame,
    encoder_name: str = "BAAI/bge-large-en-v1.5",
    t_cap: int = 128,
    cache_dir: str | Path = "out/phase1/cache",
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-token encode → (tokens (N, T, d) fp16, mask (N, T) int8) with caching.

    Operation-cycle pivot (ENGINE_A_DESIGN §1): NO pooling. Two npy files (tokens + mask)
    keyed by encoder + T cap + text hash. Re-running the same encoder + corpus + T reuses.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    enc_tag, text_hash = _text_cache_tag(corpus, encoder_name)
    tok_file = cache_dir / f"token_emb_{enc_tag}_T{t_cap}_{text_hash}.npy"
    mask_file = cache_dir / f"token_mask_{enc_tag}_T{t_cap}_{text_hash}.npy"

    if tok_file.exists() and mask_file.exists():
        print(f"[encode] reusing cached tokens: {tok_file}")
        return np.load(tok_file), np.load(mask_file)

    print(f"[encode] per-token encoding {len(corpus)} samples with {encoder_name} (T={t_cap})...")
    from .cycle import FrozenEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = FrozenEncoder(encoder_name).to(device)
    tokens, mask = encoder.encode_tokens_batched(
        corpus["text"].tolist(), batch_size=batch_size, t_cap=t_cap,
    )

    np.save(tok_file, tokens)
    np.save(mask_file, mask)
    print(f"[encode] cached → {tok_file}  tokens={tokens.shape} mask={mask.shape}")
    return tokens, mask


# ----- PyTorch dataset ---------------------------------------------------------------

class Phase1Dataset(Dataset):
    """(fact_emb, user_id) Dataset — Phase 1/2 공통.

    fact_emb 는 frozen encoder cache. Phase 1 model 은 user_id 무시 (use_user=False).
    """

    def __init__(self, fact_emb: np.ndarray, user_id: np.ndarray):
        assert fact_emb.shape[0] == user_id.shape[0]
        # Zero-copy view when dtype already matches (np.float32, np.int64) — saves ~400 MB
        # peak for 100k×1024 float32 cache vs torch.tensor(np_array) (which copies).
        self.fact_emb = torch.from_numpy(np.ascontiguousarray(fact_emb, dtype=np.float32))
        self.user_id = torch.from_numpy(np.ascontiguousarray(user_id, dtype=np.int64))

    def __len__(self) -> int:
        return self.fact_emb.size(0)

    def __getitem__(self, idx: int):
        return self.fact_emb[idx], self.user_id[idx]


def make_loaders(
    corpus: pd.DataFrame,
    fact_emb: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool | None = None,
):
    """Returns (train_loader, val_loader, test_loader) on the corpus splits.

    `pin_memory` defaults to True when CUDA is available — enables overlapped H→D
    transfer with `tensor.to(device, non_blocking=True)` in the train loop.
    """
    from torch.utils.data import DataLoader

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    splits = {}
    for split in SPLITS:
        mask = (corpus["split"] == split).values
        ds = Phase1Dataset(fact_emb[mask], corpus["user_id"].values[mask])
        splits[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == SPLIT_TRAIN),
            num_workers=num_workers, pin_memory=pin_memory,
        )
    return splits[SPLIT_TRAIN], splits[SPLIT_VAL], splits[SPLIT_TEST]


# ----- Convenience: full pipeline ----------------------------------------------------

def load_phase1_corpus(
    cfg: CorpusConfig | None = None,
    corpus_cache: str | Path | None = None,
    enc_cache_dir: str | Path = "out/phase1/cache",
):
    """Build corpus → encode. Returns (corpus_df, fact_emb) — no DataLoaders.

    Use this for eval / post-hoc analysis where you slice the corpus manually.
    """
    cfg = cfg or CorpusConfig()
    corpus = build_corpus(cfg, cache_path=corpus_cache)
    fact_emb = encode_or_load(
        corpus,
        encoder_name=cfg.encoder_name,
        encoder_max_length=cfg.encoder_max_length,
        cache_dir=enc_cache_dir,
    )
    return corpus, fact_emb


def load_phase1_data(
    cfg: CorpusConfig | None = None,
    corpus_cache: str | Path | None = None,
    enc_cache_dir: str | Path = "out/phase1/cache",
    batch_size: int = 32,
):
    """Build corpus → encode → loaders. Returns (corpus_df, fact_emb, train/val/test loaders)."""
    corpus, fact_emb = load_phase1_corpus(cfg, corpus_cache, enc_cache_dir)
    train_loader, val_loader, test_loader = make_loaders(
        corpus, fact_emb, batch_size=batch_size,
    )
    return corpus, fact_emb, train_loader, val_loader, test_loader
