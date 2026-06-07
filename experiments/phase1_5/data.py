"""Phase 1.5 MC corpus loader — LogiQA 2.0 + ReClor + Q/P-separated encoding.

Paper §5.4 / §7.4 commit: single-domain logical-reasoning corpus, MC-contrastive
LLM-free objective. Q (question + candidates) and P (passage) are encoded
*separately* — Q feeds the operation-KG bottleneck, P enters only as the
cross-attention side-channel.

Three encoding modes implement the Gap A ablation ladder (paper §7.4 row A.1/A.2/A.3):

- ``Q_only`` (A.1 default): q_tokens = encode("query: " + question).
  P does not influence the encoder input — bottleneck respected.
- ``Q_pmask`` (A.2 strict control): q_tokens = encode(
  "query: " + question + " [SEP] " + pad_repeats_for_P_positions). Positions
  preserved, P content erased (partial-Q-leak control per Reviewer 3 / paper §7.1).
- ``Q_full`` (A.3 bottleneck violation upper bound): q_tokens = encode(
  "query: " + question + " [SEP] " + passage). Full P content in the encoder
  input — ceiling against which the Q-only configuration is measured.

P is always encoded separately into ``p_tokens`` for the cross-attention KV input,
regardless of mode.

HF mirror chain (per Stage 1 verification, fallback at runtime):

- LogiQA 2.0 primary: ``csitfun/LogiQA2.0`` (Liu et al. 2023 TASLP).
- LogiQA fallback: ``lucasmccabe/logiqa`` (original 2020 release, weaker labels).
- ReClor primary: ``metaeval/reclor`` (mirror of Yu et al. 2020 ICLR).

Schema-drift defensives follow the ``phase1/data.py`` pattern
(``_first_present_in``).
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# ----- Constants ----------------------------------------------------------------------

SOURCE_LOGIQA2 = "logiqa2"
SOURCE_RECLOR = "reclor"

# Encoding mode tags (Gap A ablation row IDs)
MODE_Q_ONLY = "Q_only"
MODE_Q_PMASK = "Q_pmask"
MODE_Q_FULL = "Q_full"
ENCODING_MODES = (MODE_Q_ONLY, MODE_Q_PMASK, MODE_Q_FULL)

# Split labels
SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"

# Schema-drift defensives for LogiQA 2.0 (csitfun mirror primarily)
LOGIQA2_PASSAGE_CANDS = ("text", "context", "passage")
LOGIQA2_QUESTION_CANDS = ("question",)
LOGIQA2_OPTIONS_CANDS = ("options", "choices", "candidates")
LOGIQA2_ANSWER_CANDS = ("answer", "label", "correct_option")
LOGIQA2_TYPE_CANDS = ("type", "question_type", "reasoning_type")

# LogiQA fallback mirror (lucasmccabe/logiqa)
LOGIQA1_PASSAGE_CANDS = ("context", "passage")
LOGIQA1_QUESTION_CANDS = ("query", "question")
LOGIQA1_OPTIONS_CANDS = ("options", "choices")
LOGIQA1_ANSWER_CANDS = ("correct_option", "label", "answer")

# ReClor
RECLOR_PASSAGE_CANDS = ("context", "passage")
RECLOR_QUESTION_CANDS = ("question",)
RECLOR_OPTIONS_CANDS = ("answers", "options", "choices")
RECLOR_ANSWER_CANDS = ("label", "answer", "correct")
RECLOR_TYPE_CANDS = ("question_type", "type")


# ----- Config -------------------------------------------------------------------------


@dataclass
class MCCorpusConfig:
    """Phase 1.5 MC corpus + encoding config.

    HF mirror choices verified at Stage 1 (see
    ``STAGE4_5_FINAL_INTEGRITY_REPORT.md``). Encoder defaults to e5-large-v2 per
    paper §5.4 (Stage 1 raw-encoder ceiling 0.60 evidence).
    """

    # LogiQA 2.0 sources — 1순위 = GitHub raw (HF datasets v3 의 script 거부 우회 +
    # 인간 annotated 5-way reasoning type label 확보). 2순위 = HF mirror chain.
    logiqa2_github_base_url: str = (
        "https://raw.githubusercontent.com/csitfun/LogiQA2.0/main/logiqa/DATA/LOGIQA"
    )
    logiqa2_hf: str = "csitfun/LogiQA2.0"
    logiqa2_fallbacks_hf: tuple[str, ...] = (
        "datatune/LogiQA2.0",
        "baber/logiqa2",
        "lucasmccabe/logiqa",
    )
    # 1.5 plan §9 commits to v2; fallbacks 마지막은 v1 (original 2020).
    # (Note: the singular ``logiqa2_fallback_hf`` field from pre-pass-3 was
    # removed — it was never read by the new ``load_logiqa2`` loop and a non-
    # default value silently invalidated the corpus cache. Use the
    # ``logiqa2_fallbacks_hf`` tuple instead.)
    reclor_hf: str = "metaeval/reclor"

    # Corpus selector (Phase 1.5 1b pivot, 2026-05-31):
    #   "musique"  = MuSiQue multi-hop → 4-way MC (compositional substrate, primary)
    #   "logic_mc" = LogiQA 2.0 + ReClor (single-op CONTROL arm)
    #   "both"     = all three (transfer test)
    corpus: str = "musique"
    musique_hf: str = "dgslibisey/MuSiQue"
    # MuSiQue has no labelled public test split → carve a deterministic holdout
    # from train into `test` for the σ-gate / causal battery.
    musique_holdout_test_frac: float = 0.1

    # Sample caps
    max_train_samples: int = 20_000  # combined across loaded sources (cfg.corpus)
    max_val_samples: int = 2_000
    max_test_samples: int = 2_000

    # Sequence caps
    t_cap_q: int = 128
    t_cap_p: int = 256
    t_cap_cand: int = 48  # per-candidate, pooled
    n_candidates: int = 4

    # Encoder
    encoder_name: str = "intfloat/e5-large-v2"
    # Prefixes are derived by encoder family in ``encode_or_load_mc``; kept on
    # the config for backwards compat with callers that read them but NOT used
    # by the encoder path. Use ``encoders.default_q_prefix`` / ``default_p_prefix``
    # to override family detection programmatically.

    # Seed
    seed: int = 42

    # Cache root (override at test time)
    cache_root: str = "out/phase1_5/cache"

    def cache_key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha1(payload).hexdigest()[:10]


# ----- Helpers ------------------------------------------------------------------------


def _first_present_in(available: tuple[str, ...], candidates: tuple[str, ...]) -> str | None:
    """Return the first candidate column name that appears in ``available``."""
    return next((c for c in candidates if c in available), None)


def _empty_mc_part() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["passage", "question", "options", "answer_idx", "reasoning_type", "source", "split"]
    )


def _normalize_answer_idx(raw, n_candidates: int) -> int | None:
    """Coerce a wide variety of answer encodings to a 0..(n-1) integer index.

    Accepts:
    - int / numpy int (0-based or 1-based detected via range);
    - str digit ("0".."3" or "1".."4");
    - "a"/"b"/"c"/"d" (case-insensitive).
    Returns None if unparseable or out-of-range.

    ``bool`` is rejected explicitly (``isinstance(True, int)`` is True in Python,
    so True would otherwise map to candidate index 1 and False to 0 — a silent
    schema-drift footgun).
    """
    if raw is None or isinstance(raw, (bool, np.bool_)):
        # ``np.bool_`` is NOT a Python ``bool`` subclass on NumPy 2.x; reject
        # both kinds explicitly so a boolean column never silently maps to
        # candidate index 0/1.
        return None
    if isinstance(raw, (int, np.integer)):
        v = int(raw)
    elif isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        if s.isdigit():
            v = int(s)
        elif len(s) == 1 and s.lower() in "abcdefgh":
            v = ord(s.lower()) - ord("a")
        else:
            return None
    else:
        return None
    # Heuristic: if 1..n found (no zero ever seen), treat as 1-based.
    # We just try v as-is first; if out-of-range, try v-1.
    if 0 <= v < n_candidates:
        return v
    if 1 <= v <= n_candidates:
        return v - 1
    return None


# LSAT reasoning-type patterns for ``infer_reasoning_type`` — paper §9 corpus
# design's operation axis ("how to reason from premise to answer"). Order = match
# priority; first match wins.
#
# Disambiguation notes (LSAT taxonomy):
#   - "supports the [argument/conclusion/claim]"  → STRENGTHEN (answer supports argument)
#   - "supported by the [passage/statements]"     → INFERENCE (answer follows from passage)
#   - "most strongly support" alone is ambiguous — pattern routing relies on the
#     surrounding noun. We include the disambiguating phrases explicitly and
#     leave the bare "support" out.
# Each pattern is matched as a regex with word boundaries where applicable, so
# bare common verbs ("depend", "resolve", "flaw") do NOT match morphological
# neighbours ("independent", "unresolved", "flawless").
_REASONING_TYPE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("strengthen", (
        r"\bstrengthen\w*\b",
        # Allow 0-3 words between "the" and the argument noun so possessive /
        # adjectival stems ("supports the speaker's argument") still match — the
        # bare "the argument" form was leaking these to 'other'.
        r"supports? the (?:[\w'’-]+\s+){0,3}(?:argument|conclusion|claim|position|hypothesis|view)",
        r"most strongly justif\w+", r"most help(?:s)? to justify",
        r"most strongly support\w* the (?:[\w'’-]+\s+){0,3}(?:argument|conclusion|claim|view|position)",
    )),
    ("weaken", (
        r"\bweaken\w*\b", r"cast(?:s)? doubt", r"\bundermin\w+\b",
        r"call\w* into question", r"\bleast supports?\b",
    )),
    ("inference", (
        r"\bmust be true\b", r"\bmost likely true\b", r"\bfollows? logically\b",
        r"\bcan be (?:properly )?inferred\b", r"\bproperly inferred\b",
        r"supported by the (?:passage|statements|claims|information)",
        r"\bsupported by\b",
    )),
    ("assumption", (
        r"\bassumption\b", r"\bassume\w*\b", r"\bpresuppos\w+\b",
        r"\bdepends? (?:on|upon)\b",
    )),
    ("paradox", (
        r"\bparadox\w*\b", r"\bdiscrepanc\w+\b", r"apparent contradiction",
        r"explain the (?:result|finding|outcome|fact)",
        r"\bresolves? the\b",
    )),
    ("method", (
        r"method of reasoning", r"technique of reasoning",
        r"argumentative (?:strategy|technique)",
        r"proceeds? by\b",
    )),
    ("flaw", (
        r"\bflaw\b", r"\bflawed?\b", r"\bvulnerable\b", r"\bquestionable\b",
        r"error in reasoning", r"reasoning error", r"logical error",
    )),
    ("principle", (
        r"\bprinciple\b", r"general rule", r"conforms most closely",
    )),
    ("parallel", (
        r"\bparallel\b", r"similar reasoning",
        r"most similar in its reasoning",
    )),
    ("main_point", (
        r"main point", r"main conclusion", r"main idea", r"primary purpose",
    )),
    ("evaluate", (
        r"\bevaluate\b", r"helpful to know", r"useful to determine",
    )),
)


def _compile_patterns() -> list[tuple[str, re.Pattern]]:
    """Compile pattern alternation per label once at module load."""
    return [
        (label, re.compile("|".join(patterns), flags=re.IGNORECASE))
        for label, patterns in _REASONING_TYPE_PATTERNS
    ]


_REASONING_TYPE_REGEX: list[tuple[str, re.Pattern]] = _compile_patterns()


def infer_reasoning_type(question: str) -> str:
    """Heuristic operation-axis label from question text.

    Returns one of the LSAT-style reasoning-type names (``strengthen``,
    ``weaken``, ``inference``, ``assumption``, ``paradox``, ``method``, ``flaw``,
    ``principle``, ``parallel``, ``main_point``, ``evaluate``) if a stem
    pattern matches, else ``'other'``. Used as a fallback when the HF mirror
    omits a question_type column.

    Note: returns a single ``'other'`` bucket for unmatched stems — does NOT
    fall through to a wh-word label (``'which'`` / ``'what'`` / ...). wh-word
    routing reintroduces stem-format artefacts (Phase 1's F3 confound at the
    label-construction layer), so the heuristic deliberately collapses
    unmatched stems into a single class.
    """
    q = question or ""
    if not q.strip():
        return "other"
    for label, regex in _REASONING_TYPE_REGEX:
        if regex.search(q):
            return label
    return "other"


# Back-compat alias (cell-9 of notebook imports the underscored name).
_infer_reasoning_type = infer_reasoning_type


def _options_to_list(raw, n_candidates: int) -> list[str] | None:
    """Coerce a HF row's ``options``/``answers``/``choices`` field to ``list[str]``.

    Drops the row (returns None) if the candidate count does not match ``n_candidates``.
    """
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        opts = [str(o) for o in raw if isinstance(o, (str, bytes))]
    elif isinstance(raw, np.ndarray):
        opts = [str(o) for o in raw.tolist() if isinstance(o, (str, bytes))]
    elif isinstance(raw, dict):
        # Some mirrors store {"a": "...", "b": "...", ...}
        keys = sorted(raw.keys())
        opts = [str(raw[k]) for k in keys if isinstance(raw[k], str)]
    else:
        return None
    if len(opts) != n_candidates:
        return None
    return opts


# ----- LogiQA 2.0 GitHub raw loader -----------------------------------------------

# Priority order for single-label extraction from LogiQA 2.0's multi-label ``type``
# dict (paper §9 commit's 5 formal reasoning types). First True key wins; if all
# False or the field is missing, caller falls back to ``infer_reasoning_type``.
_LOGIQA2_TYPE_PRIORITY: tuple[str, ...] = (
    "Sufficient Conditional Reasoning",
    "Necessary Conditional Reasoning",
    "Disjunctive Reasoning",
    "Conjunctive Reasoning",
    "Categorical Reasoning",
)

_LOGIQA2_SPLIT_FILES: tuple[tuple[str, str], ...] = (
    # (HF-style split label, GitHub raw filename)
    (SPLIT_TRAIN, "train.txt"),
    (SPLIT_VAL, "dev.txt"),
    (SPLIT_TEST, "test.txt"),
)

_LOGIQA2_GITHUB_SCHEMA_WARNED: set[str] = set()


def _logiqa2_type_to_label(type_dict) -> str | None:
    """Single-label extraction from LogiQA 2.0's ``type`` dict.

    Returns the first ``True`` key in ``_LOGIQA2_TYPE_PRIORITY`` order, converted
    to snake_case (e.g., ``"Sufficient Conditional Reasoning"`` → ``"sufficient_conditional"``).
    Returns ``None`` if input is not a dict, all values are False, or no priority
    key is present — caller falls back to ``infer_reasoning_type(question)``.
    """
    if not isinstance(type_dict, dict):
        return None
    for key in _LOGIQA2_TYPE_PRIORITY:
        if type_dict.get(key) is True:
            # "Sufficient Conditional Reasoning" → "sufficient_conditional"
            # (drop trailing "Reasoning" + collapse spaces → underscores).
            stripped = key.replace(" Reasoning", "").strip()
            return stripped.lower().replace(" ", "_")
    return None


def _logiqa2_github_row_to_record(
    row: dict,
    split: str,
    n_candidates: int,
) -> dict | None:
    """Convert one LogiQA 2.0 GitHub-raw row to our schema dict, or None on validation fail."""
    if not isinstance(row, dict):
        return None
    passage = row.get("text")
    question = row.get("question")
    if not isinstance(passage, str) or not isinstance(question, str):
        return None
    opts = _options_to_list(row.get("options"), n_candidates)
    if opts is None:
        return None
    ans = _normalize_answer_idx(row.get("answer"), n_candidates)
    if ans is None:
        return None
    qtype = _logiqa2_type_to_label(row.get("type"))
    if not qtype:
        qtype = infer_reasoning_type(question)
    return {
        "passage": passage,
        "question": question,
        "options": opts,
        "answer_idx": ans,
        "reasoning_type": qtype,
        "source": SOURCE_LOGIQA2,
        "split": split,
    }


def _load_logiqa2_from_github(cfg: MCCorpusConfig) -> pd.DataFrame:
    """Download LogiQA 2.0 train/dev/test JSON-Lines from the official GitHub
    raw URL, parse to our schema. Locally caches each split file to
    ``<cache_root>/logiqa2_github_<filename>.jsonl`` so re-runs don't re-download.

    Returns an empty DataFrame if every split fetch fails (caller falls back to
    the HF mirror chain).
    """
    import urllib.error
    import urllib.request

    cache_dir = Path(cfg.cache_root)
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for split_label, filename in _LOGIQA2_SPLIT_FILES:
        url = f"{cfg.logiqa2_github_base_url.rstrip('/')}/{filename}"
        cache_path = cache_dir / f"logiqa2_github_{filename}l"  # .txt → .txtl marker
        # Use cached file if present (avoid network).
        if cache_path.exists():
            try:
                raw = cache_path.read_text(encoding="utf-8")
            except OSError as e:
                print(f"[corpus] LogiQA GitHub cache read failed for {cache_path.name} ({type(e).__name__}); refetching")
                cache_path.unlink(missing_ok=True)
                raw = None
        else:
            raw = None
        if raw is None:
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    raw = resp.read().decode("utf-8")
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                print(
                    f"[corpus] LogiQA GitHub fetch failed for {filename} "
                    f"({type(e).__name__}: {e}); next split"
                )
                continue
            except Exception as e:
                print(
                    f"[corpus] LogiQA GitHub unexpected error for {filename} "
                    f"({type(e).__name__}: {e}); next split"
                )
                continue
            try:
                cache_path.write_text(raw, encoding="utf-8")
            except OSError as e:
                print(f"[corpus] LogiQA GitHub cache write failed ({type(e).__name__}); ignored")

        parsed_for_split = 0
        for line_no, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec = _logiqa2_github_row_to_record(row, split_label, cfg.n_candidates)
            if rec is not None:
                rows.append(rec)
                parsed_for_split += 1
        if parsed_for_split == 0 and filename not in _LOGIQA2_GITHUB_SCHEMA_WARNED:
            print(
                f"[corpus] LogiQA GitHub {filename} yielded 0 valid rows "
                f"(schema drift or empty source)"
            )
            _LOGIQA2_GITHUB_SCHEMA_WARNED.add(filename)

    if not rows:
        return _empty_mc_part()
    df = pd.DataFrame(rows)
    print(
        f"[corpus] LogiQA 2.0 loaded from GitHub: {len(df)} items / "
        f"{df['reasoning_type'].nunique()} reasoning types"
    )
    return df


# ----- Loaders ------------------------------------------------------------------------


def load_logiqa2(cfg: MCCorpusConfig) -> pd.DataFrame:
    """Load LogiQA 2.0 — try GitHub raw first, then HF mirror chain.

    Strategy:
      1. **GitHub raw** (``_load_logiqa2_from_github``) — primary, paradigm-faithful:
         direct JSON-Lines download from the official csitfun repo, preserves the
         human-annotated 5-way reasoning type label via ``_logiqa2_type_to_label``.
         HF datasets v3.x rejected most custom-script mirrors (csitfun/LogiQA2.0,
         baber/logiqa2) and the GitHub raw bypasses that entirely.
      2. **HF mirror chain** (``_load_logiqa2_from_hf``) — fallback. Tries each
         mirror with ``get_dataset_config_names`` enumeration (handles
         multi-config mirrors like datatune/LogiQA2.0).

    Returns a DataFrame; empty on total failure.
    """
    df = _load_logiqa2_from_github(cfg)
    if len(df):
        return df
    print("[corpus] GitHub LogiQA 2.0 path empty; trying HF mirror chain")
    return _load_logiqa2_from_hf(cfg)


def _load_logiqa2_from_hf(cfg: MCCorpusConfig) -> pd.DataFrame:
    """HF mirror chain fallback (factored from the previous ``load_logiqa2``)."""
    from datasets import load_dataset

    try:
        from datasets import get_dataset_config_names
    except ImportError:
        get_dataset_config_names = None  # type: ignore[assignment]

    mirrors = [cfg.logiqa2_hf, *cfg.logiqa2_fallbacks_hf]
    seen: set[str] = set()
    for mirror in mirrors:
        if mirror in seen:
            continue
        seen.add(mirror)
        is_v1_fallback = "lucasmccabe" in mirror

        # Enumerate configs (handles multi-config mirrors); fall back to [None].
        configs: list[str | None] = [None]
        if get_dataset_config_names is not None:
            try:
                resolved = list(get_dataset_config_names(mirror)) or [None]
                configs = resolved  # type: ignore[assignment]
            except Exception:
                configs = [None]

        loaded = False
        for cfg_name in configs:
            try:
                ds = (
                    load_dataset(mirror, cfg_name) if cfg_name else load_dataset(mirror)
                )
            except Exception as e:
                print(
                    f"[corpus] LogiQA mirror {mirror!r} cfg={cfg_name!r} load failed "
                    f"({type(e).__name__}); next"
                )
                continue
            loaded = True
            df = _logiqa2_to_df(ds, cfg, fallback=is_v1_fallback)
            if len(df):
                print(
                    f"[corpus] LogiQA loaded from {mirror!r} cfg={cfg_name!r}"
                )
                return df
            print(
                f"[corpus] LogiQA mirror {mirror!r} cfg={cfg_name!r} "
                f"returned empty rows (schema drift); next"
            )
        if not loaded:
            print(f"[corpus] LogiQA mirror {mirror!r} had no usable config")
    print(
        "[corpus] all LogiQA mirrors/configs failed; corpus = ReClor only. "
        "Phase 1.5 plan §9 'LogiQA + ReClor' commitment partially unmet — "
        "operator may want to acquire LogiQA 2.0 separately and place rows "
        "into the parquet cache manually."
    )
    return _empty_mc_part()


def _logiqa2_to_df(ds, cfg: MCCorpusConfig, *, fallback: bool) -> pd.DataFrame:
    """Convert a LogiQA 2.0 (or v1 fallback) HF DatasetDict to our MC schema."""
    passage_cands = LOGIQA1_PASSAGE_CANDS if fallback else LOGIQA2_PASSAGE_CANDS
    question_cands = LOGIQA1_QUESTION_CANDS if fallback else LOGIQA2_QUESTION_CANDS
    options_cands = LOGIQA1_OPTIONS_CANDS if fallback else LOGIQA2_OPTIONS_CANDS
    answer_cands = LOGIQA1_ANSWER_CANDS if fallback else LOGIQA2_ANSWER_CANDS
    type_cands = LOGIQA2_TYPE_CANDS  # only present in v2

    sample_split_name = next(iter(ds.keys()))
    sample_cols = tuple(ds[sample_split_name].column_names)
    p_col = _first_present_in(sample_cols, passage_cands)
    q_col = _first_present_in(sample_cols, question_cands)
    o_col = _first_present_in(sample_cols, options_cands)
    a_col = _first_present_in(sample_cols, answer_cands)
    t_col = _first_present_in(sample_cols, type_cands)
    if p_col is None or q_col is None or o_col is None or a_col is None:
        print(f"[corpus] LogiQA schema mismatch (cols={list(sample_cols)}); skipping")
        return _empty_mc_part()

    rows: list[dict] = []
    for hf_split_name, hf_split in ds.items():
        split_label = _logiqa_split_label(hf_split_name)
        for r in hf_split:
            passage = r.get(p_col)
            question = r.get(q_col)
            if not isinstance(passage, str) or not isinstance(question, str):
                continue
            opts = _options_to_list(r.get(o_col), cfg.n_candidates)
            if opts is None:
                continue
            ans = _normalize_answer_idx(r.get(a_col), cfg.n_candidates)
            if ans is None:
                continue
            qtype = r.get(t_col) if t_col else None
            qtype = qtype if isinstance(qtype, str) and qtype else _infer_reasoning_type(question)
            rows.append(
                {
                    "passage": passage,
                    "question": question,
                    "options": opts,
                    "answer_idx": ans,
                    "reasoning_type": qtype,
                    "source": SOURCE_LOGIQA2,
                    "split": split_label,
                }
            )

    if not rows:
        return _empty_mc_part()
    df = pd.DataFrame(rows)
    print(
        f"[corpus] LogiQA{'(v1 fallback)' if fallback else ' 2.0'}: {len(df)} items / "
        f"{df['reasoning_type'].nunique()} reasoning types"
    )
    return df


_UNKNOWN_SPLITS_WARNED: set[str] = set()


def _logiqa_split_label(hf_split: str) -> str:
    """Map an HF split name to {train, val, test}.

    Unknown split names route to ``SPLIT_TEST`` (held-out), NOT to train —
    polluting train with rows from an unrecognised split (``hidden_test``,
    ``dev2``, ``test_easy``, etc.) is a silent contamination class. Emit a
    one-time warning per unknown split name so the operator can investigate.
    """
    s = hf_split.lower()
    if s in ("train", "training"):
        return SPLIT_TRAIN
    if s in ("validation", "val", "dev"):
        return SPLIT_VAL
    if s in ("test", "eval"):
        return SPLIT_TEST
    if hf_split not in _UNKNOWN_SPLITS_WARNED:
        print(
            f"[corpus] warning: unknown HF split name {hf_split!r}; "
            f"routing rows to SPLIT_TEST (held-out, NOT train)"
        )
        _UNKNOWN_SPLITS_WARNED.add(hf_split)
    return SPLIT_TEST


def load_reclor(cfg: MCCorpusConfig) -> pd.DataFrame:
    """Load ReClor from HF (metaeval/reclor primary).

    Returns the same schema as ``load_logiqa2``. ReClor's test split has hidden
    labels on the canonical leaderboard; the mirror may drop it. We use whichever
    splits expose answer labels.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset(cfg.reclor_hf)
    except Exception as e:
        print(f"[corpus] ReClor load failed ({type(e).__name__}: {e}); skipping")
        return _empty_mc_part()

    sample_split_name = next(iter(ds.keys()))
    sample_cols = tuple(ds[sample_split_name].column_names)
    p_col = _first_present_in(sample_cols, RECLOR_PASSAGE_CANDS)
    q_col = _first_present_in(sample_cols, RECLOR_QUESTION_CANDS)
    o_col = _first_present_in(sample_cols, RECLOR_OPTIONS_CANDS)
    a_col = _first_present_in(sample_cols, RECLOR_ANSWER_CANDS)
    t_col = _first_present_in(sample_cols, RECLOR_TYPE_CANDS)
    if p_col is None or q_col is None or o_col is None:
        print(f"[corpus] ReClor schema mismatch (cols={list(sample_cols)}); skipping")
        return _empty_mc_part()

    rows: list[dict] = []
    for hf_split_name, hf_split in ds.items():
        split_label = _logiqa_split_label(hf_split_name)
        for r in hf_split:
            passage = r.get(p_col)
            question = r.get(q_col)
            if not isinstance(passage, str) or not isinstance(question, str):
                continue
            opts = _options_to_list(r.get(o_col), cfg.n_candidates)
            if opts is None:
                continue
            if a_col is None or r.get(a_col) is None:
                # No answer label — likely the official test split. Skip for training,
                # but keep with answer_idx=-1 if we ever want it for blind eval.
                continue
            ans = _normalize_answer_idx(r.get(a_col), cfg.n_candidates)
            if ans is None:
                continue
            qtype = r.get(t_col) if t_col else None
            qtype = qtype if isinstance(qtype, str) and qtype else _infer_reasoning_type(question)
            rows.append(
                {
                    "passage": passage,
                    "question": question,
                    "options": opts,
                    "answer_idx": ans,
                    "reasoning_type": qtype,
                    "source": SOURCE_RECLOR,
                    "split": split_label,
                }
            )

    if not rows:
        return _empty_mc_part()
    df = pd.DataFrame(rows)
    print(
        f"[corpus] ReClor: {len(df)} items / {df['reasoning_type'].nunique()} reasoning types"
    )
    return df


def build_mc_corpus(cfg: MCCorpusConfig | None = None) -> pd.DataFrame:
    """Build the combined LogiQA 2.0 + ReClor MC corpus with caching.

    Cached parquet under ``out/phase1_5/cache/corpus_{cfg.cache_key()}.parquet``.
    Re-running same cfg reuses cache.

    Per-split caps are applied *after* concatenation so each split is uniformly
    sampled across both sources (preserves source mix).
    """
    cfg = cfg or MCCorpusConfig()
    cache_dir = Path(cfg.cache_root)
    cache_path = cache_dir / f"corpus_{cfg.cache_key()}.parquet"

    if cache_path.exists():
        print(f"[corpus] reusing cache: {cache_path}")
        return pd.read_parquet(cache_path)

    # Dispatch on the corpus selector (Phase 1.5 1b pivot). load_musique is
    # imported lazily to avoid a circular import (data_musique imports from data).
    if cfg.corpus == "musique":
        from .data_musique import load_musique

        parts = [load_musique(cfg)]
    elif cfg.corpus == "logic_mc":
        parts = [load_logiqa2(cfg), load_reclor(cfg)]
    elif cfg.corpus == "both":
        from .data_musique import load_musique

        parts = [load_musique(cfg), load_logiqa2(cfg), load_reclor(cfg)]
    else:
        raise ValueError(
            f"MCCorpusConfig.corpus must be 'musique'/'logic_mc'/'both'; got {cfg.corpus!r}"
        )
    parts = [p for p in parts if len(p) > 0]
    if not parts:
        raise ValueError("MCCorpusConfig: every source loader failed; corpus is empty.")

    corpus = pd.concat(parts, ignore_index=True)
    # Per-split cap (uniform sample within split → preserves source mix).
    rng = np.random.default_rng(cfg.seed)
    caps = {
        SPLIT_TRAIN: cfg.max_train_samples,
        SPLIT_VAL: cfg.max_val_samples,
        SPLIT_TEST: cfg.max_test_samples,
    }
    kept = []
    for split, cap in caps.items():
        sub = corpus[corpus["split"] == split]
        if len(sub) > cap:
            idx = rng.choice(len(sub), size=cap, replace=False)
            sub = sub.iloc[idx]
        kept.append(sub)
    corpus = pd.concat(kept, ignore_index=True)
    sources = "+".join(sorted(corpus["source"].unique())) if len(corpus) else cfg.corpus
    print(
        f"[corpus] {sources} (corpus={cfg.corpus!r}): {len(corpus)} samples "
        f"(train={int((corpus['split'] == SPLIT_TRAIN).sum())}, "
        f"val={int((corpus['split'] == SPLIT_VAL).sum())}, "
        f"test={int((corpus['split'] == SPLIT_TEST).sum())})"
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(cache_path)
    print(f"[corpus] cached → {cache_path}")
    return corpus


# ----- Encoding -----------------------------------------------------------------------


def _text_cache_tag(texts: list[str], encoder_name: str) -> tuple[str, str]:
    """Encoder-tag + chunked SHA1 of the joined text list (Phase 1 pattern)."""
    h = hashlib.sha1()
    for t in texts:
        h.update(t.encode("utf-8", errors="ignore"))
        h.update(b"\n")
    enc_tag = encoder_name.replace("/", "_").replace(":", "_")
    return enc_tag, h.hexdigest()[:10]


def _build_q_text(
    question: str,
    passage: str,
    sep: str,
    encoding_mode: str,
    pmask_placeholder: str = "<pad>",
) -> str:
    """Compose the Q-side text per encoding mode.

    A.1 (Q_only):  question only.
    A.2 (Q_pmask): ``question + [SEP] + placeholder_repeated``.
                   Position tokens preserved, content erased. ``pmask_placeholder``
                   must be the tokenizer's pad-token surface form (``<pad>`` for
                   XLM-RoBERTa / e5 family, ``[PAD]`` for BERT / BGE family). The
                   earlier ``[unused0]`` literal was wordpiece-split into 4
                   subwords (``[`` ``unused`` ``##0`` ``]``) by both tokenizer
                   families, leaking the substring "unused" as content. The caller
                   (``encode_or_load_mc``) supplies the family-correct placeholder.
    A.3 (Q_full):  ``question + [SEP] + passage`` (full text).

    The Q-side prefix ("query: " for e5) is prepended by ``FrozenEncoder.encode_tokens``
    at call time, not here.
    """
    if encoding_mode == MODE_Q_ONLY:
        return question
    if encoding_mode == MODE_Q_FULL:
        return f"{question} {sep} {passage}"
    if encoding_mode == MODE_Q_PMASK:
        n_words = max(1, len(passage.split()))
        masked = " ".join([pmask_placeholder] * n_words)
        return f"{question} {sep} {masked}"
    raise ValueError(f"unknown encoding_mode: {encoding_mode}")


@functools.lru_cache(maxsize=8)
def _encoder_pad_token_text(encoder_name: str) -> str:
    """Return the surface-form pad token for the given encoder, family-aware.

    Strategy:
      1. Try ``AutoTokenizer.from_pretrained(encoder_name).pad_token`` /
         ``unk_token``. If non-empty, use that (the canonical single-token surface
         form for the model).
      2. If tokenizer load fails (offline, HF hub unreachable), fall back per
         family — ``[PAD]`` for BERT-family encoders (BGE), ``<pad>`` otherwise
         (XLM-RoBERTa / e5). Using the wrong family-fallback re-introduces the
         leakage class the ``[unused0]`` fix eliminated: ``<pad>`` is BPE-split
         by BERT WordPiece into ``<``, ``pad``, ``>`` — leaking "pad" subword
         content into the A.2 strict control.

    Memoised per ``encoder_name`` so the tokenizer is only loaded once across
    all rows.
    """
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(encoder_name)
        pad = tok.pad_token or tok.unk_token
        if isinstance(pad, str) and pad:
            return pad
    except Exception:
        pass
    # Family-aware fallback when tokenizer load failed or returned no pad/unk.
    name = encoder_name.lower()
    if "bge" in name or "bert" in name:
        return "[PAD]"
    return "<pad>"


def encode_or_load_mc(
    corpus: pd.DataFrame,
    cfg: MCCorpusConfig,
    *,
    encoder_override: str | None = None,
    encoding_mode: str = MODE_Q_ONLY,
    batch_size: int = 32,
    device: str | None = None,
) -> dict:
    """Encode corpus rows into Q + P + candidate tensors with caching.

    Args:
        corpus: output of ``build_mc_corpus``.
        cfg: ``MCCorpusConfig``.
        encoder_override: HF model id; if set, overrides ``cfg.encoder_name``
            (Row C ablation = BGE swap).
        encoding_mode: ``MODE_Q_ONLY`` / ``MODE_Q_PMASK`` / ``MODE_Q_FULL``.
        batch_size: encoder forward batch.
        device: torch device override.

    Returns a dict with:
        q_tokens (N, T_q, d_emb) fp16, q_mask (N, T_q) int8,
        p_tokens (N, T_p, d_emb) fp16, p_mask (N, T_p) int8,
        cand_pooled (N, 4, d_emb) fp32 (L2-normalized),
        answer_idx (N,) int64,
        reasoning_type (N,) object,
        source (N,) object,
        split (N,) object.
    """
    if encoding_mode not in ENCODING_MODES:
        raise ValueError(f"encoding_mode must be in {ENCODING_MODES}; got {encoding_mode}")

    encoder_name = encoder_override or cfg.encoder_name
    cache_dir = Path(cfg.cache_root)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Derive prefix per encoder family (never inherit cfg.encoder_prefix_* values
    # set for a *different* family — e.g. cfg.encoder_prefix_q='query: ' is e5's
    # default but must NOT be applied to BGE). Family detection lives in
    # ``encoders.default_q_prefix``/``default_p_prefix``.
    from .encoders import default_p_prefix, default_q_prefix

    q_prefix = default_q_prefix(encoder_name)
    p_prefix = default_p_prefix(encoder_name)
    cand_prefix = default_q_prefix(encoder_name)

    # We use the encoder family's [SEP] equivalent token literal; the tokenizer
    # handles real special-token insertion, so a textual "[SEP]" is sufficient.
    sep_literal = "[SEP]"
    # pmask_placeholder is only used by ``MODE_Q_PMASK``; lazy-load to skip the
    # tokenizer fetch on the Q_only / Q_full + cache-hit paths.
    pmask_placeholder = (
        _encoder_pad_token_text(encoder_name) if encoding_mode == MODE_Q_PMASK else ""
    )

    q_texts = [
        _build_q_text(q, p, sep_literal, encoding_mode, pmask_placeholder=pmask_placeholder)
        for q, p in zip(corpus["question"].tolist(), corpus["passage"].tolist())
    ]
    p_texts = corpus["passage"].tolist()
    # Flatten candidates: 4 per row.
    cand_texts: list[str] = []
    for opts in corpus["options"]:
        cand_texts.extend(opts)

    # Cache keys — include the prefix in the encoder tag so changing only the
    # prefix value invalidates stale embeddings (the prefix is applied inside
    # ``FrozenEncoder.encode_tokens`` and is NOT in ``_text_cache_tag``'s payload).
    enc_tag, q_hash = _text_cache_tag(q_texts, encoder_name)
    _, p_hash = _text_cache_tag(p_texts, encoder_name)
    _, c_hash = _text_cache_tag(cand_texts, encoder_name)

    def _prefix_tag(prefix: str) -> str:
        return hashlib.sha1(prefix.encode("utf-8", errors="ignore")).hexdigest()[:6]

    q_pref_tag = _prefix_tag(q_prefix)
    p_pref_tag = _prefix_tag(p_prefix)
    c_pref_tag = _prefix_tag(cand_prefix)

    q_tok_file = cache_dir / f"q_tok_{enc_tag}_T{cfg.t_cap_q}_{encoding_mode}_p{q_pref_tag}_{q_hash}.npy"
    q_msk_file = cache_dir / f"q_msk_{enc_tag}_T{cfg.t_cap_q}_{encoding_mode}_p{q_pref_tag}_{q_hash}.npy"
    p_tok_file = cache_dir / f"p_tok_{enc_tag}_T{cfg.t_cap_p}_p{p_pref_tag}_{p_hash}.npy"
    p_msk_file = cache_dir / f"p_msk_{enc_tag}_T{cfg.t_cap_p}_p{p_pref_tag}_{p_hash}.npy"
    c_pool_file = cache_dir / f"cand_pool_{enc_tag}_T{cfg.t_cap_cand}_p{c_pref_tag}_{c_hash}.npy"

    cache_hit = all(f.exists() for f in (q_tok_file, q_msk_file, p_tok_file, p_msk_file, c_pool_file))

    if cache_hit:
        print(f"[encode] reusing cache: {q_tok_file.name} + p + cand")
        q_tokens = np.load(q_tok_file)
        q_mask = np.load(q_msk_file)
        p_tokens = np.load(p_tok_file)
        p_mask = np.load(p_msk_file)
        cand_pool_flat = np.load(c_pool_file)
    else:
        print(
            f"[encode] {len(corpus)} rows × {encoder_name} "
            f"(Q mode={encoding_mode}, T_q={cfg.t_cap_q}, T_p={cfg.t_cap_p})..."
        )
        from .encoders import FrozenEncoder
        from .train import resolve_device

        torch_device = resolve_device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        encoder = FrozenEncoder(encoder_name).to(torch_device)
        # q_prefix / p_prefix / cand_prefix derived above by encoder family.

        q_tokens, q_mask = encoder.encode_tokens_batched(
            q_texts, prefix=q_prefix, batch_size=batch_size, t_cap=cfg.t_cap_q
        )
        p_tokens, p_mask = encoder.encode_tokens_batched(
            p_texts, prefix=p_prefix, batch_size=batch_size, t_cap=cfg.t_cap_p
        )
        cand_pool_flat = encoder.encode_pooled_batched(
            cand_texts,
            prefix=cand_prefix,
            batch_size=batch_size * 2,
            max_length=cfg.t_cap_cand,
            l2_normalize=True,
        )

        np.save(q_tok_file, q_tokens)
        np.save(q_msk_file, q_mask)
        np.save(p_tok_file, p_tokens)
        np.save(p_msk_file, p_mask)
        np.save(c_pool_file, cand_pool_flat)
        print(
            f"[encode] cached q={q_tokens.shape} p={p_tokens.shape} cand={cand_pool_flat.shape}"
        )

    n = len(corpus)
    # ``-1`` reshape cannot infer the trailing dim from a zero-size array
    # (``ValueError: cannot reshape array of size 0 into shape (0,4,newaxis)``);
    # fall back to the encoder's declared hidden size when the corpus is empty.
    d_emb = cand_pool_flat.shape[-1] if cand_pool_flat.size else 0
    cand_pooled = cand_pool_flat.reshape(n, cfg.n_candidates, d_emb)

    return {
        "q_tokens": q_tokens,
        "q_mask": q_mask,
        "p_tokens": p_tokens,
        "p_mask": p_mask,
        "cand_pooled": cand_pooled,
        "answer_idx": corpus["answer_idx"].values.astype(np.int64),
        "reasoning_type": corpus["reasoning_type"].values,
        "source": corpus["source"].values,
        "split": corpus["split"].values,
    }


# ----- PyTorch Dataset ---------------------------------------------------------------


class MCDataset(Dataset):
    """Pre-encoded MC dataset. Returns dict of tensors per ``__getitem__``."""

    def __init__(
        self,
        q_tokens: np.ndarray,
        q_mask: np.ndarray,
        p_tokens: np.ndarray,
        p_mask: np.ndarray,
        cand_pooled: np.ndarray,
        answer_idx: np.ndarray,
    ):
        assert q_tokens.shape[0] == q_mask.shape[0] == p_tokens.shape[0]
        assert p_mask.shape[0] == cand_pooled.shape[0] == answer_idx.shape[0]
        # Keep fp16 / int8 on CPU; cast to fp32 / long on __getitem__ (cheap).
        self.q_tokens = q_tokens
        self.q_mask = q_mask
        self.p_tokens = p_tokens
        self.p_mask = p_mask
        self.cand_pooled = cand_pooled
        self.answer_idx = answer_idx

    def __len__(self) -> int:
        return int(self.q_tokens.shape[0])

    def __getitem__(self, idx: int) -> dict:
        return {
            "q_tokens": torch.from_numpy(self.q_tokens[idx]).float(),
            "q_mask": torch.from_numpy(self.q_mask[idx]).float(),
            "p_tokens": torch.from_numpy(self.p_tokens[idx]).float(),
            "p_mask": torch.from_numpy(self.p_mask[idx]).float(),
            "cand_pooled": torch.from_numpy(self.cand_pooled[idx]).float(),
            "answer_idx": torch.tensor(int(self.answer_idx[idx]), dtype=torch.long),
        }


def make_mc_loaders(
    data: dict,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Returns (train_loader, val_loader, test_loader) split by ``data['split']``."""
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loaders = {}
    for split in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST):
        mask = data["split"] == split
        if not mask.any():
            loaders[split] = None
            continue
        ds = MCDataset(
            q_tokens=data["q_tokens"][mask],
            q_mask=data["q_mask"][mask],
            p_tokens=data["p_tokens"][mask],
            p_mask=data["p_mask"][mask],
            cand_pooled=data["cand_pooled"][mask],
            answer_idx=data["answer_idx"][mask],
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == SPLIT_TRAIN),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return loaders[SPLIT_TRAIN], loaders[SPLIT_VAL], loaders[SPLIT_TEST]
