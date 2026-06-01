"""MuSiQue (multi-hop QA) → 4-way MC conversion for Phase 1.5 1b.

Why this module (plan M1, 2026-05-31):
    LogiQA/ReClor are single-operation-per-question and stem-announce the
    operation (raw-Q ceiling 0.98) → no "composition of distinct operations"
    substrate for the emergent-NMN (1b chain-router) thesis. MuSiQue is built by
    *composing* single-hop questions (anti-shortcut by construction) and ships
    per-hop decomposition with intermediate answers. We convert it to the generic
    7-column MC schema (passage, question, options, answer_idx, reasoning_type,
    source, split) so the existing encode/cache/Dataset path (``data.encode_or_load_mc``)
    is a drop-in.

Hard-distractor design (LLM-free, IRON-faithful):
    The intermediate-hop answers are used as HARD distractors — a single-hop
    shortcut picks an intermediate answer and is therefore wrong, which forces
    the model to execute the full composition. Shortfall (2-hop has only one
    intermediate) is back-filled from a same-answer-type pool (random within
    type), never with an LLM. ``musique_distractor_leak_check`` (M1 GATE) is the
    empirical guard on distractor quality — if it flags trivial/leaky distractors
    on real data, the backfill can be sharpened to TF-IDF mid-band. ``reasoning_type``
    carries the hop/structure label for eval/S1 ONLY — never a training signal.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .data import MCCorpusConfig, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL, _empty_mc_part

# Source tag (parallels SOURCE_LOGIQA2 / SOURCE_RECLOR in data.py).
SOURCE_MUSIQUE = "musique"


# ----- answer normalisation (equality / dedupe ONLY; option surface keeps casing) ----

_WS_RE = re.compile(r"\s+")


def _normalize_answer(s: str) -> str:
    """Lowercase, strip surrounding quotes/punct, collapse whitespace. Used only
    for equality + dedupe + leak-guard comparisons — the stored option surface
    keeps original casing (entities encode better cased under e5)."""
    if not isinstance(s, str):
        return ""
    t = s.strip().strip("\"'").strip()
    t = t.rstrip(".,;:!?").strip()
    t = _WS_RE.sub(" ", t)
    return t.lower()


# ----- answer-type bucketing + backfill pool -----------------------------------------

_YEAR_RE = re.compile(r"^\d{3,4}$|\b(1\d{3}|20\d{2})\b")
_MONTH_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.IGNORECASE
)
_NUM_RE = re.compile(r"^[\d.,]+$")

TYPE_DATE = "DATE"
TYPE_NUM = "NUM"
TYPE_SHORT_ENTITY = "SHORT_ENTITY"
TYPE_OTHER = "OTHER"


def _answer_type(s: str) -> str:
    """Coarse answer-type bucket (no NER) — DATE / NUM / SHORT_ENTITY / OTHER."""
    t = (s or "").strip()
    if _YEAR_RE.search(t) or _MONTH_RE.search(t):
        return TYPE_DATE
    if _NUM_RE.match(t):
        return TYPE_NUM
    if 1 <= len(t.split()) <= 4:
        return TYPE_SHORT_ENTITY
    return TYPE_OTHER


class AnswerPool:
    """Same-answer-type candidate pool for distractor backfill. Built once per
    split from the universe of gold + intermediate answers — deduped by answer
    type at construction so ``sample`` is an O(bucket) filter (no per-call regex
    normalisation or re-dedup). ``sample`` draws same-type surfaces, excluding a
    blocked set, deterministically."""

    def __init__(self, by_type: dict[str, list[tuple[str, str]]]):
        # by_type[type] = list of (surface, normalised) pairs, already deduped.
        self._by_type = by_type
        self._all = [pair for lst in by_type.values() for pair in lst]

    def sample(self, gold: str, *, exclude: set[str], k: int, rng_seed: int) -> list[str]:
        bucket = self._by_type.get(_answer_type(gold)) or self._all
        cands = [s for (s, n) in bucket if n not in exclude]
        if not cands:  # rare type fully excluded → widen to the global universe
            cands = [s for (s, n) in self._all if n not in exclude]
        if not cands:
            return []
        rng = np.random.default_rng(abs(hash((rng_seed, gold))) % (2**32))
        idx = rng.permutation(len(cands))[:k]
        return [cands[i] for i in idx]


def _build_answer_pool(rows: list[dict]) -> AnswerPool:
    """Collect the universe of gold + intermediate answers across ``rows``, dedupe
    by answer type (normalised), and bucket for same-type distractor backfill."""
    by_type: dict[str, list[tuple[str, str]]] = {}
    seen_per_type: dict[str, set[str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        answers = [row.get("answer")]
        for d in row.get("question_decomposition") or []:
            if isinstance(d, dict):
                answers.append(d.get("answer"))
        for a in answers:
            if not isinstance(a, str) or not a.strip():
                continue
            t = _answer_type(a)
            n = _normalize_answer(a)
            seen = seen_per_type.setdefault(t, set())
            if n not in seen:
                seen.add(n)
                by_type.setdefault(t, []).append((a, n))
    return AnswerPool(by_type)


# ----- per-row conversion ------------------------------------------------------------


def _musique_row_to_record(
    row: dict,
    split: str,
    cfg: MCCorpusConfig,
    answer_pool=None,
    rng_seed: int = 0,
) -> dict | None:
    """Convert one MuSiQue row → generic 7-column MC record, or None to drop.

    Distractors = intermediate-hop answers (all decomposition answers whose
    normalised form differs from the gold and from any answer alias), deduped.
    Shortfall to 3 distractors is back-filled from ``answer_pool`` (None ⇒ no
    backfill available, so a row that cannot reach 4 unique options is dropped).
    """
    if not isinstance(row, dict):
        return None
    if row.get("answerable") is False:
        return None  # unanswerable rows have no reliable gold
    gold = row.get("answer")
    if not isinstance(gold, str) or not gold.strip():
        return None
    gold_norm = _normalize_answer(gold)
    aliases_norm = {
        _normalize_answer(a) for a in (row.get("answer_aliases") or []) if isinstance(a, str)
    }
    blocked = {gold_norm, *aliases_norm}

    # Intermediate-hop answers → hard distractors (leak-guarded + deduped).
    distractors: list[str] = []
    seen = {gold_norm}
    for d in row.get("question_decomposition") or []:
        ans = d.get("answer") if isinstance(d, dict) else None
        if not isinstance(ans, str) or not ans.strip():
            continue
        n = _normalize_answer(ans)
        if n in blocked or n in seen:
            continue  # equals gold/alias (leak) or already chosen (dedupe)
        seen.add(n)
        distractors.append(ans)

    n_cand = cfg.n_candidates
    need = (n_cand - 1) - len(distractors)
    if need > 0:
        # Backfill from the same-answer-type pool (None ⇒ cannot fill → drop row).
        if answer_pool is None:
            return None
        for cand in answer_pool.sample(gold, exclude=blocked | seen, k=need, rng_seed=rng_seed):
            n = _normalize_answer(cand)
            if n in blocked or n in seen:
                continue
            seen.add(n)
            distractors.append(cand)
    distractors = distractors[: n_cand - 1]
    if len(distractors) < n_cand - 1:
        return None  # could not reach 4 unique options → drop (never pad duplicates)

    options = [gold, *distractors]
    # Deterministic per-row shuffle so gold is not always slot 0.
    order = _deterministic_perm(len(options), rng_seed, row.get("id", ""))
    options = [options[i] for i in order]
    answer_idx = options.index(gold)

    return {
        "passage": _assemble_passage(row, cfg),
        "question": row.get("question", ""),
        "options": options,
        "answer_idx": answer_idx,
        "reasoning_type": infer_musique_op_label(row),
        "source": SOURCE_MUSIQUE,
        "split": split,
    }


def _deterministic_perm(n: int, rng_seed: int, row_id: str) -> "np.ndarray":
    """A permutation of range(n) seeded by (rng_seed, row_id) — reproducible
    across runs without depending on global RNG state."""
    h = abs(hash((rng_seed, row_id))) % (2**32)
    return np.random.default_rng(h).permutation(n)


# ----- passage assembly --------------------------------------------------------------


def _assemble_passage(row: dict, cfg: MCCorpusConfig, sep: str = "[SEP]") -> str:
    """Assemble P from paragraphs: supporting paragraphs first (in hop order),
    then the remaining paragraphs by idx, accumulating until the ``t_cap_p`` word
    budget is reached. Keeps the Q-only bottleneck (P → KV side-channel only)."""
    paras = {
        p["idx"]: p
        for p in (row.get("paragraphs") or [])
        if isinstance(p, dict) and "idx" in p
    }
    # Supporting paragraph idxs in hop order (dedup, preserve first occurrence).
    support_order: list[int] = []
    for d in row.get("question_decomposition") or []:
        idx = d.get("paragraph_support_idx") if isinstance(d, dict) else None
        if isinstance(idx, int) and idx in paras and idx not in support_order:
            support_order.append(idx)
    rest = [i for i in sorted(paras) if i not in support_order]
    ordered = support_order + rest

    budget = cfg.t_cap_p
    out: list[str] = []
    used = 0
    for i in ordered:
        p = paras[i]
        chunk = f"{p.get('title', '')}: {p.get('paragraph_text', '')}".strip()
        w = len(chunk.split())
        if out and used + w > budget:
            break  # supporting paras lead, so truncation drops trailing non-supporting
        out.append(chunk)
        used += w
    return f" {sep} ".join(out)


# ----- top-level loader --------------------------------------------------------------


def _hf_load_musique(cfg: MCCorpusConfig) -> dict:
    """Load MuSiQue from HF → ``{split_name: list[row_dict]}``. Isolated so tests
    can monkeypatch it without faking the full HF DatasetDict API."""
    from datasets import load_dataset

    ds = load_dataset(cfg.musique_hf)
    return {name: list(split) for name, split in ds.items()}


def load_musique(cfg: MCCorpusConfig) -> pd.DataFrame:
    """Load MuSiQue → generic 7-column MC schema.

    - validation → ``val``; a deterministic ``musique_holdout_test_frac`` slice of
      train → ``test`` (MuSiQue's public test is unlabelled).
    - Distractors built per-row (intermediate answers + same-type backfill from a
      per-split answer pool). Rows that cannot reach ``n_candidates`` options are
      dropped.
    """
    try:
        raw = _hf_load_musique(cfg)
    except Exception as e:  # pragma: no cover - network/HF failure path
        print(f"[corpus] MuSiQue load failed ({type(e).__name__}: {e}); skipping")
        return _empty_mc_part()

    train_rows = list(raw.get("train", []))
    val_rows = list(raw.get("validation", raw.get("dev", [])))

    # Deterministic train→test holdout (shuffle by seed, take tail frac).
    if train_rows and cfg.musique_holdout_test_frac > 0:
        rng = np.random.default_rng(cfg.seed)
        perm = rng.permutation(len(train_rows))
        n_test = int(round(len(train_rows) * cfg.musique_holdout_test_frac))
        test_ids = set(perm[:n_test].tolist())
        held_train = [r for i, r in enumerate(train_rows) if i not in test_ids]
        test_rows = [r for i, r in enumerate(train_rows) if i in test_ids]
    else:
        held_train, test_rows = train_rows, []

    records: list[dict] = []
    for split_label, rows in (
        (SPLIT_TRAIN, held_train),
        (SPLIT_VAL, val_rows),
        (SPLIT_TEST, test_rows),
    ):
        if not rows:
            continue
        pool = _build_answer_pool(rows)  # per-split universe (no cross-split leak)
        for r in rows:
            rec = _musique_row_to_record(r, split_label, cfg, answer_pool=pool, rng_seed=cfg.seed)
            if rec is not None:
                records.append(rec)

    if not records:
        return _empty_mc_part()
    df = pd.DataFrame(records)
    print(
        f"[corpus] MuSiQue: {len(df)} items "
        f"(train={int((df['split'] == SPLIT_TRAIN).sum())}, "
        f"val={int((df['split'] == SPLIT_VAL).sum())}, "
        f"test={int((df['split'] == SPLIT_TEST).sum())}) / "
        f"hops {dict(df['reasoning_type'].value_counts())}"
    )
    return df


# ----- operation label (eval / S1 ONLY — never a training signal) --------------------


def infer_musique_op_label(row: dict) -> str:
    """Operation-axis label from a MuSiQue row. Hop-count from the ``id`` prefix
    (``2hop``/``3hop``/``4hop``). Eval/S1 only."""
    rid = row.get("id", "") if isinstance(row, dict) else ""
    m = re.match(r"(\d+)hop", str(rid))
    return f"{m.group(1)}hop" if m else "unknown"


# ----- distractor leak-check (M1 GATE) ----------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = set(
    "the a an of to in is are was were be been being and or but if then that this "
    "these those it its as at by for from on with which who whom whose what when "
    "where why how not no nor so than too very can will just".split()
)


def _toks(s: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall((s or "").lower()) if t not in _STOP and len(t) > 1}


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / (len(a | b) or 1)


def musique_distractor_leak_check(corpus_df) -> dict:
    """M1 GATE: lexical-Jaccard argmax baselines. For each row pick the option
    most lexically similar to (passage) and to (question); report accuracy + per-hop.

    Interpretation (plan R2): intermediate distractors ARE passage spans, so
    ``passage_to_option_acc`` is *intentionally* raised by them — that is the
    anti-shortcut design, NOT a leak. The real tripwire is
    **``question_to_option_acc``**: if the question's surface picks the answer,
    the distractors leak and the conversion must be fixed (target ≤ ~0.30).

    Single pass over the rows: each row's passage/question/options are tokenised
    once and both baselines + the per-hop breakdown are tallied together.
    """
    has_hop = "reasoning_type" in corpus_df.columns
    # tally per key = [n, passage_correct, question_correct]; "" = overall.
    tally: dict[str, list[int]] = {"": [0, 0, 0]}
    for r in corpus_df.itertuples(index=False):
        opts = list(r.options)
        if len(opts) < 2:
            continue
        opt_toks = [_toks(o) for o in opts]
        ans = int(r.answer_idx)
        p_ok = int(np.argmax([_jaccard(_toks(r.passage), t) for t in opt_toks]) == ans)
        q_ok = int(np.argmax([_jaccard(_toks(r.question), t) for t in opt_toks]) == ans)
        for key in ("", r.reasoning_type) if has_hop else ("",):
            t = tally.setdefault(key, [0, 0, 0])
            t[0] += 1
            t[1] += p_ok
            t[2] += q_ok

    def _acc(t: list[int]) -> dict:
        return {
            "n": t[0],
            "passage_to_option_acc": t[1] / max(t[0], 1),
            "question_to_option_acc": t[2] / max(t[0], 1),
        }

    overall = _acc(tally[""])
    return {
        "n": int(len(corpus_df)),
        "chance": 1.0 / max(len(corpus_df["options"].iloc[0]), 1) if len(corpus_df) else 0.0,
        "passage_to_option_acc": overall["passage_to_option_acc"],
        "question_to_option_acc": overall["question_to_option_acc"],
        "per_hop": {k: _acc(v) for k, v in sorted(tally.items()) if k},
    }


_HOP_REF_RE = re.compile(r"#\d+")


def infer_musique_structure(row: dict) -> str:
    """Decomposition-structure label (eval/S1 only). ``"chain"`` (bridge) if any
    sub-question references an earlier answer via ``#k``; else ``"comparison"``
    (independent sub-questions)."""
    decomp = row.get("question_decomposition") or [] if isinstance(row, dict) else []
    for d in decomp:
        q = d.get("question", "") if isinstance(d, dict) else ""
        if _HOP_REF_RE.search(q or ""):
            return "chain"
    return "comparison"
