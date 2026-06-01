"""Throwaway diagnostic (grill-with-docs, 2026-05-31): discriminate 1a-weakness
cause (A) flat-arch-limit vs (B) corpus-limit.

CPU-only, no e5 encoding. Three cheap cuts:
  1. corpus stats (reasoning_type / source / split sizes).
  2. LEXICAL passage-only baseline — pick the candidate with highest lexical
     similarity to the passage (Q ignored). If >> chance(0.25), the answer is
     recoverable by surface P<->cand matching => (B) evidence. LSAT logic MC is
     designed so distractors are lexically close, so genuine logic should sit
     near chance here.
  3. dump N sample items for manual composition-depth inspection.

Run: python -m research.demo.phase1_5._diag_composition  (from repo root)
  or python _diag_composition.py  (from phase1_5/ with sys.path hack below)
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# Allow running as a loose script from inside phase1_5/.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from research.demo.phase1_5.data import MCCorpusConfig, build_mc_corpus
    from research.demo.phase1_5.data_musique import _toks
else:
    from .data import MCCorpusConfig, build_mc_corpus
    from .data_musique import _toks  # shared lexical tokeniser (single source)


def lexical_jaccard_baseline(corpus) -> dict:
    """For each row, pick the candidate maximizing Jaccard(passage_tokens,
    candidate_tokens). Report accuracy + how often the *correct* answer is the
    lexically-closest. Tie -> first max (argmax) so it's a fair single guess."""
    n = 0
    correct = 0
    # also: rank of the true answer by lexical sim (1=closest) to see if signal
    # is weak-but-present vs absent.
    true_rank_counts = Counter()
    for _, row in corpus.iterrows():
        opts = list(row["options"])
        if len(opts) != 4:
            continue
        p_tok = _toks(row["passage"])
        sims = []
        for o in opts:
            o_tok = _toks(o)
            inter = len(p_tok & o_tok)
            union = len(p_tok | o_tok) or 1
            sims.append(inter / union)
        sims = np.asarray(sims)
        pred = int(sims.argmax())
        ans = int(row["answer_idx"])
        n += 1
        if pred == ans:
            correct += 1
        # rank of true answer (1 = highest sim). ties broken by stable sort.
        order = np.argsort(-sims, kind="stable").tolist()
        true_rank_counts[order.index(ans) + 1] += 1
    return {
        "n": n,
        "acc": correct / max(n, 1),
        "chance": 0.25,
        "true_answer_rank_hist": dict(sorted(true_rank_counts.items())),
    }


def dump_samples(corpus, k: int, seed: int, out_path: Path) -> None:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(corpus), size=min(k, len(corpus)), replace=False)
    lines: list[str] = []
    for j, i in enumerate(idx):
        row = corpus.iloc[int(i)]
        lines.append(f"===== SAMPLE {j+1}  (source={row['source']} type={row['reasoning_type']}) =====")
        lines.append(f"[PASSAGE] {row['passage']}")
        lines.append(f"[QUESTION] {row['question']}")
        for c, o in enumerate(row["options"]):
            mark = " <== ANSWER" if c == int(row["answer_idx"]) else ""
            lines.append(f"   ({chr(97+c)}) {o}{mark}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[dump] {len(idx)} samples -> {out_path}")


def main() -> None:
    cfg = MCCorpusConfig()
    corpus = build_mc_corpus(cfg)
    print("\n========== CORPUS STATS ==========")
    print(f"total: {len(corpus)}")
    print("split:", dict(Counter(corpus["split"])))
    print("source:", dict(Counter(corpus["source"])))
    print("reasoning_type (all):", dict(Counter(corpus["reasoning_type"]).most_common()))

    # eval-relevant split = test (LogiQA labelled) else val
    for split in ("test", "val"):
        sub = corpus[corpus["split"] == split]
        if not len(sub):
            continue
        print(f"\n--- LEXICAL passage-only baseline on split={split} (n={len(sub)}) ---")
        res = lexical_jaccard_baseline(sub)
        print(f"  acc={res['acc']:.4f}  (chance={res['chance']})  n={res['n']}")
        print(f"  true-answer lexical-rank hist (1=closest to passage): {res['true_answer_rank_hist']}")
        # per-source breakdown
        for src in sorted(set(sub["source"])):
            ss = sub[sub["source"] == src]
            r = lexical_jaccard_baseline(ss)
            print(f"    [{src}] acc={r['acc']:.4f} n={r['n']}")

    out_path = Path(__file__).resolve().parent / "_diag_samples.txt"
    dump_samples(corpus[corpus["split"].isin(["test", "val"])], k=40, seed=7, out_path=out_path)


if __name__ == "__main__":
    main()
