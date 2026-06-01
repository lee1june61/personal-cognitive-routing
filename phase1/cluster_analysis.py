"""Cluster analysis utility for Phase 1 expert specialization study.

Notebook 의 반복 cluster cell + 비교 table 을 한 줄 호출로 추출.
각 expert 의 top-N activating sample 의 source distribution + TF-IDF top keywords.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader

from .cycle import CycleConfig, Phase1Cycle, collect_activations
from .data import CorpusConfig, Phase1Dataset, load_phase1_corpus
from .train import CONFIG_NAME, MODEL_NAME

TOP_N = 100
ALPHA_THRESHOLD = 0.01
TFIDF_MAX_FEATURES = 5000
TFIDF_TOP_KEYWORDS = 10
BATCH_SIZE = 128


def analyze_experts(
    run_id: str,
    top_n: int = TOP_N,
    alpha_threshold: float = ALPHA_THRESHOLD,
    out_dir: str = "out/phase1",
) -> dict:
    """Phase 1 run 의 expert specialization analysis.

    Prints per-expert n_active + source distribution + TF-IDF top keywords.
    Returns {run_id, K, k_active_mean, k_active_dist}.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(out_dir) / run_id
    config_path = run_dir / CONFIG_NAME
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
    else:
        print(f"[warn] {config_path} not found — using defaults (v4 setting)")
        cfg = {
            "encoder_name": "BAAI/bge-large-en-v1.5",
            "encoder_max_length": 256,
            "k_routed": 16,
            "k_shared": 4,
            "d_hidden": 2048,
            "d_bottleneck": 64,
            "lambda_lb": 0.1,
            "lambda_ortho": 0.0,
            "use_user": False,
        }

    ccfg = CorpusConfig(
        encoder_name=cfg["encoder_name"], encoder_max_length=cfg["encoder_max_length"],
    )
    corpus, fact_emb = load_phase1_corpus(cfg=ccfg)

    model = Phase1Cycle(
        n_users=int(corpus["user_id"].nunique()),
        encoder_name=cfg["encoder_name"],
        k_routed=cfg.get("k_routed", 16),
        k_shared=cfg.get("k_shared", 4),
        d_hidden=cfg.get("d_hidden", 2048),
        config=CycleConfig(
            lambda_lb=cfg.get("lambda_lb", 0.05),
            lambda_ortho=cfg.get("lambda_ortho", 0.05),
            d_bottleneck=cfg.get("d_bottleneck", 64),
        ),
        use_user=cfg.get("use_user", False),
    ).to(device)
    model.load_state_dict(torch.load(run_dir / MODEL_NAME, map_location=device), strict=False)
    model.eval()

    loader = DataLoader(
        Phase1Dataset(fact_emb, corpus["user_id"].values),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    routed_alpha = collect_activations(model, loader, device)["routed_alpha"].numpy()
    K = routed_alpha.shape[1]

    k_active = (routed_alpha > alpha_threshold).sum(axis=1)
    kact_mean = float(k_active.mean())
    kact_dist = np.bincount(k_active, minlength=K + 1)[: K + 1].tolist()
    print(f"Mean K_active = {kact_mean:.2f} / {K}")
    print(f"K_active dist: {kact_dist}")

    print(f'\n{"=" * 80}\nExpert specialization (top-{top_n})\n{"=" * 80}')
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES, stop_words="english", ngram_range=(1, 1),
    )
    tfidf = vectorizer.fit_transform(corpus["text"].astype(str).tolist())
    feature_names = vectorizer.get_feature_names_out()
    all_sources = corpus["source"].tolist()

    for k in range(K):
        active_mask = routed_alpha[:, k] > alpha_threshold
        n_active = int(active_mask.sum())
        if n_active == 0:
            print(f"\nExpert {k:>2}: DEAD")
            continue
        top_idx = np.argsort(-routed_alpha[:, k])[:top_n]
        mean_alpha = float(routed_alpha[active_mask, k].mean())
        src_counts = Counter(all_sources[i] for i in top_idx)
        src_str = ", ".join(f"{s}={c}" for s, c in src_counts.most_common(5))
        avg_tfidf = np.asarray(tfidf[top_idx].mean(axis=0)).ravel()
        keywords = ", ".join(
            feature_names[i] for i in np.argsort(-avg_tfidf)[:TFIDF_TOP_KEYWORDS]
        )
        print(f"\nExpert {k:>2}: n_active={n_active:>6} alpha={mean_alpha:.3f}")
        print(f"  Top sources: {src_str}")
        print(f"  Keywords: {keywords}")

    return {
        "run_id": run_id,
        "K": K,
        "k_active_mean": kact_mean,
        "k_active_dist": kact_dist,
    }


def compare_runs(run_ids: list[str], out_dir: str = "out/phase1") -> None:
    """N-way 비교 table — epoch, recon_cos, K_active, alpha_F1, simbench."""
    header = (
        f'{"run_id":<20} {"epoch":>6} {"recon_cos":>10} '
        f'{"K_active":>9} {"alpha_F1":>9} {"simbench":>9}'
    )
    print(header)
    print("-" * len(header))
    for rid in run_ids:
        rd = Path(out_dir) / rid
        if not (rd / "history.json").exists():
            print(f"{rid:<20}  MISSING")
            continue
        h = json.loads((rd / "history.json").read_text())
        ev = json.loads((rd / "eval.json").read_text()) if (rd / "eval.json").exists() else {}
        sb = (
            json.loads((rd / "simbench_eval.json").read_text())
            if (rd / "simbench_eval.json").exists() else {}
        )
        cfg = (
            json.loads((rd / CONFIG_NAME).read_text())
            if (rd / CONFIG_NAME).exists() else {}
        )
        K = cfg.get("k_routed", 16)
        n_ep = len(h)
        recon = ev.get("cycle_reconstruction", {}).get("mean_cosine", float("nan"))
        active_frac = h[-1].get("active_frac")
        kact = active_frac * K if active_frac is not None else float("nan")
        f1 = ev.get("linear_probe_routed_alpha", {}).get("macro_f1", float("nan"))
        sba = sb.get("test", {}).get("argmax_acc", float("nan"))
        print(
            f"{rid:<20} {n_ep:>6} {recon:>10.4f} {kact:>9.2f} {f1:>9.4f} {sba:>9.4f}"
        )
