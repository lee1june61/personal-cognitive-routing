"""Phase 1 evaluation — embedding-only cycle, no LLM (revision 4).

Three eval forms:
  1. Cycle reconstruction quality on held-out (cosine / MSE on fact_emb vs fact_emb_recon)
  2. Linear probe separability (Nikolic 2025 §5.1) — context label predictability from
     routed_alpha. D1 post-hoc analysis preview.
  3. SimBench feature transfer prep — sub_kg + routed_alpha → classifier head trainable
     for SimBench multi-choice (separate downstream script).

No text BLEU (text-level cycle abandoned in revision 4). HyperRED external eval moved
to Phase 3 future work (post-hoc text rendering via external LLM).

Run from `research/demo/`:
    python -m phase1.eval --run_id ph1_v0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .cycle import CycleConfig, Phase1Cycle, collect_activations
from .data import CorpusConfig, load_phase1_corpus
from .train import MODEL_NAME, CONFIG_NAME

EVAL_NAME = "eval.json"


# ----- Linear probe separability (Nikolic 2025 §5.1) ---------------------------------

def linear_probe_separability(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """SVM linear probe accuracy + macro F1 on features → labels.

    Nikolic 2025 finding: unsupervised expert assignment 의 linear probe accuracy 가
    supervised assignment 보다 +8.3pp on QuickDraw. 우리 paradigm 의 *emergent
    relation-type expert specialization* 을 *post-hoc* 으로 측정하는 sanity metric.
    """
    try:
        from sklearn.svm import LinearSVC
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score
    except ImportError:
        return {"error": "sklearn missing"}

    features = np.asarray(features)
    labels = np.asarray(labels)
    assert features.shape[0] == labels.shape[0]

    X_tr, X_te, y_tr, y_te = train_test_split(
        features, labels, test_size=test_size, random_state=seed, stratify=labels,
    )
    clf = LinearSVC(C=1.0, max_iter=5000, dual="auto")
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    return {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "macro_f1": float(f1_score(y_te, y_pred, average="macro")),
        "n_samples": int(features.shape[0]),
        "n_classes": int(len(set(labels.tolist()))),
        "n_features": int(features.shape[1]),
    }


# ----- Cycle reconstruction quality --------------------------------------------------

@torch.no_grad()
def eval_cycle_reconstruction(
    model: Phase1Cycle,
    fact_emb: np.ndarray,
    user_id: np.ndarray | None,
    device: torch.device,
    batch_size: int = 64,
) -> dict:
    """Cosine + MSE on held-out fact_emb vs fact_emb_recon."""
    model.eval()
    # One zero-copy view of the whole array; per-batch slice is then a view, not a copy.
    fact_emb_t = torch.from_numpy(np.ascontiguousarray(fact_emb, dtype=np.float32))
    user_id_t = (
        torch.from_numpy(np.ascontiguousarray(user_id, dtype=np.int64))
        if user_id is not None and model.use_user else None
    )
    sum_cos = 0.0
    sum_mse = 0.0
    n = 0
    for i in range(0, len(fact_emb_t), batch_size):
        fe = fact_emb_t[i:i+batch_size].to(device, non_blocking=True)
        uid = user_id_t[i:i+batch_size].to(device, non_blocking=True) if user_id_t is not None else None
        out = model(fe, user_id=uid)
        cos = torch.nn.functional.cosine_similarity(fe, out["recon"], dim=-1)
        mse = ((fe - out["recon"]) ** 2).mean(dim=-1)
        sum_cos += float(cos.sum())
        sum_mse += float(mse.sum())
        n += fe.size(0)
    return {
        "n_samples": n,
        "mean_cosine": sum_cos / max(n, 1),
        "mean_mse": sum_mse / max(n, 1),
        "cycle_loss_recon": 1.0 - sum_cos / max(n, 1),
    }


# ----- Main eval ---------------------------------------------------------------------

def eval_phase1(
    run_id: str = "ph1_v0",
    out_dir: str = "out/phase1",
    encoder_name: str = "BAAI/bge-large-en-v1.5",
    encoder_max_length: int = 256,
    batch_size: int = 32,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(out_dir) / run_id
    if not (run_dir / MODEL_NAME).exists():
        raise FileNotFoundError(f"{run_dir/'model.pt'} not found")

    # Load config
    cfg_path = run_dir / CONFIG_NAME
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    use_user = cfg.get("use_user", False)
    k_routed = cfg.get("k_routed", 16)
    k_shared = cfg.get("k_shared", 4)
    d_hidden = cfg.get("d_hidden", 2048)

    # Corpus + fact_emb cache (no DataLoaders — we slice manually below)
    ccfg = CorpusConfig(encoder_name=encoder_name, encoder_max_length=encoder_max_length)
    corpus, fact_emb = load_phase1_corpus(cfg=ccfg)
    n_users = int(corpus["user_id"].nunique())

    # Rebuild model
    model = Phase1Cycle(
        n_users=n_users, encoder_name=encoder_name,
        k_routed=k_routed, k_shared=k_shared, d_hidden=d_hidden,
        config=CycleConfig(
            lambda_lb=cfg.get("lambda_lb", 0.1),
            lambda_ortho=cfg.get("lambda_ortho", 0.0),
            d_bottleneck=cfg.get("d_bottleneck", 64),
        ),
        use_user=use_user,
    ).to(device)
    model.load_state_dict(torch.load(run_dir / MODEL_NAME, map_location=device), strict=False)
    model.eval()

    # Held-out test set
    test_mask = (corpus["split"] == "test").values
    test_fact_emb = fact_emb[test_mask]
    test_user_id = corpus["user_id"].values[test_mask]

    # 1. Cycle reconstruction
    recon = eval_cycle_reconstruction(
        model, test_fact_emb,
        user_id=test_user_id if use_user else None,
        device=device, batch_size=batch_size,
    )
    print(f"[recon] mean_cosine={recon['mean_cosine']:.4f}  "
          f"mse={recon['mean_mse']:.6f}  cycle_loss={recon['cycle_loss_recon']:.4f}")

    # 2. Linear probe separability — `source` label (Pennebaker vs Reddit)
    #    as coarse cognitive context proxy. Nikolic 2025 §5.1 metric.
    from torch.utils.data import DataLoader
    from .data import Phase1Dataset
    test_ds = Phase1Dataset(test_fact_emb, test_user_id)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    acts = collect_activations(model, test_loader, device)

    source_labels = (corpus["source"].values[test_mask] == "pennebaker").astype(np.int64)
    probe_alpha = linear_probe_separability(
        acts["routed_alpha"].numpy(), source_labels,
    )
    probe_subkg = linear_probe_separability(
        acts["sub_kg"].numpy(), source_labels,
    )
    print(f"[probe routed_alpha] acc={probe_alpha.get('accuracy', 'n/a')}  "
          f"macro_f1={probe_alpha.get('macro_f1', 'n/a')}")
    print(f"[probe sub_kg]       acc={probe_subkg.get('accuracy', 'n/a')}  "
          f"macro_f1={probe_subkg.get('macro_f1', 'n/a')}")

    result = {
        "run_id": run_id, "use_user": use_user,
        "cycle_reconstruction": recon,
        "linear_probe_routed_alpha": probe_alpha,
        "linear_probe_sub_kg": probe_subkg,
        "n_test": int(test_mask.sum()),
    }
    (run_dir / EVAL_NAME).write_text(json.dumps(result, indent=2))
    print(f"[eval] saved → {run_dir / 'eval.json'}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", default="ph1_v0")
    p.add_argument("--out_dir", default="out/phase1")
    p.add_argument("--encoder_name", default="BAAI/bge-large-en-v1.5")
    p.add_argument("--encoder_max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()
    eval_phase1(**vars(args))


if __name__ == "__main__":
    main()
