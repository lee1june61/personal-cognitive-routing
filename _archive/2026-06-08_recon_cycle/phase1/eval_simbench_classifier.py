"""SimBench Path A — LLM-free distributional user simulator (paper §4.1 primary eval).

Pipeline:
    text (persona + question) → BGE-large frozen → fact_emb (1024d)
                              → Phase 1 (frozen)  → sub_kg (1024d) + routed_alpha (K_routed)
                              → classifier head   → logits (MAX_OPTIONS)
                              → softmax over valid options → predicted distribution

Loss: KL(human_aggregate || predicted). Eval: KL + argmax accuracy on held-out.

Dataset: SimBench (Hu et al. ICML 2025, arXiv 2510.17516). HuggingFace
`pitehu/SimBench` has known DatasetGenerationCastError on combined load — we bypass
`load_dataset` and pull the two CSVs directly with pandas.

Usage (from research/demo/):
    python -m phase1.eval_simbench_classifier --phase1_run_id ph1_v3_minimal

Or from a notebook:
    from phase1.eval_simbench_classifier import train_simbench_classifier
    train_simbench_classifier(phase1_run_id='ph1_v3_minimal', epochs=20)
"""

from __future__ import annotations

import argparse
import ast
import json
import string
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .baselines.generic_baseline import GenericBaseline
from .baselines.standard_moe_baseline import StandardMoEBaseline
from .cycle import CycleConfig, Phase1Cycle
from .train import CONFIG_NAME, MODEL_NAME


# ----- Constants ---------------------------------------------------------------------

SIMBENCH_BASE = "https://huggingface.co/datasets/pitehu/SimBench/resolve/main"
POP_URL = f"{SIMBENCH_BASE}/SimBenchPop.csv"
GROUPED_URL = f"{SIMBENCH_BASE}/SimBenchGrouped.csv"

# 'A'..'J' covers Pop 95%, Grouped 98%; 11+ option rows dropped.
MAX_OPTIONS = 10
LETTER_TO_IDX = {c: i for i, c in enumerate(string.ascii_uppercase[:MAX_OPTIONS])}

# Numerical / split tunables.
KL_EPS = 1e-12                          # log-prob safety floor in masked KL
DEFAULT_SPLIT = (0.8, 0.1, 0.1)         # train / val / test
FEATURE_BATCH_DEFAULT = 64              # BGE-large GPU throughput sweet spot

CLASSIFIER_NAME = "simbench_classifier_head.pt"
EVAL_NAME = "simbench_eval.json"


# ----- Data loading + parsing --------------------------------------------------------

def parse_simbench_dict(s):
    """SimBench dict columns are stringified Python dicts; tolerate both JSON and repr.

    Public so notebooks can reuse without redefining the same fallback ladder.
    """
    if not isinstance(s, str):
        return s
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return None


def load_simbench(
    include_pop: bool = True,
    include_grouped: bool = True,
    cache_dir: str | Path = "out/phase1/cache/simbench",
) -> pd.DataFrame:
    """Download SimBench CSVs from HF, parse dict columns, drop rows beyond MAX_OPTIONS.

    Concatenates Pop + Grouped with column union (Grouped has 7 extra cols not in Pop).
    Caches to parquet for instant reload.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"simbench_pop{int(include_pop)}_grouped{int(include_grouped)}.parquet"
    if cache_path.exists():
        print(f"[simbench] reusing cache: {cache_path}")
        return pd.read_parquet(cache_path)

    parts = []
    if include_pop:
        pop = pd.read_csv(POP_URL)
        pop["_split_source"] = "pop"
        parts.append(pop)
        print(f"[simbench] loaded Pop: {len(pop)} rows")
    if include_grouped:
        grouped = pd.read_csv(GROUPED_URL)
        grouped["_split_source"] = "grouped"
        parts.append(grouped)
        print(f"[simbench] loaded Grouped: {len(grouped)} rows")
    if not parts:
        raise ValueError("At least one of include_pop / include_grouped must be True.")

    df = pd.concat(parts, ignore_index=True, sort=False)

    # Resolve persona text by formatting template with variable_map (mostly empty).
    # Track format failures so corpus quality regressions are visible at load time.
    n_format_fail = 0
    def _resolve_persona(row):
        nonlocal n_format_fail
        template = row["group_prompt_template"]
        var_map = parse_simbench_dict(row["group_prompt_variable_map"]) or {}
        if not isinstance(var_map, dict) or not var_map:
            return template
        try:
            return template.format(**var_map)
        except (KeyError, IndexError, ValueError):
            n_format_fail += 1
            return template
    df["_persona"] = df.apply(_resolve_persona, axis=1)
    df["_text"] = df["_persona"].astype(str) + "\n" + df["input_template"].astype(str)
    if n_format_fail:
        print(f"[simbench] {n_format_fail} persona templates failed to format — kept raw")

    # Parse human_answer dict + count valid options (non-None entries only —
    # some SimBench rows have {'A': 0.5, 'B': None, ...} where None means "no data").
    df["_human_answer"] = df["human_answer"].apply(parse_simbench_dict)
    df["_n_options"] = df["_human_answer"].apply(
        lambda d: sum(1 for v in d.values() if v is not None) if isinstance(d, dict) else 0
    )

    n_before = len(df)
    df = df[(df["_n_options"] >= 2) & (df["_n_options"] <= MAX_OPTIONS)].reset_index(drop=True)
    print(f"[simbench] kept {len(df)}/{n_before} rows "
          f"(dropped {n_before - len(df)} with <2 or >{MAX_OPTIONS} valid options, or parse fail)")

    df.to_parquet(cache_path)
    print(f"[simbench] cached → {cache_path}")
    return df


def build_targets(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialize (dist, mask, correct_idx) arrays from parsed human_answer dicts.

    dist:    (N, MAX_OPTIONS) — normalized probabilities over valid options
    mask:    (N, MAX_OPTIONS) bool — True for options present in human_answer
    correct: (N,) — argmax of human distribution (== correct_answer for accuracy tasks)
    """
    N = len(df)
    dist = np.zeros((N, MAX_OPTIONS), dtype=np.float32)
    mask = np.zeros((N, MAX_OPTIONS), dtype=bool)
    correct = np.zeros(N, dtype=np.int64)
    for i, ha in enumerate(df["_human_answer"]):
        for letter, pct in ha.items():
            up = letter.upper()
            if up not in LETTER_TO_IDX or pct is None:
                continue
            idx = LETTER_TO_IDX[up]
            dist[i, idx] = float(pct)
            mask[i, idx] = True
        s = dist[i].sum()
        if s > 0:
            dist[i] /= s
        correct[i] = int(dist[i].argmax())
    return dist, mask, correct


# ----- Feature extraction (frozen Phase 1) -------------------------------------------

@torch.no_grad()
def extract_features(
    df: pd.DataFrame,
    model: Phase1Cycle | GenericBaseline | StandardMoEBaseline,
    device: torch.device,
    batch_size: int = FEATURE_BATCH_DEFAULT,
    encoder_max_length: int = 256,
    cache_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode every SimBench text → fact_emb → cycle model forward → (sub_kg, routed_alpha).

    Works with Phase 1, B0 (no routed_alpha), and B1. For B0 the returned routed_alpha
    has shape (N, 0). Pre-allocates output arrays, writes in place. Cached uncompressed.
    """
    cache_path = Path(cache_path) if cache_path else None
    if cache_path is not None and cache_path.exists():
        d = np.load(cache_path)
        print(f"[features] reusing cache: {cache_path}")
        return d["sub_kg"], d["routed_alpha"]

    model.eval()
    texts = df["_text"].tolist()
    N = len(texts)

    # Probe one batch to learn output shapes (k_routed may be 0 for B0).
    probe = model(model.encoder(texts[:1], max_length=encoder_max_length))
    d_model = probe["sub_kg"].shape[-1]
    ra_probe = probe.get("routed_alpha")
    k_routed = ra_probe.shape[-1] if ra_probe is not None else 0
    sub_kg = np.empty((N, d_model), dtype=np.float32)
    routed_alpha = np.empty((N, k_routed), dtype=np.float32)

    t0 = time.time()
    print(f"[features] encoding {N} samples with BGE + cycle model forward "
          f"(batch={batch_size}, d_model={d_model}, k_routed={k_routed}) ...")
    for i in range(0, N, batch_size):
        batch = texts[i:i + batch_size]
        fact_emb = model.encoder(batch, max_length=encoder_max_length)
        out = model(fact_emb)
        sub_kg[i:i + len(batch)] = out["sub_kg"].cpu().numpy()
        if k_routed > 0:
            routed_alpha[i:i + len(batch)] = out["routed_alpha"].cpu().numpy()
        if (i // batch_size) % 50 == 0 and i > 0:
            elapsed = time.time() - t0
            rate = (i + batch_size) / elapsed
            eta = (N - i - batch_size) / rate
            print(f"  {i}/{N}  ({rate:.0f} samp/s, eta {eta:.0f}s)")
    print(f"[features] done in {time.time() - t0:.0f}s — "
          f"sub_kg={sub_kg.shape}, routed_alpha={routed_alpha.shape}")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, sub_kg=sub_kg, routed_alpha=routed_alpha)
        print(f"[features] cached → {cache_path}")
    return sub_kg, routed_alpha


# ----- Classifier head ---------------------------------------------------------------

class SimBenchHead(nn.Module):
    """MLP over concat(sub_kg, routed_alpha) → MAX_OPTIONS logits.

    B0 baselines have no routing distribution — set `routed_alpha_dim=0` and pass
    routed_alpha=None at forward time. Head input dim collapses to sub_kg_dim.
    """

    def __init__(
        self,
        sub_kg_dim: int = 1024,
        routed_alpha_dim: int = 16,
        hidden: int = 512,
        num_options: int = MAX_OPTIONS,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.routed_alpha_dim = routed_alpha_dim
        self.head = nn.Sequential(
            nn.Linear(sub_kg_dim + routed_alpha_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_options),
        )

    def forward(
        self, sub_kg: torch.Tensor, routed_alpha: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.routed_alpha_dim == 0:
            return self.head(sub_kg)
        return self.head(torch.cat([sub_kg, routed_alpha], dim=-1))


# ----- Dataset / loaders -------------------------------------------------------------

class SimBenchDataset(Dataset):
    def __init__(self, sub_kg, routed_alpha, dist, mask, correct):
        self.sub_kg = torch.from_numpy(np.ascontiguousarray(sub_kg, dtype=np.float32))
        self.routed_alpha = torch.from_numpy(np.ascontiguousarray(routed_alpha, dtype=np.float32))
        self.dist = torch.from_numpy(np.ascontiguousarray(dist, dtype=np.float32))
        self.mask = torch.from_numpy(np.ascontiguousarray(mask, dtype=bool))
        self.correct = torch.from_numpy(np.ascontiguousarray(correct, dtype=np.int64))

    def __len__(self):
        return self.sub_kg.size(0)

    def __getitem__(self, idx):
        return (
            self.sub_kg[idx], self.routed_alpha[idx],
            self.dist[idx], self.mask[idx], self.correct[idx],
        )


def split_dataset(n: int, seed: int = 42, ratios: tuple[float, float, float] = DEFAULT_SPLIT):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]


# ----- Loss + eval -------------------------------------------------------------------

def kl_div_masked(
    logits: torch.Tensor,
    target_dist: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """KL(target || pred) restricted to valid options.

    logits:      (B, K) raw scores from head
    target_dist: (B, K) human distribution, sums to 1 over `mask`, 0 elsewhere
    mask:        (B, K) bool — True for options present in this sample

    KL = Σ_valid target * (log target - log pred). Invalid positions contribute 0 because
    target_dist is 0 there.
    """
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    log_probs = F.log_softmax(masked_logits, dim=-1)
    target_safe = target_dist.clamp_min(KL_EPS)
    kl_per_pos = target_dist * (torch.log(target_safe) - log_probs)
    # invalid positions: target=0 and log_probs=-inf → 0*-inf = NaN; mask to 0.
    kl_per_pos = torch.where(mask, kl_per_pos, torch.zeros_like(kl_per_pos))
    return kl_per_pos.sum(dim=-1).mean()


def evaluate(model: SimBenchHead, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    # Accumulators stay on GPU; single sync at the end avoids per-batch D→H.
    total_kl = torch.zeros((), device=device)
    total_acc = torch.zeros((), dtype=torch.long, device=device)
    total_n = 0
    with torch.no_grad():
        for batch in loader:
            sk, ra, d, m, c = [x.to(device) for x in batch]
            logits = model(sk, ra)
            kl = kl_div_masked(logits, d, m)
            pred = logits.masked_fill(~m, float("-inf")).argmax(dim=-1)
            total_kl += kl.detach() * sk.size(0)
            total_acc += (pred == c).sum()
            total_n += sk.size(0)
    return {
        "kl": float(total_kl) / total_n,
        "argmax_acc": int(total_acc) / total_n,
        "n": total_n,
    }


# ----- Model factory + train entry --------------------------------------------------

def _build_eval_model(cfg: dict, device: torch.device, n_users_fallback: int = 11892):
    """Reconstruct Phase 1 / B0 / B1 from a saved config.json. `model_type` field
    selects the class; legacy phase1 runs lack the field and default to 'phase1'.
    Caller loads weights via `state_dict` separately."""
    model_type = cfg.get("model_type", "phase1")
    encoder_name = cfg["encoder_name"]
    d_bottleneck = cfg.get("d_bottleneck", 64)
    if model_type == "B0":
        model = GenericBaseline(
            encoder_name=encoder_name,
            d_bottleneck=d_bottleneck,
            mlp_width=cfg.get("mlp_width", 4500),
            mlp_n_hidden=cfg.get("mlp_n_hidden", 3),
        )
    elif model_type == "B1":
        model = StandardMoEBaseline(
            encoder_name=encoder_name,
            k_routed=cfg.get("k_routed", 20),
            d_hidden=cfg.get("d_hidden", 2048),
            d_bottleneck=d_bottleneck,
        )
    else:
        model = Phase1Cycle(
            n_users=cfg.get("n_users", n_users_fallback),
            encoder_name=encoder_name,
            k_routed=cfg.get("k_routed", 16),
            k_shared=cfg.get("k_shared", 4),
            d_hidden=cfg.get("d_hidden", 2048),
            config=CycleConfig(
                lambda_lb=cfg.get("lambda_lb", 0.1),
                lambda_ortho=cfg.get("lambda_ortho", 0.0),
                d_bottleneck=d_bottleneck,
            ),
            use_user=cfg.get("use_user", False),
        )
    return model.to(device), model_type


def train_simbench_classifier(
    run_id: str | None = None,
    phase1_run_id: str | None = None,  # backward compat for notebooks pinned to old name
    out_dir: str = "out/phase1",
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden: int = 512,
    dropout: float = 0.1,
    include_pop: bool = True,
    include_grouped: bool = True,
    encoder_max_length: int = 256,
    feature_batch_size: int = FEATURE_BATCH_DEFAULT,
    seed: int = 42,
):
    if run_id is None:
        run_id = phase1_run_id
    if run_id is None:
        raise ValueError("run_id (or phase1_run_id) required")
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(out_dir) / run_id
    if not (run_dir / MODEL_NAME).exists():
        raise FileNotFoundError(f"model.pt not found at {run_dir / MODEL_NAME}")

    cfg = json.loads((run_dir / CONFIG_NAME).read_text())
    model, model_type = _build_eval_model(cfg, device)
    model.load_state_dict(torch.load(run_dir / MODEL_NAME, map_location=device), strict=False)
    model.eval()
    print(f"[model] loaded {run_id} (type={model_type})")

    df = load_simbench(include_pop=include_pop, include_grouped=include_grouped)
    dist, mask, correct = build_targets(df)
    print(f"[simbench] {len(df)} samples, MAX_OPTIONS={MAX_OPTIONS}, "
          f"avg options/sample={mask.sum() / len(df):.2f}")

    feat_cache = Path(out_dir) / "cache" / "simbench" / f"features_{run_id}.npz"
    sub_kg, routed_alpha = extract_features(
        df, model, device,
        batch_size=feature_batch_size,
        encoder_max_length=encoder_max_length,
        cache_path=feat_cache,
    )

    train_idx, val_idx, test_idx = split_dataset(len(df), seed=seed)
    print(f"[split] train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    def _loader(idx, shuffle):
        ds = SimBenchDataset(
            sub_kg[idx], routed_alpha[idx], dist[idx], mask[idx], correct[idx],
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          pin_memory=device.type == "cuda")
    train_loader = _loader(train_idx, shuffle=True)
    val_loader = _loader(val_idx, shuffle=False)
    test_loader = _loader(test_idx, shuffle=False)

    model = SimBenchHead(
        sub_kg_dim=sub_kg.shape[1],
        routed_alpha_dim=routed_alpha.shape[1],
        hidden=hidden,
        num_options=MAX_OPTIONS,
        dropout=dropout,
    ).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[head] trainable params: {n_trainable:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_kl = float("inf")
    best_epoch = 0
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = torch.zeros((), device=device)
        n_seen = 0
        for batch in train_loader:
            sk, ra, d, m, c = [x.to(device) for x in batch]
            logits = model(sk, ra)
            loss = kl_div_masked(logits, d, m)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * sk.size(0)
            n_seen += sk.size(0)
        train_kl = float(loss_sum) / max(n_seen, 1)
        val_metrics = evaluate(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_kl": train_kl,
            "val_kl": val_metrics["kl"],
            "val_argmax_acc": val_metrics["argmax_acc"],
        })
        print(f"epoch {epoch:>3}/{epochs}  train_kl={train_kl:.4f}  "
              f"val_kl={val_metrics['kl']:.4f}  val_acc={val_metrics['argmax_acc']:.4f}")
        if val_metrics["kl"] < best_val_kl:
            best_val_kl = val_metrics["kl"]
            best_epoch = epoch
            torch.save(model.state_dict(), run_dir / CLASSIFIER_NAME)

    # Restore best checkpoint and report on held-out test set.
    model.load_state_dict(torch.load(run_dir / CLASSIFIER_NAME, map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    print(f"\n[test] kl={test_metrics['kl']:.4f}  argmax_acc={test_metrics['argmax_acc']:.4f}")
    print(f"[test] best val_kl={best_val_kl:.4f} at epoch {best_epoch}")

    result = {
        "run_id": run_id,
        "model_type": model_type,
        "best_val_kl": best_val_kl,
        "best_epoch": best_epoch,
        "test": test_metrics,
        "history": history,
        "config": {
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "hidden": hidden, "dropout": dropout,
            "include_pop": include_pop, "include_grouped": include_grouped,
            "encoder_max_length": encoder_max_length,
            "max_options": MAX_OPTIONS,
        },
    }
    (run_dir / EVAL_NAME).write_text(json.dumps(result, indent=2))
    print(f"[simbench] saved → {run_dir / EVAL_NAME}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", default="ph1_v3_minimal",
                   help="Phase 1 or B0/B1 baseline run directory under out_dir")
    p.add_argument("--out_dir", default="out/phase1")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no_pop", dest="include_pop", action="store_false")
    p.add_argument("--no_grouped", dest="include_grouped", action="store_false")
    p.add_argument("--encoder_max_length", type=int, default=256)
    p.add_argument("--feature_batch_size", type=int, default=FEATURE_BATCH_DEFAULT)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train_simbench_classifier(**vars(args))


if __name__ == "__main__":
    main()
