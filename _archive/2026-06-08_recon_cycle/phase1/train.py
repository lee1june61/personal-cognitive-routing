"""Phase 1 training entry — embedding-level cycle on Pennebaker + Reddit.

Revision 4 (2026-05-21): LLM 제거, embedding-only cycle (BGE-large-en frozen + MoE +
FactDecoder + cycle loss). Stage 1 spec default: K_routed=16, K_shared=4, d_hidden=2048.

Usage (Colab / local, from research/demo/):

    # Phase 1 cycle pretrain (use_user=False, default)
    python -m phase1.train --run_id ph1_v0_ortho_off --epochs 10 --lambda_ortho 0.0
    python -m phase1.train --run_id ph1_v0_ortho_on  --epochs 10 --lambda_ortho 0.05

    # Phase 2 cycle pretrain (use_user=True, after Phase 1 통과)
    python -m phase1.train --run_id ph2_v0 --use_user --epochs 10 \
        --resume_from out/phase1/ph1_v0_ortho_on/model.pt

Outputs (out/phase1/<run_id>/):
  - model.pt        (full Phase1Cycle state)
  - history.json    (per-epoch loss / recon / lb / ortho / k_active / λ)
  - config.json     (run config + use_user)
  - sanity.json     (mini-run gate verdict)
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from transformers import get_cosine_with_min_lr_schedule_with_warmup

from .cycle import CycleConfig, Phase1Cycle
from .data import CorpusConfig, load_phase1_data


# Per-run artifact filenames (shared with eval.py).
MODEL_NAME = "model.pt"           # trainable state only, eval-facing
CKPT_NAME = "ckpt.pt"             # full training state (model + opt + RNG + epoch), auto-resume
HISTORY_NAME = "history.json"
SANITY_NAME = "sanity.json"
CONFIG_NAME = "config.json"

# LR scheduler / sanity gate tunables.
WARMUP_MAX_FRACTION = 0.05        # cap warmup at 5% of total training steps
MIN_LR_RATIO = 0.01               # cosine decay floor as fraction of peak lr
SANITY_POST_WARMUP_MIN = 200      # sanity_steps must allow this many post-warmup steps
SANITY_POST_WARMUP_BUMP = 500     # if check fails, bump sanity_steps to warmup + this


def _atomic_save(obj, path: Path, *, json_mode: bool = False) -> None:
    """Atomic write via tmp file + os.replace — protects against partial writes
    if the process is killed mid-save (Colab session timeout)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    if json_mode:
        tmp.write_text(json.dumps(obj, indent=2, default=str))
    else:
        torch.save(obj, tmp)
    os.replace(tmp, path)


# ----- Sanity gate -------------------------------------------------------------------

class SanityGate:
    """Streaming sanity checker (RESEARCH_PLAN §7 Q7).

    Gates: (a) loss finite, (b) no OOM, (c) K_active mean > 1e-3,
    (d) loss last quarter < first quarter (decreasing trend over window).
    """

    def __init__(self, window: int, k_active_min: float = 1e-3):
        self.window = window
        self.k_active_min = k_active_min
        self.losses: list[float] = []
        self.k_active: list[float] = []
        self.first_oom = False
        self.first_nonfinite_step: int | None = None

    def update(self, step: int, loss: float, routed_alpha: torch.Tensor | None = None):
        if not math.isfinite(loss):
            if self.first_nonfinite_step is None:
                self.first_nonfinite_step = step
        self.losses.append(loss)
        if routed_alpha is not None:
            with torch.no_grad():
                self.k_active.append(float((routed_alpha > 1e-6).float().mean()))

    def mark_oom(self):
        self.first_oom = True

    def verdict(self) -> dict:
        losses = self.losses
        result = {
            "n_steps": len(losses),
            "first_oom": self.first_oom,
            "first_nonfinite_step": self.first_nonfinite_step,
        }
        if not losses:
            return {**result, "verdict": "no_data", "reasons": ["no batches"]}
        reasons: list[str] = []
        if self.first_nonfinite_step is not None:
            reasons.append(f"non-finite loss at step {self.first_nonfinite_step}")
        if self.first_oom:
            reasons.append("OOM during sanity window")
        if self.k_active:
            k_mean = float(np.mean(self.k_active))
            result["k_active_mean"] = k_mean
            if k_mean <= self.k_active_min:
                reasons.append(f"K_active collapsed (mean={k_mean:.4f})")
        q = max(1, len(losses) // 4)
        fq = float(np.mean(losses[:q])); lq = float(np.mean(losses[-q:]))
        result["loss_first_quarter"] = fq
        result["loss_last_quarter"] = lq
        if lq >= fq:
            reasons.append(f"loss not decreasing ({fq:.4f} → {lq:.4f})")
        result["reasons"] = reasons
        result["verdict"] = "pass" if not reasons else "fail"
        return result


# ----- Train entry -------------------------------------------------------------------

def train_phase1(
    run_id: str = "ph1_v0",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    k_routed: int = 16,
    k_shared: int = 4,
    d_hidden: int = 2048,
    lambda_recon: float = 1.0,
    lambda_lb: float = 0.1,
    lambda_ortho: float = 0.0,
    d_bottleneck: int = 64,
    encoder_name: str = "BAAI/bge-large-en-v1.5",
    encoder_max_length: int = 256,
    cos_loss: bool = True,
    use_user: bool = False,
    warmup_steps: int = 1000,
    sanity_steps: int = 300,
    continue_on_sanity_fail: bool = False,
    reddit_splits: tuple[str, ...] | None = None,
    reddit_max_rows_per_split: int = 80_000,
    corpus_cache: str | None = None,
    enc_cache_dir: str = "out/phase1/cache",
    resume_from: str | None = None,
    grad_clip: float = 1.0,
    out_dir: str = "out/phase1",
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] device={device}, out={run_dir}, use_user={use_user}")

    # --- Corpus + fact_emb cache + loaders ---
    ccfg_kwargs = dict(
        reddit_max_rows_per_split=reddit_max_rows_per_split,
        encoder_name=encoder_name, encoder_max_length=encoder_max_length, seed=seed,
    )
    if reddit_splits is not None:
        ccfg_kwargs["reddit_splits"] = tuple(reddit_splits)
    ccfg = CorpusConfig(**ccfg_kwargs)
    corpus, fact_emb, train_loader, val_loader, test_loader = load_phase1_data(
        cfg=ccfg, corpus_cache=corpus_cache, enc_cache_dir=enc_cache_dir,
        batch_size=batch_size,
    )
    n_users = int(corpus["user_id"].nunique())
    print(f"[train] n_users={n_users}, fact_emb={fact_emb.shape}")

    # --- Model ---
    cycle_cfg = CycleConfig(
        lambda_recon=lambda_recon, lambda_lb=lambda_lb, lambda_ortho=lambda_ortho,
        cos_loss=cos_loss, d_bottleneck=d_bottleneck,
    )
    model = Phase1Cycle(
        n_users=n_users,
        encoder_name=encoder_name,
        k_routed=k_routed, k_shared=k_shared, d_hidden=d_hidden,
        config=cycle_cfg, use_user=use_user,
    ).to(device)

    if resume_from is not None:
        if use_user:
            report = model.load_phase1_weights(resume_from, map_location=device)
            print(f"[train] loaded Phase 1 weights from {resume_from}; "
                  f"missing={report['missing']}, unexpected={report['unexpected']}")
        else:
            ckpt = torch.load(resume_from, map_location=device)
            model.load_state_dict(ckpt, strict=False)
            print(f"[train] resumed Phase 1 from {resume_from}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[train] trainable params: {n_trainable:,}")

    optimizer = torch.optim.AdamW(trainable, lr=lr)

    # LR scheduler — linear warmup → cosine decay to MIN_LR_RATIO * peak. total_train_steps
    # spans the full epoch budget so resume-after-interrupt picks up at the right point on
    # the cosine curve.
    total_train_steps = epochs * len(train_loader)
    effective_warmup = min(warmup_steps, max(1, int(total_train_steps * WARMUP_MAX_FRACTION)))
    effective_sanity_steps = sanity_steps
    if effective_sanity_steps < effective_warmup + SANITY_POST_WARMUP_MIN:
        effective_sanity_steps = effective_warmup + SANITY_POST_WARMUP_BUMP
        print(f"[train] sanity_steps ({sanity_steps}) < warmup ({effective_warmup}) + "
              f"{SANITY_POST_WARMUP_MIN}; using {effective_sanity_steps} so loss-decrease "
              f"check sees post-warmup steps")
    scheduler = get_cosine_with_min_lr_schedule_with_warmup(
        optimizer,
        num_warmup_steps=effective_warmup,
        num_training_steps=total_train_steps,
        min_lr_rate=MIN_LR_RATIO,
    )

    # Auto-resume: explicit `resume_from` (Phase 1 → Phase 2 weight transfer) takes
    # precedence; otherwise pick up any `ckpt.pt` left by a previous interrupted run.
    # Clean restart = delete `run_dir/` or pass a fresh `run_id`.
    start_epoch = 1
    history: list[dict] = []
    sanity_verdict: dict | None = None
    ckpt_path = run_dir / CKPT_NAME
    if ckpt_path.exists() and resume_from is None:
        # weights_only=False: ckpt holds np RNG tuple + history list-of-dicts +
        # sanity_verdict dict — trusted local file. map_location='cpu' avoids a mild
        # VRAM spike when load_state_dict will route to device anyway.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        history = ckpt.get("history", [])
        sanity_verdict = ckpt.get("sanity_verdict")
        # Restore RNG so DataLoader shuffle resumes deterministically.
        for key, restore_fn in (("rng_torch", torch.set_rng_state),
                                ("rng_numpy", np.random.set_state)):
            if ckpt.get(key) is not None:
                restore_fn(ckpt[key])
        if torch.cuda.is_available() and ckpt.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(ckpt["rng_cuda"])
        print(f"[train] auto-resume from {ckpt_path} — start at epoch {start_epoch}/{epochs}"
              f"  (history={len(history)} entries, sanity_verdict={sanity_verdict and sanity_verdict.get('verdict')})")
        if start_epoch > epochs:
            print(f"[train] already complete — nothing to do")
            return run_dir

    # --- Sanity + training loop ---
    sanity = SanityGate(window=effective_sanity_steps)

    total_steps = (epochs - start_epoch + 1) * len(train_loader)
    with tqdm(total=total_steps, desc="train", unit="batch", dynamic_ncols=True) as pbar:
        global_step = 0
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            t0 = time.time()
            # On-device sum accumulators — sync once per epoch (not per batch).
            sums_dev = {k: torch.zeros((), device=device) for k in
                        ("loss", "loss_recon", "loss_lb", "loss_ortho", "active_frac", "lambda_mean")}
            n_seen = 0

            for batch in train_loader:
                fact_emb_b, user_id_b = batch
                fact_emb_b = fact_emb_b.to(device, non_blocking=True)
                user_id_b = user_id_b.to(device, non_blocking=True)
                try:
                    losses = model.cycle_loss(
                        fact_emb_b,
                        user_id=user_id_b if use_user else None,
                    )
                    optimizer.zero_grad()
                    losses["loss"].backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(trainable, max_norm=grad_clip)
                    optimizer.step()
                    scheduler.step()
                except torch.cuda.OutOfMemoryError as e:
                    if sanity_verdict is None and global_step < effective_sanity_steps:
                        sanity.mark_oom()
                    print(f"  OOM at batch step {global_step}: {e}")
                    optimizer.zero_grad()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

                with torch.no_grad():
                    active_frac_t = (losses["routed_alpha"] > 1e-6).float().mean()
                    lam_t = losses["lam"].mean()
                    b = fact_emb_b.size(0)
                    zero = torch.zeros((), device=device)
                    sums_dev["loss"] += losses["loss"].detach() * b
                    sums_dev["loss_recon"] += losses["loss_recon"] * b
                    sums_dev["loss_lb"] += losses.get("loss_lb", zero) * b
                    sums_dev["loss_ortho"] += losses.get("loss_ortho", zero) * b
                    sums_dev["active_frac"] += active_frac_t * b
                    sums_dev["lambda_mean"] += lam_t * b
                    n_seen += b

                # Sanity gate runs inside its window — small extra sync OK there.
                if sanity_verdict is None and global_step < effective_sanity_steps:
                    sanity.update(global_step, float(losses["loss"]), losses["routed_alpha"])

                # Throttle progress-bar D2H sync to every 50 steps to keep hot path async.
                if global_step % 50 == 0:
                    pbar.set_postfix(
                        ep=epoch, loss=f"{float(losses['loss']):.4f}",
                        act=f"{float(active_frac_t):.2f}", λ=f"{float(lam_t):+.3f}",
                    )
                pbar.update(1)
                global_step += 1

            sums = {k: float(v) / max(n_seen, 1) for k, v in sums_dev.items()}

            if sanity_verdict is None and global_step >= effective_sanity_steps:
                sanity_verdict = sanity.verdict()
                _atomic_save(sanity_verdict, run_dir / SANITY_NAME, json_mode=True)
                print(f"[sanity] verdict={sanity_verdict['verdict']}")
                for r in sanity_verdict.get("reasons", []):
                    print(f"[sanity]   - {r}")
                if sanity_verdict["verdict"] != "pass" and not continue_on_sanity_fail:
                    print("[sanity] failing — aborting before remaining epochs. "
                          "Re-run with --continue_on_sanity_fail to ignore.")
                    break

            entry = {"epoch": epoch, "seconds": time.time() - t0, **sums}
            history.append(entry)
            print(f"epoch {epoch}/{epochs}  loss={entry['loss']:.4f}  "
                  f"recon={entry['loss_recon']:.4f}  lb={entry['loss_lb']:.4f}  "
                  f"ortho={entry['loss_ortho']:.4f}  K_active={entry['active_frac']*k_routed:.1f}  "
                  f"λ={entry['lambda_mean']:+.3f}  ({entry['seconds']:.0f}s)")

            # Two checkpoint files per epoch:
            #  model.pt = trainable weights only (encoder.* stripped — public BGE weights);
            #             used by eval.py for inference.
            #  ckpt.pt  = full training state (model + optimizer + epoch + RNG + history +
            #             sanity verdict); used for auto-resume across Colab session timeouts.
            # Atomic writes (tmp + os.replace) so a kill mid-save doesn't corrupt the file.
            trainable_state = {
                k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")
            }
            _atomic_save(trainable_state, run_dir / MODEL_NAME)
            _atomic_save({
                "model": trainable_state,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "history": history,
                "sanity_verdict": sanity_verdict,
                "rng_torch": torch.get_rng_state(),
                "rng_numpy": np.random.get_state(),
                "rng_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            }, run_dir / CKPT_NAME)
            _atomic_save(history, run_dir / HISTORY_NAME, json_mode=True)

    # Derive config_dump from locals — survives signature changes without drift.
    _signature_params = inspect.signature(train_phase1).parameters
    config_dump = {k: locals()[k] for k in _signature_params if k in locals()}
    config_dump["n_users"] = n_users
    # JSON-safe coercion for non-primitive defaults (e.g. tuple).
    config_dump = {k: (list(v) if isinstance(v, tuple) else v) for k, v in config_dump.items()}
    _atomic_save(config_dump, run_dir / CONFIG_NAME, json_mode=True)
    print(f"[train] saved → {run_dir}")
    return run_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", default="ph1_v0")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--k_routed", type=int, default=16)
    p.add_argument("--k_shared", type=int, default=4)
    p.add_argument("--d_hidden", type=int, default=2048)
    p.add_argument("--lambda_recon", type=float, default=1.0)
    p.add_argument("--lambda_lb", type=float, default=0.1)
    p.add_argument("--lambda_ortho", type=float, default=0.0)
    p.add_argument("--d_bottleneck", type=int, default=64)
    p.add_argument("--encoder_name", default="BAAI/bge-large-en-v1.5")
    p.add_argument("--encoder_max_length", type=int, default=256)
    p.add_argument("--cos_loss", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use_user", action="store_true")
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--sanity_steps", type=int, default=300)
    p.add_argument("--continue_on_sanity_fail", action="store_true")
    p.add_argument("--reddit_splits", nargs="*", default=None,
                   help="space-separated subreddit list (default = 8 diverse subreddits)")
    p.add_argument("--reddit_max_rows_per_split", type=int, default=80_000)
    p.add_argument("--corpus_cache", default=None)
    p.add_argument("--enc_cache_dir", default="out/phase1/cache")
    p.add_argument("--resume_from", default=None)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--out_dir", default="out/phase1")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train_phase1(**vars(args))


if __name__ == "__main__":
    main()
