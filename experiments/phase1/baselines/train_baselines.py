"""B0 / B1 baseline cycle pretrain — paper §4.1 risky test 의 비교 대상.

B0 = Generic Encoder + CapacityMLP + FactDecoder (no MoE, capacity-matched MLP)
B1 = Generic Encoder + StandardMoE (Switch top-1) + FactDecoder (no KG-bottleneck-routing
     decoupling, no shared experts)

같은 cycle pretrain setup (corpus, encoder, batch, lr, epochs) 으로 학습 → SimBench Path A
의 비교 대상. Phase 1 (ph1_v3_minimal) 의 71.2% argmax accuracy 가 *MoE-KG-cycle
architecture 의 contribution* 인지 검증.

Usage (from research/demo/):
    from phase1.baselines.train_baselines import train_baseline
    train_baseline('B0', run_id='b0_v0', epochs=30, batch_size=32, lr=1e-4)
    train_baseline('B1', run_id='b1_v0', epochs=30, batch_size=32, lr=1e-4, lambda_lb=0.1)
"""

from __future__ import annotations

import inspect
import json
import math
import os
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import get_cosine_with_min_lr_schedule_with_warmup

from ..data import CorpusConfig, load_phase1_data
from ..train import (
    CKPT_NAME, CONFIG_NAME, HISTORY_NAME, MIN_LR_RATIO, MODEL_NAME,
    SANITY_NAME, SANITY_POST_WARMUP_BUMP, SANITY_POST_WARMUP_MIN,
    SanityGate, WARMUP_MAX_FRACTION, _atomic_save,
)
from .generic_baseline import GenericBaseline
from .standard_moe_baseline import StandardMoEBaseline


BaselineName = Literal["B0", "B1"]


def _build_model(
    baseline: BaselineName,
    encoder_name: str,
    d_bottleneck: int,
    mlp_width: int,
    mlp_n_hidden: int,
    k_routed: int,
    d_hidden: int,
):
    if baseline == "B0":
        return GenericBaseline(
            encoder_name=encoder_name,
            d_bottleneck=d_bottleneck,
            mlp_width=mlp_width,
            mlp_n_hidden=mlp_n_hidden,
        )
    if baseline == "B1":
        return StandardMoEBaseline(
            encoder_name=encoder_name,
            k_routed=k_routed,
            d_hidden=d_hidden,
            d_bottleneck=d_bottleneck,
        )
    raise ValueError(f"baseline must be 'B0' or 'B1', got {baseline!r}")


def train_baseline(
    baseline: BaselineName,
    run_id: str,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-4,
    cos_loss: bool = True,
    d_bottleneck: int = 64,
    # B0-specific
    mlp_width: int = 4500,
    mlp_n_hidden: int = 3,
    # B1-specific
    k_routed: int = 20,
    d_hidden: int = 2048,
    lambda_lb: float = 0.1,
    # shared
    encoder_name: str = "BAAI/bge-large-en-v1.5",
    encoder_max_length: int = 256,
    warmup_steps: int = 300,
    sanity_steps: int = 300,
    continue_on_sanity_fail: bool = False,
    reddit_splits: tuple[str, ...] | None = None,
    reddit_max_rows_per_split: int = 80_000,
    corpus_cache: str | None = None,
    enc_cache_dir: str = "out/phase1/cache",
    grad_clip: float = 1.0,
    out_dir: str = "out/phase1",
    seed: int = 42,
):
    """Cycle pretrain entry for B0 / B1. Auto-resume + LR scheduler aligned with
    `train_phase1`, but model class swapped + cycle_loss signature differs."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] baseline={baseline} device={device}, out={run_dir}")

    # Corpus + cached fact_emb + loaders (same as Phase 1 — sanity guarantees a fair
    # comparison since all three models see the exact same pretrain signal).
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

    model = _build_model(
        baseline, encoder_name=encoder_name, d_bottleneck=d_bottleneck,
        mlp_width=mlp_width, mlp_n_hidden=mlp_n_hidden,
        k_routed=k_routed, d_hidden=d_hidden,
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[train] trainable params: {n_trainable:,}")

    optimizer = torch.optim.AdamW(trainable, lr=lr)

    # LR scheduler — same shape as Phase 1 (warmup + cosine decay to MIN_LR_RATIO).
    total_train_steps = epochs * len(train_loader)
    effective_warmup = min(warmup_steps, max(1, int(total_train_steps * WARMUP_MAX_FRACTION)))
    effective_sanity_steps = sanity_steps
    if effective_sanity_steps < effective_warmup + SANITY_POST_WARMUP_MIN:
        effective_sanity_steps = effective_warmup + SANITY_POST_WARMUP_BUMP
        print(f"[train] sanity_steps ({sanity_steps}) < warmup ({effective_warmup}) + "
              f"{SANITY_POST_WARMUP_MIN}; using {effective_sanity_steps}")
    scheduler = get_cosine_with_min_lr_schedule_with_warmup(
        optimizer,
        num_warmup_steps=effective_warmup,
        num_training_steps=total_train_steps,
        min_lr_rate=MIN_LR_RATIO,
    )

    # Auto-resume from any prior ckpt.pt for this run_id.
    start_epoch = 1
    history: list[dict] = []
    sanity_verdict: dict | None = None
    ckpt_path = run_dir / CKPT_NAME
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        history = ckpt.get("history", [])
        sanity_verdict = ckpt.get("sanity_verdict")
        for key, restore_fn in (("rng_torch", torch.set_rng_state),
                                ("rng_numpy", np.random.set_state)):
            if ckpt.get(key) is not None:
                restore_fn(ckpt[key])
        if torch.cuda.is_available() and ckpt.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(ckpt["rng_cuda"])
        print(f"[train] auto-resume — start at epoch {start_epoch}/{epochs}")
        if start_epoch > epochs:
            print(f"[train] already complete — nothing to do")
            return run_dir

    sanity = SanityGate(window=effective_sanity_steps)

    # cycle_loss signature differs per baseline — B1 takes lambda_lb, B0 doesn't.
    def call_cycle_loss(fact_emb_b):
        if baseline == "B0":
            return model.cycle_loss(fact_emb_b, cos_loss=cos_loss)
        return model.cycle_loss(fact_emb_b, cos_loss=cos_loss, lambda_lb=lambda_lb)

    total_steps = (epochs - start_epoch + 1) * len(train_loader)
    with tqdm(total=total_steps, desc=f"train[{baseline}]", unit="batch", dynamic_ncols=True) as pbar:
        global_step = 0
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            t0 = time.time()
            sums_dev = {k: torch.zeros((), device=device) for k in
                        ("loss", "loss_recon", "loss_lb", "active_frac")}
            n_seen = 0

            for batch in train_loader:
                fact_emb_b, _user_id_b = batch
                fact_emb_b = fact_emb_b.to(device, non_blocking=True)
                try:
                    losses = call_cycle_loss(fact_emb_b)
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
                    b = fact_emb_b.size(0)
                    zero = torch.zeros((), device=device)
                    sums_dev["loss"] += losses["loss"].detach() * b
                    sums_dev["loss_recon"] += losses["loss_recon"] * b
                    sums_dev["loss_lb"] += losses.get("loss_lb", zero) * b
                    ra = losses.get("routed_alpha")
                    if ra is not None:
                        sums_dev["active_frac"] += (ra > 1e-6).float().mean() * b
                    n_seen += b

                if sanity_verdict is None and global_step < effective_sanity_steps:
                    sanity.update(global_step, float(losses["loss"]), losses.get("routed_alpha"))

                if global_step % 50 == 0:
                    postfix = {"ep": epoch, "loss": f"{float(losses['loss']):.4f}"}
                    if losses.get("routed_alpha") is not None:
                        postfix["act"] = f"{float((losses['routed_alpha'] > 1e-6).float().mean()):.2f}"
                    pbar.set_postfix(**postfix)
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
                    print("[sanity] failing — aborting before remaining epochs.")
                    break

            entry = {"epoch": epoch, "seconds": time.time() - t0, **sums}
            history.append(entry)
            act_str = f"  K_active={entry['active_frac']*max(k_routed, 1):.1f}" if baseline == "B1" else ""
            print(f"epoch {epoch}/{epochs}  loss={entry['loss']:.4f}  "
                  f"recon={entry['loss_recon']:.4f}  lb={entry['loss_lb']:.4f}"
                  f"{act_str}  ({entry['seconds']:.0f}s)")

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

    # Config dump — eval_simbench_classifier reads `baseline` / `model_type` here to
    # rebuild the right class. Keep the field stable.
    _signature_params = inspect.signature(train_baseline).parameters
    config_dump = {k: locals()[k] for k in _signature_params if k in locals()}
    config_dump["n_users"] = n_users
    config_dump["model_type"] = baseline                      # B0 or B1
    config_dump["encoder_name"] = encoder_name
    config_dump = {k: (list(v) if isinstance(v, tuple) else v) for k, v in config_dump.items()}
    _atomic_save(config_dump, run_dir / CONFIG_NAME, json_mode=True)
    print(f"[train] saved → {run_dir}")
    return run_dir
