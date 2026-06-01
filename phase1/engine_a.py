"""Engine-A runner — train the operation-cycle, then run the selectivity gate.

This is the go/no-go orchestrator (ENGINE_A_DESIGN §5): given per-token embeddings of
the operation-axis corpus (Super-NaturalInstructions) and the probe set (QuAIL), it
trains `OpCycleMoE` on the unsupervised reconstruction cycle, extracts sequence codes
(`mean_t α`) on the probe set, and emits a Hewitt-Liang selectivity verdict.

The frozen-BGE encode is *not* done here — callers pass pre-computed token arrays
(`data.encode_or_load_tokens`). That keeps this module LLM-free, device-agnostic, and
unit-testable on tiny synthetic tensors, and lets Colab cache the (expensive) encode once.
"""

from __future__ import annotations

import numpy as np
import torch

from .eval_opcycle import sequence_code, selectivity_report
from .model_opcycle import OpCycleMoE, opcycle_loss, update_l1_lambda


def _cpu_tokens(x) -> torch.Tensor:
    """Keep the (N, T, d) token array on CPU in its NATIVE dtype (typically fp16 from the
    cache). We deliberately do NOT upcast-to-float32-on-device here: at N≈40k, T=128,
    d=1024 that single allocation is ~21 GB and OOMs the GPU before training starts. Each
    batch is sliced and moved to the device as float32 inside the loop instead (~134 MB)."""
    if isinstance(x, torch.Tensor):
        return x.cpu()
    return torch.as_tensor(np.asarray(x))


def _cpu_mask(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.cpu().long()
    return torch.as_tensor(np.asarray(x)).long()


def train_opcycle(
    model: OpCycleMoE,
    tokens,
    mask,
    *,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    lambdas: dict | None = None,
    k_target: float | None = None,
    return_best: bool = False,
    log_every: int = 0,
    device: str | torch.device = "cpu",
):
    """Train the operation-cycle on per-token embeddings (unsupervised reconstruction).

    tokens: (N, T, d_model), mask: (N, T). Returns per-epoch history of (example-weighted)
    mean loss components. AdamW + (recon + ReMoE L1 + z-loss + load-balance).

    Tokens stay on CPU in their native dtype; each batch is moved to `device` as float32
    (avoids a multi-GB up-front device allocation). If `k_target` is given, the L1
    coefficient is adapted each epoch toward that mean K_active (ReMoE adaptive sparsity),
    overriding any fixed lambdas['lambda_l1']; the running λ is logged as history['lam_l1'].

    `return_best=True` → returns `(history, best_state)` where `best_state` is a CPU
    deep-copy of the model's state_dict at the epoch with the lowest recon. The ReMoE
    adaptive-L1 controller can over-shoot sparsity in late epochs and degrade recon (recon
    rises after its minimum); the best-recon checkpoint lets the caller probe the model at
    its reconstruction peak rather than the over-sparsified final weights. Default
    (`return_best=False`) returns just `history` (unchanged contract).

    `log_every>0` prints a one-line per-epoch progress summary (recon / k_active / λ_l1)
    every `log_every` epochs (and the last) — useful on Colab where a 40-epoch run is
    otherwise silent until the verdict. Default 0 = silent (unchanged).
    """
    device = torch.device(device)
    model.to(device).train()
    tokens = _cpu_tokens(tokens)
    mask = _cpu_mask(mask)
    lambdas = dict(lambdas or {})
    lam_l1 = lambdas.pop("lambda_l1", 1e-2)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    n = tokens.size(0)
    history: list[dict] = []
    best_recon = float("inf")
    best_state: dict | None = None
    for ep in range(epochs):
        perm = torch.randperm(n)
        epoch_parts: dict[str, float] = {}
        n_seen = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            h_b = tokens[idx].to(device=device, dtype=torch.float32)
            m_b = mask[idx].to(device)
            out = model(h_b, m_b)
            total, parts = opcycle_loss(out, h_b, m_b, lambda_l1=lam_l1, **lambdas)
            opt.zero_grad()
            total.backward()
            opt.step()
            parts["total"] = float(total.detach())
            bs = h_b.size(0)
            for kkey, v in parts.items():
                epoch_parts[kkey] = epoch_parts.get(kkey, 0.0) + v * bs
            n_seen += bs
        epoch = {k: v / max(n_seen, 1) for k, v in epoch_parts.items()}
        epoch["lam_l1"] = lam_l1
        history.append(epoch)
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            print(f"  [epoch {ep + 1:>3}/{epochs}] recon={epoch['recon']:.4f} "
                  f"k_active={epoch['k_active']:.2f} λ_l1={lam_l1:.5f} "
                  f"total={epoch['total']:.4f}", flush=True)
        if return_best and epoch["recon"] < best_recon:
            best_recon = epoch["recon"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if k_target is not None:
            lam_l1 = update_l1_lambda(lam_l1, epoch["k_active"], k_target)
    if return_best:
        # best_state is set whenever ≥1 epoch ran (recon < inf); fall back to final weights
        # for the degenerate epochs=0 case so the contract (a dict) always holds.
        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return history, best_state
    return history


@torch.no_grad()
def compute_codes(
    model: OpCycleMoE,
    tokens,
    mask,
    *,
    batch_size: int = 64,
    agg: str = "mean",
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Sequence codes (N, K) [or (N, 2K) for agg='meanmax'] over the probe set.

    Per-batch device transfer (same memory rationale as train_opcycle). Returns a correctly
    shaped empty (0, width) array when the probe set is empty rather than crashing on
    np.concatenate of an empty list."""
    device = torch.device(device)
    model.to(device).eval()
    tokens = _cpu_tokens(tokens)
    mask = _cpu_mask(mask)
    n = tokens.size(0)
    width = model.k * (2 if agg == "meanmax" else 1)
    if n == 0:
        return np.empty((0, width), dtype=np.float32)
    chunks = []
    for i in range(0, n, batch_size):
        h_b = tokens[i:i + batch_size].to(device=device, dtype=torch.float32)
        m_b = mask[i:i + batch_size].to(device)
        out = model(h_b, m_b)
        chunks.append(sequence_code(out["alpha"], m_b, agg=agg))
    return np.concatenate(chunks, axis=0)


def run_engine_a(
    train_tokens,
    train_mask,
    probe_tokens,
    probe_mask,
    operation_labels,
    control_label_sets: dict | None = None,
    *,
    d_model: int | None = None,
    d_z: int = 256,
    k: int = 16,
    d_hidden: int = 512,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    lambdas: dict | None = None,
    k_target: float | None = None,
    use_best_recon: bool = False,
    route_on_deviation: bool = False,
    log_every: int = 0,
    agg: str = "mean",
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> dict:
    """Full go/no-go pipeline: train cycle → probe codes → selectivity verdict.

    Returns {"history", "model", "codes", "report"}. `report["verdict"]` is PASS iff the
    chance-adjusted operation probe accuracy beats every control (random / topic / token /
    geometry) with topic+geometry present. `agg` selects the sequence-code aggregation
    ('meanmax' recommended for the real probe — see sequence_code). `k_target` enables the
    ReMoE adaptive-L1 controller. `use_best_recon=True` extracts the probe codes from the
    best-recon checkpoint (the reconstruction peak) rather than the final, possibly
    over-sparsified, weights. `route_on_deviation=True` routes on within-sequence local
    structure instead of sequence-global content (the F3b topic-collapse lever).
    """
    n_probe = probe_tokens.shape[0] if hasattr(probe_tokens, "shape") else len(probe_tokens)
    if n_probe == 0 or len(operation_labels) == 0:
        raise ValueError(
            "run_engine_a: empty probe set or operation_labels — QuAIL load likely failed."
        )
    n_train = train_tokens.shape[0] if hasattr(train_tokens, "shape") else len(train_tokens)
    if n_train == 0:
        # An empty train set silently skips every batch loop, leaving epoch dicts without
        # recon/k_active/total → a cryptic KeyError downstream (log_every print, return_best,
        # k_target controller). Fail fast with the likely cause instead (mirrors the probe
        # guard) — same failure mode as the Super-NI load returning an empty corpus.
        raise ValueError(
            "run_engine_a: empty train set — Super-NI load likely failed."
        )
    if d_model is None:
        d_model = (
            train_tokens.shape[-1]
            if hasattr(train_tokens, "shape")
            else np.asarray(train_tokens).shape[-1]
        )
    torch.manual_seed(seed)
    model = OpCycleMoE(
        d_model=d_model, d_z=d_z, k=k, d_hidden=d_hidden,
        route_on_deviation=route_on_deviation,
    )

    train_out = train_opcycle(
        model, train_tokens, train_mask,
        epochs=epochs, batch_size=batch_size, lr=lr, lambdas=lambdas,
        k_target=k_target, return_best=use_best_recon, log_every=log_every, device=device,
    )
    if use_best_recon:
        history, best_state = train_out
        model.load_state_dict(best_state)
    else:
        history = train_out
    codes = compute_codes(
        model, probe_tokens, probe_mask, batch_size=batch_size, agg=agg, device=device
    )
    report = selectivity_report(
        codes, operation_labels, control_label_sets=control_label_sets, seed=seed
    )
    return {"history": history, "model": model, "codes": codes, "report": report}
