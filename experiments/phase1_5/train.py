"""Phase 1.5 1a training loop — answer-prediction CE + ReMoE sparsity + z-loss.

Loss composition (paper §5.1 row 1, §5.4 row 7; ``λ_ortho=0`` per memory
``project_lambda_ortho_collapse``):

    L = mc_ce + λ_l1 · L1(α)  +  λ_z · z_loss

- ``mc_ce``  : F.cross_entropy(logits, answer_idx). Paper §5.1 row 1.
- ``L1(α)``  : mean per-token ‖α‖₁ over masked Q tokens. ReMoE adaptive L1
  drives K_active toward ``k_target`` via ``update_l1_lambda`` (copy of phase1).
- ``z_loss`` : ST-MoE (Zoph 2022) router z-loss = mean logsumexp(logits)² over
  masked Q tokens. Keeps router logits numerically stable.

Layer-1 LB (DeepSeek-V3 / Wang 2408.15664) updates a per-expert bias buffer
after each ``optimizer.step``. The bias is added to router logits *before*
ReLU and *before* ``router_z_loss``, so z-loss penalises the bias-shifted
logits. This is intentional: it caps the magnitude of (gate(z) + bias) and
prevents the LB drift from blowing up unboundedly. The practical trade-off
is that on a sustained-dead expert, z-loss exerts a downward gradient on
the gate weights that partially counteracts the LB's upward bias drift —
but with paper-default λ_z=1e-3 and lr_bias=1e-3, both forces are O(1e-3)
per step and the LB-controlled equilibrium remains the dominant attractor.
If post-Layer-1 evaluation shows the recovery is being undone post-warmup,
Layer 2 (k_target up / λ_z permanently off) is the documented escalation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Loss primitives now live in core.loss_primitives (canonical source =
# phase1/model_opcycle.py; phase1_5 held verbatim copies). Imported here so the names
# stay available to this module and its tests unchanged. ``_masked_token_mean`` is the
# back-compat underscore alias used in the train/eval loops below.
from core.loss_primitives import (
    _masked_token_mean,
    remoe_l1_loss,
    router_z_loss,
    update_l1_lambda,
)


# ----- Loss primitives (phase1_5-specific) ------------------------------------------


def mc_ce_loss(logits: torch.Tensor, answer_idx: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over MC candidates against the gold index. Phase 1.5 1a primary loss."""
    return F.cross_entropy(logits, answer_idx)


# ----- Train config + main loop -----------------------------------------------------


def resolve_device(device: str | torch.device) -> torch.device:
    """Resolve a requested torch device, warning on silent demote.

    Behaviour:
      - ``cuda`` requested but no CUDA visible → warn + fall back to CPU.
      - ``mps`` requested but Apple MPS unavailable → warn + fall back to CPU.
      - ``cpu`` requested → CPU.
      - Anything else (``xpu``, ``rocm``, typos) → warn + fall back to CPU.
      - Already a ``torch.device`` instance → pass through (caller intent honoured).

    Accepts both ``str`` and ``torch.device`` for callers that have already
    resolved the device (notebook idiom ``dev = torch.device('cuda')``).
    """
    if isinstance(device, torch.device):
        return device
    requested = str(device).lower()
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device)
        print(f"[device] warning: {device!r} requested but CUDA unavailable; using CPU")
        return torch.device("cpu")
    if requested == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        print(f"[device] warning: {device!r} requested but MPS unavailable; using CPU")
        return torch.device("cpu")
    if requested == "cpu":
        return torch.device("cpu")
    print(f"[device] warning: unknown device {device!r}; using CPU")
    return torch.device("cpu")


# Back-compat alias (the underscore-prefixed name was cross-module imported in
# the first review pass — keep it as a thin alias so existing imports still work).
_resolve_device = resolve_device


@dataclass
class TrainConfig:
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-2
    k_target: float = 4.0
    # Per-expert L1 penalty scales with K. Phase 1 K=16 used λ_l1=1e-2.
    # Phase 1.5 Layer-1 (K=128, λ_l1=1e-3) showed an ep3 cliff (K_active 52→1)
    # despite the 8× scale-down — adaptive λ ratio at K=128 is too aggressive
    # for the post-warmup step. Layer 2-B drops a further 10× to 1e-4. Wang
    # 2412.14711 §3.2 (ReMoE) calls out adaptive K-scaling explicitly; the
    # adaptive controller (``update_l1_lambda``) still nudges multiplicatively
    # from this initial value, so the lower start does not pin sparsity — it
    # just buys the router headroom to leave warmup without collapsing.
    lam_l1_init: float = 1e-4
    # L1 OFF for the first ``lam_l1_warmup_epochs`` — lets the router establish
    # a non-zero gate distribution before the sparsity penalty kicks in. Without
    # warmup, λ_l1 applied from epoch 0 over a freshly-init router pushes ReLU
    # gates negative within one step → dead router with no gradient to recover.
    lam_l1_warmup_epochs: int = 3
    # If K_active drops below this at end-of-epoch, λ_l1 is multiplied by
    # ``dead_rescue_factor`` (release pressure). The adaptive controller alone
    # cannot recover from K_active=0 because dead ReLU gates produce no gradient.
    dead_rescue_k_active: float = 0.5
    dead_rescue_factor: float = 0.1
    # Layer 2-A: λ_z forced to 0 for the LB-inclusive setting (Phase 1.5 1a
    # default = lb_strategy='aux_free'). Phase 1.5 Layer-1 Colab run showed
    # z_loss × LB conflict — LB drifts bias up on dead experts, which lifts
    # logsumexp(logits)²; z_loss's gradient then pushes gate weights DOWN to
    # cancel the bias drift, locking (gate(z) + bias) near zero and re-creating
    # the K_active=0 collapse the LB was meant to break. Paper §7.4 Gap D
    # ("z_loss redundancy in LB-inclusive setting") is the lit-grounded
    # justification. Row F (lb_strategy='off') should explicitly set this back
    # to 1e-3 if the z-loss baseline is desired.
    lam_z: float = 0.0
    grad_clip: float = 1.0
    log_every: int = 1
    use_best_val: bool = True
    # Metric to select the best checkpoint: "loss" (min val CE) or "acc" (max val
    # MC accuracy). With overfitting, val CE explodes (confident-wrong) while val
    # accuracy holds — so "acc" selects a usefully-trained model where "loss"
    # would pick the underfit epoch-0 checkpoint.
    best_metric: str = "loss"
    seed: int = 0


def _forward_normalized(model: nn.Module, batch: dict):
    """Run the model and normalise flat (1a) vs chain (1b) outputs to a common
    shape: ``(logits, alpha_list, router_logits_list, k_active_list)`` where the
    lists have one entry for flat and ``chain_steps`` entries for the chain. Lets
    the train/eval loops treat both regimes uniformly (per-step L1/z-loss are
    *averaged* so the adaptive-L1 K-target scale is comparable across regimes)."""
    if getattr(model, "chain_steps", 1) > 1:
        # ⚠ 1b chain path DEPRECATED (2026-06-08): seq layout = setup-failure →
        # direction-1 parallel. Kept for reproduction; flat 1a (chain_steps=1) is live.
        out = model.forward_chain(batch)
        return out["logits"], out["alpha_steps"], out["router_logits_steps"], out["k_active_steps"]
    out = model(batch)
    return out["logits"], [out["alpha"]], [out["router_logits"]], [out["k_active"]]


def train_phase15(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    val_loader: DataLoader | None = None,
    cfg: TrainConfig | None = None,
    device: str = "cuda",
    progress: bool = False,
) -> dict:
    """Train Phase15MoE on MC-CE + ReMoE L1 + z-loss.

    Returns a dict with:
        ``history``      : list of per-epoch metrics dicts;
        ``best_state``   : state_dict at the lowest val loss (or last epoch
                           if ``val_loader`` is None / ``use_best_val=False``);
        ``best_val_loss``: float or None;
        ``model``        : the (possibly best-state-restored) trained model.
    """
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)

    torch_device = resolve_device(device)
    model = model.to(torch_device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1))

    lam_l1 = cfg.lam_l1_init
    history: list[dict] = []
    best_val_loss = float("inf")
    best_score = float("-inf")  # for best_metric="acc" (higher is better)
    best_state: dict | None = None

    pbar = None
    if progress:
        try:
            from tqdm.auto import tqdm

            pbar = tqdm(range(cfg.epochs), desc="train_phase15")
        except ImportError:
            pbar = None

    for epoch_idx in (pbar if pbar is not None else range(cfg.epochs)):
        model.train()
        sum_loss = 0.0
        sum_ce = 0.0
        sum_l1 = 0.0
        sum_zl = 0.0
        sum_k_active = 0.0
        sum_correct = 0
        n_samples = 0

        # Warmup: BOTH λ_l1 AND λ_z are 0 for the first ``lam_l1_warmup_epochs``.
        # z_loss (logsumexp(logits)²) pushes the router logits toward zero, which
        # combined with the ReLU gate amplifies dead-expert collapse at warmup
        # time — so we suppress it during warmup along with the L1 penalty.
        in_warmup = int(epoch_idx) < cfg.lam_l1_warmup_epochs
        effective_lam_l1 = 0.0 if in_warmup else lam_l1
        effective_lam_z = 0.0 if in_warmup else cfg.lam_z

        for batch in train_loader:
            batch = {k: v.to(torch_device) for k, v in batch.items()}
            logits, alpha_list, rlogits_list, kact_list = _forward_normalized(model, batch)

            ce = mc_ce_loss(logits, batch["answer_idx"])
            n_steps = len(alpha_list)
            l1 = sum(remoe_l1_loss(a, batch["q_mask"]) for a in alpha_list) / n_steps
            zl = sum(router_z_loss(r, batch["q_mask"]) for r in rlogits_list) / n_steps
            loss = ce + effective_lam_l1 * l1 + effective_lam_z * zl

            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(trainable, max_norm=cfg.grad_clip)
            optimizer.step()

            # Aux-loss-free LB bias update (DeepSeek-V3 §2.2.2 / Wang 2408.15664).
            # ``model.lb`` is None when ``lb_strategy="off"`` — no-op then. The
            # ``alpha`` here is the bias-applied ReLU output from the same forward
            # pass, so the update rule is self-consistent with the routing the
            # gradient just acted on.
            lb = getattr(model, "lb", None)
            if lb is not None:
                for a in alpha_list:  # one load observation per routing call (L for chain)
                    lb.step(a.detach(), batch["q_mask"])

            with torch.no_grad():
                # Mean K_active over real Q tokens (mask-weighted), averaged over steps.
                k_active_mean = float(
                    sum(_masked_token_mean(k.float(), batch["q_mask"]) for k in kact_list)
                    / n_steps
                )
                pred = logits.argmax(dim=-1)
                correct = (pred == batch["answer_idx"]).sum().item()

            b = batch["answer_idx"].size(0)
            sum_loss += float(loss.detach()) * b
            sum_ce += float(ce.detach()) * b
            sum_l1 += float(l1.detach()) * b
            sum_zl += float(zl.detach()) * b
            sum_k_active += k_active_mean * b
            sum_correct += correct
            n_samples += b

        scheduler.step()
        n_safe = max(n_samples, 1)
        train_metrics = {
            "epoch": int(epoch_idx),
            "loss": sum_loss / n_safe,
            "ce": sum_ce / n_safe,
            "l1": sum_l1 / n_safe,
            "z_loss": sum_zl / n_safe,
            "k_active_mean": sum_k_active / n_safe,
            "mc_acc": sum_correct / n_safe,
            "lam_l1": effective_lam_l1,
            "lam_l1_target": lam_l1,
            "warmup": in_warmup,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }

        if val_loader is not None:
            val_metrics = _evaluate(model, val_loader, torch_device)
            train_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            if cfg.use_best_val:
                if cfg.best_metric == "acc":
                    # maximise val accuracy (robust to overfit-inflated val CE)
                    if val_metrics["mc_acc"] > best_score:
                        best_score = val_metrics["mc_acc"]
                        best_val_loss = val_metrics["loss"]
                        best_state = copy.deepcopy(model.state_dict())
                elif val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    best_state = copy.deepcopy(model.state_dict())

        history.append(train_metrics)
        if cfg.log_every and (int(epoch_idx) % cfg.log_every == 0):
            msg_val = ""
            if val_loader is not None:
                msg_val = (
                    f" val_loss={train_metrics['val_loss']:.4f}"
                    f" val_acc={train_metrics['val_mc_acc']:.3f}"
                )
            warm_tag = " (warmup)" if in_warmup else ""
            print(
                f"[train] ep={int(epoch_idx):3d} loss={train_metrics['loss']:.4f} "
                f"ce={train_metrics['ce']:.4f} acc={train_metrics['mc_acc']:.3f} "
                f"k_active={train_metrics['k_active_mean']:.2f} "
                f"lam={effective_lam_l1:.5f}{warm_tag}{msg_val}"
            )

        # Dead-rescue: K_active collapsed below threshold → multiplicatively
        # release the L1 pressure faster than the standard 1/factor nudge.
        # Only fires post-warmup (during warmup λ is 0 anyway).
        if (
            not in_warmup
            and train_metrics["k_active_mean"] < cfg.dead_rescue_k_active
        ):
            old_lam = lam_l1
            lam_l1 = max(lam_l1 * cfg.dead_rescue_factor, 1e-6)
            print(
                f"[train] dead-rescue: K_active={train_metrics['k_active_mean']:.2f} "
                f"< {cfg.dead_rescue_k_active}; lam {old_lam:.5f} → {lam_l1:.5f}"
            )
        else:
            lam_l1 = update_l1_lambda(
                lam_l1, train_metrics["k_active_mean"], cfg.k_target
            )

    if cfg.use_best_val and best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_state = copy.deepcopy(model.state_dict())

    return {
        "history": history,
        "best_state": best_state,
        "best_val_loss": (best_val_loss if best_val_loss < float("inf") else None),
        "model": model,
    }


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Single-pass evaluation: loss (CE only) + MC accuracy + mean K_active."""
    model.eval()
    sum_loss = 0.0
    sum_correct = 0
    sum_k_active = 0.0
    n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits, _alpha_list, _rl_list, kact_list = _forward_normalized(model, batch)
        ce = mc_ce_loss(logits, batch["answer_idx"])
        pred = logits.argmax(dim=-1)
        b = batch["answer_idx"].size(0)
        sum_loss += float(ce) * b
        sum_correct += int((pred == batch["answer_idx"]).sum())
        sum_k_active += float(
            sum(_masked_token_mean(k.float(), batch["q_mask"]) for k in kact_list) / len(kact_list)
        ) * b
        n += b
    n_safe = max(n, 1)
    return {
        "loss": sum_loss / n_safe,
        "mc_acc": sum_correct / n_safe,
        "k_active_mean": sum_k_active / n_safe,
    }
