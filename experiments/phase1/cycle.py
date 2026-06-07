"""Phase 1 embedding-level cycle (LLM-free, paradigm intent: latent only).

Architecture (RESEARCH_PLAN_2026-05-19 §2.1 revision 4):

    text (B, list[str])
        ↓ frozen encoder (BGE-large-en-v1.5, 1024d)
    fact_emb (B, 1024)
        ↓ Phase1MoE (use_user toggle for Phase 1 / Phase 2)
    sub_kg (B, 1024)   ★ latent KG (distributional, emergent relation-type)
        ↓ FactDecoder (tight 64d bottleneck)
    fact_emb_recon (B, 1024)
        ↓
    L_recon = 1 − cos(fact_emb, fact_emb_recon)
    L_lb    = λ_lb · K · Σ F_k · P_k                (Switch / LD-MoLE Eq. 9)
    L_ortho = λ_ortho · ‖M^T M − I‖_F²              (SIMoE, optional)

No LLM in cycle. KG = sub_kg latent + routed_alpha distribution. Wikidata bracket
text rendering deferred to Phase 3 future work (post-hoc render via external LLM
for HyperRED external eval if needed).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .model import (
    Phase1MoE,
    FactDecoder,
    FrozenEncoderHost,
    recon_loss,
    routing_load_balance,
    routing_orthogonality,
)

# FrozenEncoder is now the shared superset in core (it absorbed phase1's pooled
# forward / encode_batched / no-prefix encode_tokens AND phase1_5's prefix kwarg +
# encode_pooled paths). phase1 callers construct it with the BGE model name explicitly,
# so the core default model name (e5) is never reached here. `prefix` defaults to ""
# (no prefix) → identical to phase1's original encode_tokens behaviour. Re-exported so
# `phase1.cycle.FrozenEncoder` (used by baselines, data.py, tests) keeps resolving here.
from core.encoders import FrozenEncoder


# ----- Cycle config ------------------------------------------------------------------

@dataclass
class CycleConfig:
    lambda_recon: float = 1.0       # cycle reconstruction primary weight
    lambda_lb: float = 0.1           # Switch / LD-MoLE Eq. 9 load balance (Switch default α=1.0,
                                    # we use 0.1 — recon_loss scale ~1 makes 1.0 too aggressive)
    lambda_ortho: float = 0.0        # SIMoE orthogonality (off by default, ablation 0.05)
    cos_loss: bool = True            # True = 1 − cosine, False = MSE
    d_bottleneck: int = 64           # FactDecoder tight bottleneck


# ----- Phase 1 cycle wrapper ---------------------------------------------------------

class Phase1Cycle(FrozenEncoderHost):
    """MoE + FactDecoder + cycle loss — embedding-only, no LLM.

    Phase 1 / Phase 2 분기: `use_user` flag. `load_phase1_weights(path)` helper for
    Phase 1 → Phase 2 weight transfer (use_user=False checkpoint into use_user=True
    model, leaving user_logits / lambda_mlp at fresh init).
    """

    def __init__(
        self,
        n_users: int,
        encoder_name: str = "BAAI/bge-large-en-v1.5",
        k_routed: int = 16,
        k_shared: int = 4,
        d_hidden: int = 2048,
        config: CycleConfig | None = None,
        use_user: bool = False,
    ):
        super().__init__()
        cfg = config or CycleConfig()
        self.config = cfg
        self.use_user = use_user

        self.encoder = FrozenEncoder(encoder_name)
        d_model = self.encoder.d_model

        self.moe = Phase1MoE(
            n_users=n_users,
            d_model=d_model,
            k_routed=k_routed,
            k_shared=k_shared,
            d_hidden=d_hidden,
            use_user=use_user,
        )
        self.decoder = FactDecoder(d_model=d_model, d_bottleneck=cfg.d_bottleneck)

    @torch.no_grad()
    def encode(self, texts: list[str], max_length: int = 256) -> torch.Tensor:
        """Convenience: text → fact_emb (frozen encoder)."""
        return self.encoder(texts, max_length=max_length)

    def forward(
        self,
        fact_emb: torch.Tensor,
        user_id: torch.Tensor | None = None,
        return_expert_outs: bool = False,
    ) -> dict:
        """fact_emb: (B, d_model). user_id: (B,) long (Phase 2) or None (Phase 1)."""
        moe_out = self.moe(fact_emb, user_id=user_id, return_expert_outs=return_expert_outs)
        recon = self.decoder(moe_out["sub_kg"])
        return {**moe_out, "recon": recon}

    def cycle_loss(
        self,
        fact_emb: torch.Tensor,
        user_id: torch.Tensor | None = None,
    ) -> dict:
        """Forward + multi-term loss aggregation."""
        out = self.forward(fact_emb, user_id=user_id)
        cfg = self.config

        recon = recon_loss(fact_emb, out["recon"], cos_loss=cfg.cos_loss)
        lb_loss = routing_load_balance(out["routed_alpha"], lambda_lb=cfg.lambda_lb)
        ortho_loss = routing_orthogonality(out["routed_alpha"], lambda_ortho=cfg.lambda_ortho)
        total = cfg.lambda_recon * recon + lb_loss + ortho_loss

        return {
            "loss": total,
            "loss_recon": recon.detach(),
            "loss_lb": lb_loss.detach(),
            "loss_ortho": ortho_loss.detach(),
            "routed_alpha": out["routed_alpha"],
            "lam": out["lam"],
            "k_active": out["k_active"],
        }

    def load_phase1_weights(self, path: str | Path, map_location=None) -> dict:
        """Load a Phase 1 (use_user=False) checkpoint into a Phase 2 (use_user=True) model.

        Copies expert / fact_gate / shared_experts / decoder weights. Leaves user_logits /
        lambda_mlp at fresh init for Stage B2 (Phase 2 finetune from Phase 1 cycle pretrain)
        per RESEARCH_PLAN §4.

        Allow-list (which keys are *expected* to mismatch) is derived dynamically by
        comparing a fresh Phase 1 vs Phase 2 state_dict — survives module renames.
        """
        if not self.use_user:
            raise RuntimeError(
                "load_phase1_weights targets a Phase 2 (use_user=True) model. "
                "Phase 1 → Phase 1 = standard load_state_dict."
            )
        state = torch.load(path, map_location=map_location)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k: v for k, v in state.items() if not k.startswith("encoder.")}
        report = self.load_state_dict(state, strict=False)

        allowed_missing, allowed_unexpected = self._phase_diff_keys()
        # Encoder keys are frozen and stripped on save — always tolerate missing.
        allowed_missing |= {k for k in report.missing_keys if k.startswith("encoder.")}

        bad_missing = [k for k in report.missing_keys if k not in allowed_missing]
        bad_unexpected = [k for k in report.unexpected_keys if k not in allowed_unexpected]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                f"Phase 1 → Phase 2 load mismatch beyond expected ablation keys.\n"
                f"  unexpected_missing  = {bad_missing}\n"
                f"  unexpected_extra    = {bad_unexpected}"
            )
        return {
            "missing": list(report.missing_keys),
            "unexpected": list(report.unexpected_keys),
        }

    def _phase_diff_keys(self) -> tuple[set[str], set[str]]:
        """Derive the Phase 1↔Phase 2 state_dict key delta from MoE constructor args."""
        from .model import Phase1MoE
        moe = self.moe
        p2 = Phase1MoE(
            n_users=moe.user_logits.num_embeddings, d_model=moe.d_model,
            k_routed=moe.k_routed, k_shared=moe.k_shared,
            d_hidden=moe.routed_experts[0].fc1.out_features,
            use_user=True,
        ).state_dict().keys()
        p1 = Phase1MoE(
            n_users=1, d_model=moe.d_model,
            k_routed=moe.k_routed, k_shared=moe.k_shared,
            d_hidden=moe.routed_experts[0].fc1.out_features,
            use_user=False,
        ).state_dict().keys()
        only_p2 = {f"moe.{k}" for k in p2 - p1}     # present in Phase 2, missing in Phase 1 ckpt
        only_p1 = {f"moe.{k}" for k in p1 - p2}     # present in Phase 1 ckpt, unexpected in Phase 2
        return only_p2, only_p1


# ----- Training utilities ------------------------------------------------------------

def train_epoch(
    model: Phase1Cycle,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    progress=None,
    grad_clip: float = 1.0,
) -> dict:
    """Train one epoch on a (fact_emb, user_id) DataLoader.

    Loader yields (fact_emb (B, d), user_id (B,)) — user_id ignored if model.use_user=False.

    `phase1.train.train_phase1` runs an inline loop with sanity-gate logic; this is the
    notebook-friendly standalone variant. Both apply grad_clip identically (default 1.0).
    """
    model.train()
    n = 0
    sum_loss = 0.0
    sum_recon = 0.0
    sum_lb = 0.0
    sum_ortho = 0.0
    sum_active = 0.0
    sum_lam = 0.0
    trainable = [p for p in model.parameters() if p.requires_grad]

    for batch in loader:
        if len(batch) == 2:
            fact_emb, user_id = batch
        else:
            fact_emb = batch[0]
            user_id = None

        fact_emb = fact_emb.to(device)
        user_id = user_id.to(device) if user_id is not None else None

        losses = model.cycle_loss(
            fact_emb,
            user_id=user_id if model.use_user else None,
        )

        optimizer.zero_grad()
        losses["loss"].backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=grad_clip)
        optimizer.step()

        with torch.no_grad():
            active_frac = (losses["routed_alpha"] > 1e-6).float().mean()
            mean_lam = float(losses["lam"].mean())

        b = fact_emb.size(0)
        sum_loss += float(losses["loss"]) * b
        sum_recon += float(losses["loss_recon"]) * b
        sum_lb += float(losses["loss_lb"]) * b
        sum_ortho += float(losses["loss_ortho"]) * b
        sum_active += float(active_frac) * b
        sum_lam += mean_lam * b
        n += b

        if progress is not None:
            progress.update(1)

    return {
        "loss": sum_loss / max(n, 1),
        "loss_recon": sum_recon / max(n, 1),
        "loss_lb": sum_lb / max(n, 1),
        "loss_ortho": sum_ortho / max(n, 1),
        "active_frac": sum_active / max(n, 1),
        "lambda_mean": sum_lam / max(n, 1),
    }


@torch.no_grad()
def collect_activations(
    model: Phase1Cycle,
    loader,
    device: torch.device,
    return_expert_outs: bool = False,
) -> dict:
    """Gather sub_kg + routed_alpha (+ optional routed_outs) over a loader.

    `return_expert_outs=True` adds per-expert pre-mixture outputs to the result
    (~K·d_model floats per sample — ~6.5 GB CPU RAM for 100k samples × K=16 × d=1024).
    Default off — only enable for D1 disentanglement post-hoc analysis.
    """
    model.eval()
    all_sub_kg = []
    all_alpha = []
    all_user = []
    all_routed_outs: list = [] if return_expert_outs else []

    for batch in loader:
        if len(batch) == 2:
            fact_emb, user_id = batch
        else:
            fact_emb = batch[0]
            user_id = None

        fact_emb = fact_emb.to(device, non_blocking=True)
        user_id_dev = user_id.to(device, non_blocking=True) if user_id is not None else None

        out = model(
            fact_emb,
            user_id=user_id_dev if model.use_user else None,
            return_expert_outs=return_expert_outs,
        )
        all_sub_kg.append(out["sub_kg"].cpu())
        all_alpha.append(out["routed_alpha"].cpu())
        if return_expert_outs:
            all_routed_outs.append(out["routed_outs"].cpu())
        if user_id is not None:
            all_user.append(user_id.cpu())

    gu = model.moe.gu_table()
    result = {
        "sub_kg": torch.cat(all_sub_kg, dim=0),
        "routed_alpha": torch.cat(all_alpha, dim=0),
        "user_id": torch.cat(all_user, dim=0) if all_user else None,
        "gu_table": gu.cpu() if gu is not None else None,
    }
    if return_expert_outs:
        result["routed_outs"] = torch.cat(all_routed_outs, dim=0)
    return result
