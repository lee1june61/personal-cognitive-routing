"""Causal operation-specialization battery (plan: causal functional verification).

The selectivity gate is confounded by the raw-Q ceiling (operation label is
surface-decodable from the frozen encoder, ceiling≈0.98), so routing-selectivity
cannot separate "experts discovered operations" from "router inherits the
stem-signal e5 already encodes". These interventions test the *causal* claim of
C0 — that experts implement operation-specific *computation* — which is downstream
of routing and not defeated by the encoder ceiling:

- ``lesion_specificity``: zero an operation's signature experts → does accuracy
  drop *selectively* on that operation's questions (diagonal-dominant)?
- ``operation_swap``: route operation X's questions through operation Y's expert
  signature → does the answer collapse off-diagonal (X|Y ≪ X|X)?

Both use ``Phase15MoE.forward(batch, alpha_override=...)``. Operation labels are
the (evaluation-only) regex axis; the *functional* drop — not the label — is the
evidence, so a pure surface-tagger (no operation-specific transform) yields null
matrices.
"""

from __future__ import annotations

import numpy as np
import torch

from .train import resolve_device


@torch.no_grad()
def _routed_alpha(model, q_tokens: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Per-token routed alpha (N, T_q, K) from the trained router."""
    return model.compute_alpha(q_tokens.to(device))["alpha"]


def _per_question_alpha(
    model, q_tokens: torch.Tensor, q_mask: torch.Tensor, device: torch.device
) -> np.ndarray:
    """Masked-mean of routed alpha over Q tokens → (N, K)."""
    alpha = _routed_alpha(model, q_tokens, device)  # (N, T, K)
    m = q_mask.to(device).to(alpha.dtype).unsqueeze(-1)  # (N, T, 1)
    pq = (alpha * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)  # (N, K)
    return pq.cpu().numpy()


def operation_signature(
    model, q_tokens, q_mask, op_labels, *, k_top: int = 4, device="cpu"
) -> tuple[dict, dict]:
    """Per-operation expert signature (mean per-question alpha) + top-``k_top`` experts.

    Returns ``(signatures, top_experts)`` where ``signatures[op]`` is a (K,) numpy
    array and ``top_experts[op]`` is a list of the op's most-activated expert ids.
    """
    dev = resolve_device(device)
    pq = _per_question_alpha(model.to(dev).eval(), q_tokens, q_mask, dev)  # (N, K)
    op_labels = np.asarray(op_labels)
    signatures, top_experts = {}, {}
    for op in sorted(set(op_labels.tolist())):
        sig = pq[op_labels == op].mean(axis=0)  # (K,)
        signatures[op] = sig
        top_experts[op] = np.argsort(-sig)[:k_top].tolist()
    return signatures, top_experts


@torch.no_grad()
def _accuracy_by_op(model, batch, op_labels, dev, *, alpha_override=None) -> dict:
    """MC accuracy per operation under an optional alpha_override."""
    b = {k: v.to(dev) for k, v in batch.items()}
    ov = alpha_override.to(dev) if alpha_override is not None else None
    pred = model(b, alpha_override=ov)["logits"].argmax(dim=-1).cpu().numpy()
    ans = batch["answer_idx"].cpu().numpy()
    op_labels = np.asarray(op_labels)
    return {
        op: float((pred[op_labels == op] == ans[op_labels == op]).mean())
        for op in sorted(set(op_labels.tolist()))
    }


def lesion_specificity(model, batch, op_labels, top_experts, *, device="cpu") -> dict:
    """Lesion each operation's signature experts (zero their alpha) and measure the
    per-operation accuracy drop.

    Returns ``{"baseline": {op: acc}, "drop": {op_X: {op_Y: baseline-lesioned}}}``.
    Diagonal-dominant ``drop`` (lesioning X's experts hurts X's questions most) =
    operation-specific function. Uniform drop = non-specific.
    """
    dev = resolve_device(device)
    model = model.to(dev).eval()
    baseline = _accuracy_by_op(model, batch, op_labels, dev)
    routed = _routed_alpha(model, batch["q_tokens"], dev)  # (N, T, K)
    drop = {}
    for op_x, experts in top_experts.items():
        ov = routed.clone()
        if experts:
            ov[..., list(experts)] = 0.0
        acc = _accuracy_by_op(model, batch, op_labels, dev, alpha_override=ov)
        drop[op_x] = {op_y: baseline[op_y] - acc[op_y] for op_y in baseline}
    return {"baseline": baseline, "drop": drop}


# ----- 1b chain-of-experts causal battery + S1 motif ---------------------------------


@torch.no_grad()
def _chain_alpha_steps(model, batch, dev) -> list:
    """Per-step routed alpha (list of (N, T, K)) from ``forward_chain``."""
    b = {k: v.to(dev) for k, v in batch.items()}
    return model.forward_chain(b)["alpha_steps"]


@torch.no_grad()
def _accuracy_by_op_chain(model, batch, op_labels, dev, *, alpha_override_steps=None) -> dict:
    """MC accuracy per operation under an optional per-step alpha override (1b)."""
    b = {k: v.to(dev) for k, v in batch.items()}
    ov = [a.to(dev) for a in alpha_override_steps] if alpha_override_steps is not None else None
    pred = model.forward_chain(b, alpha_override_steps=ov)["logits"].argmax(dim=-1).cpu().numpy()
    ans = batch["answer_idx"].cpu().numpy()
    ol = np.asarray(op_labels)
    return {
        op: float((pred[ol == op] == ans[ol == op]).mean())
        for op in sorted(set(ol.tolist()))
    }


def lesion_step_specificity(model, batch, op_labels, *, k_top: int = 4, device="cpu") -> dict:
    """1b causal compositional battery. For each chain step ℓ and operation X, zero
    X's signature experts *at step ℓ only* (other steps keep their routed alpha)
    and measure the per-operation accuracy drop.

    Returns ``{"baseline": {op: acc}, "drop_by_step": {ℓ: {op_X: {op_Y: drop}}},
    "n_steps": L}``. The 1b PASS reading (engine_1b): lesioning a *late* step must
    hurt *deep-hop* (3/4hop) questions strictly more than shallow (2hop) — a
    hop-depth-selective gradient. Uniform drop = non-compositional (FAIL).
    """
    dev = resolve_device(device)
    model = model.to(dev).eval()
    alpha_steps = _chain_alpha_steps(model, batch, dev)
    L = len(alpha_steps)
    m = batch["q_mask"].to(dev).to(alpha_steps[0].dtype).unsqueeze(-1)  # (N, T, 1)
    pq_steps = [(a * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0) for a in alpha_steps]  # (N,K)
    baseline = _accuracy_by_op_chain(model, batch, op_labels, dev)
    ol = np.asarray(op_labels)
    ops = sorted(set(ol.tolist()))

    drop_by_step: dict = {}
    for ell in range(L):
        pq = pq_steps[ell].cpu().numpy()  # (N, K)
        step_drop = {}
        for op_x in ops:
            sig = pq[ol == op_x].mean(axis=0)  # (K,) mean activation for op-X at step ℓ
            experts = np.argsort(-sig)[:k_top].tolist()
            ov = [a.clone() for a in alpha_steps]
            if experts:
                ov[ell][..., experts] = 0.0
            acc = _accuracy_by_op_chain(model, batch, op_labels, dev, alpha_override_steps=ov)
            step_drop[op_x] = {op_y: baseline[op_y] - acc[op_y] for op_y in baseline}
        drop_by_step[ell] = step_drop
    return {"baseline": baseline, "drop_by_step": drop_by_step, "n_steps": L}


def chain_motif_codes(model, batch, *, device="cpu") -> np.ndarray:
    """Per-question S1 motif code = concat over steps of the masked-mean active
    alpha (the active sub-path signature), shape (N, L*K). Feed with the
    decomposition-structure labels to ``eval.operation_consistency`` (topic-
    controlled) for the S1 motif-consistency criterion."""
    dev = resolve_device(device)
    model = model.to(dev).eval()
    alpha_steps = _chain_alpha_steps(model, batch, dev)
    m = batch["q_mask"].to(dev).to(alpha_steps[0].dtype).unsqueeze(-1)
    per_step = [
        ((a * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)).cpu().numpy() for a in alpha_steps
    ]
    return np.concatenate(per_step, axis=1)


def operation_swap(model, batch, op_labels, signatures, *, device="cpu") -> dict:
    """Route *all* questions through each operation's fixed signature and read the
    accuracy per source operation.

    Returns ``acc[op_X][op_Y]`` = accuracy of op-X questions when forced to route
    through op-Y's signature. Diagonal-dominant (X|X ≫ X|Y) = experts implement
    operation-specific transforms (causal positive); X|Y ≈ X|X = interchangeable
    (surface tagging). ``N`` forwards (one per signature).
    """
    dev = resolve_device(device)
    model = model.to(dev).eval()
    n, t = batch["q_tokens"].shape[0], batch["q_tokens"].shape[1]
    ops = sorted(signatures)
    acc = {op_x: {} for op_x in ops}
    for op_y in ops:
        sig_y = torch.as_tensor(np.asarray(signatures[op_y]), dtype=torch.float32)  # (K,)
        ov = sig_y.view(1, 1, -1).expand(n, t, -1).contiguous()
        acc_y = _accuracy_by_op(model, batch, op_labels, dev, alpha_override=ov)
        for op_x in ops:
            acc[op_x][op_y] = acc_y[op_x]
    return acc
