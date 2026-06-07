"""ReMoE router — ReLU gate, per-expert independent, no simplex normalisation.

Superset of the two copies (phase1/model_opcycle.py ``ReMoERouter`` [subset: ``(d_z, k)``,
bias init zeros, ``forward(z)``] and phase1_5/model.py ``ReMoERouter`` [superset: adds
``bias_init``, ``routing``, ``k_active``, ``external_bias``, top-k mode]).

Backward-compat for phase1: construct with ``bias_init=0.0`` and call ``forward(z)`` (no
external_bias). With ``routing="relu_l1"`` + ``external_bias=None`` + ``bias_init=0.0`` the
forward returns ``(relu(logits), logits)`` — identical to phase1's old behaviour (the
bias buffer is all-zeros, so ``logits + 0 == logits``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReMoERouter(nn.Module):
    """ReMoE routing (Wang et al. 2024): replace TopK/softmax with a per-expert ReLU
    gate. Each expert i gets an independent gate g_i = ReLU(W z + b)_i — non-negative,
    exactly zero when the logit is ≤ 0, and *not* normalised to a simplex. Sparsity
    (K_active) is emergent and controlled at train-time by an adaptive L1 penalty on the
    gates (``remoe_l1_loss``), so it self-adjusts rather than being fixed by TopK.

    Gate bias is initialised to ``bias_init``:
      - ``bias_init=0.0`` (phase1 spec) → initial logits ~ N(0, 0.01); ReLU kills ~50%
        of tokens at epoch 0. This is the phase1 OpCycleMoE behaviour.
      - ``bias_init=0.5`` (phase1_5 default) → keeps initial logits in the active
        half-plane so dead gates can recover (the K_active=0 trap).

    Returns ``(alpha, logits)``: alpha = ReLU(biased logits) are the mixing gates; logits
    are kept raw (bias-applied) so the ST-MoE router z-loss and the load-balance loss can
    read them.
    """

    def __init__(
        self,
        d_z: int = 256,
        k: int = 128,
        bias_init: float = 0.5,
        routing: str = "relu_l1",
        k_active: int = 4,
    ):
        super().__init__()
        if routing not in ("relu_l1", "topk"):
            raise ValueError(f"routing must be 'relu_l1' or 'topk'; got {routing!r}")
        self.k = k
        self.routing = routing
        self.k_active = min(int(k_active), k)
        self.gate = nn.Linear(d_z, k)
        nn.init.normal_(self.gate.weight, std=0.01)
        nn.init.constant_(self.gate.bias, bias_init)

    def forward(
        self,
        z: torch.Tensor,
        external_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute ``alpha`` and ``router_logits`` from latent ``z``.

        ``external_bias`` is the aux-loss-free LB bias buffer (DeepSeek-V3 /
        Wang 2408.15664).

        - ``routing="relu_l1"`` (default): ``alpha = relu(gate(z) + bias)``;
          sparsity comes from the train-time adaptive L1. bias is added before
          ReLU so dead gates recover via positive drift. Returned logits are
          bias-applied (z-loss / metrics consistent).
        - ``routing="topk"`` (Phase 3 diversity): select the top-``k_active``
          experts by ``gate(z) + bias`` (bias steers *selection* only, its
          designed aux-free use), weight them by softmax of the *original*
          logits, zero the rest → K_active ≡ k_active by construction (no
          collapse). Returned logits are the original (un-biased) gate logits.
        """
        logits = self.gate(z)  # (B, T, K)
        biased = logits if external_bias is None else logits + external_bias
        if self.routing == "topk":
            # bias steers selection only; weight selected by softmax of original logits.
            topk_idx = biased.topk(self.k_active, dim=-1).indices  # (B, T, k_active)
            sel_w = F.softmax(logits.gather(-1, topk_idx), dim=-1)
            alpha = torch.zeros_like(logits).scatter(-1, topk_idx, sel_w)
            return alpha, logits
        # relu_l1 (default): bias added before ReLU; biased logits returned so
        # z-loss / observability metrics are bias-consistent.
        return F.relu(biased), biased
