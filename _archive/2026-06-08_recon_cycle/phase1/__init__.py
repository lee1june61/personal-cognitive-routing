"""Phase 1 — Emergent Distributional KG via Unsupervised Expert Specialization.

Revision 4 (2026-05-21): embedding-only cycle, no LLM. See README.md for full overview.
"""

from . import _hf_setup  # noqa: F401  — auto-loads HF_TOKEN from Colab Secret / env (optional)
from .model import (
    Phase1MoE,
    ExpertFFN,
    FactDecoder,
    sparsegen,
    routing_load_balance,
    routing_orthogonality,
)
from .cycle import (
    Phase1Cycle,
    FrozenEncoder,
    CycleConfig,
    train_epoch,
    collect_activations,
)

