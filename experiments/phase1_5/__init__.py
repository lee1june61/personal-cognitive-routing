"""Phase 1.5 — Operation-axis architectural pivot (2026-05-28).

answer-prediction objective + info-bottleneck-3 + emergent NMN target + 단일 logic 도메인
+ K=128 sparse routing. See README.md and `research/PHASE1_5_LIT_POSITIONING_PAPER.md`
for full lit-grounded warrant per component.

Phase 1 (recon-cycle) = closed evidence; phase1/ is NOT modified by this package.
Where helpful, hardened modules from phase1 are *copied* into this package
(eval primitives), not imported, to keep phase1_5 self-contained on Colab.
"""

from . import _hf_setup  # noqa: F401  — auto-loads HF_TOKEN from Colab Secret / env (optional)
