"""Phase 1 baselines (revision 4, embedding-only, no LLM).

- B0 (`generic_baseline.GenericBaseline`): frozen encoder + FactDecoder.
  No MoE, no expert routing — tests "MoE-KG-cycle adds value over plain encoder+decoder".

- B1 (`standard_moe_baseline.StandardMoEBaseline`): frozen encoder + Switch-style
  top-1 hard routing MoE + FactDecoder. Tests "MoE alone vs MoE-KG-cycle (sparsegen +
  KG bottleneck + cycle)".

Phase 1 risky test (RESEARCH_PLAN §3.4 revision 4):
  Phase 1 (our) > B0 + Phase 1 > B1  =  MoE-KG-cycle generic value confirmed.
"""

from .generic_baseline import GenericBaseline
from .standard_moe_baseline import StandardMoEBaseline, StandardMoE

__all__ = ["GenericBaseline", "StandardMoEBaseline", "StandardMoE"]
