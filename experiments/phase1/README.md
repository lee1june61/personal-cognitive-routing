# Phase 1 — closed (informative negative result, evidence preserved)

> **⚠️ DEPRECATED 2026-05-28** — Phase 1 = closed. 활성 작업 폴더 = **`../phase1_5/`**.
> 현 single source = `../../RESEARCH_PLAN_2026-05-28_phase1_5.md` + `../../CONTEXT.md` + `../../LIT_REVIEW_MASTER_2026-05-28_phase1_5.md`. Paradigm = `../../draft/KG_Project_Vision.md` §6.1 (C0-C3 + C3 sharpening).
>
> Phase 1 의 전체 narrative (5-19 plan → 5-21 revision 4 (latent-only, no LLM) → 5-22~5-24 5-run ablation → 5-27 operation-cycle pivot → 5-28 Engine-A Stage1 F3 confirmed → Phase 1.5 reframe) = **`../../ARCHIVE.md §23`** (전부 흡수). 본 README 의 §8 evidence 만 아래 보존.

---

## §8 (preserved). Phase 1 ablation final — Phase 1.5 의 직접 motivation

| run | epoch | K_active | recon_cos | SimBench | hyperparam | 의미 |
|---|---|---|---|---|---|---|
| v3_minimal | 30 | 13.88 | 0.8864 | **0.7120** | λ_lb=0.1, λ_ortho=0, 4-source | dense routing, best SimBench (narrow corpus 우연 align) |
| v4_diverse | 30 | 6.56 | 0.8786 | 0.7029 | + diverse 7-source | source-level cluster (narrative/discussion/finance) |
| v5_arch | 30 | **1.00** | 0.8788 | 0.7067 | λ_lb=0.05, λ_ortho=0.05 | **K=1 hard 1-hot, paradigm degenerate** |
| **v6_long** | 79 | 5.66 | 0.8789 | **0.6901** | v4 setting + epoch 79 (plateau at 15) | **paradigm-faithful + sharp cluster** (narrative 4-way + programming 2-way + reddit 다양), lowest SimBench |
| b0_v0 (B0) | 30 | — | 0.8687 | 0.7082 | no MoE | baseline |

### 핵심 evidence

- ✓ **recon +1pp vs B0** confirmed (architectural superiority on cycle reconstruction).
- ✗ **SimBench parity / inverse trade-off** — sharper cluster → lower SimBench. **Pretext-downstream geometry mismatch** (Loaiza-Ganem 2020).
- ✓ **v6_long fine-grained sub-cluster** emergence at *source/format level*, NOT operation level (Zoph ST-MoE token-type pattern).
- ✓ Stage 1 plateau at epoch 15 (architecture capacity ceiling).
- ✗ B1 (Switch top-1) sanity fail — expert collapse.
- ✗ **Engine-A Stage1 (5-28)** — operation adj +0.18 < topic adj 0.58~0.97 모든 config (raw e5 ceiling 에서도) → **F3 구조적 = recon objective 가 content/topic 보상**.

**결론** = recon-primary 가 operation 목표에 self-defeating → 5-28 grill 에서 paradigm pivot → Phase 1.5 (answer-prediction + info-bottleneck + emergent NMN + 단일 logic 도메인 + K=128).

## Hardened 코드 자산 (phase1_5/ 에서 재사용)

`FrozenEncoder` (per-token encode, fp16+int8 mask, OOM-safe batched) · `selectivity_report` + 4 control (chance-normalized) · adaptive L1 controller (`update_l1_lambda`) · Herfindahl load-balance · `agg=meanmax` · code review 5종 hardening — `ENGINE_A_DESIGN.md §9` + `phase1_5/README §6` 매핑.

---

*Phase 1 의 architectural spec / Stage 1/2 plan / 데이터셋 / 즉시 다음 작업 / 진입점 등은 모두 superseded. 필요 시 git history 또는 `../../ARCHIVE.md §23`.*
