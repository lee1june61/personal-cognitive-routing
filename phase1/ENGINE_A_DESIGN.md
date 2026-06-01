# Engine-A Design (Phase 1 recon-cycle, 2026-05-27) · ⚠️ SUPERSEDED 2026-05-28

> **DEPRECATED**: Phase 1 Engine-A spec (SMoE-VAE식 decode-side experts · K=16 · ReMoE · per-token · sequence-level readout · recon-primary). Stage1 결과 F3 collapse 구조적 (operation adj +0.18 < topic 0.58~0.97). 5-28 paradigm pivot → Phase 1.5.
>
> 현 spec = `../phase1_5/README.md` (answer-prediction + info-bottleneck + emergent NMN + K=128 + cross-attn + LogiQA/ReClor). 전체 narrative = `../../ARCHIVE.md §23`.
>
> 본 문서의 architecture / data / loss / hyperparam / trajectory / go-no-go spec (§1-§8) 은 모두 superseded. **§9 code review hardening 만 보존** — phase1_5/ 에서 재사용할 technical asset.

---

## §9 (preserved). Code review hardening — 2026-05-27 post-xhigh review

high-effort 코드리뷰(5 angle) 에서 real-scale/real-data 버그 다수 발견·수정 (tiny CPU 테스트는 통과했지만 real run 에서 터질 종류). 진단 사다리 추가 후 **120 passed**.

- **OOM 수정**: `train_opcycle`/`compute_codes` 가 전체 (N,T,1024) 를 float32 로 GPU 에 통째 올리던 것(~21GB) → **CPU native dtype 보관 + per-batch device 이동**. `forward` 도 (B,T,K,d) expert stack 제거 → running-sum (peak ↓).
- **load_balance 재설계 (B1)**: `softmax(logits)` 가 아니라 **실제 ReLU 게이트 α** 기반 importance/Herfindahl `K·Σ P_i²` (uniform=1, collapse=K). softmax 형은 죽은 expert(logit≤0)에도 mass 줘서 collapse 못 잡음. top-1 Switch 형은 multi-active 라 uniform·collapse 둘 다 K 로 degenerate.
- **adaptive L1 (B2)**: `update_l1_lambda` 컨트롤러 — k_target 으로 λ_l1 자가조절. 고정 λ 는 ReMoE "self-adjusting sparsity" 와 불일치·brittle. phase1_5 k_target=4.
- **sequence_code agg (B3)**: mean 외 **max/meanmax**. mean-pool 이 소수 salient-token operation 신호 희석 → `agg='meanmax'` 채택.
- **selectivity 강건화 (A2/B4)**: stratify singleton-class crash → safe fallback + <2 class ValueError; degenerate control 자동 skip + warning; **chance 정규화** `(acc−chance)/(1−chance)` 로 cardinality 다른 control 공정 비교; topic+geometry 없으면 PASS 보류.
- **empty-probe (A4)**: probe 로드 실패 시 명확한 ValueError, `compute_codes` (0,width) 반환.
- **잔존 (의도)**: GPU 비결정성 — margin 근처 verdict 재현성 한계. random control single-permutation variance. 필요 시 seed-sweep.
