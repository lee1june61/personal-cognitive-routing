# Phase 1.5 — Engine-A operation router (1a) · 2026-05-28

> **★ 2026-05-31 — 1b corpus pivot.** 1a(logic-MC) = weak-ceiling negative; 진단 = LogiQA/ReClor 가 단일-op·stem-announced(composition substrate 부재). → **MuSiQue(multi-hop→4지 MC, intermediate-answer hard distractor) = 주 compositional corpus, logic-MC = 단일-op control arm.** 1b = `model.Phase15MoE.forward_chain`(L-step chain-of-experts; no-bypass = 출력 누적기 Σmod_ℓ 가 z_q 제외; per-step `_chain_modulations`). corpus selector = `MCCorpusConfig.corpus ∈ {musique, logic_mc, both}`, 변환 = `data_musique.py`. 구현 M1/M2/M4 완료(181 tests), M3/M5/M6 = Colab. 단일 출처 = `../../CONTEXT.md §Corpus` + ADR `../../docs/adr/0003` + memory `project_phase15_corpus_pivot_2026_05_31` + plan `~/.claude/plans/hashed-orbiting-cerf.md`. 아래 §1~§7 은 1a 시점 기록(historical).

> **Status**: skeleton (디렉토리 신설, 결정 박음). 구현 = 다음 session via `tdd` skill.
> **Spec source**: `../../RESEARCH_PLAN_2026-05-28_phase1_5.md` (full plan) · `../../CONTEXT.md` (domain glossary) · `../../LIT_REVIEW_MASTER_2026-05-28_phase1_5.md` (lit positioning).
> 본 README = 코드 디렉토리 진입점. Colab BASE = `/content/drive/MyDrive/sideproject/phase1_5`.

## 1. 1a 본질 (현 step)

flat-α operation router. K_routed=128 fine-grained experts, K_active=4. graph 없음 (1b 에서 chain 추가). 목적 = **"fine-grained operation primitive 가 *애초에* emergent 하나?"** go/no-go.

> 명명: 1a 에선 "operation router" — "operation-KG" 호칭은 1b chain-router 부터 (활성 path = literal subgraph 일 때 실체).

## 2. 아키텍처 (1a)

```
Q (text)
  ↓ frozen e5-large-v2
h(Q) per-token (B, T_q, 1024)
  ↓ SharedEncoderHead (1024 → 256)
z(Q) (B, T_q, 256)
  ↓ masked mean pool (sequence-level)
z_q (B, 256)
  ↓ ReMoERouter (K=128, ReLU + adaptive L1 → K_active≈4)
α (B, 128)  ◀── flat operation activation = 1a 의 "KG (aspirational)"
  ↓ KG construction:
kg_vec = Σ_k α_k · op_token_k   ; op_token_k = learnable embedding (K=128, dim=256)
  ↓
P (passage text)
  ↓ frozen e5-large-v2
h(P) per-token (B, T_p, 1024)
  ↓ Linear 1024 → 256
P_repr (B, T_p, 256)
  ↓ KGHypernetModulation: h_P=Attn(Q=kg_vec, K/V=P_repr); out=U(s(kg_vec)⊙V(h_P))
    (no-bypass: kg_vec=0 ⟹ out=0; bias-free·no-residual. cross-attn/FiLM 폐기 = ADR 0001)
predicted_answer_repr (B, 256)
  ↓ candidate scoring (4 MC options 각 임베딩과 cosine 또는 inner-product)
logits (B, 4)
  ↓ Cross-entropy (정답 index)
loss
```

**bottleneck 보장 (info-bottleneck 3중 강제)**:
1. **Q-only encoding**: P 는 인코더 안 보임. ✓
2. **P side-channel**: P 는 decoder cross-attn 의 key/value 로만. ✓
3. **KG modulates** (no-bypass): ⚠ 원래 "cross-attn query → KG 없으면 attn 정의 불가" 주장은 **경험적으로 거짓** (zero-query attn=uniform→passage 누출; rev2 K_active=0 인데 acc 0.35). 수정 불변식 = `kg_vec=0 ⟹ modulation 출력=0` (KGHypernetModulation, bias-free·no-residual). 상세 = ADR 0001 + CONTEXT.md #3.

## 3. Hyperparameters (closed 2026-05-28)

| 항목 | 값 | 근거 |
|---|---|---|
| encoder | **e5-large-v2** (frozen) | Stage1 raw ceiling op adj 0.60 vs BGE |
| K_routed | **128** | sub-sub-skill emergence 기대, 단일도메인 (사용자) |
| K_active target | **4** | DeepSeek-V3 ratio 8/256 = 3.1% 매칭 |
| d_z | 256 | Phase 1 default |
| d_hidden (expert internal) | 512 | Phase 1 default |
| op_token dim | 256 | = d_z, alignment |
| router | ReMoE (ReLU + adaptive L1, k_target=4) | Phase 1 v6 검증 |
| modulation | **kg_hypernet** (gated low-rank hypernet, no-bypass; 1a default, L=3 stack 1b) | 5-29 grill / ADR 0001 (cross-attn·FiLM=Gap B baseline) |
| rank r (kg_hypernet) | **d_z // 4 = 64** | low-rank KG-생성 변환 |
| MC candidates | 4 | LogiQA/ReClor 구조 |
| corpus | **LogiQA 2.0 (TASLP 2023) + ReClor** | hard distractor, 단일 logic 도메인 |
| optimizer | AdamW + cosine | Phase 1 |
| seq cap T_q | 128 (question, 짧음) | LogiQA 통계 확인 후 조정 |
| seq cap T_p | 256 (passage, 더 김) | LogiQA premise 길이 |

## 4. PASS bar (1a gate)

LogiQA + ReClor test set 에서 selectivity (Hewitt-Liang style):
- **target** = reasoning-type label (necessary / sufficient / disjunctive / etc. — corpus annotation)
- **acc(operation)** > 4 control 전부 :
  1. random label
  2. topic (passage cluster)
  3. token-type (length bucket / POS)
  4. geometry (code shuffle baseline)

**절대 bar 유지** — ceiling-relative 재정의 X (5-28 grill 에서 forcing 으로 자체 철회). FAIL 시 F1/F2/F3 framework 으로 진단 (Phase 1 에서 hardened).

**1b 진입 조건** = 1a PASS + S1 (motif 공유) 측정 가능 형태로 코드 준비.

## 5. 디렉토리 구조 (skeleton)

```
phase1_5/
├── README.md               # 본 문서 — 진입점
├── data.py                 # LogiQA 2.0 + ReClor MC loader · e5 caching
├── model.py                # Phase15MoE (1a flat-α + cross-attn decoder + MC scorer)
├── eval.py                 # operation selectivity gate · 4 control · S1 placeholder
├── train.py                # MC-contrastive 학습 loop · adaptive L1 controller
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_eval.py
│   └── test_train.py
└── notebooks/
    └── 01_engine_a_1a.ipynb   # Colab driver (BASE = /content/drive/MyDrive/sideproject)
```

## 6. 재사용 (phase1/ 에서 가져옴)

phase1/ 의 hardened 코드 (120 passed tests, code review xhigh) 에서 다음은 그대로 또는 살짝 어댑트:

| phase1/ | phase1_5/ | 변경 |
|---|---|---|
| `eval_opcycle.py:selectivity_report` + `probe_accuracy` + `geometry_baseline` + `sequence_code` | `eval.py` | 그대로 재사용. probe_accuracy 의 codes 인자만 (B,K) → (B,K) flat α 통일. |
| `data.py:FrozenEncoder.encode_tokens` + `encode_or_load_tokens` (per-token, fp16 + int8 mask, batched) | `data.py` | encoder name 만 e5 로. caching key 자동 갱신. |
| `engine_a.py:train_opcycle` 의 OOM-안전 batching (CPU native dtype 보관 + per-batch device 이동) | `train.py` | answer-pred loss 로 교체하되 batching 구조 유지. |
| `engine_a.py:update_l1_lambda` (k_target adaptive controller) | `train.py` | 그대로, k_target=4. |
| `ENGINE_A_DESIGN.md §9` 의 5 종 hardening (load-balance Herfindahl, masked recon, OOM, agg=meanmax, selectivity 강건화) | (적용 가능 시 유지) | recon→ans 로 일부 무관 |
| `model_opcycle.py:OpCycleMoE.forward` (per-token expert running-sum, peak↓) | `model.py` | 1a 는 sequence-level pooled 이라 더 단순. expert stack 패턴은 1b 에서 재사용. |

## 7. 다음 작업 (다음 session, tdd skill)

1. **`data.py`** — LogiQA 2.0 (HF `datasets`) + ReClor loader, MC 4지 normalize, e5 caching (`FrozenEncoder` adapt). corpus_key 캐시 키. test: shape, label 분포, MC normalize.
2. **`model.py`** — Phase15MoE (위 §2 spec). tdd red→green. SharedHead + ReMoERouter(K=128) + op_token + CrossAttn + MC scorer. test: shapes, α exact-zero (ReLU), backward.
3. **`train.py`** — MC CE loss (cross-entropy over 4 candidates), AdamW+cosine, adaptive L1 controller (k_target=4), per-batch device 이동. test: tiny CPU end-to-end.
4. **`eval.py`** — selectivity_report adapter (LogiQA reasoning-type label loader, 4 control 재사용). test: tiny synthetic.
5. **`notebooks/01_engine_a_1a.ipynb`** — Colab driver (Drive mount, BASE path, e5 인코딩, 학습, eval).
6. **1a 실행** → go/no-go.

각 단계 `tdd` skill 명시 호출. 실험은 Colab.

## 8. References

- Plan: `../../RESEARCH_PLAN_2026-05-28_phase1_5.md`
- Context: `../../CONTEXT.md`
- Lit: `../../LIT_REVIEW_MASTER_2026-05-28_phase1_5.md` (특히 §6.1 Reasoning as Compression / Conditional IB · §6.2 Iterated Learning explicit pressure · §6.4 AdaLoGN 대비축)
- Phase 1 evidence (motivation): `../phase1/notebooks/04_diverse.ipynb`, `05_ablation.ipynb`, `06_baselines.ipynb`, `07_engine_a.ipynb`
- Memory: `~/.claude/projects/.../memory/project_objective_pivot_2026_05_28.md`
