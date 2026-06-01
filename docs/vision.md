# Project Vision: Structural Personalization via Per-User MoE Activation over a Universal KG-Generation Mechanism (v2)

> ℹ️ **Reproduced working note.** This is an internal design document, written bilingually (mostly Korean) as the project's single source of truth for the paradigm and architecture. It is included for transparency and depth. Some cross-references point to files in the private working repository and may not resolve here — the public entry points are the [README](../README.md) and [`research-journey.md`](research-journey.md).

> **Core claim**: 기존 LLM personalization (LaMP, Per-Pcs, OPPU, Lee 2025) 은 user identity 를 *whole object* — flat embedding 또는 isolated per-user PEFT, 어느 cognitive context 에서도 동일한 single vector / single module — 로 다룬다. 우리는 user identity 가 **structurally decomposable** 하다고 주장한다: *shared* 한 interpretive operation pool + ***per-user, cognitive-context-conditional*** activation distribution 으로. 같은 fact 입력에 대해 user 마다 다른 cognitive output (user-specific sub-KG) 을 생성하면서, *같은 user 도* fact 의 cognitive context (예술 / 분석 / 사회적 / 회상 mode etc.) 에 따라 다른 expert subset 활성. User identity = *어느 context 에서 어느 primitive 를 활성하는가* 의 conditional distribution. 이 메커니즘을 자연어로부터 unsupervised (cycle reconstruction primary) 학습한다.

> **2026-05-13 v1 → v2 paradigm shift**: v2 본격 rewrite. v1 의 "외부 universal KG + per-user edge selection" framing 폐기. v1 archive = `research/PROPOSAL_PLANNING_v1_archive.md` §1-2.
>
> **2026-05-17 framing refresh + hypothesis scope sharpening**: Primary anchor = LLM personalization 학계의 structural-vs-flat 대비 (PCT/cogsci 는 사후 inspiration footnote, load-bearing 아님). 직접 실험 = **H2-a + S1 + H3 (3)** — H1 은 published baseline (Mairesse 2007 / Majumder 2017 / Boyd & Schwartz 2023) ref 흡수, 짜잘 가설 (S1 보완 motif/edge-type, Expert K ablation, gating reg 조합, cognitive complexity statistical form, expert frame match, cold-start, Lee 2025 flat-ablation) 전부 ref-only. 변경 근거 = `research/ARCHIVE.md §12`.
>
> **2026-05-19 paradigm framing shift + cognitive context conditioning sharpening**: paradigm framing 을 *epistemic falsifier* (S1 mechanism universality 직접 측정) 에서 *ML-standard multi-task performance evaluation + multi-axis post-hoc analysis* 로 evolution. paradigm 자체 (KG = cognitive output, shared K-expert × per-user G_u, cycle primary) 는 유지. *Per-user activation* wording 을 *per-user, cognitive-context-conditional activation* 으로 sharpening — user identity 가 *static vector* 가 아니라 *어느 cognitive context 에서 어느 primitive 활성* 의 conditional distribution 임을 명시. S1 의 primary falsifier 위상 강등 (→ §5 post-hoc analysis 의 expert disentanglement metric). 변경 근거 = `research/ARCHIVE.md §18`. (이 note 는 2026-05-19 시점 기록 — **현 single source = `research/RESEARCH_PLAN_2026-05-27_operation_cycle.md`**; 05-19 plan 은 archive. paradigm 자체는 본 문서가 계속 진실원천.)
>
> **2026-05-19 (revision 2) — Layered Phase 1 / Phase 2 structure**: paper-level structure 를 *layered sub-thesis* form 으로 reorganize. **Phase 1 (architectural value, no user)** = MoE-KG-cycle inductive bias 자체의 generic value 검증 (B2 > Generic LM, B2 > standard MoE — cheap gate). **Phase 2 (personalization value, + user 분화)** = Phase 1 위에 per-user, cognitive-context-conditional activation 추가 시 Phase 1 model + existing personalization baselines 능가 (added value test). 두 sub-thesis 는 같은 architecture 의 *with/without user dim ablation* — `phase2/model.py` 의 `use_user: bool` flag 추가로 minimal isolation. Historical phase numbering (framing_c_real/docs/02_phase1 / 03_phase2 / 04_phase3) 는 deprecated, ARCHIVE only marking (변경 근거 = `ARCHIVE.md §19`).

---

## 1. Core Hypothesis (v2)

**User identity 는 structurally decomposable 하다.** 기존 personalization 의 *flat embedding* (Lee 2025) 와 *isolated per-user PEFT* (Per-Pcs, OPPU) 는 user 를 whole object 로 다룬다. 우리는 user 를 두 축으로 분해한다:

- **사실(facts) = 공통 입력**: 자연어 corpus 의 진술 또는 entity-pair context. 모든 user 가 같은 fact 접근. 단 각 fact 는 *cognitive context* (예술 / 분석 / 사회적 / 회상 mode etc.) 의 latent signal 담음.
- **해석(interpretation) = per-user, cognitive-context-conditional 표상**: 같은 fact 에 대해 user 마다 *어떤 관계화 패턴 (interpretive operation) 을 활성화* 하는가가 다르며, *같은 user 도* fact 의 cognitive context 에 따라 다른 expert subset 활성. User identity = *어느 context 에서 어느 primitive 활성* 의 conditional distribution — single static vector 가 아님.

이 두 가지가 **공유 KG-generation mechanism + per-user, cognitive-context-conditional MoE activation distribution G_u** 로 분리된다는 게 v2 핵심 가설. KG 는 입력이 아니라 *cognitive output* — user 의 G_u 가 expert 들을 (fact 의 cognitive context 와 함께) 활성하여 user-specific sub-KG 를 생성한다. Architectural implementation 에서 *fact_gate(fact_emb) × user_logits(user_id)* 의 *combined gating* 이 이 conditional distribution 의 자연 form — explicit task_id input 또는 별도 context label 불필요, cognitive context 는 fact_emb 안에 implicit.

**Layered sub-structure (2026-05-19 revision 2)**:
- **Phase 1 (architectural value)** — `phase2/model.py` 의 `use_user=False` ablation: user dim 제거, fact_gate only, global λ scalar. 모든 user 가 같은 routing 활성 — generic MoE-KG-cycle architecture. Falsifier = Generic LM + standard MoE 능가.
- **Phase 2 (personalization value)** — `use_user=True` full form: 위에 `user_logits` + `lambda_mlp` 추가, per-user G_u + λ_u 의 cognitive-context-conditional modulation. Falsifier = Phase 1 model 능가 + existing personalization baselines (Lee 2025 / OPPU / PROPER) 와 competitive.

두 Phase 는 *같은 architecture 의 with/without user dim*, *minimal architectural isolation*. paper-level evidence 는 Phase 1 의 architectural value isolation + Phase 2 의 marginal contribution measurement 의 conjunction.

```
[Fact corpus (자연어)]  ← 공통 입력 (모든 user 가 같은 fact 접근)
       │
       ▼
[KG-generation mechanism]   ← 모든 user 공유 (universal)
   ├── Expert pool  e_1, e_2, ..., e_K  (관계화 패턴, fully learnable, K=8-12)
   └── Gating
       │
       ▼
[G_u — per-user MoE activation distribution]   ← user u 의 cognitive structure
   (K-dim distribution: 어떤 expert 를 얼마나 활성하는가)
       │
       ▼
[User-specific sub-KG]   ← user u 의 "해석" (cognitive output)
       │
       ▼
[Text]   ← 생성된 표면 형태
```

→ 학습 대상은 (a) universal expert pool + gating 메커니즘, (b) per-user activation distribution G_u. KG 는 위 둘의 함수로서 *출력*.

### v1 → v2 핵심 차이

| 측면 | v1 (deprecated) | v2 (현재) |
|---|---|---|
| KG 의 위치 | 입력 (외부 ConceptNet/Wikidata) | 출력 (cognitive output, 학습된 메커니즘이 생성) |
| User 표상 | edge selection over fixed graph | activation distribution over learnable expert pool |
| Expert 의 정체 | relation type (외부 taxonomy) | 관계화 패턴 (fully learnable interpretive operation) |
| Expert 개수 | ConceptNet 30+ relation | K=8-12 잠정 |
| 학습 신호 | cycle (auxiliary 또는 contrastive 보조) | cycle reconstruction (primary, 유일) |
| Personalization 위치 | per-relation-type weight (외부 taxonomy 의존) | structural decomposition: shared operations + per-user activation (LaMP/Per-Pcs 등 whole-object personalization 의 대비) |

### K_routed=8 의 정당화 (2026-05-19 fresh lit review)

**Nikolic et al. 2025** (arXiv 2509.10025 — *Exploring Expert Specialization through Unsupervised Training in Sparse MoE*): unsupervised expert routing 의 optimal K 가 num_classes 와 다름. QuickDraw 5-class 에서 optimal unsupervised K=7, linear probe separability +8.3pp vs class-label supervised K=5. **K=8 은 cognitive cardinality 의 강한 claim 이 아닌 empirical hyperparameter** — sub-categorical specialization 가능성을 열어둔 design choice. (Schank CD 11 primitives, Wachowiak 39 image schemas, Big Five 5 — 어떤 인지과학 cardinality 도 strong anchor 로 lock-in 하지 않음.)

추가 reference: **Bertotti et al. 2024** (arXiv 2403.11786) hyper-relation format precedent, **Chen et al. 2025 SIMoE** (arXiv 2506.12597) adaptive sparsity orthogonality loss alternative, **P-React (Dan 2025)** Phase 2 closest supervised counterpart (PSL on Big Five labels), **Facet-Level SAE (Tang 2026)** D5 cognitive context discovery method. 본 fresh lit review: `research/LIT_REVIEW_phase1_counterparts_2026-05-19.md`.

### v2 의 falsification

> ⚠️ **2026-05-19 update**: S1 mechanism universality 의 *primary falsifier 위상 강등*. `out/phase2/metric_v2_eval.json` reveal 에서 W_A ≈ 1 trivially (architectural identity, `routed_outs[k] = e_k(fact_emb)` 가 user_id-free function) — S1 v2 metric (A+C) 이 *paradigm universality* 가 아니라 *expert disentanglement* 측정함을 확인. paradigm 의 진짜 falsifier 는 *multi-task user simulation performance* (paper §4) + *multi-axis post-hoc analysis* (paper §5: D1 expert disentanglement / D2 mechanism universality 의 routed_alpha pattern β-form / D3 Yarkoni cognitive interpretation / D5 cognitive context discovery + user-conditional modulation). 변경 근거 = `research/ARCHIVE.md §18`, 새 plan §6.3 scope evolution table.
>
> 본 section 의 historical content (2026-05-17 measurement scope) 는 reference 보존.

**Historical (2026-05-17 wording)**: 같은 expert k 가 다른 user 에서 활성화될 때, expert 가 생성하는 sub-KG 의 형식적 property (motif, structural pattern, edge type distribution) 가 user 간 유사해야 한다. 이 prediction 이 깨지면 v2 의 "universal KG-generation mechanism" 가설이 *K개 user-specific subnet 집합* 으로 환원 → v2 falsified. Measurement: **Contrastive MI (MoMoK ExID, user axis 확장)** 가 *직접 측정 primary*. Motif / edge-type / structural pattern 은 *ref-only 보완*. MoMoK ICLR 2025 의 ExID estimator (eq 7-10) 의 user-axis 일반화가 v2 의 직접 contribution. (단 2026-05-18 PM 의 S1 v2 metric reform 후 2026-05-19 reveal 에서 *architectural triviality* 확인 — 새 plan 의 post-hoc analysis 로 강등.)

---

## 2. Framing 정리 — 세 가지 framing의 차이 (v2)

| 차원 | Framing A: 기존 PKG | Framing B (deprecated, [ARCHIVE §1](../ARCHIVE.md)): 사람마다 다른 KG | **Framing C v2 (확정): Universal KG-generation mechanism + Per-user G_u** |
|---|---|---|---|
| KG = 무엇? | 객관적 사실 표현 (입력) | 그 사람만의 schema (입력) | **cognitive output — 학습된 메커니즘이 fact 입력으로부터 생성** |
| "personal"의 의미 | 이 사람에 대한 사실들 | 추상화 구조가 다름 | **per-user MoE activation distribution G_u 가 다름** |
| 다른 사람과 비교 | 같은 schema, 다른 instance | 비교 어려움 | **같은 expert pool 의 다른 activation 분포** |
| 사실 vs 해석 | 구분 없음 (둘 다 사실) | 구분 없음 (둘 다 해석) | **명확히 구분 — fact = corpus, interpretation = G_u** |
| Sample efficiency | 좋음 | 나쁨 (사람당 많은 데이터) | **좋음 (mechanism 은 corpus 전체로, G_u 만 user data)** |
| ML 구현 | RAG·추천 | 막연함 | **MoE activation distribution + cycle reconstruction primary** |
| Comparability | 쉬움 | 어려움 | **쉬움 (G_u 가 K-dim distribution)** |
| Evaluation | precision/recall | 거의 불가 | **G_u distribution similarity·consistency + S1 (mechanism universality)** |

---

## 3. 왜 이 자리가 비어있나 (v2)

### KG + MoE 작업은 있는데 routing 의 동기가 다름

| 기존 작업 | Routing은 무엇 기반인가 | KG 의 역할 |
|---|---|---|
| MoKGR (2025) | Query 기반 | 입력 (외부 KG) |
| MixRAG (2025) | Query intent 기반 | 입력 (외부 KG) |
| MoSE (2025) | Node local context 기반 | 입력 (외부 KG) |
| MOEE (2026) | Relation type 기반 | 입력 (외부 KG) |

→ **Per-user routing 을 cognitive structure 로 해석한 작업 부재 + KG 를 cognitive output 으로 본 작업 부재**. v2 는 두 측면 모두 새로움.

### Adaptive-K routing 작업도 있는데 per-user identity 로 conditioning 한 작업 부재 (2026-05-18 lit-review)

`research/LIT_REVIEW_adaptive_routing_2026-05-18.md` 가 12 paper survey:

| Routing input axis | Paper |
|---|---|
| Token features (query) | Switch, GShard, DeepSeekMoE, AdaMoE, DynMoE, ReMoE, LD-MoLE, MoE++, Harder Tasks, DynaMoE |
| Expert demand (reverse direction) | Expert Choice |
| Per-slot soft assignment over all tokens | Soft MoE |
| **Per-user identity** | **(none)** |

→ **Per-user identity 를 routing 의 *axis* 로 사용하고 그 routing 이 *cognitive identity decomposition* 임을 주장한 작업 부재**. ReMoE / LD-MoLE 같은 fully-differentiable variable-K routing 의 *technical infrastructure* 는 모두 있지만, *그것을 per-user identity 로 사용 + cognitive interpretation* 은 v2 의 자리. Cardinality(K_active per fact, user) 가 Bieri 1955 / Scott 1962 의 cognitive complexity 의 자연스러운 implementation 이라는 framing 도 v2 specific.

**2026-05-18 실측 evidence**: Phase 2 v1 (`pen_reddit_v1_regfix`) 가 *naive per-user conditioning* (concat `fact_gate(fact) + user_logits(user)` before softmax) 을 시도했고 실패함 — routing 이 fact-driven 으로 학습되고 user identity 무시 (S1 W ≈ 0 across all 8 experts, K_active = K trivially since softmax). 이게 위 gap claim 의 실측 evidence: **per-user identity 가 routing 의 *axis* 로 architecturally privileged 되어야 함, naive concat 불충분**. v2 (`pen_reddit_v2_remoe`) 가 ReMoE drop-in + user_logits scale boost 로 이 architectural privileging 시도. 자세히는 `research/ARCHIVE.md §16`.

### Personal KG 작업은 사실 저장만 함

PersonalAI, CIKG, FedTREK-LM 모두 *그 사람에 대한 사실* 저장 (Framing A). *그 사람의 해석 방식 (관계화 패턴)* 은 안 다룸.

### LLM personalization 의 두 갈래 모두 user 를 whole object 로 다룸

| 갈래 | 대표작 | User 표상 |
|---|---|---|
| Flat user embedding | Lee 2025 (Nat Hum Behav), LaMP retrieval | user = vector (one point in space) |
| Isolated per-user adapter | Per-Pcs, OPPU (Tan EMNLP 2024) | user = independent PEFT module |

두 갈래 모두 user identity 가 *어떻게 fact 를 해석하는가* 의 **structural decomposition** 을 다루지 않음. 본 작업의 자리.

### 인지과학과의 관계 (inspiration only)

Personal Construct Theory (Kelly 1955) 의 person-as-*constructor* 발상이 우리 *KG-as-cognitive-output* framing 과 우연찮게 isomorphic. 그러나 본 작업은 PCT measurement 의 70년 explicit-elicitation 한계 해결을 *목표* 로 하지 않는다 — 그것은 cogsci 학계의 문제이고, 우리는 CS 학계의 LLM personalization 문제 안에서 작업한다. PCT 는 사후 inspiration 으로만 언급.

### 인지구조 통합 ontology 부재 = unsupervised discovery 의 정당성 (2026-05-18 lit-review)

인지과학에 *통합 정론* 의 cognitive operation taxonomy 는 없다 (Poldrack & Yarkoni 2016, *Annual Review of Psychology*). Schank's Conceptual Dependency (11 ACTs), Lakoff/Johnson image schemas (~30), ACT-R modules (~7), Cognitive Atlas (~868), Schank → image schemas → FrameNet → Talmy 등 competing schools 가 존재. 학계 메타-입장은 *coordinated pluralism* (Sullivan 2017; Burnston 2016) — 다른 연구 목적에는 다른 ontology 가 필요하다는 입장. 이는 우리 *unsupervised discovery* 의 근본적 정당성: 어느 학파에도 commit 하지 않고 (labeled-expert variant 의 한계), data 가 decomposition 을 발견하게 두고, 학습된 expert 를 다수 ontology 와 *post-hoc cross-validate*. 자세히는 `research/LIT_REVIEW_cognitive_ontology_2026-05-18.md`, `research/LIT_REVIEW_labeled_expert_2026-05-18.md`.

---

## 4. Theoretical Foundation (v2)

### LLM Personalization 의 structural-vs-flat 대비 (★ primary anchor)
- **Flat-vector personalization**: Lee 2025 (Nat Hum Behav) belief embedding centroid, LaMP (Salemi ACL 2024) retrieval-augmented personalization, Sentence-BERT style user embeddings
- **Per-user adapter personalization**: OPPU, Per-Pcs (Tan EMNLP 2024) — each user gets own PEFT module
- 공통 한계: user 가 *어떻게 fact 를 interpret 하는가* 의 structural decomposition 부재
- → **본 작업의 자리**: user identity 를 *shared operations + per-user activation* 으로 분해

### Mixture of Experts (Jacobs 1991, Shazeer 2017)
- Shared base + specialized experts + routing
- → **Universal mechanism (shared expert pool) + per-user G_u (activation distribution) 을 구현하는 ML 인프라**. v1 의 "routing over fixed graph" 가 아니라 v2 의 "activation distribution over learnable mechanism"

### Cycle Consistency (Guo et al. 2020, CycleGT)
- Text → structure → text reconstruction 으로 unsupervised 학습
- → **Primary learning signal in v2**. 외부 KG ground truth 부재의 자연스러운 귀결

### Expert disentanglement (MoMoK, Zhang et al., ICLR 2025)
- KG+MoE 에서 expert 간 contrastive MI 로 disentanglement
- → **S1 mechanism universality 의 직접 measurement precedent**. user axis 로 확장

### Cognitive-modeling inspiration (footnote only, not load-bearing)
- Personal Construct Theory (Kelly 1955), Cognitive Complexity (Bieri 1955, Scott 1962) 의 *person-as-constructor* + *differentiation/integration* 개념이 우리의 *shared operations + per-user activation* 구조와 isomorphic
- 본 작업은 PCT 의 measurement 문제 해결을 *목표* 로 하지 않음. 사후 inspiration 언급만

### Cognitive decomposition principle (메타-anchor, ★ strong consensus)
- *부분 기능들의 조합으로 인지를 설명하는 것 자체* 가 cognitive science 의 학계 정통 메타 원리 — **new mechanistic philosophy** (Bechtel & Richardson, *Discovering Complexity*, MIT Press 1993/2010) + **mechanistic explanation** (Craver, *Explaining the Brain*, Oxford UP 2007) + **modularity tradition** (Fodor 1983 *Modularity of Mind* → Carruthers 2006 *Massive Modularity*) → 30+년 dominant explanatory framework. Decomposition + localization 이 cognitive science 의 두 핵심 heuristic
- 우리 framing 의 *decomposition principle* 은 이 라인의 직접 후예. *어떤 부분들로 분해되는가* 는 pluralism (앞 §3 참조), *decomposition 한다는 것 자체* 는 학계 consensus
- 기존 modularity 라인 (Fodor 1983 ~ Carruthers 2006, Spelke core knowledge) 은 *universal mind* 만 다룸. 우리는 같은 decomposition principle 위에 **per-user activation distribution** 이라는 새 axis 를 더한다 — 이게 *decomposition 라인의 새로운 axis*
- 자세히는 `research/LIT_REVIEW_mechanistic_decomposition_2026-05-18.md` (별도 lit-review)

---

## 5. Closest Prior Work

### Cycle-training framework
| 논문 | 활용 |
|---|---|
| **CycleGT** (Guo et al., 2020) | Text↔KG cycle training의 학습 framework |
| **CycleCVAE / Fork or Fail** (Guo 2021) | Many-to-one mapping의 이론적 함정 (해석은 본질적으로 many-to-one) |
| **INFINITY** (ACL 2023) | 단일 seq2seq로 unsupervised 양방향, 가장 가까운 baseline |
| **ReGen** (EMNLP 2021) | SCST로 discrete bottleneck 처리 |
| **LAGRANGE** (LREC-COLING 2024, Apple) | Cyclic generation을 평가지표로 사용 |

### KG + MoE
| 논문 | 활용 |
|---|---|
| **MixRAG** (2025) — *정리 완료, `MixRAG_Summary.md` 참조* | MoE Graph-RAG의 직접 baseline. Soft node-wise gating + query-aware GNN + LLM 통합 패턴 |
| **MOEE** (2026) | "Shared embedding + specialized experts + routing" 패턴의 깔끔한 reference |
| **MoSE** (2025) | Subgraph expert 개념 |
| **MoKGR** (2025) | KG reasoning + personalization (per-query) |

### Adaptive-K / variable-active-count routing (2026-05-18 lit-review)
> 12 paper survey 정리: [`../LIT_REVIEW_adaptive_routing_2026-05-18.md`](../LIT_REVIEW_adaptive_routing_2026-05-18.md). v2 의 G_u activation cardinality 를 *cognitive complexity (Bieri 1955, Scott 1962) 의 measurement* 로 해석할 기술적 기반.

| 논문 | 활용 |
|---|---|
| **ReMoE** (Wang et al., ICLR 2025, arXiv 2412.14711) | ★ Primary drop-in for v2 Case B fallback. ReLU gate + adaptive L1, fully differentiable, per-(fact, user) variable K_active |
| **LD-MoLE** (Zhuang et al., ICLR 2026, arXiv 2509.25684) | Sparsegen + λ-MLP per-token sparsity. Sharper variant (LoRA experts 용, FFN 포팅 필요). Phase 3 후보 |
| **AdaMoE** (Zeng/Miao et al., EMNLP Findings 2024, arXiv 2406.13233) | Null experts + top-k, minimum architectural change. ReMoE 대안 |
| **DynMoE** (Guo et al., ICLR 2025, arXiv 2405.14297) | Top-Any Gating, auto-tunes K_total (우리 K commitment 와 conflict, 채택 X) |
| **MoE++** (Jin/Skywork, ICLR 2025 Oral, arXiv 2410.07348) | Zero/copy/constant heterogeneous experts |
| **Harder Tasks Need More Experts** (Huang et al., ACL 2024, arXiv 2403.07652) | Cumulative-probability threshold (top-p analogue), narrative match 가장 강함 |
| **Soft MoE** (Puigcerver et al., ICLR 2024, arXiv 2308.00951) | Dense opposite endpoint. v1 (loss-primary fix) 가 수렴하면 결과가 이쪽에 가까울 가능성 |
| **Expert Choice** (Zhou et al., NeurIPS 2022, arXiv 2202.09368) | Reverse direction (expert→token), variable-k. Routing semantics 반대라 우리 framing 과 mismatch |
| **DeepSeekMoE** (Dai et al., ACL 2024, arXiv 2401.06066) | Shared + routed expert + fine-grained. 우리 `K_routed=8 + K_shared=2` 가 이미 inherit. NOT adaptive count 자체 |
| **Switch / GShard** (Fedus JMLR 2022, Lepikhin ICLR 2021) | Rigid baseline. Switch 의 load_balance 가 v0 collapse 의 직접 inheritance |

### Cognitive structure inference from language
| 논문 | 활용 |
|---|---|
| **COKE** (ACL Findings 2024) | Cognitive KG의 개념적 baseline. 단 dataset-level, user-specific 아님 |
| **Capturing Human Cognitive Styles with Language** (2025) | Language → cognitive style inference 가능성의 정량 증거 (AUC 0.8) |

### Belief / cognitive representation in LMs (2024-2025, lit-review 추가)
| 논문 | 활용 |
|---|---|
| **Wu et al. 2025** (npj AI) — *ToM via Sparse Parameter Patterns* | Substrate-level sparse subnet evidence. 사회 인지 = 0.001% 파라미터 + low-rank subnet. **task-level**이라 우리 *per-user* routing과 differentiation 안전 |
| **Lee et al. 2025** (Nature Human Behaviour) — *Semantic Embedding Space for Human Beliefs* | **Closest unsupervised baseline**. Vote co-occurrence triplet → S-BERT contrastive fine-tuning → user = belief vector mean. Flat embedding + facts/interpretation 미분리 한계 |
| **Zhu et al. 2024** (ICML) — *LMs Represent Beliefs of Self and Others* | Belief가 LM 내부에서 linearly decodable + causal (intervention으로 ToM acc 0.33→0.66). R_u가 linear projection으로 충분할 가능성 시사. 단 supervised + narrative |
| **Shai et al. 2024** (arXiv 2405.15943) — *Belief State Geometry in Residual Stream* | R_u를 residual subspace projection으로 정식화하는 이론적 anchor (정독 예정) |
| **Chen et al. 2024a** — *Dashboard for Transparency* | Author demographic linear probe — 가장 위협적 인접 작업. supervised probe vs 우리 unsupervised routing 차이 명시 필요 (정독 예정) |

### Evaluation framework
| 논문 | 활용 |
|---|---|
| **Ribeiro et al. 2020** (Applied Network Science) — *Semantic Frame Induction via Community Detection* | **Routing similarity 평가의 직접 baseline**. Chinese Whispers + cosine-thresholded graph. Threshold = granularity dial. Phase 2 evaluation pipeline에 prototype |

### Cognitive decomposition tradition (메타-이론 anchor, 2026-05-18 lit-review)
> 본격 정리: [`../LIT_REVIEW_mechanistic_decomposition_2026-05-18.md`](../LIT_REVIEW_mechanistic_decomposition_2026-05-18.md). 학계 정통 라인의 *decomposition principle* 이 우리 framing 의 메타 정당성.

| 논문 / 저작 | 활용 |
|---|---|
| **Bechtel & Richardson 1993/2010** — *Discovering Complexity* (MIT Press) | "New mechanistic philosophy" — *decomposition + localization* 이 cognitive science 의 두 핵심 heuristic. 우리 framing 메타-원리의 철학적 anchor |
| **Craver 2007** — *Explaining the Brain* (Oxford UP) | Mechanism = *entities + activities organized → behaviors*. 신경과학 표준 explanatory framework. 우리 expert pool × G_u 의 mechanistic 정의 |
| **Fodor 1983** — *The Modularity of Mind* (MIT Press) | 변곡점 paper. Mind = modules + central system. 우리 *shared operations* 의 학계 ancestor |
| **Carruthers 2006** — Massive Modularity | Fodor 확장 — 전체 mind 가 modular. 우리 decomposition 의 가장 가까운 contemporary stance |
| **Spelke (1994+)** — Core Knowledge Systems | 4-5 core systems (object, agent, number, space, social). Modularity 구체 instantiation, *universal-mind* 가정 example |
| **Poldrack & Yarkoni 2016** — *From Brain Maps to Cognitive Ontologies* (Annu Rev Psychol) | 인지구조 통합 정론 *부재* 의 학계 공식 진단 — pluralism 메타-입장 |
| **Sullivan 2017** — Coordinated Pluralism | 우리 multi-ontology cross-validation 의 학계 메타-anchor |

### Cognitive operation ontologies (post-hoc cross-validation 후보, 2026-05-18 lit-review)
> 본격 정리: [`../LIT_REVIEW_cognitive_ontology_2026-05-18.md`](../LIT_REVIEW_cognitive_ontology_2026-05-18.md) + [`../LIT_REVIEW_labeled_expert_2026-05-18.md`](../LIT_REVIEW_labeled_expert_2026-05-18.md). H3-(c) 의 dual-ontology cross-validation.

| 논문 / 저작 | 활용 |
|---|---|
| **Schank 1972, 1975** — Conceptual Dependency (11 ACTs) | H3-(c) primary axis A — categorial cognitive operations |
| **Lakoff 1987 / Johnson 1987** — Image Schemas | H3-(c) primary axis B — embodied relational primitives |
| **Wachowiak & Gromann 2022** (COLING) — mBERT image-schema detector | H3-(c) axis B 의 자동 검출 도구 |
| **Halford, Wilson, Phillips 1998** (BBS) — Relational Complexity | H3-(c) secondary metric — relational arity (1-4), Bieri/Scott modern formalization |
| **Anderson 2007 / Anderson et al. 2004** (Psychol Rev) — ACT-R modules | Labeled-variant primary candidate (K=7 architectural homology) |
| **Poldrack et al. 2011** (Front Neuroinform) — Cognitive Atlas (868 concepts) | Cross-discipline ontology anchor; labeled-variant secondary candidate |
| **Koh, Liang et al. 2020** (ICML) — Concept Bottleneck Models | Labeled-variant 의 직접 architectural template (~30 lines port) |
| **Cambria et al. 2022** (LREC) — SenticNet 7 | CD-primitive neurosymbolic revival, methodology precedent |
| **Templeton et al. 2024** (transformer-circuits.pub) — Scaling Monosemanticity | SAE expert-interpretation precedent (free-form labels, no ontology cross-validation) |
| **Li & Zhou 2025** (ICLR Oral) — MoE-Embedding | MoE routing 이 semantic content carry 한다는 precedent |

---

## 6. Contribution One-Liner (v2)

> "Structural personalization 을, **'user identity 는 whole object 가 아니라 shared operations × per-user activation 으로 분해된다' 라는 하나의 commitment 에서 연역되는 단일 설계**로 모델링한다. 같은 fact 입력에 대한 per-user 해석을 — 외부 ontology 도, supervised label 도, explicit elicitation 도 없이 — 자연어로부터 unsupervised 하게 학습한다. 기여는 'MoE + KG generator' 라는 컴포넌트 조합이 아니라, **그 조합이 임의적이지 않다는 것** — 각 설계 결정이 앞 결정에서 강제되며, 그 사슬이 falsifiable 한 예측(S1)을 낳는다는 데 있다."

### 6.1. 왜 이 conjunction 은 임의적이지 않은가 — 연역 사슬

이 프로젝트의 4개 핵심 결정은 *쌓아 올린* 것이 아니라 *연역된* 것이다. 출발점은 단 하나의 commitment 다.

**[C0] User identity 는 structurally decomposable 하다: shared interpretive operations × per-user activation.**
기존 personalization (Lee 2025 의 flat embedding, Per-Pcs/OPPU 의 isolated per-user PEFT) 은 user 를 whole object 로 다룬다. 우리는 user 를 *어떤 operation 들을 얼마나 활성하는가* 의 distribution 으로 분해한다. 이 프로젝트가 받아들이는 유일한 전제. 나머지는 전부 여기서 따라 나온다.

*(historical note: 이 분해 구조는 Kelly 1955 의 person-as-constructor 발상과 isomorphic — 단, 우리는 PCT 가 아니라 LLM personalization 학계의 flat-vs-structural 대비에서 출발한다.)*

**[C0] → [C1] Per-user activation distribution = G_u.**
Decomposition 을 모델링하려면 "이 user 가 어떤 operation 을 활성하는가" 의 distribution 이 필요. 공유 operation pool 위의 per-user activation distribution = G_u. *MoE 는 이 G_u 를 구현하는 ML 인프라일 뿐, 출발점이 아니다.* (MoKGR / MixRAG / MoSE / MoMoK 가 incremental 로 읽히는 이유: 그들은 MoE 에서 출발해 routing 의 *대상* 만 바꾼다. 우리는 C0 에서 출발해 routing 의 *해석* 을 바꾼다.)

**[C0] → [C2] Operation 의 결과물 = cognitive output ⟹ KG 는 입력이 아니라 출력이다.**
Operation 이 fact 에 작용해 무언가를 생성한다 — 그 생성물이 user 의 *interpretation*, 즉 user-specific sub-KG. KG 가 외부에서 주어진 입력이면 user 가 할 수 있는 일은 edge 를 *고르는* 것뿐 (selector 로 환원). 이것이 정확히 v1 의 실수였다 (`외부 universal KG + per-user edge selection`). C0 의 decomposition 을 진지하게 받으면 KG 는 G_u × operations 가 fact 에 작용해 *만들어내는* 출력.

**[C2] → [C3] KG 가 출력이다 ⟹ 외부 ground-truth KG 가 없다 ⟹ cycle reconstruction 이 (선택이 아니라) 유일하게 가능한 primary signal 이다.**
KG 가 cognitive output 이면 정답 KG 를 어디서도 가져올 수 없다. supervised 학습이 원천 봉쇄된다. 학습 신호는 관측 가능한 것 — 텍스트 — 에서만 올 수 있다. ⟹ `text → G_u → user-specific sub-KG → text` 의 cycle reconstruction. cycle 이 primary 인 것은 우리가 그것을 *선호*해서가 아니라, C2 가 다른 모든 선택지를 닫아버렸기 때문이다.

**[C1] + [C2] → [S1] 검증 가능한 예측.**
G_u 가 *공유* mechanism 위의 activation 이라면, 같은 expert k 가 다른 user 에서 활성될 때 그것이 생성하는 sub-KG 의 형식적 property (motif / edge-type 분포) 가 user 간 유사해야 한다. 깨지면 "universal mechanism" 가설이 *K개의 user-specific subnet* 으로 환원 → 설계 전체가 falsified. S1 은 C0 의 commitment (shared operations 가 진짜로 *shared* 인가) 를 직접 시험하는 예측이다 — 그래서 framing 이 *장식이 아니다*.

**2026-05-17 실험 scope**: 위 연역 사슬 [C0]-[C3] 은 그대로 유지. 단 *직접 실험으로 검증할 것* 은 **H2-a (G_u group-spec ∧ user-distinct) + S1 (mechanism universality, contrastive MI) + H3 (full pipeline cycle quality + transfer + post-hoc frame match) 3 가지로 축소**. H1 (Phase 1 encoder + routing sanity, 사슬의 prerequisite unit test) 는 published baseline ref 인용으로 흡수, 별도 실험 폐기. 짜잘 가설 (motif/edge-type 보완, K ablation, gating reg 조합 등) 도 전부 ref-only. 자세히는 `research/ARCHIVE.md §12`.

### 6.2. 따라서 컴포넌트를 빼면 — 설계의 falsification

이 사슬이 임의적이지 않다는 가장 강한 증거: **어떤 컴포넌트를 빼도 사슬이 끊긴다.**

| 빼는 것 | 무너지는 것 |
|---|---|
| per-user G_u | structural decomposition 의 *per-user* 축 사라짐 — C1 붕괴, 일반 KG construction 으로 환원 |
| KG-as-output (입력으로 되돌림) | user 가 selector 로 환원 — C2 붕괴, v1 으로 후퇴 |
| cycle primary | KG 가 출력인데 학습 신호가 없음 — C3 붕괴, 학습 불가능 |
| S1 | framing 이 falsifiable 예측을 못 낳음 — "shared operations" 주장이 decorative 로 전락 |

→ 4개는 "MoE + KG + cycle 을 합쳤다" 가 아니라, **C0 (structural decomposition) 하나를 받아들이면 강제되는 단일 설계**다. Relabeling 비판 ("Per-Pcs / OPPU 와 뭐가 다르냐") 과 arbitrariness 비판 ("왜 하필 이 조합이냐") 을 동시에 막는 방어선이 여기다 — Per-Pcs/OPPU 는 user 를 *isolated PEFT* 로 다루므로 C0 의 "shared operations" 축이 없고, 따라서 C2 도 C3 도 S1 도 갖지 않는다.

### 6.3. Lit-review 반영 확장

> "**Lee et al. 2025** (Nat Hum Behav) 가 vote co-occurrence triplet 으로 *user-as-centroid* belief embedding space 를 구축했고, **Per-Pcs / OPPU** (Tan EMNLP 2024) 가 user 마다 *isolated PEFT* 로 personalization 을 구현했다. 두 갈래 모두 user 를 *whole object* 로 다룬다. 별도로, **Wu et al. 2025** (npj AI) 가 ToM 능력이 sparse low-rank subnet 에 *task-level* 로 localized 됨을, **Zhu et al. 2024** (ICML) 가 belief 가 LM 내부에서 *linearly decodable* 함을 보였다 — 단 두 작업 모두 *task-level*, *supervised*. 우리는 이 위에서 (i) **structural** (user = shared operations × per-user activation, whole object 아님), (ii) **per-user** identity-conditioned, (iii) **unsupervised** (cycle reconstruction primary), (iv) **fact / interpretation 명시적 분리**, (v) **KG-as-cognitive-output** 의 **per-user MoE activation distribution G_u** 를 universal KG-generation mechanism 위에서 학습한다. (Cognitive-modeling 관점에서는 Kelly 1955 의 person-as-constructor 와 isomorphic 한 inspiration footnote — load-bearing 아님.)"

### 6.4. 학계 정통 라인의 새 axis (decomposition tradition 안에서의 자리)

우리 framing 의 *decomposition principle 자체* 는 cognitive science 의 학계 정통 — **new mechanistic philosophy** (Bechtel & Richardson 1993/2010; Craver 2007) + **modularity tradition** (Fodor 1983 → Carruthers 2006; Spelke core knowledge) — 의 직접 후예다. 학계 메타-합의는 "*인지는 부분 기능들의 조합으로 설명되어야 한다*"; *어떤 부분들로 분해되는가* 는 pluralism (Poldrack & Yarkoni 2016; Sullivan 2017). 우리 contribution 의 위치는:

1. **Decomposition principle 자체는 발명 아님** — Bechtel/Craver/Fodor 라인 직접 후예.
2. **어떤 decomposition 인지 commit 회피** — pluralism 따름; data 가 unsupervised 발견; multi-ontology cross-validation.
3. **새로 더하는 axis**: 기존 modularity 라인 (Fodor 이래 30+년) 은 *universal mind* 만 다뤘다. 같은 decomposition principle 위에 **per-user activation distribution G_u** 라는 새로운 axis 를 더한다. 이게 *decomposition tradition 안에서의 우리 contribution*.

즉 우리는 *학계 변두리* 가 아니라 *학계 정통 (mechanistic decomposition) 라인의 새 axis* 위치. 자세히는 `research/LIT_REVIEW_mechanistic_decomposition_2026-05-18.md`.

---

## 7. What This Project Is / Is Not (v2)

**Is**:
- Structural personalization: user identity 를 shared operations × per-user activation 으로 분해
- Universal KG-generation mechanism + per-user MoE activation distribution G_u 의 학습
- 사실(facts, 자연어 corpus) 과 해석(interpretation, G_u) 의 명시적 분리
- KG = cognitive output (입력 아닌 출력)
- MoE expert = 관계화 패턴 (interpretive operation), fully learnable
- Cycle reconstruction primary signal 기반 unsupervised 학습
- S1 (mechanism universality) falsification 명시
- LLM personalization 학계의 flat-vs-structural 대비 (Lee 2025, Per-Pcs/OPPU 와의 대비)

**Is Not**:
- Per-user 별개 KG 학습 (Framing B는 deprecated)
- 외부 KG (ConceptNet/Wikidata) 위의 edge selection (v1 폐기)
- 외부 taxonomy (relation type) 기반 expert pool (v1 폐기)
- 사실 추출 시스템 (corpus 자체가 fact, mechanism 은 *KG 생성*)
- General-purpose KG construction (cognitive 의도 명시)
- RAG·추천 시스템 (downstream application 은 별개)
- 단순 personalization (preference filtering)

---

## 8. Architecture Sketch (v2 — KG-as-cognitive-output)

```
                  [Fact corpus (자연어, 모든 user 공통 입력)]
                                    │
                                    ▼
                    [Frozen text encoder]
                       (input fact 를 dense 표현으로)
                                    │
                                    ▼
        ┌─────── Expert pool (관계화 패턴, fully learnable, K=8-12) ───────┐
        │     e_1     e_2     ...     e_K                                    │
        │  각 expert = (fact → sub-KG 일부) 의 generation operator             │
        └──────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
              [Gating + per-user G_u activation distribution]
                G_u ∈ R^K (K-dim distribution, softmax 또는 sparse)
                u_emb = G_u 자체 (cycle 학습 결과)
                                    │
                                    ▼
              [User-specific sub-KG]
                = Σ_k G_u[k] · e_k(fact_context)
                (또는 hard top-k 선택, 잠정 soft)
                                    │
                                    ▼
              [LLM decoder (frozen + LoRA, KG-conditioned)]
                                    │
                                    ▼
                              text 복원

Cycle (primary signal): text(u) → frozen encoder → G_u(u, fact) → user-specific sub-KG → LLM → text 복원
Loss: cycle reconstruction (primary) + G_u sparsity / entropy / load-balancing regularization (auxiliary)
Personalization: G_u 가 per-user 학습 대상 (free table 또는 short-text encoded)
Falsification (S1): 같은 expert k 가 다른 user 에서 활성될 때 생성하는 sub-KG 의 motif/edge-type 분포 user 간 유사 검증
```

### 컴포넌트별 구현 결정 (v2)

| 컴포넌트 | 결정 (v2) | 근거 |
|---|---|---|
| **Fact source** | 자연어 corpus (Pennebaker Essays 등). entity 공간 anchor 로만 외부 KG (ConceptNet) 빌림 가능 | v2 의 핵심 — KG 는 출력이지 입력이 아님 |
| **Expert pool** | **관계화 패턴**, **fully learnable**, K=8-12 잠정. 외부 taxonomy 의존성 명시적 제거 | v1 의 ConceptNet relation 30+ 폐기. 인지언어학 sub-20 정합 + 사후 cluster 분석 가능 |
| **Routing = G_u** | per-user K-dim activation distribution (softmax 또는 sparse). user_emb = G_u 자체 | v1 의 per-relation-type weight 폐기 |
| **User embedding** | G_u 자체 (cycle 학습 결과). Cold-start 는 short text → encoder → initial G_u | v1 의 frozen encoder text encoding 은 *input fact 처리* 에만 한정 |
| **Routing 방식** | Soft (softmax) 잠정. 후속 sparse top-k 또는 entropy reg 로 sparsify | MixRAG soft 만으로 SOTA, 단 cognitive complexity (sparsity) 측정 위해 reg 필요 |
| **Backbone** | Pretrained LM frozen + LoRA, KG-conditioned (cycle 의 decoder side) | MixRAG+LoRA 패턴 동일 |
| **Primary loss** | **Cycle reconstruction** (Guo 2020) — auxiliary 아니라 *유일한 primary* | 외부 KG ground truth 부재의 귀결 |

---

## 9. Architecture 결정 상태 (v2)

### 확정 (v2)
- **방법론 기조**: Cycle consistency primary, GAN 빠짐
- **Framing**: C v2 (Universal KG-generation mechanism + per-user G_u, KG-as-cognitive-output)
- **Fact source**: 자연어 corpus (v1 의 외부 KG-as-input 폐기)
- **Expert pool**: 관계화 패턴, **fully learnable**, K=8-12 잠정 (v1 의 ConceptNet relation 30+ 폐기)
- **Routing = G_u**: per-user K-dim activation distribution. user_emb = G_u 자체
- **실험 scope (2026-05-17)**: 직접 실험 = H2-a + S1 + H3 (3). H1 + 짜잘 가설 전부 ref-only. K=10 commit, gating reg = MoMoK/Switch standard, cold-start = P2P hypernetwork, S1 metric = contrastive MI primary (motif/edge-type ref-only). ARCHIVE §12
- **Discrete Bottleneck**: **Soft routing**, Gumbel/STE 불필요 *(MixRAG 인사이트 1)*. 사후 sparsify
- **Encoder/Decoder Backbone**: Pretrained LM frozen + LoRA + KG-conditioned soft prompt *(MixRAG 인사이트 5)*
- **Tokenizer**: pretrained + KG special tokens
- **Primary loss**: Cycle reconstruction (v2 명시 — auxiliary 아닌 primary)
- **Falsification (S1)**: same expert k 가 다른 user 에서 활성될 때 sub-KG 형식적 property (motif/edge-type) user 간 유사

### Framing C v2 로 인해 명확해진 카테고리
| # | 카테고리 | 결정 방향 (v2) |
|---|---|---|
| 2 | KG Representation | 출력 — universal mechanism (shared) + per-user G_u 가 생성 |
| 4 | Loss | Cycle reconstruction primary. G_u sparsity·entropy·load-balancing regularization 보조 |
| 6 | Personalization | **G_u 가 personalization 그 자체** (K-dim activation distribution) |
| 7 | Data | corpus 는 모든 user 공통, G_u 는 user-specific text 로 학습 |
| 8 | Evaluation | G_u distribution similarity (cross-user), consistency (within-user), cycle reconstruction quality, **S1 mechanism universality** |

### 여전히 미정 (정독 → 결정 필요)
- **Expert 개수 K 의 정확 값**: 8-12 잠정. Ablation 으로 결정
- **MoE gating regularization**: sparsity (L1) / entropy / load-balancing — Background Survey §2 후 표준 조합 채택
- **S1 의 정확 metric**: motif / structural pattern / edge-type distribution 중 — Background Survey §1, §5 후 결정
- **Cold-start**: meta-learning vs population mean prior vs few-shot adaptation
- **Expert 의미 사후 해석 reference**: 인지언어학 frame / ConceptNet relation / FrameNet 중 어느 것과 post-hoc match

---

## 10. Open Questions

→ Component-anchored sharper form 은 [`V2_ARCHITECTURE_SHARPENING.md` §12](../V2_ARCHITECTURE_SHARPENING.md) (Q1.1-Q7.2) 참조. 본 §10 의 generic 5-question 원본은 [`ARCHIVE.md` §10](../ARCHIVE.md) 보존.

---

## 11. Next Steps (v2)

### 즉시·단기 정독 일정

→ [`V2_ARCHITECTURE_SHARPENING.md` §11 (Tier 1-4 정독 순서)](../V2_ARCHITECTURE_SHARPENING.md) + §15 (Recommended action items) 가 decision-anchor 기반 sharper form. 본 §11 의 원래 즉시·단기 step (2026-05-13 작성) 은 [`ARCHIVE.md` §11](../ARCHIVE.md) 보존.

### 중기

- **Minimal working baseline**:
  - G-Retriever 코드 (https://github.com/XiaoxinHe/G-Retriever) 를 fork → user_emb + per-user gate 추가 patch
  - 또는 CycleGT 재현 위에 v2 MoE activation 얹기
- **Cold-start 전략 실험**
- **Cognitive psych assessment 와의 correlation 실험**

(완료된 next steps — MixRAG 정독, Phase 1 lit-review, Background Survey 작성, 교집합 검증 — 은 [ARCHIVE §6 / §9](../ARCHIVE.md) 또는 `BACKGROUND_LIT_REVIEW.md §7` 참조)
