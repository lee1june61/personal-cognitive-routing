# Curated reading list — v2 paradigm

> ℹ️ **Reproduced working note (bilingual).** A curated list of papers organized by role in the project. The PDFs themselves are not included in this repository (large, re-downloadable from arXiv); this preserves the organization and rationale. Public entry points: [README](../README.md), [`research-journey.md`](research-journey.md).

> v2 paradigm (Universal KG-generation mechanism + per-user MoE activation G_u, KG-as-cognitive-output, cycle reconstruction primary) 의 정독 후보 paper 들. `BACKGROUND_LIT_REVIEW.md` 의 fresh 4-Layer Progressive Strategy search 결과 (★★★ 23편) + 2026-05-15 unsupervised/contrastive MoE 보강 6편 + 2026-05-18 cognitive_decomposition track 19편 (Angle A 9 + Angle B 8 + adaptive routing 2) 를 영역별 폴더로 구조화.

## 폴더 구조

| 폴더 | 영역 | paper 수 | 정독 우선순위 |
|---|---|---|---|
| `A_KG_Construction/` | Universal mechanism backbone — KG construction from text | 3 | ★★★ |
| `B_Sparse_MoE/` | Sparse Mixture-of-Experts + adaptive routing 의존 foundational | 5 *(+1, 2026-05-15)* *(+2, 2026-05-18)* *(−1 강등 2026-05-18 PM: MixER → _lookup_v2)* = 7 PDF + 2 txt | ★★★ |
| `C_Expert_Disentanglement/` | Expert disentanglement, routing diversity, **expert-level interpretation** | 6 *(+3, 2026-05-15)* *(+4, 2026-05-18)* *(−1 강등 2026-05-18 PM: POS Sensitivity → _lookup_v2)* = 9 | ★★★ |
| `D_Personalization/` | Personalization & per-user adaptation (G_u inject point) | 4 | ★★★ |
| `E_Cycle_Training/` | Cycle / dual learning / round-trip (primary learning signal) | 3 | ★★★ |
| `F_Cognitive_Representation/` | Cognition & belief representation in NNs (G_u 의 의미) + **brain-cognition mapping + mechanistic frame localization** | 3 *(+4, 2026-05-18)* *(−1 강등 2026-05-18 PM: Beger spatial → _lookup_v2)* = 6 | ★★★ |
| `G_Self_Supervised_Structural/` | Self-supervised structural / discrete bottleneck (KG-as-output 학습) | 5 *(+2, 2026-05-15)* | ★★★ |
| **`H_Labeled_Cognitive_Supervision/`** | **Labeled cognitive supervision baseline (MBTI / CBT / brain-network / Big Five) — v2 의 supervised counterpart** | **9** *(신규 2026-05-18)* *(−3 강등 2026-05-18 PM: Cai CBT + PerDet-R1 + Sun CD → _lookup_v2)* = 6 | ★★★/★★ |
| `_PCT_track_already_read/` | PCT/personality 트랙 (이미 정독, `PROPOSAL_PLANNING.md §4` 15편 중 PDF 확보 분) | 3 | (정독 완료) |
| `_lookup_v2/` | lookup-only (★★ skim / ★ lookup / paradigm independent baseline / 2026-05-18 demoted). 정독 안 함. | 8 *(원래 12 → 2026-05-18 cleanup: 10 삭제 + 6 강등 영입 = 8)* | ★★ 이하 |
| `_tools/` | 재사용 도구 (`_extract.py` 등) — paper 아님 | — | — |

**A-G prefix** = `BACKGROUND_LIT_REVIEW.md §1` 의 영역 재정의 순서. 알파벳 정렬 시 자연스럽게 영역 순서로 정렬됨.

## 각 영역 ★★★ paper

### A_KG_Construction (3)
- REBEL (Huguet Cabot & Navigli 2021)
- GraphRAG (Edge et al. 2024) — *From Local to Global*
- EDC (Zhang & Soh 2024)

### B_Sparse_MoE (5 + 1 + 2 − 1)
- Sparsely-Gated MoE (Shazeer et al. 2017)
- Switch Transformers (Fedus et al. 2022)
- DeepSeekMoE (Dai et al. 2024)
- MoE++ (Fedus et al. 2025)
- ~~*(added 2026-05-15)* MixER (Nzoyem et al. 2025 ICLR SCOPE) — *Hierarchical Meta-Learning via MoE*. Lit review 정리 완료 → `research/LIT_REVIEW_unsup_moe_2026-05-15.md` §5~~ **→ 2026-05-18 PM 강등: dynamical system reconstruction domain-specific. `_lookup_v2/_demoted_2026-05-18/` 로 이동.**
- *(added 2026-05-18)* **LD-MoLE** (Zhuang et al. 2026 ICLR) — *Learnable Dynamic Routing for Mixture of LoRA Experts*. Adaptive-K routing via Sparsegen + λ-MLP. Lit review → `LIT_REVIEW_adaptive_routing_2026-05-18.md` §1.11
- *(added 2026-05-18)* **Sparsegen** (Laha et al. 2018 NeurIPS) — *On Controllable Sparse Alternatives to Softmax*. Foundational for LD-MoLE 의 closed-form variable-K. Lit review → `LIT_REVIEW_adaptive_routing_2026-05-18.md` §1.13. `ld_mole.txt` / `sparsegen.txt` 는 추출 텍스트.

### C_Expert_Disentanglement (6 + 3 + 4)
- MoMoK (Zhang et al. 2025 ICLR) — *Multiple Heads are Better than One*
- OMoE (Chen et al. 2025) — *Orthogonal Finetuning*
- Specialization (Liu et al. 2025) — *Advancing Expert Specialization*
- *(added 2026-05-15)* ADaMoRE (Chu et al. 2025) — *Adaptive Graph MoE, Unsupervised, Heterogeneous*. Lit review 정리 완료 → §1
- *(added 2026-05-15)* CoMoE (Feng et al.) — *Contrastive MI gap for top-k routing*. Lit review 정리 완료 → §2 — **S1 metric 의 method-level precedent**
- *(added 2026-05-15)* MiCE (Tsai et al. 2021 ICLR) — *Mixture of Contrastive Experts, EM*. Lit review 정리 완료 → §4 — **G_u 정식화 base 후보**
- *(added 2026-05-18)* **MoE-X** (Yang et al. 2025 ICML) — *Mixture of Experts Made Intrinsically Interpretable*. Sparse routing → expert monosemanticity. Lit review → `LIT_REVIEW_labeled_expert_2026-05-18.md` §2.2 — **K=8-12 + sparsity reg 결정의 정당화**
- *(added 2026-05-18)* **Expert Strikes Back** (Herbst, Lee, & Wermter 2026 preprint ❓) — *Interpreting MoE LMs at Expert Level*. K-sparse probing methodology. Lit review → `LIT_REVIEW_cognitive_decomposition_2026-05-18.md` §1.10 — **Phase 3 post-hoc expert interpretation 방법론**
- *(added 2026-05-18)* **What Gets Activated: Domain & Driver Experts** (Yan et al. 2026 preprint ❓) — Entropy + causal-effect metrics. Lit review → `LIT_REVIEW_cognitive_decomposition_2026-05-18.md` §1.11 — **brain functional specialization motivation, S1 complement (causal effect)**
- ~~*(added 2026-05-18)* **POS Sensitivity** (Lo et al. 2024) — *Routers exhibit POS-class sensitivity*. Lit review → `LIT_REVIEW_cognitive_decomposition_2026-05-18.md` §1.12 — first-rung evidence of unsupervised emergent routing structure~~ **→ 2026-05-18 PM 강등: first-rung evidence only, Herbst/Yan 가 더 깊은 분석 제공. `_lookup_v2/_demoted_2026-05-18/` 로 이동.**

### D_Personalization (4)
- OPPU (Tan et al. 2024 EMNLP) — *Democratizing LLMs via Personalized PEFT*
- Per-Pcs (Tan et al. 2024 EMNLP) — *Personalized Pieces*
- PROPER (Zhang et al. 2025 ACL) — *Progressive Learning Framework, Group-Level Adaptation*
- P2P (Tan et al. 2025) — *Instant Personalized LLM via Hypernetwork*

### E_Cycle_Training (3)
- CycleGT (Guo et al. 2020)
- INFINITY (Xu et al. 2023)
- Dual Learning (He et al. 2016 NeurIPS)

### F_Cognitive_Representation (3 + 4 − 1)
- Shai et al. (2024 NeurIPS) — *Transformers represent belief state geometry*
- COKE (Wu et al. 2024 ACL) — *Cognitive Knowledge Graph for ToM*
- ~~Beger et al. (2026) — *Do LLMs Build Spatial World Models*~~ **→ 2026-05-18 PM 강등: spatial reasoning grid-world ≠ G_u user-level decomposition. `_lookup_v2/_demoted_2026-05-18/` 로 이동.**
- *(added 2026-05-18)* **Neurosynth** (Yarkoni et al. 2011 Nature Methods) — *Large-Scale Automated Synthesis of fMRI*. Forward/reverse inference framework. Lit review → `LIT_REVIEW_cognitive_decomposition_2026-05-18.md` §1.15 — **expert post-hoc cognitive interpretation 의 brain-side algorithmic isomorphism**
- *(added 2026-05-18)* **From Brain Maps to Cognitive Ontologies** (Poldrack & Yarkoni 2016 Annu Rev Psychol) — Cognitive Atlas / pluralism review. Lit review → `LIT_REVIEW_mechanistic_decomposition_2026-05-18.md` §1.11 — expert vs G_u 분리의 isomorphism
- *(added 2026-05-18)* **Mechanistic Interpretability of Socio-Political Frames** (Asghari & Nenno 2024 AIMLAI workshop ECML/PKDD) — Lakoff "strict father / nurturing parent" frame 의 hidden-state localization. Lit review → `LIT_REVIEW_cognitive_decomposition_2026-05-18.md` §1.13 — **expert → Lakoff frame mapping 방법론**
- *(added 2026-05-18)* **Linear Representations of Political Perspective** (Kim, Evans, & Schein 2025 ICLR) — Liberal-conservative axis linear decodable from attention heads. Lit review → `LIT_REVIEW_cognitive_decomposition_2026-05-18.md` §1.14 — R_u linear projection 의 두 번째 evidence

### G_Self_Supervised_Structural (5)
- Gumbel-Softmax (Jang et al. 2017)
- VGAE (Kipf & Welling 2016)
- Latent Mixture (Chen et al. 2025) — *Fine-Grained Graph Generation*
- *(added 2026-05-15)* SMoE-VAE (Nikolic et al. 2025) — *Unsupervised expert routing > supervised*. Lit review 정리 완료 → §3 — **v2 paradigm 정당화**
- *(added 2026-05-15)* MMVAE (Shi et al. 2019 NeurIPS) — *Variational MoE for multi-modal generation*. Lit review 정리 완료 → §6 — **cycle reconstruction 의 정신적 조상**

### H_Labeled_Cognitive_Supervision (9, 신규 2026-05-18)

v2 의 supervised counterpart — *labeled cognitive structure (MBTI / CBT / brain-network / Big Five) 를 ground truth 로 supervised baseline*. Lit review = `LIT_REVIEW_cognitive_decomposition_2026-05-18.md` Angle A (§1.1-1.9). v2 = unsupervised cycle + per-user G_u 의 inverse, 본 9편이 가장 가까운 비교 anchor.

- **★★★ MiCRo** (AlKhamissi et al. 2026 ICLR) — *Mixture of Cognitive Reasoners*. 4 brain networks (language/logic/social/world) 로 curriculum post-train. Angle A+B 교차 anchor — **load-bearing**. Lit review §1.1
- **★★★ Machine Mindset** (Lu et al. 2024) — MBTI 16-cell SFT+DPO. Categorical identity baseline. Lit review §1.2
- **★★★ CD Survey** (Sage, Keppens, Yannakoudakis 2025 EMNLP Findings) — 38 studies × 2 decades CD-NLP 종합 survey. *Line-level anchor*. Lit review §1.3
- **★★★ Mairesse 2007** (JAIR) — Big Five from text foundational. **첫 프로젝트 entry** (BACKGROUND_LIT_REVIEW / PROPOSAL_PLANNING_papers 에 없었음). H1 ref source. Lit review §1.9
- ★★ MbtiBench (Li et al. 2024 COLING 2025) — soft-labeled MBTI dataset, categorical→distributional drift. Lit review §1.7
- ★★ From Post to Personality (Chen et al. 2025 CIKM) — attention distribution ↔ MBTI dominant/auxiliary alignment. Lit review §1.8
- ~~★★ AI-Enhanced CBT (Cai et al. 2024) — cognitive pathway extraction. Lit review §1.5~~ **→ 2026-05-18 PM 강등 `_lookup_v2/_demoted_2026-05-18/`**
- ~~★★ ❓ PerDet-R1 (Lan et al. 2026, AAAI Bridge submission) — MBTI ranking + GRPO. Lit review §1.6.~~ **→ 2026-05-18 PM 강등 (핵심 claim 미검증)**
- ~~★★ ❓ Towards Consistent CD Detection (Sun et al. 2025 preprint) — LLM-as-annotator + κ evaluation. Sage 후속. Lit review §1.4~~ **→ 2026-05-18 PM 강등 ("standalone weight is limited")**

→ 최초 9편 중 3편 강등 → 폴더 남은 6편 (★★★ 4 + ★★ 2). Lit review 의 entry 는 모두 유지 — 분석은 남아 있고, PDF 만 `_lookup_v2/_demoted_2026-05-18/` 로.

### _PCT_track_already_read (3) — `PROPOSAL_PLANNING.md §4` 의 15편 중 PDF 확보 분
- Lee et al. (2025 *Nature Human Behaviour*) — semantic embedding space for beliefs
- Wu et al. (2025 *npj AI*) — ToM via sparse parameter patterns
- Zhu et al. (2024 ICML) — LMs Represent Beliefs of Self and Others

→ 이미 정독 완료. fresh ★★★ list 와 별도 track. `BACKGROUND_LIT_REVIEW.md` 에서 중복 금지.

## 정독 시작 가이드

`BACKGROUND_LIT_REVIEW.md §6` 의 3-주 schedule 따라 진행. **Week 1 Day 1 추천 3편**:
1. **C_Expert_Disentanglement/Multiple Heads are Better than One (MoMoK)** — S1 falsification metric 결정에 critical
2. **F_Cognitive_Representation/Transformers represent belief state geometry (Shai)** — G_u 의 mathematical formulation anchor
3. **E_Cycle_Training/CycleGT** — primary signal implementation 시작점

## Phase 1 / Phase 2 attribution (2026-05-19 revision 2)

layered paper structure (`RESEARCH_PLAN_2026-05-27_operation_cycle.md` §2-3) 의 Phase 1 (architectural value, no user — operation-cycle) + Phase 2 (+ user 분화 personalization value) 의 paper attribution. 정독 우선순위는 *어느 Phase 가 현재 작업 단계* 인가 에 따라.

### Phase 1 attribution (architectural value backbone — MoE-KG-cycle generic value)

| 영역 | Paper 수 | 정독 핵심 (Phase 1) |
|---|---|---|
| **A_KG_Construction** | 3 | REBEL (relation extraction), GraphRAG, EDC — universal KG-generation mechanism |
| **B_Sparse_MoE** | 7 | Switch (Fedus 2022), DeepSeekMoE (Dai 2024), LD-MoLE (Zhuang 2026 ICLR), Sparsegen (Laha 2018) — sparse routing architecture, **Phase 1 baseline B1 의 reference** |
| **E_Cycle_Training** | 3 | INFINITY (Xu 2023 ACL) — cycle reconstruction primary supervision, **Phase 1 cycle 의 직접 ancestor**; CycleGT (Guo 2020); Dual Learning (He 2016) |
| **G_Self_Supervised_Structural** | 5 | Gumbel-Softmax, VGAE, Latent Mixture, SMoE-VAE (Nikolic 2025), MMVAE — discrete bottleneck + unsupervised structural |

### Phase 2 attribution (+ user 분화 personalization value)

| 영역 | Paper 수 | 정독 핵심 (Phase 2) |
|---|---|---|
| **D_Personalization** | 4 | OPPU (Tan 2024 EMNLP, **BP2 baseline**), Per-Pcs, PROPER (Zhang 2025, **BP3 후보**), P2P (Tan 2025, hypernetwork) |
| **F_Cognitive_Representation** | 6 | Shai 2024 NeurIPS (belief geometry, **D5 cognitive context discovery anchor**), Yarkoni 2011 (D3 Yarkoni protocol), Poldrack & Yarkoni 2016, Asghari Lakoff frame, Kim ICLR 2025 (linear political axis), COKE |
| **H_Labeled_Cognitive_Supervision** | 6 | MiCRo (AlKhamissi 2026 ICLR, **supervised counterpart, MoB 4-block**), Machine Mindset MBTI, Mairesse 2007 (Phase 2 의 *trait acc auxiliary* anchor), MbtiBench, From Post to Personality |
| **_PCT_track_already_read** | 3 | Lee 2025 NHB (**BP1 flat centroid baseline**), Wu 2025 npj AI, Zhu 2024 ICML — 이미 정독 완료 |

### Cross-Phase (Phase 1 + Phase 2 둘 다 적용)

| 영역 | Paper 수 | 정독 핵심 |
|---|---|---|
| **C_Expert_Disentanglement** | 9 | MoMoK (Zhang 2025 ICLR, S1 metric ancestor), OMoE, Specialization, ADaMoRE, CoMoE (S1 method-level precedent), MiCE (G_u formalization), MoE-X (sparsity reg), Herbst Expert Strikes Back (K-sparse probing), Yan Domain/Driver Experts. Phase 1 의 D1 expert disentanglement (general MoE specialization) + Phase 2 의 D2 user-conditional disentanglement 둘 다 reference |

### 정독 시작 가이드 (revision 2)

**Phase 1 시작 시 우선** (Stage A 의 prerequisite):
1. **E_Cycle_Training/INFINITY** — Phase 1 cycle 의 직접 ancestor, architecture 변경 분명화
2. **B_Sparse_MoE/Switch (Fedus 2022)** + **DeepSeekMoE (Dai 2024)** — Phase 1 의 B1 baseline 의 standard reference
3. **A_KG_Construction/EDC** — KG-as-output 의 unsupervised construction

**Phase 2 시작 시 우선** (Stage B2 의 prerequisite, Phase 1 통과 후):
1. **D_Personalization/OPPU** ★ — **사용자 본인 정독 필수** (BP2 baseline 의 per-user LoRA param count + setup 정확화, paper claim 의 weak point)
2. **_PCT_track_already_read/Lee 2025 NHB** — BP1 baseline, *paper §2 prior work 의 core anchor* (venue final verify 필요)
3. **H_Labeled_Cognitive_Supervision/MiCRo** — supervised counterpart, Phase 2 의 D4 future work 의 r=0.7 anchor

**D5 cognitive context discovery 시작 시 우선** (Stage D5 의 prerequisite):
1. **F_Cognitive_Representation/Yarkoni 2011** — forward/reverse inference + FDR
2. **F_Cognitive_Representation/Asghari Lakoff frame** — frame localization 의 방법론
3. **H_Labeled_Cognitive_Supervision/Mairesse 2007** — observer > self-reports + LIWC 88 feature ceiling

(이전 "Week 1 Day 1 추천 3편" — MoMoK / Shai / CycleGT — 는 historical, 새 layered structure 의 priority 와 다름.)

---

## 2026-05-19 Fresh lit review — 7 new papers (Phase 1 counterparts)

`LIT_REVIEW_phase1_counterparts_2026-05-19.md` 의 7 paper 다운 + 영역별 attribution. 사용자 정독 priority 순:

| Paper | arXiv | Folder | Priority |
|---|---|---|---|
| **Bertotti 2024 — Hyper-Relational KG by LLM** | 2403.11786 | `A_KG_Construction/Bertotti_2024_HyperRelational_KG_LLM.pdf` | ★★★ Phase 1 hyper-relation format ancestor |
| **Nikolic 2025 — Unsupervised Expert Specialization in Sparse MoE** | 2509.10025 | `G_Self_Supervised_Structural/Han_2025_Unsupervised_Expert_Specialization.pdf` | ★★★ Phase 1 paradigm core anchor (K hyperparameter 정당화) |
| **P-React (Dan 2025) — Personality × LoRA Experts** | 2406.12548 | `D_Personalization/Wei_2024_PReact_Personality_LoRA.pdf` | ★★★ Phase 2 closest supervised counterpart (PSL) |
| **Facet-Level SAE (Tang 2026) — Contrastive SAE persona routing** | 2602.19157 | `D_Personalization/FacetLevel_Persona_Routing_SAE.pdf` | ★★★ Phase 2 most direct architectural competitor (inference-time SAE) |
| **SIMoE (Chen 2025) — Sparse Interpolated MoE upcycling** | 2506.12597 | `B_Sparse_MoE/Lin_2025_SIMoE_Interpolated_MoE.pdf` | ★★ Phase 1 orthogonality loss source |
| **FLEx (Liu 2025) — Federated Personalized MoE Expert Grafting** | 2506.00965 | `D_Personalization/Chen_2025_FLEx_Federated_MoE.pdf` | ★★ Phase 2 federated counterpart |
| **Personality Subnetworks (Ye 2026 ICLR) — Train-free pruning** | 2602.07164 | `F_Cognitive_Representation/Personality_Subnetworks_LLM.pdf` | ★★ Phase 2 train-free alternative paradigm |

전체 annotated bibliography + Phase 1 / Phase 2 보강 implications = `../LIT_REVIEW_phase1_counterparts_2026-05-19.md`.

## 변경 이력

- **2026-05-13**: v1 → v2 paradigm shift 후 fresh search → 23 ★★★ paper 정독 후보 확보. 잉여 12편 → `_lookup_v2/`, PCT 트랙 3편 → `_PCT_track_already_read/`. 영역 폴더 A-G prefix 로 rename. 폐기된 v1-era `trash/` 폴더는 `/.archive/papers_trash_v1_era/` 로 이동 (`ARCHIVE.md §9` 참조).
- **2026-05-15**: Unsupervised / contrastive MoE specialization 라인 6편 추가 (ADaMoRE, CoMoE, MiCE → C; SMoE-VAE, MMVAE → G; MixER → B). `new/` staging 폴더에서 영역 폴더로 이동. Lit review 는 `research/LIT_REVIEW_unsup_moe_2026-05-15.md` 가 single source. 사용자 정독 priority 는 `research/READING_PRIORITY_2026-05-15.md` 참조. `_tools/` 신규 (`_extract.py` 재사용 가능 PDF 추출 스크립트).
- **2026-05-18**: Cognitive_decomposition track (Angle A 9편 + Angle B 6편) + adaptive routing 2편 + 기존 lit review cross-listed 3편, 총 19편 `new/` → 영역 폴더 분산. **신규 `H_Labeled_Cognitive_Supervision/` 폴더 생성** (Angle A 9편 — MBTI / CBT / brain-network / Big Five labeled-supervision baseline). B/C/F 각 영역 확장. Lit review 다섯 개 동시 생성 — `LIT_REVIEW_cognitive_decomposition_2026-05-18.md` (본 19편 중 15편 본체) + `LIT_REVIEW_adaptive_routing_2026-05-18.md` + `LIT_REVIEW_cognitive_ontology_2026-05-18.md` + `LIT_REVIEW_labeled_expert_2026-05-18.md` + `LIT_REVIEW_mechanistic_decomposition_2026-05-18.md`. PREP doc = `LIT_REVIEW_PREP_cognitive_decomposition_2026-05-18.md`. 정독 후보 우선순위: §6 의 "Suggested next-step actions" 참조 — MiCRo (1.1) + Yarkoni 2011 (1.15) + Mairesse 2007 (1.9) 가 cognitive_decomposition 의 load-bearing top-3. **PM PDF-deep 보강**: pdftoppm/pdftotext (Poppler 25.07) 로 MiCRo/Mairesse/Neurosynth 본문 정독 — §1.1 MiCRo 가 standard MoE 가 아닌 **Mixture-of-Blocks (MoB)** 임을 정정, §1.9 Mairesse 의 정확한 feature counts (LIWC 88 / MRC 14 / utt 4 / pros 10) + ceiling (Openness 63%, Extraversion 73%) 추가, §1.15 Neurosynth term source 가 abstract 가 아닌 *full article text* 임을 정정.
- **2026-05-19 (revision 2)**: layered Phase 1 / Phase 2 paper attribution 추가 (위 section). 영역별 paper 들을 architectural value (Phase 1: A/B/E/G) + personalization value (Phase 2: D/F/H/_PCT) + cross-Phase (C) 로 매핑. 정독 시작 가이드도 새 layered structure 의 priority 로 update. Historical "Week 1 Day 1 추천 3편" 은 deprecated. 변경 근거 = `../ARCHIVE.md §19`.
- **2026-05-18 PM cleanup**: scope drift / lookup 가치 약한 paper 정리.
  - **삭제 10편**: `_lookup_v2/{universal_KG_v1_era,user_activation_v1_era,architecture,cognitive,persona}/` 5 sub-folder 전체 (3 + 1 + 2 + 2 + 2 = 10 PDFs). 이유: v2 KG-as-output 와 정반대 (v1-era KG retrieval), v1-era discrete latent (G 폴더가 직접 baseline), 너무 광범위 (ToM survey, Ribeiro frame), PAS ref-only per CLAUDE.md. lit review reference 없음 — 분석 손실 없음.
  - **강등 6편 (lit review entry 있음, PDF 만 `_lookup_v2/_demoted_2026-05-18/` 신규 sub-folder 로)**: B/MixER (dynamical system reconstruction, ODE domain) · C/POS Sensitivity (first-rung evidence only) · F/Beger spatial (spatial reasoning ≠ G_u user decomposition) · H/AI-Enhanced CBT · H/PerDet-R1 (핵심 claim 미검증) · H/Sun Consistent CD ("standalone weight is limited"). Lit review entry 는 *유지* — 분석은 남고, PDF 위치만 강등.
