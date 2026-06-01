# Lit Review — MASTER (Phase 1.5 lens) · positioning against prior work

> ℹ️ **Reproduced working note (bilingual).** Annotated positioning of this project against ~38 papers, re-verdicted under the Phase 1.5 lens (answer-prediction · information bottleneck · emergent neural module networks · single logic domain). Included for transparency; some cross-references point to the private working repo. Public entry points: [README](../README.md), [`research-journey.md`](research-journey.md).

> **목적**: 5-27 master 를 *Phase 1.5 lens* (answer-prediction · info-bottleneck · emergent NMN · 단일 logic 도메인 · K=128) 로 재verdict + targeted WebSearch 의 신규 카드 통합. 단일 진실 원천.
>
> **historical**: `LIT_REVIEW_MASTER_2026-05-27.md` (recon-cycle lens) 보존 — 카드 본체·arXiv ID 가 그쪽에 있고 본 문서는 *verdict 갱신 + 신규 카드 추가* 중심. 양쪽 참조.
>
> **verdict 범례**: ★★ TIER-1 직접 precedent / forced-design 핵심 인용 · ★ STILL-LOAD-BEARING · ◐ PARTIAL · ✗ OBSOLETE (Phase 1.5 에서). arXiv ID 는 5-27 master 와 본 §6 신규 카드 verbatim.

---

## §1. Positioning (Phase 1.5, 2026-05-28)

**모든 메커니즘 점유** — novelty 주장 = "empty seat" 아닌 **forced-design configuration + 경험적 결과**. Phase 1.5 의 핵심 입장:

1. **C0 single commitment 의 연역 사슬 (paradigm)**: C0(structural decomp) → C1(G_u) → C2(KG-as-output) → C3(observable-text target). [C3] sharpening 으로 answer-prediction 이 정당 인스턴스 (vision §6.1, 5-28).
2. **"Explicit pressure 필요"의 학계 합의**: Iterated Learning (2105.01119) 가 직접 명시 — "NMN 의 compositionality bias 만으론 emergent layout X, explicit pressure 필요." 우리 3 prior (NMN modules emergent · MC hard-negative · info-bottleneck modulation) 가 정확히 그 explicit pressure. **우리 설계의 lit-grounded 정당화 = 이 카드.**
3. **Conditional IB-for-reasoning 의 학계 이름**: Reasoning as Compression (2603.08462) — "reasoning trace = computational bridge containing only information about response not directly accessible from prompt". **우리 info-bottleneck (Q-only · P side-channel · KG modulation) 의 학계 정확한 이름이 Conditional IB. paradigm 의 가장 직접 precedent.**
4. **Labeled counterpart**: MiCRo (2506.13331) — brain-like supervised modular reasoning. 우리 unsupervised emergence vs MiCRo supervised 가 직접 비교축.
5. **단일도메인 logic 정당화**: LogiQA/ReClor MC 의 distractor 가 same-context-different-reasoning hard-negative → operation isolation 의 ideal corpus.

**경계 (Phase 3, 안 함)**: AdaLoGN (2203.08992) = neural-symbolic explicit logic graph. 우리 latent operation-KG 와 대비축, Phase 3 readout 시 거리 측정 대상.

---

## §2. Thematic Bibliography (Phase 1.5 verdict)

### 2.1 Cycle / Dual Learning (★→◐ — recon-cycle 대비축으로 강등)
- ★→◐ He (2016) Dual Learning · Guo (2020) **CycleGT** · Xu (2023) INFINITY · Wang (2023) Faithful Data-to-Text
  → input=output autoencoder 형. 우리 answer-prediction 으로 분기 = *직접 선례 아닌 대비 카드*. forced-design framing 의 "이전 가정"으로 인용 (C3 의 미서술 가정 부분).

### 2.2 MoE foundations & routing (★ 유지)
- ★ Shazeer (1701.06538) · Fedus Switch (2101.03961) · Dai DeepSeekMoE (2401.06066) · Laha Sparsegen (1810.11975) · Wang ReMoE (2412.14711) · Peters α-entmax (1905.05702) · Sparsemax (2016)
- ◐ Lepikhin GShard · Zhou Expert Choice · Puigcerver Soft MoE · Huang Harder Tasks · Zeng AdaMoE · Guo DynMoE · Antoniak MoT · Hazimeh DSelect-K · Zhuang LD-MoLE · Chen SIMoE · Fedus MoE++
- ★ aux-loss-free LB (2408.15664) · normalized sigmoid gating ❓
- **5-28 추가**: Phase 1.5 K=128 router 선택 후보 = ReMoE / α-entmax / sparsegen. 1a 시작은 Phase 1 의 sparsegen 그대로 (v6 검증), 1a 진행 중 swap-ablation.

### 2.3 Emergent / unsupervised expert specialization (★★ 유지·강화)
- ★★ Nikolic **SMoE-VAE** (2509.10025) — 핵심 anchor 유지. 단 mechanism = recon → **answer-pred** 로 교체 (Phase 1.5 sharpening).
- **★→◐ Chu ADaMoRE** (2510.21207) — "reconstruction-primary + diversity" anchor 였으나 recon-primary 자체가 Phase 1.5 에서 대체됨. 인접 카드로 강등 (diversity 부분만 유지).
- ★ Han/Liu (2505.22323) Advancing Expert Specialization
- ★ Zhang **MoMoK ExID** (2405.16869) — S1 contrastive-MI 선례 유지
- ★ Feng **CoMoE** (2505.17553) · Tsai MiCE (2105.01899)
- ◐ Chen OMoE (orthogonal) · Shi MMVAE · Bristol MixER

### 2.4 Expert trajectory / sequence / composition (★→★★ — Phase 1.5 1b target 그 자체)
- **★★ Polysemantic Experts, Monosemantic Paths** (2604.17837) — 1b 활성 path 의 *해석 단위* 정통 precedent.
- **★★ Route Experts by Sequence, Not Token** (2511.06494) — per-token cycle 의 sequence-level readout 근거.
- **★★ Chain-of-Experts** (2506.18945) — **1b graph-router 의 직접 architectural precedent**.
- **★★ Andreas et al. NMN over-text** (1912.04971) · **NMN structure-learning** (1905.11532) — **emergent NMN target 의 정통 라인**. Phase 1.5 의 우리 입장 = 이들의 *unsupervised end-to-end emergent* 변형.
- ★ Hao **Coconut** (2412.06769) — latent reasoning trajectory.

### 2.5 Verification — probing / interpretability / selectivity / geometry (★★/★ 유지)
- ★★ Hewitt-Liang **Control Tasks / selectivity** (1909.03368) — Engine-A 핵심 metric, Phase 1.5 1a/1b gate.
- ★★ **Myth of Expert Specialization** (2604.09780) — geometry control 필수.
- ★★ Zoph **ST-MoE** (2202.08906) — token-type 경고. Phase 1.5 selectivity 가 4 control 로 직접 차단.
- ★ Lo **POS Sensitivity** (2412.16971) · Templeton Scaling Monosemanticity · Bricken Monosemanticity (transformer-circuits.pub)
- ★ Herbst Expert Strikes Back ❓ · Yan What Gets Activated ❓
- **★★→★★ Stuhr & Brauer** (2009.02383) **Objective Mismatch** — Phase 1 F3 collapse 의 *학계 정확한 이름*. forced-design framing 의 핵심 인용 ★★. (5-28 originally mis-attributed as "Loaiza-Ganem"; corrected at Stage 5 of academic-pipeline run 2026-05-28 — actual authors per arXiv = Bonifaz Stuhr & Jürgen Brauer, *Don't miss the Mismatch*, 2020-09-04.)
- ◐ Elhage / Olsson Induction Heads / Tenney BERT pipeline / Conmy / Geva / Gurnee / Belinkov & Glass

### 2.6 Scaling (★ — Phase 1.5 K=128 직접 anchor 로 승격)
- **★ Ludziejewski** (2402.07871) Scaling Laws Fine-Grained MoE — G≈8.
- **★ DeepSeek-V2** (2405.04434) — K_routed=64.
- **★ DeepSeek-V3** (2412.19437) — K_routed=256 (Phase 1.5 K=128 의 sweep target).
- **★ Muennighoff OLMoE** (2409.02060) · Nguyen Statistical Benefits Shared Experts (2505.10860).
- ◐ Mixtral · Chinchilla · Kudugunta task-MoE.

### 2.7 Personalization (◐ — Phase 2 deferred 유지)
- ◐ OPPU (2402.04401) · Per-Pcs (2406.10471) · P2P (2510.16282) · DEP (2507.20849)
- ◐ MoPE · FLEx · Facet-Aware · Facet-Level SAE · Mixture-of-Tastes · PROPER · P-React · Personality Subnetworks · MoLE
- ✗ LaMP / LongLaMP

### 2.8 Injection (★→◐ — Engine-B deferred, Phase 1.5 = MC-contrastive LLM-free)
- ◐ BLIP-2 · Frozen · Persistent Memory · Graph-as-Memory Cross-Attn (Phase 1.5 통과 후)
- ◐ Prefix-Tuning · Prompt Tuning · Knowledge Prompts · ActAdd · CAA
- ◐ KnowLA · KG-Adapter · GRIP · Text-to-LoRA · HypeLoRA (1b modulation 형태 후보 카드 — FiLM 대안)
- **★ FiLM 검토 필요 (5-27 master 누락)** — Perez (2017) Feature-wise Linear Modulation, modulation 의 정통 precedent. §6 신규 카드 후보.

### 2.9 Symbolic KG grounding (✗(now)/◐(Phase 3) 유지)
- ✗(now)/◐(Phase 3) GraphRAG · Think-on-Graph · QA-GNN · GreaseLM
- **5-28 추가**: **★ AdaLoGN (2203.08992)** — LogiQA/ReClor 의 neural-symbolic SOTA. **Phase 1.5 의 *명시적 대비축*** (우리 latent vs AdaLoGN 명시 logic graph). Phase 3 readout 시 거리 측정 대상. §6 신규.

### 2.10 KG construction / representation (◐ 유지)
- ◐ REBEL · EDC ❓ · HyperRED · COKE · VGAE · Gumbel-Softmax · Latent Mixture (2605.02780)

### 2.11 Cognitive-science grounding + labeled counterpart (Phase 1.5 입장 변화)
**Decomposition warrant**: Cummins · Marr · Bechtel & Richardson · Craver · Piccinini · Fodor · Carruthers · Spelke Core Knowledge (★ paradigm §1)
**Pluralism warrant**: Poldrack & Yarkoni · Sullivan · Burnston (★ unsupervised-discovery 정당화)
**Per-user axis 대비**: Per-Pcs / OPPU · Lee 2025 NHB belief embedding ❓
**Operation ontology (Phase 3 deferred)**: Schank · image schemas (Lakoff/Johnson/Wachowiak) · Talmy · FrameNet · Halford · Cognitive Atlas · ACT-R · VerbNet · Bloom · Baddeley · SOAR · Kelly PCT ✗(footnote)
**Belief/structure in nets**: Shai Belief Geometry · Park Linear Representation · Yarkoni Neurosynth
- **◐→★★ MiCRo** (AlKhamissi 2506.13331) — **Phase 1.5 의 *직접 labeled counterpart*** 로 승격. brain-like 4-expert supervised modular reasoning. 우리 unsupervised emergence 의 비교축 = MiCRo. ablation only 가 아닌 **paper level head-to-head 후보**.
- **★→★ Concept Bottleneck** (Koh 2007.04612) — supervised CBM 의 bottleneck-for-prediction. 우리 = unsupervised + reasoning-type bottleneck. 직접 precedent 로 승격.
- **5-28 신규**: **★★ Reasoning as Compression — Conditional IB** (2603.08462) §6.
- ◐ MoE-as-embedding (Li & Zhou 2410.10814)

---

## §6. 신규 카드 (2026-05-28 WebSearch GAP 통합)

### 6.1 ★★ **Reasoning as Compression: Unifying Budget Forcing via the Conditional Information Bottleneck** — arXiv:2603.08462
- CoT 생성을 Conditional IB 로 모델링. **"reasoning trace = computational bridge containing only the information about the response not directly accessible from the prompt."**
- **우리 Phase 1.5 의 학계 정확한 이름** — info-bottleneck (Q-only · P side-channel · KG modulation) 가 정확히 Conditional IB(Q→bottleneck→A 에서 P 는 conditioning). KG = reasoning trace 의 latent 버전.
- **forced-design framing 의 가장 강한 인용** — Phase 1.5 info-bottleneck 이 임의가 아니고 *학계 정통 IB 의 reasoning instantiation*.

### 6.2 ★★ **Iterated Learning for Emergent Systematicity in VQA** — arXiv:2105.01119
- 핵심 finding: "NMN 의 compositionality bias 만으론 emergent layout X. layout+module joint learning 시 **explicit pressure 필요**."
- **우리 3 prior (NMN modules emergent · MC hard-negative · info-bottleneck modulation) = 정확히 그 explicit pressure** — paper 의 design 이 lit-grounded 라는 정당화.
- **Phase 1.5 의 "왜 단일 도메인 + K=128 + bottleneck 셋이 다 필요한가" 학계 정답**: emergent layout 은 자발 X, 강한 inductive bias 필요. 우리는 그것을 architecture + corpus 양쪽으로 박음.
- **gate FAIL 시 framing**: "explicit pressure 다 깔고도 안 됨 = 학계 open problem 의 강한 negative". PASS 시 framing: "explicit pressure 조합의 경험적 성공 evidence".

### 6.3 ★ **End-to-End Module Networks (N2NMN)** — arXiv:1704.05526
- Hu et al. layout 을 parser 없이 직접 예측. 단 imitation (expert demonstrations) + downstream loss 사용.
- 우리 = imitation 없이 (operation label 없이) downstream answer-pred loss + info-bottleneck 만. **더 약한 supervision**.
- 직접 architectural precedent + 우리와 supervision 강도 차이 명시 인용.

### 6.4 ★★ **AdaLoGN — Adaptive Logic Graph Network** — arXiv:2203.08992
- LogiQA/ReClor 에서 **neural-symbolic explicit logic graph** 로 추론. message passing on relation graph.
- **Phase 1.5 의 직접 대비축**: 우리 latent operation-KG vs AdaLoGN explicit logic graph. *같은 corpus(LogiQA/ReClor), 다른 KG 표현*.
- Phase 1.5 gate PASS 후 Phase 3 readout 의 거리 측정 대상 (latent → explicit 사상 가능성).
- LogiQA SOTA 추적의 reference point.

### 6.5 ★ **LogiQA 2.0** — TASLP 2023 (Liu et al.) — `dl.acm.org/doi/10.1109/TASLP.2023.3293046`
- LogiQA 원본 dataset 의 cleaned/expanded 버전. Phase 1.5 corpus 선택 시 v2 사용 검토.

### 6.6 ◐ **Graph-Integrated Multimodal Concept Bottleneck Model** — arXiv:2510.00701
- CBM + graph transformer 로 answer-concept / answer-question graph 구성. 우리 1b graph router 의 multimodal precedent.
- Phase 1.5 1b 형태 결정 시 reference.

### 6.7 ★ **Stack Neural Module Networks** — arXiv:1807.08556
- NMN 의 stack-based composition. 1b chain depth 선택 시 reference.

### 6.8 ★ **Information Bottleneck (foundational)** — Tishby, Pereira & Bialek (1999/2000)
- IB principle 의 원전. Conditional IB 의 base. 인용 anchor.

---

## §7. 5-27 → 5-28 verdict delta summary

**승격 (★ → ★★)**:
- Stuhr & Brauer Objective Mismatch (Phase 1 F3 의 학계 이름; mis-attributed as "Loaiza-Ganem" pre-Stage 5)
- §2.4 trajectory 카드 4종 (Polysemantic-Paths, Route-by-Seq, CoE, NMN ×2) — 1b target 그 자체
- MiCRo (labeled counterpart → 직접 비교축)

**강등 (★ → ◐)**:
- §2.1 Cycle/Dual (CycleGT, Dual Learning) — input=output autoencoder, 대비축으로
- §2.3 ADaMoRE — "recon-primary anchor" 가 Phase 1.5 에서 교체됨
- §2.8 Injection 전반 — Engine-B deferred (Phase 1.5 = LLM-free MC-contrastive)

**◐ → ★ 승격**:
- Concept Bottleneck (CBM) — 우리 bottleneck-for-prediction 의 supervised counterpart

**신규 ★★ (5-28 WebSearch)**:
- **Reasoning as Compression / Conditional IB** (2603.08462) — info-bottleneck 의 학계 이름
- **Iterated Learning Emergent Systematicity** (2105.01119) — explicit pressure 정당화
- **AdaLoGN** (2203.08992) — LogiQA/ReClor neural-symbolic 대비축

**신규 ★**: N2NMN (1704.05526) · LogiQA 2.0 · Stack NMN (1807.08556) · IB foundational (Tishby)

---

## §8. Citation-check (5-27 master §3 그대로 유지)
신규 카드 arXiv ID 6.1~6.7 = WebSearch 결과 verbatim (preprint 상태 미검증, paper write 전 재확인 필요). Tishby IB = 원전 1999/2000, exact venue 확인 권고.

## §9. Archived
5-27 master + 그 § 4 의 모든 sub-archive 그대로. Phase 1.5 historical = `LIT_REVIEW_MASTER_2026-05-27.md`.
