# Research Context — 용어집 / 불변식 (Glossary & Invariants)

> ℹ️ **Reproduced working note (bilingual, mostly Korean).** The project's domain glossary and its "IRON rules" (invariants kept across every experiment). Included for transparency; some cross-references point to the private working repo. Public entry points: [README](../README.md), [`research-journey.md`](research-journey.md).

> 이 프로젝트(Research 트랙)의 도메인 언어 단일 출처. 구현 디테일 X, 도메인 전문가에게 의미 있는 term 만.
> Product 트랙(`mvp/`)은 별개 context.

## 핵심 commitment (KG_Project_Vision §6.1 연역 사슬)

- **C0** — User identity 는 whole object 가 아니라 *shared interpretive operations × per-user activation distribution* 으로 structural decomposition. 유일한 전제, 나머지는 연역.
- **C1** — per-user activation distribution = **G_u** (공유 operation pool 위의 분포). MoE 는 G_u 의 구현 인프라일 뿐.
- **C2** — operation 의 결과물 = cognitive output ⟹ **KG 는 입력이 아니라 출력**. (외부 KG 위 edge selection = v1 의 실수.)
- **C3** — KG 가 출력 ⟹ 외부 ground-truth KG 없음 ⟹ supervised KG 학습 봉쇄 ⟹ **학습 신호는 관측 가능한 텍스트에서만**.

## 용어집

### Cycle reconstruction (C3)
학습 신호의 형식. **정의 (2026-05-28 sharpening)**: C3 가 강제하는 것은 *"타깃이 관측 가능한 텍스트여야 한다"* 뿐 — supervised KG 가 불가능하기 때문. C3 는 **input=output(autoencoder) 을 강제하지 않는다**. 원래 vision 의 `text → KG → text` 서술에서 "같은 텍스트"는 *미서술 가정*이었지 연역이 아님.

- 따라서 정당한 타깃 = 관측 가능한 어떤 텍스트든. **입력 복원(autoencoder)** 과 **정답 예측(`Q → operation-KG → 정답-text`)** 둘 다 C3 의 인스턴스.
- 정답 예측이 오히려 **C0 에 더 충실**: 입력 복원은 content+operation 을 뭉뚱그려 topic 이 지배(Stage1 evidence: operation adj +0.18 vs topic adj 0.58~0.97). 정답 예측은 "유저의 Q→A 변환"을 타깃으로 삼아 **operation 을 isolate**.
- ⚠️ 여전히 위반인 것 = supervised KG label, 외부 ground-truth KG, explicit elicitation. 이건 어떤 타깃 선택으로도 열면 안 됨.

### Operation (vs Topic)
expert 가 분화하길 원하는 축. **operation = "어떻게 처리/추론하나"** (Q→A 변환). **topic = "무엇에 관한가"** (content). 둘은 같은 topic·다른 reasoning 으로 분리됨 (예: causal vs predictive question on 같은 passage). topic 으로 분화하면 G_u 가 bag-of-interests 로 붕괴 = flat baseline = novelty 소멸 → operation 이 필요조건.

### Answer-prediction objective (Engine-A, 2026-05-28 pivot)
Engine-A 의 학습 objective. **입력 복원(autoencoder) 폐기 → 정답 예측으로 full replace** (hybrid 아님 — recon 이 topic 을 *유발*하므로 auxiliary 로도 도로 들이지 않음; collapse 관측 시에만 ablation 으로 복귀).
- **info-bottleneck (3종, 아키텍처로 강제)**:
  1. **Q-only 인코딩** — operation-KG = encode(Q). passage P 는 인코더가 안 봄 → P content 가 KG 에 못 들어감 = topic 오염 구조적 차단.
  2. **P = decoder side-channel** — facts 는 decode 시 직접 주입, KG 가 facts 외울 필요 없음.
  3. **KG modulation (concat 아님, no-bypass 강제)** — operation-KG 가 "P→정답 변환"을 파라미터화. **불변식 (2026-05-29 sharpening, 경험적 수정)**: 원래 문장 "KG 빠지면 변환 미정의 → bypass 차단"은 **거짓**으로 판명 (zero-query cross-attn = uniform attention → passage 평균이 residual 로 누출; FiLM 의 `+β` = 순수 passage 함수). canonical 재정의:
     - **(i) no-bypass (검증 테스트)**: `kg_hidden := 0` 이면 modulation 출력 = 0 ⟹ P-파생 정보가 답에 0 기여. P 가 답에 닿는 유일 경로 = KG 가 파라미터화한 변환. **강제**: P→answer 경로의 KG-독립 additive 항 전부 금지 (LN affine-bias / Linear bias / residual 제거) — 하나라도 남으면 상수 누출로 bypass 부활. `forward(kg=0)==0` 단위테스트로 회귀 방지.
     - **(ii) non-degeneracy (사후 검출)**: KG 는 input(Q)-의존. 상수-KG 붕괴는 (i) 만족해도 operation 미포함(P-gate 만 학습) → training pressure 로 강제 X (emergence 편향), selectivity gate + "K_active vs acc 상관" 으로 검출. **(2026-05-30: bypass 수정 후 val_acc 0.59. adj_op=0 은 (ii) 가 아니라 operation-라벨 confound 였음 — probe(LogiQA) type-dict 95% 한 클래스, 모든 K 에서 0. 수정 = operation 축을 regex(drop-other/희소)로, type-dict 폐기. K→1(z-loss) 은 별개 diversity 과제. operation emergence 는 regex 축 재측정 전까지 미판정.)**
     근거 = Phase 1.5 rev2 (K_active=0 인데 acc 0.35 = KG 무관 = (i) 위반 직접 증거; memory `project_phase15_collapse_diagnosis_2026_05_29`). **현 cross-attn·FiLM 둘 다 (i) 위반 → modulation 재설계 진행 중 (2026-05-29).**
- **MC-contrastive (LLM-free)** — QuAIL 4지선다 후보 임베딩 점수화 → argmax → CE. distractor 가 "P 만으론 구별 불가, Q reasoning 필요"하게 설계됨 → topic-매칭 bypass 차단. frozen BGE 임베딩 공간, 생성 없음 → "LLM in cycle 금지" 불변식 유지.
- bypass 이중 방어선: distractor(topic-매칭 차단) + modulation(KG-우회 차단 — ⚠ 위 #3 (i) 만족하는 modulation 이어야 실제 성립; 현 구현은 미달).

### S1 (mechanism universality)
falsifiable 예측: 같은 expert k 가 다른 user 에서 활성될 때 생성 sub-KG 의 형식적 property(motif/edge-type 분포) 가 user 간 유사. 깨지면 "universal mechanism" → K 개 user-specific subnet 으로 환원 → 설계 falsified. **타깃이 input 이든 answer 든 S1 은 불변**.

**측정 재정의 (2026-05-31, 1b)**: contrastive-MI 변형(MoMoK ExID)은 trivial(W_A≈1, architectural identity) 로 판명되어 **폐기**. 1b PASS bar 의 S1 은 **causal compositional battery + motif-consistency** 로 operationalize:
- **(a) causal**: hop-k expert(signature) lesion → hop-k 필요 문항에서 *선택적* 정확도 하락(hop-depth diagonal-dominant, >1σ). uniform 하락 = FAIL. (`intervention.lesion_step_specificity`, ceiling-robust)
- **(b) motif-consistency**: 같은 decomposition-structure(2hop/3hop/chain/comparison) 문항이 유사 active sub-path. `operation_consistency(motif_code, structure_labels, topic_control)`, op_purity>chance ∧ topic 이김.
- MuSiQue decomposition 주석이 S1 을 *처음으로* 측정가능하게 함(structure label = eval/S1 전용, 학습 신호 X).

### Phase 분류 (2026-05-28 reframe)
| Phase | 상태 | 본질 |
|---|---|---|
| Phase 1 (recon-cycle) | closed (informative negative) | 5-run + Stage1 → recon 이 content/topic 보상해 F3 collapse 구조적. 결과 = Phase 1.5 motivation. |
| **Phase 1.5** (operation-axis architectural) ★ active | answer-prediction + info-bottleneck + emergent NMN + 단일 logic 도메인 + K=128. Engine-A go/no-go. `RESEARCH_PLAN_2026-05-28_phase1_5.md` 단일 출처. |
| Phase 2 (personalization) | deferred (1.5 통과 후) | per-user G_u, use_user=True. |
| Phase 3 (KG readout) | future | latent → symbolic. |

### Engine-A target architecture (Phase 1.5)
**⚠ 2026-06-01: 순차 NMN north star 는 NEGATIVE (아래 §Sequencing 1b).** 원 target = emergent NMN(순차): `Q→enc→z(Q)→graph-router→operation-program(활성 path)→[execute on P side-channel]→predicted-answer→MC contrastive`. modules=experts(emergent), layout=path(answer-pred 압력으로 창발). → 순차 layout 창발 X 확인. **현 north star (direction 1) = 병렬 co-activation**: flat mixture `Σα_k expert_k`(K_active~5) 의 활성-분포 = vision C1 **G_u** 직접 구현. graph-router/순차 path 폐기, operation = *동시활성 분포*의 구조로 재정의.

### Corpus (2026-05-31, 1b pivot)
- **MuSiQue** (multi-hop QA → 4지 MC) = **주 compositional corpus**. single-hop 을 *합성*해 만든 anti-shortcut(single-hop 모델 F1 −30) + per-hop decomposition·intermediate answer 주석. operation 이 latent(stem-announce X). chain-router L steps ↔ 2~4 hops, active path = literal multi-hop program. distractor = **intermediate-hop answer**(single-hop shortcut 이 중간답 골라 틀림 = composition 구조적 강제, LLM-free) + same-type 백필. (`data_musique.py`, HF `dgslibisey/MuSiQue`)
- **LogiQA 2.0 + ReClor** = **단일-op control arm**. 진단(2026-05-31 그릴): 문항당 *단일 operation* 을 깊게 적용(구별-op 합성 아님), 모든 stem 이 operation announce(raw-Q ceiling 0.98), lexical passage→option ≈ chance(0.27) — operation 은 필요하나 *합성 substrate 부재*. → emergent-NMN(구별-op 합성) thesis 는 logic-MC 에 substrate 없음. control 로 보존: 1b 가 MuSiQue 엔 emergence·logic-MC 엔 X → **composition-substrate 가 driver 라는 attribution**.
- 진단 근거 = memory [[project_phase15_corpus_pivot_2026_05_31]]. corpus selector = `MCCorpusConfig.corpus ∈ {musique, logic_mc, both}`.

### Sequencing (ablation 규율)
- **1a**: flat α (1-step), 명명 = **operation router** ("KG" aspirational). gate = operation primitive 가 애초에 emergent 한지. **결과 = weak-ceiling negative (logic-MC); 진단 = corpus 가 단일-op → 1b 는 MuSiQue 로.**
- **1b**: + chain router (CoE pure chain, L=2~4; `model.forward_chain`, weight-tied experts + per-step `modulation_steps`, no-bypass=출력 누적기 z_q 제외), 명명 = **operation-KG** (literal). gate = composition(causal battery) + S1(motif). corpus = MuSiQue.
  - **결과 = NEGATIVE (2026-06-01, confound 제거 후)**: MuSiQue substrate 견고(flat val 0.598)나 1b breadth={2,2,2} = **깊이-적응 순차 composition 창발 X**. ★ 단 이 negative는 **순차 chain** 부정이지 **병렬/동시활성** 부정 아님 — flat mixture(`Σα_k expert_k`, K_active~5)가 이미 병렬 co-activation이고 이게 **vision C1 G_u(활성 *분포*)** 와 대응. **순차 chain은 detour, composition은 병렬일 수 있음 → G_u 병렬-활성 프레임 회귀가 live 대안.** [[project_phase15_corpus_pivot_2026_05_31]]. (잔여: L=3<4hop, frozen-e5 MC 천장→overfit 암기.)
- **1c**: + tree/DAG. 표현력 확장.
- 각 step 한 부품만 → attribution.

## 불변식 (IRON, 깨면 sync)
- KG = latent·distributional·emergent (explicit schema X, text form X = Phase 3 future). **Phase 1.5 1a 에서 "KG" 호칭 보류 — 1b chain-router 부터 실체 (활성 path = literal subgraph)**.
- Cycle = 관측가능 텍스트 reconstruction (C3 sharpening, 2026-05-28). input=output autoencoder X — answer-prediction 도 C3 정당 인스턴스. **LLM in cycle = Phase 1.5 금지** (vision end-state 는 LLM decoder, 단 phase 1.5 = MC-contrastive embedding-only).
- unsupervised emergent = **operation label supervision 금지** (intent). end-task supervision (gold answer) 은 허용 — 이게 emergence 압력의 출처. "unsupervised cycle" → "answer-supervised, operation-emergent" 로 sharpening.
- Phase 1 / 1.5 / 2 / 3 엄격 분리. user dim 은 Phase 2 진입 전 동결.
- novelty 주장 금지, forced-design + 결과 framing. 부품은 ablation 으로 규율.
