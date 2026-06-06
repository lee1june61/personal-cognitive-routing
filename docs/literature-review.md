# Literature Review — a graded gap map (where to aim, what to claim, what to borrow)

> ℹ️ **Reproduced working note.** Annotated positioning of this project against prior work, organized as a *graded gap map*. This is the current positioning doc (2026-06-06); it replaces an earlier Phase-1.5 thematic review that framed the target as a *sequential* neural module network — that direction was falsified (see [research-journey](research-journey.md)), so the positioning here is built around **parallel co-activation** and a **persona, not preference** framing. Public entry points: [README](../README.md), [research-journey](research-journey.md), [vision](vision.md), [glossary](glossary.md).
>
> **Framing (no novelty claim).** Every single axis below is occupied by prior work; what is underexplored is their *conjunction*, framed as *forced* by the project's C0–C3 commitments. Positioning is by **axis** (supervised vs unsupervised; predefined vs emergent), never "they treat the user as a whole object." Negative results to date are a maturity signal about how hard the precondition is.
>
> **Card format:** `[arXiv ID] Title — one-line verdict — TAG`, `TAG ∈ {adopt, beat, contrast, borrow-infra, borrow-target}`. **adopt** = import the technique; **beat** = baseline to outperform/differentiate; **contrast** = foil; **borrow-infra** = reuse plumbing, no claim; **borrow-target** = take the question, not the method.

---

## §A. Where we aim — underexplored, well-posed sub-questions

### A1 — Unsupervised emergence of *interpretive operations* (not predefined, not supervised)

The nearest work supervises the operation set (P-React: Big-Five labels) or hand-defines it (DS-MoE: shallow/compositional/logical experts); Expert Strikes Back *observes* expert≈operation post-hoc in a pretrained LM but neither induces nor verifies it. Inducing operations with no operation labels, and verifying them, is the open part.

- `[2105.01119]` Iterated Learning for Emergent Systematicity (ICLR 2021) — compositional structure needs *explicit pressure*; our bottleneck + hard-negatives is that pressure — **adopt**.
- `[2509.10025]` SMoE-VAE — unsupervised routing finds structure beyond label boundaries — **adopt**.
- `[2603.08462]` Reasoning as Compression / Conditional IB — the formal name for our bottleneck — **adopt**.
- `[2509.20577]` DS-MoE / Dynamic Reasoning Chains — same target, but expert *types predefined* — **beat**.
- `[2406.12548]` P-React — operation-like experts tied to Big-Five *labels* — **beat**.
- `[2202.08906]` ST-MoE — left alone, specialization drifts to token-type/topic — **contrast**.

### A2 — Per-user routing that is *unsupervised AND read as cognitive structure*

Per-user routing infrastructure already exists and is, in representation, close to our `G_u`. The missing axis: existing per-user routers are trained on **supervised preference/domain** signal and aren't read as a cognitive-operation decomposition.

- `[2503.01658]` CoPL (ICLR 2025) — shared LoRA expert + individual experts, user-preference gate ≈ our Phase-2 `use_user=True` — **borrow-infra** / **beat** (supervised axis).
- `[2409.13931]` CoMiGS — per-user routing over generalist/specialist experts + an emergent-role analysis worth reusing — **borrow-infra**.
- `[2510.08256]` Mix-/MoE-DPO — gate-side personalization without retraining experts (multi-preference, not per-user-identity exclusive) — **borrow-infra**.
- `[2402.04401]` OPPU · `[2406.10471]` Per-Pcs — isolated-module pole — **contrast**.

### A3 — Corpus design that *isolates operation from topic* and forces composition

Most reasoning corpora announce the operation (logic-MC stem ceiling ~0.98) or use a single operation, so they never force separating *how* from *what*. This was the root cause of the 1a negative.

- `[2108.00573]` MuSiQue (TACL 2022) — anti-shortcut multi-hop (single-hop model −30 F1), intermediate-answer distractors, latent operations — **adopt** (the project's main compositional corpus).
- *(open)* An operation-vs-topic *isolation* corpus has no clean off-the-shelf instance; building one is part of the contribution surface.

## §B. What we are positioned to claim — a methodology gap the field calls open

### B1 — A *causal* battery that decides whether an expert is a reasoning OPERATION or a topic/shortcut

The field mostly shows "experts cluster by *something*" — usually topic/token-type (ST-MoE; POS-sensitivity). Rare is a *causal* test: lesion a candidate operation → selective accuracy drop only on problems that need it, plus operation-vs-topic disentanglement and motif-consistency. B1 is the tool that turns "experts specialized" into "experts specialized *by operation*" — a results-independent contribution candidate, framed as proof-of-success (not a "valuable either way" hedge).

- `[1909.03368]` Hewitt & Liang, control tasks / selectivity — **adopt**.
- `[2604.09780]` The Myth of Expert Specialization — mandatory geometry control — **adopt**.
- `[2604.02178]` Herbst et al., The Expert Strikes Back (ICML 2026) — peer-reviewed evidence that experts carry operations (feasibility, not proof of *emergence*) — **adopt**.
- `[2012.14913]` Geva (FFN key-value) · `[2304.14997]` Conmy (ACDC) — causal localization / patching methods — **adopt**.
- `[2009.02383]` Stuhr & Brauer, objective mismatch — names why a content-rewarding objective never yields an operation axis — **contrast** (diagnostic).

## §C. Borrow but do not claim

**Infrastructure (borrow-infra):** Chain-of-Experts `[2506.18945]` (sequential chaining — note: the project's *sequential* variant was falsified; this remains an architectural reference only) · N2NMN `[1704.05526]` · Stack-NMN `[1807.08556]` (operations-as-program lineage) · Sparse-MoE `[1701.06538]` · Switch `[2101.03961]` · DeepSeekMoE `[2401.06066]` · ReMoE `[2412.14711]` (routing substrate/math).

**Baselines to beat:** MiCRo `[2506.13331]` (supervised modular reasoning — the labeled twin) · P-React `[2406.12548]` · CoPL `[2503.01658]`.

**Contrast foils:** Lee 2025 (Nature Human Behaviour, flat belief vector) · OPPU `[2402.04401]` · Per-Pcs `[2406.10471]` · ST-MoE `[2202.08906]` / POS-sensitivity `[2412.16971]`.

**Adjacent — borrow the target/paradigm, not the method (borrow-target):** PB&J `[2504.17993]` (reasoning behind a user's judgment via psychological scaffolds + LM rationales — borrow the *target*, not the supervised-rationale method) · AI-psychometrics `[2505.08245]`/`[2406.17675]` (borrow the *psychometric-validity paradigm*; that field measures the *model's* psychology, so adapt the validation paradigm to per-user cognitive style).

## §D. The real target — the integration, and why B1 makes it count

Each single axis is borrowable: per-user mixtures (A2), operation-experts (A1/B1), unsupervised specialization (A1), composition substrate (A3), verification primitives (B1). What no line instantiates is the **conjunction**: A1 (unsupervised emergent operations) × A2 (a per-user activation distribution read as cognitive structure) × B1 (a causal test that the experts are operations), with **KG-as-output** as the deferred Phase-3 telos. This is "combination novelty" — the weakest kind on its own — which is exactly why the project doesn't lead with it. The load-bearing move is B1: the integration earns its keep only if the causal battery shows the combined design *buys something* a flat or supervised baseline does not.

### Positioning & evaluation — persona, not preference

The framing is *cognitive persona / user-modeling*, not preference-prediction personalization. Preference accuracy (LaMP, recommendation, preference data) is exactly where the project is weak — that turf belongs to supervised methods (CoPL/OPPU/P-React) that hold a preference signal the project doesn't. Read as persona/cognitive user-modeling, the strengths land on axes the field actually cares about, and the evaluation is set accordingly:

- **Primary headline = sample-efficiency / cold-start.** Mechanism learned once from the corpus; per-user cost is only `G_u`. If `G_u` matches per-user adapters at a smaller per-user data budget, that is the top-tier contribution — shown via a *controlled per-user data-budget head-to-head* (vs OPPU/Per-Pcs), not assumed.
- **Interpretability / comparability / auditability** — a `K`-dim activation distribution is comparable and auditable across users; opaque adapters/embeddings are not.
- **Evaluation axis = user-simulation / persona-faithfulness**; preference-accuracy is reported only as a secondary baseline. (Defining the user-simulation protocol is itself part of the Phase-2 work — it is less standardized than preference leaderboards.)
- **Field trajectory.** The field is moving from flat preference vectors toward psychology/reasoning-grounded persona (PB&J; AI-psychometrics). This project pushes one step further — an *emergent, comparable operation decomposition* — so it has an audience and is not empty (the sub-field is converging fast).

> One-line: persona/cognitive user-modeling is a timely bet with real strengths (efficiency, interpretability); "beating preference prediction" is a weak bet fought on a supervised field with an unbuilt mechanism. Same research, framing decides its fate.

---

*The full internal corpus (per-paper verdicts, ~38 cards, citation-check) lives in the private working repo; this is the public positioning synthesis. All arXiv IDs here were web-confirmed 2026-06-06.*
