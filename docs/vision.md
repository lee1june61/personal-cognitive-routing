# Project Vision: Structural Personalization via Per-User MoE Activation (v2, 2026-06-06 lean rewrite)

> ℹ️ **Reproduced working note.** This mirrors the project's internal paradigm single-source. It is the *lean* current version — earlier-era material (cycle-reconstruction-primary, K=8–12, contrastive-MI S1, MoKGR/MixRAG framing) has been archived as stale. Public entry points: [README](../README.md), [research-journey](research-journey.md), [glossary](glossary.md), [literature-review](literature-review.md).

---

## 0. Research Direction (clean-bet statement)

> *Proposal-grade. A forward bet, not a contribution claim — the success case is the baseline, the risk is stated once as the open scientific question, and the verification method is the tool that would demonstrate success.*

I want to find out whether the way a person interprets information can be modeled as a structured, emergent decomposition: a small set of reusable reasoning operations, discovered without labels, that different people recruit in different combinations. If that holds, it gives a structural alternative to how personalization works today, where a user is a single embedding or an isolated adapter. Identity becomes a distribution over shared operations, and the same facts yield different interpretations because different operations fire. It also offers a handle on the reasoning structure inside a model that doesn't depend on hand-defined skill taxonomies.

The hypothesis I want to establish first sits underneath all of that: do interpretive operations emerge at all, unsupervised, when a model is trained only to predict answers? I train a mixture-of-experts under an architecture that separates *how* from *what*. The operation structure is encoded from the question alone, while the supporting facts enter only as a side-channel, so the structure that forms is about reasoning rather than topic. To show it is real rather than an artifact, I verify it causally: lesioning a candidate operation should degrade accuracy only on the problems that require it, and problems sharing a reasoning pattern should recruit the same structure. The closest existing work either supervises these operations with trait or preference labels (P-React; CoPL; Mix-/MoE-DPO) or predefines the reasoning experts by hand (DS-MoE). The open question is whether they can emerge on their own, which is what my current experiments test.

If they do, the next step is the one the whole direction is for: conditioning that activation distribution on the individual, to get a comparable and structural account of how different people construe the same information, and eventually of how groups do.

---

## 1. Core claim

Mainstream LLM personalization treats user identity as a *whole object* — a flat embedding (Lee 2025, LaMP) or an isolated per-user PEFT module (Per-Pcs, OPPU). Some recent work (CoPL, CoMiGS, Mix-/MoE-DPO) already uses a *shared-expert + per-user mixture* representation, but its routing axis is **supervised preference/domain**. The claim here is that user identity is **structurally decomposable**: a *shared* pool of interpretive operations + a **per-user, context-conditional** activation distribution `G_u`. **The difference is not the architecture but the *axis*** — unsupervised interpretive operations (no operation labels) with the knowledge graph as a generated *output*.

- **Facts = shared input**: every user sees the same facts; each fact carries a latent cognitive-context signal (analytic / narrative / social / recall mode).
- **Interpretation = per-user representation**: which interpretive operation a user activates on the same fact differs per user (and per context). User identity = the conditional distribution of *which primitive fires in which context* — not a static vector.

```
[fact corpus] → [shared K-expert pool + gate] → [G_u activation distribution] → [user-specific interpretation] → [observable text / answer]
                (mechanism learned from whole corpus)   (per-user)                  (KG = cognitive output)        (training signal)
```

## 2. Forced design — the C0→C3 deduction

The four core decisions are *deduced* from a single commitment, not stacked (no novelty claim):

- **[C0]** User identity = shared interpretive operations × per-user activation. *The one premise.* (Per-Pcs/OPPU's isolated PEFT has no "shared operations" axis.)
- **[C0]→[C1]** the per-user activation distribution = **G_u** (a distribution over the shared pool). MoE is only the implementation substrate for `G_u`, not the starting point.
- **[C0]→[C2]** an operation's product = a cognitive output ⟹ **the KG is an output, not an input**. (Edge selection over an external KG was the v1 mistake.)
- **[C2]→[C3]** if the KG is an output, there is no external ground-truth KG ⟹ supervised KG training is blocked ⟹ **the learning signal can only come from observable text**. (The Phase-1.5 instance is answer-prediction.)
- **[C1]+[C2]→[S1]** a falsifiable prediction: when the same expert `k` is active for two different users, the formal signature of what it generates should be similar across them. If not, the "universal mechanism" reduces to `K` user-specific subnetworks and the design is falsified.

**Component falsification** (remove one and the chain breaks): drop per-user `G_u` → C1 collapses (reduces to generic KG construction); drop KG-as-output → C2 collapses (user becomes a selector, v1); drop the observable-text signal → C3 collapses (no training possible); drop S1 → the framing makes no falsifiable prediction (the "shared" claim becomes decorative).

## 3. Current architecture (Phase 1.5, direction 1)

- **Objective = answer-prediction** (`question → operation-KG → answer-text`). Cycle reconstruction was the Phase-1 attempt and was falsified (it rewards topic, making the operation objective self-defeating).
- **Three-way information bottleneck (enforced structurally)**: (1) question-only encoding, (2) passage as a decoder side-channel, (3) KG modulation with no bypass (`kg=0 ⟹ output=0`).
- **K = 128** experts (`K_active ≈ 4`), frozen `e5-large-v2`, multiple-choice contrastive (no LLM in the loop).
- **Direction 1 = parallel co-activation**: the flat mixture's simultaneously-active distribution (`K_active ≈ 5`) *is* the `G_u` from C1. (Sequential chain / emergent-NMN was tested and is NEGATIVE as of 2026-06-01.)
- **S1 verification = a causal lesion battery + motif-consistency** (selective lesion + operation-vs-topic disentanglement + motif). Contrastive-MI (MoMoK ExID) was dropped as architecturally trivial.
- **Corpus** = MuSiQue (multi-hop → 4-choice MC, intermediate-answer distractors) as primary + LogiQA/ReClor as a single-op control arm.

## 4. Phases & falsifiers

| Phase | Status | Essence | Falsifier |
|---|---|---|---|
| Phase 1 (recon-cycle) | **closed (informative negative)** | reconstruction rewards topic → operations don't emerge (structural F3) | (closed) → motivation for Phase 1.5 |
| **Phase 1.5** (operation-axis) ★ active | answer-prediction + bottleneck + K=128, **direction-1 parallel co-activation** | operations emerge unsupervised **and** the causal battery shows they are operations (≠ topic): substrate ✓ → co-activation → causal-lesion gate |
| Phase 2 (personalization) | deferred (after 1.5) | per-user `G_u`, `use_user=True` | **persona, not preference**: headline = **sample-efficiency / cold-start** (controlled per-user data budget vs OPPU/Per-Pcs) + user-simulation / persona-faithfulness + interpretability (comparable `G_u`); preference-accuracy is a secondary report only. **S1 lives here.** |
| Phase 3 (KG readout) | future telos | latent → symbolic | — (least-evidenced axis, deferred) |

(Residual risk: the frozen `e5` encoder caps MC accuracy near ~0.55, so the model may be memorizing rather than feeling compositional pressure — the frozen-encoder rule may itself be the deeper bottleneck.)

## 5. Positioning

- **Persona / cognitive user-modeling**, NOT preference-prediction personalization. Evaluation axis = user-simulation / persona-faithfulness + sample-efficiency + interpretability; the project is *not* pitched on preference-accuracy leaderboards.
- **Three classes of personalization prior** (flat / isolated adapter / shared-expert-per-user-mixture); the difference is the *axis* (unsupervised interpretive operation + KG-output), not the architecture.
- **No novelty claim**: every single pillar is occupied; what is underexplored is the *combination* (the weakest kind of novelty). The position is "an underexplored, well-posed question + a results-independent verification method," one step ahead of the field's trajectory from flat preference vectors toward psychology/reasoning-grounded persona (PB&J; AI-psychometrics). Full graded positioning: [literature-review](literature-review.md).

## 6. What this is / is not

**Is**: structural personalization (shared ops × per-user `G_u`); facts/interpretation separated; KG = cognitive output; expert = learnable interpretive operation; answer-prediction with no operation labels; explicit S1 causal falsification; persona / cognitive user-modeling.

**Is not**: a per-user separate KG; edge selection over an external KG; a relation-type expert taxonomy; general-purpose KG construction; RAG/recommendation; preference-filtering personalization.

---

*Cognitive-science footnote (not load-bearing)*: the decomposition principle descends from the mechanistic-decomposition tradition (Bechtel/Craver/Fodor); the axis this project adds is a *per-user* activation distribution. It is isomorphic to Kelly's (1955) person-as-constructor idea, but solving PCT's measurement problem is *not* a goal here.
