# Research Journey — what I tried, what broke, and why

This document is the honest, detailed log behind the [README](../README.md). It walks through each experiment as it actually happened: the hypothesis, the setup, the result, and, most importantly, **what the result ruled out** and how it set up the next attempt.

The project's north star (see [`vision.md`](vision.md)): model user identity as a **distribution over shared interpretive operations** (`G_u`), learned without operation labels, with the knowledge graph as a *generated output* rather than an input. Every experiment below is an attempt to get **operation structure to emerge** under some learning pressure.

A recurring discipline I held throughout (the project's "IRON rules", see [`glossary.md`](glossary.md)):

- **No LLM in the training loop.** Everything is frozen-encoder embeddings + scoring, so any structure is attributable to the routing mechanism, not a language model's prior.
- **Operation-label-free / emergent.** No operation labels are fed in. The only supervision is the end-task answer; operation structure has to *emerge* under that pressure, or it doesn't count.
- **Real data only.** No synthetic toy corpora used as evidence.
- **No novelty claims.** Forced design + reported results. Negative results reported as negative.

---

## Phase 1 — the reconstruction cycle (CLOSED, informative negative)

**Hypothesis.** If I train a Mixture-of-Experts to *reconstruct its own input* (a "cycle": embedding → routed experts → reconstruct the embedding), will the experts spontaneously specialize into **interpretive operations**?

**Setup.**
- Frozen sentence encoder; ~62M trainable parameters in the MoE + decoder.
- 7 deliberately diverse text sources (Reddit, Pennebaker personality essays, PANDORA, PersonaChat, ROCStories, αNLI, SocialIQA).
- Sparse routing with a controllable number of active experts (`K_active`).
- Loss = reconstruct the input representation from its own routed encoding.
- 5-run ablation sweep over routing density and regularization, plus a no-MoE baseline (B0).

**Results.**

| run | epochs | K_active | recon (cos) | downstream (SimBench) | note |
|---|---|---|---|---|---|
| v3_minimal | 30 | 13.88 | 0.8864 | **0.7120** | dense routing; best downstream (narrow corpus, lucky alignment) |
| v4_diverse | 30 | 6.56 | 0.8786 | 0.7029 | 7-source; clusters form at **source level** |
| v5_arch | 30 | **1.00** | 0.8788 | 0.7067 | with orthogonality penalty → collapses to **K=1** (degenerate) |
| **v6_long** | 79 | 5.66 | 0.8789 | **0.6901** | paradigm-faithful, sharpest clusters, **lowest** downstream |
| b0 (no MoE) | 30 | — | 0.8687 | 0.7082 | baseline |

**What went wrong.**
- ✅ The MoE *did* beat the baseline on the reconstruction task itself (+1pp). The architecture works.
- ❌ But the expert clusters formed at the **source / format level** (narrative vs. discussion vs. finance), **not** at the operation level (causal vs. analogical vs. narrative reasoning). This is the classic ST-MoE "experts specialize by surface token-type" pattern.
- ❌ **Inverse trade-off:** the *sharper* the clustering, the *worse* the downstream task performance. A direct symptom of a **pretext/downstream objective mismatch**: the reconstruction objective optimizes something that doesn't transfer.
- ❌ A follow-up diagnostic ("Stage 1") confirmed the root cause structurally: even reading selectivity off the *raw frozen encoder*, **topic** alignment (0.58–0.97) dwarfed **operation** alignment (~+0.18). The signal the objective rewards is topic, full stop.

**The lesson that mattered.** Reconstruction is **agnostic to *how* text is processed**: it only cares about *what* the text is about. Two people who reason differently about the same topic produce similar-content text, so an operation-specialized expert is never *needed* to lower the loss. **Reconstruction-as-primary is self-defeating for an operation objective.**

This is what closed Phase 1 and motivated the central pivot: **stop reconstructing the input; start predicting an answer that can only be reached by reasoning.**

> A side note on a design trap found here: adding an orthogonality penalty between experts (intended to *encourage* diversity) instead drove the sparse gate to saturate at a single active expert (`K_active = 1`), a degenerate collapse. The penalty-free configuration was the paradigm-faithful one. Small regularizers can silently destroy the very structure you're trying to create.

---

## Phase 1.5 — the objective pivot

**The reframe.** Replace reconstruction (text → text) with **answer prediction** (question + passage → multiple-choice answer), structured so the answer can *only* be reached through the routed operations.

The mechanism that enforces this is a **3-way information bottleneck**:
1. **Question-only encoding.** Only the question is encoded into the routing input; the passage is never seen by the encoder/router.
2. **Passage as side-channel.** The passage enters only as key/value in the decoder's cross-attention.
3. **KG modulates, with no bypass.** The decoder is built so that *if the operation activation is zero, the output is zero*. There is no path for the passage to leak around the operation bottleneck.

That third point came from a **real bug discovered empirically**: an earlier design assumed "the cross-attention is undefined without a routing query, so the bottleneck is automatic." False. A zero query just makes attention *uniform*, which leaks the whole passage. (A run with zero active experts still scored 0.35, clear evidence of leakage.) The fix was the strict bias-free, no-residual, no-bypass modulation invariant. **Lesson: verify your bottleneck empirically; don't assume an architectural constraint holds.**

Target architecture: `K = 128` fine-grained experts, `K_active ≈ 4`, single reasoning domain (logic), with the goal that **fine-grained operation primitives emerge**.

### Stage 1a — flat operation router (CLOSED, weak-ceiling negative)

**Hypothesis.** With the new objective, do fine-grained operation primitives emerge in a *flat* mixture (no chaining yet)?

**Setup.** `K=128` experts, `K_active≈4`, frozen `e5-large-v2`, LogiQA 2.0 + ReClor (4-choice logic QA with hard distractors), MC cross-entropy, strict no-bypass bottleneck. PASS bar = operation selectivity beating **four** control baselines (random label, topic, token-type/length, geometry shuffle), an *absolute* bar, deliberately not relaxed.

**Result. ❌ Negative, but it localized the problem.** Operation selectivity sat barely above chance (~0.05). The diagnostic that mattered: on this corpus, the **question stem alone** predicts the answer-type at a 0.98 ceiling, while lexical matching is at chance (~0.27). In other words, **every problem announces its own single operation**. There is *no composition to discover*. The corpus is a **single-operation substrate**. You cannot observe operations *composing* if each problem only ever uses one.

This reframed the next step away from the *model* and toward the *data*: I needed a corpus where a problem genuinely requires **chaining multiple operations**.

### Stage 1b — sequential chain over a compositional corpus (CLOSED, negative)

**Corpus pivot to MuSiQue.** MuSiQue is a multi-hop QA dataset where each question requires **2–4 reasoning hops**, and, crucially, the **intermediate answers are hard distractors** (correct at their own hop, wrong as the final answer). I converted it to a 4-choice MC format and kept the logic-QA set as a **single-op control arm** to isolate the effect of compositional substrate. (Decomposition / hop-depth labels are available for analysis but never fed to the model.)

**Model.** A sequential **chain-of-experts**: an `L`-step chain (`L = 2–4`) that reuses the same expert pool at each step, with per-step modulation and an accumulating output, still under the strict no-bypass invariant.

**Result (confound-controlled run). ❌ Negative on sequential composition.**
- The **substrate is healthy**: the flat model reaches **~0.598 validation accuracy** on MuSiQue (well above chance), so the corpus and pipeline are sound.
- The control arm behaves correctly (hop structure empty where it should be).
- But the **sequential chain discovers no adaptive depth**: across problems the effective breadth stays flat (≈`{2,2,2}`), and the chain performs **no better than the flat mixture**. Depth-adaptive sequential composition **did not emerge**.

**What this rules out, and what it points to.** Sequential chaining (the "chain-of-thought" inductive bias) is **not** what emerges here. But, and this is the pivot, a **negative on *sequential* composition is not a negative on *parallel* composition.** The flat mixture already activates ~5 experts *simultaneously*. That simultaneous co-activation **is itself a distribution over operations**, which is *exactly* the `G_u` (per-user activation distribution) the project set out to model. The sequential chain was, in hindsight, a **detour**.

**Honest residual limitations of this run** (stated so they're not swept under the rug):
- The chain depth tested (`L=3`) is shallower than the deepest 4-hop problems.
- The **frozen** `e5` encoder caps multiple-choice accuracy around ~0.55, leaving little headroom, so the model may be **memorizing/overfitting** dataset artifacts rather than feeling genuine compositional pressure. The frozen-encoder constraint (an IRON rule that keeps results attributable) may itself be the deeper bottleneck. A future stage may need to relax it.

---

## Direction 1 — parallel co-activation (CURRENT)

**The reframe that 1b forced.** Stop modeling operations as a *sequential path*. Model them as a **simultaneously-activated distribution**: the flat mixture's co-activation pattern, read directly as `G_u`. (The 1b negative this pivot rests on was later reproduced from scratch and *strengthened* by closing the chain-depth confound; see [Reproduction & self-audit](#reproduction--self-audit--the-discipline-check) below.)

**How to test it (planned).** Use **causal lesioning** for specificity rather than relying on labels:
- Turn off experts of one putative operation type. Does accuracy drop **only** on problems that need that operation, and not on others? That's causal evidence an expert *is* an operation, not a topic detector.
- Measure **operation consistency** via shared activation motifs across problems with the same reasoning structure, controlling for topic.
- This connects forward to the project's **S1 falsification** (mechanistic universality: the same operation should produce the same activation signature across users).

The intervention/lesion/swap harness for this is already built and tested in [`../experiments/phase1_5/`](../experiments/phase1_5).

**The alternative branch** if Direction 1 doesn't hold: move to **Phase 2**, introducing a genuine **per-user** activation distribution `G_u` (so far all "users" share one `G`), per the vision document. Phase 2 is framed as **persona / cognitive user-modeling, not preference prediction**: its intended payoff is sample-efficiency / cold-start (mechanism learned once; per-user cost is only `G_u`) and an interpretable, comparable user representation, evaluated by user-simulation / persona-faithfulness — *not* by beating preference-accuracy baselines on their own turf.

---

## Reproduction & self-audit — the discipline check

Before building anything on top of these three negatives, I stopped to ask an uncomfortable question: *do I actually believe my own results?* A full self-audit of the experiment record gave a sobering answer: **the evidence was thin.** Most of the numbers above lived only in prose (notes, READMEs, slides); **9 of 10 experiment notebooks were unexecuted skeletons** with no committed metric artifacts. The *plans and code* were solid (real unit tests, the no-bypass invariant actually enforced), but "I ran this and got *X*" was, in too many places, unverified.

So I closed the gap the only honest way: I built a single **one-click reproduction notebook** (`REPRODUCE_ALL.ipynb`) that re-trains every experiment **from scratch**, recomputes every reported number, and bakes the result into one artifact, `out/VERIFICATION.json`, that places each *documented* value next to its *freshly-measured* one. Then I ran the whole thing on a Colab A100.

**The verdict: all three pivot negatives reproduced, and the most important one got stronger.**

| claim | documented | measured (from scratch) | verdict |
|---|---|---|---|
| Phase 1 (Engine-A) F3 operation selectivity (adj_op) | 0.176 | **0.176** (ceiling 0.596, verdict FAIL) | ✅ reproduced exactly |
| 1a operation gate / σ-gates | weak, below ceiling | all σ-gates **FAIL**; gates 0.23 > 0.20 > 0.18 < ceiling 0.465 | ✅ weak-ceiling reproduced |
| 1b sequential depth (breadth) | `{2,2,2}`, non-monotone | `chain_steps=4` × **3 seeds** → all `{1,1,1}`, non-monotone | ✅✅ confound rebutted, **negative strengthened** |
| 1b intervention selectivity (col_spec) | ~0.05 (weak) | 0.0395 ≪ chance 0.14 | ✅ no operation specialization |
| SimBench — was it just a base-rate ceiling? | (open worry) | random 0.30 / majority 0.38 / model **0.70** | ✅ +32pp, not a ceiling |
| "+1pp over no-MoE on reconstruction" | superiority | every run converges to ~0.867 → **+0.0** | ⚠️ **did not reproduce** |

A few things worth calling out honestly:

- **The 1b result is now better-supported than before.** The audit's single biggest worry was a confound: 1b had only ever tested a 3-step chain on problems that can need 4 hops, which could *structurally* bias the outcome toward "no depth." The reproduction ran `chain_steps=4` across three seeds, and the chain *still* collapsed to `{1,1,1}` every time. The confound is rebutted on its own terms; the negative held. That is precisely what **licenses** the Direction 1 pivot rather than undermining it.
- **The SimBench worry is resolved.** A reasonable critic could have said "0.70 sounds high, but maybe the task is trivial." With random at 0.30 and a majority-class baseline at 0.38, the model's 0.70 is a real +32pp, not a ceiling artifact.
- **One claim weakened, and I'm keeping it on the record.** The old "+1pp over the no-MoE baseline on reconstruction" did **not** reproduce: every run converged to essentially the same reconstruction floor (~0.867), independent of architecture. The "+1pp" was noise. (Running the notebook also surfaced two embarrassing-but-useful bugs that only an *executed* notebook could catch: two runs I'd treated as distinct, `v3` and `v4`, were the same configuration up to a run-id; and a later change to the default corpus had silently been crashing the Phase 1 evaluation probe.)

The meta-point: the most valuable thing this pass produced wasn't a new result. It was **earned trust in the old ones**, plus a reproducible path for anyone (including future me) to re-check them in a single run.

---

## Summary of what each negative bought

| Experiment | Falsified | Bought |
|---|---|---|
| Phase 1 (reconstruction) | "Reconstruction yields operation specialization" | The objective pivot to answer-prediction + the information bottleneck |
| 1a (flat, logic-QA) | "Operations emerge on any reasoning corpus" | Located the failure in the **corpus** (single-op substrate) → move to multi-hop |
| 1b (sequential chain, MuSiQue) | "*Sequential* chaining is the right inductive bias" | Pointed back to **parallel co-activation** = the original `G_u` formulation |

The shape of the project is a sequence of **well-designed eliminations**, each of which was **independently reproduced from scratch** (2026-06-04, see above). None of these produced a triumphant positive result, and that's an honest description of a real research process. The value is in *which* hypotheses got cleanly ruled out, and why.

---

*For the formal architecture and the broader paradigm, see [`vision.md`](vision.md). For positioning against prior work (~38 papers), see [`literature-review.md`](literature-review.md). For domain terms and the project's invariants, see [`glossary.md`](glossary.md).*
