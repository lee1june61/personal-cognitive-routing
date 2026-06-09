# Cognitive Routing over a Universal Knowledge Graph

> ⚠️ **Live repo — under active refactor (June 2026).** This is an ongoing research project, not a finished artifact. The code is being reorganized (experiments are being archived/restructured as the framing sharpens), so some notebooks or imports may not run cleanly mid-refactor. The research narrative below is kept current; the code layout is still settling.

> Can we model *who a user is* not as a single vector, but as a **distribution over shared interpretive operations**: the reusable ways of reasoning that everyone draws from, but that each person activates differently depending on context?

This repository is a research log of that question. It is an **honest research project**: most of what is recorded here is a sequence of carefully designed experiments that **did not work the way I hoped**, what each negative result ruled out, and how it reshaped the next attempt. The negative results are the point. They are where the actual learning happened.

If you only read one thing after this page, read **[`docs/research-journey.md`](docs/research-journey.md)**, the full "what I tried, what broke, and why" narrative.

---

## 1. The vision

Most LLM personalization treats a user as a **whole, monolithic object**:

- a single flat **embedding** (a point in space, e.g. Lee et al. 2025), or
- an **isolated per-user module** (one fine-tuned adapter per person, e.g. Per-Pcs, OPPU).

Both miss something basic about people. The *same* person reasons differently in different contexts: analytically about a contract, narratively about a memory, socially about a friend. And *different* people often share reasoning patterns. A flat vector can't express "this user, in this context, leans on causal reasoning"; an isolated per-user adapter can't express what two users have *in common*.

**The hypothesis of this project** is that user identity is *structurally decomposable* into two parts:

1. **A shared pool of interpretive operations**: `K` learnable "experts," each ideally capturing a reusable reasoning primitive (comparison, causal inference, narrative reconstruction, …). These are *universal* and learned from data, shared across all users.
2. **A per-user, context-conditional activation distribution `G_u`**: given an input, which operations fire, and how strongly. The *same* operation pool, routed differently per user and per context.

The knowledge graph (KG) is treated as a **cognitive output, not an input**: it is *generated* by combining the input with the activated operation pattern. This keeps the operation axis **label-free** (no external KG, no operation labels): the only training signal is the observable answer, so any operation structure has to **emerge** from that pressure rather than being handed to the model.

```
  input text
      │
      ▼  frozen encoder (no LLM in the loop)
  representation
      │
      ▼  Mixture-of-Experts router  ──►  activation distribution  G_u
  (K shared "operation" experts)         (which ops fire, how strongly)
      │
      ▼  combine input × activated ops
  generated KG / answer  ────────────►  training signal
```

Architecturally this is a **sparse Mixture-of-Experts** whose gate learns both input-driven routing and, eventually, user-conditional modulation. Same input, different routing → different operation path → different output. For the full paradigm and architecture, see [`docs/vision.md`](docs/vision.md).

---

## 2. What's new, and what isn't

To be upfront: the building blocks here are not new, and this project claims no new method and no state-of-the-art result. Mixture-of-experts routing, generating structure from text, unsupervised expert specialization, and even *per-user* expert routing all already exist. The closest neighbors are worth naming precisely, because each one comes close on a single axis:

| Closest work | Shares with this project | Differs on |
|---|---|---|
| P-React | a person mapped to a routing distribution over experts | supervised by personality (Big Five) labels; experts predefined; a trait, not an interpretive operation |
| CoPL, MoE-DPO, CoMiGS | per-user routing over a (partly) shared expert pool | the routing axis is preference or domain, and supervised; experts are capacity modules, not operations |
| Depth-specialized reasoning MoE | reasoning operations as experts, composed across steps | experts defined by hand; not per-user; not unsupervised |
| Unsupervised sparse-MoE specialization | expert structure emerging without labels | not personalization; experts not read as interpretive operations |

What is left is a specific combination none of them occupy: interpretive operations (how a person reasons, not their preferences or traits), discovered *without operation labels*, treated as a generated *output*, and conditioned per user, with a falsifiable test that the shared operations behave consistently across users (S1). The question is therefore not "what is this user like?" but *which operations does a person apply, and when?*

Two things keep this short of a novelty claim, and neither is "someone already did it." First, a new combination of known parts is the weakest kind of novelty; its worth depends on the combination doing something the neighbors cannot. Second, that has not been shown yet: the central mechanism, operations emerging without operation labels, has not held up in the experiments below. So this repository is best read as a falsifiable *question* under investigation rather than a finished contribution. (Reconstruction was the first attempt and was falsified in Phase 1; answer-prediction replaced it; whether interpretive operations emerge at all is still open.)

Two adjacent results frame the bet precisely — and honestly, including the part that cuts against it. Herbst et al. (2026, *The Expert Strikes Back*) show experts *can* carry fine-grained operations — but **observed post-hoc in large pretrained MoEs**, and they do *not* claim the structure *emerges* from a training objective. And *Iterated Learning for Emergent Systematicity* (2021) found that purely unsupervised compositional emergence was "thus far infeasible" without some supervision. Together they make this **worth trying, not proven**: the *existence* of operation-structured experts is real, but the project's specific bet — that it **emerges from answer-prediction alone, with no operation labels** — is exactly the open, headwind-facing part. No cited paper protects that bet; that is what the experiments are for.

> **S1 (mechanistic universality).** If the same operation is active for two different users, the formal signature of what it generates should be similar across them. If not, the "shared mechanism" is really `K` user-specific subnetworks, and the decomposition is wrong. (A Phase 2 test; the per-user axis isn't built yet.)

What the project does offer is a clearly-stated and falsifiable hypothesis, a design whose parts follow from a single commitment rather than being assembled ad hoc, a battery for separating genuine operation structure from topical shortcuts, and an honest record of what has been ruled out. The broader survey of prior work is in [`docs/literature-review.md`](docs/literature-review.md).

---

## 3. Phase 1 — the reconstruction cycle *(deprecated, informative negative)*

### The questions we broke it into

The guiding question: *can interpretive operations emerge if an MoE is trained to reconstruct its own input?* When the first run failed, the failure was localized with a three-branch diagnostic ladder (F1 → F2 → F3):

- **F1 (measurement).** Was the operation signal simply *cut off* by probe truncation in evaluation?
- **F2 (encoder).** Can the frozen encoder *even encode* operation structure at all?
- **F3 (objective).** Does the reconstruction objective *itself* let topic/content drown out operation?

### Experiments, results & interpretation

A 5-run ablation over routing density and regularization, plus a no-MoE baseline (full setup in [`docs/research-journey.md`](docs/research-journey.md)):

| run | epochs | K_active | recon (cos) | downstream (SimBench) | note |
|---|---|---|---|---|---|
| v3_minimal | 30 | 13.88 | 0.8864 | **0.7120** | dense routing; best downstream |
| v4_diverse | 30 | 6.56 | 0.8786 | 0.7029 | 7-source; clusters form at **source level** |
| v5_arch | 30 | **1.00** | 0.8788 | 0.7067 | orthogonality penalty → collapses to **K=1** (degenerate) |
| **v6_long** | 79 | 5.66 | 0.8789 | **0.6901** | paradigm-faithful, sharpest clusters, **lowest** downstream |
| b0 (no MoE) | 30 | — | 0.8687 | 0.7082 | baseline |

- **F1 and F2 were rejected.** Fixing the probe didn't help, and read straight off the *raw frozen encoder*, operation alignment was real but small (~+0.18).
- **F3 was the structural cause.** Topic alignment (0.58–0.97) dwarfed operation alignment in *every* configuration. The experts clustered by **source / format** (narrative vs. discussion vs. finance), not by operation, the classic "experts specialize by surface token-type" pattern. And the **sharper** the clustering, the **worse** the downstream task, a direct symptom of a pretext/downstream **objective mismatch**.

**Interpretation.** Reconstruction is **agnostic to *how* text is processed**: it only cares about *what* the text is about. Two people who reason differently about the same topic produce similar-content text, so an operation-specialized expert is never *needed* to lower the loss.

> A design trap found here: an orthogonality penalty meant to *encourage* expert diversity instead drove the sparse gate to saturate at a single active expert (`K_active = 1`). The penalty-free configuration was the paradigm-faithful one. Small regularizers can silently destroy the structure you're trying to create.

### Why we pivoted to Phase 1.5

Reconstruction-as-primary is **self-defeating for an operation objective**. The fix: **stop reconstructing the input; start predicting an answer that can only be reached by reasoning.**

---

## 4. Phase 1.5 — the operation-axis objective *(current)*

### The questions we broke it into

**(a) The information bottleneck: three design questions.** The new objective (question + passage → multiple-choice answer) is only meaningful if the answer *must* flow through the routed operations. Three structural constraints enforce that:

1. **Question-only encoding.** Only the question is encoded into the router; the passage is never seen by the encoder.
2. **Passage as a decoder side-channel.** The passage enters only as key/value in cross-attention, never as learned KG content.
3. **KG modulates, with no bypass.** If the operation activation is zero, the output is zero. (This came from a *real bug*: an earlier design assumed a zero routing query made attention "undefined"; in fact it makes attention *uniform*, leaking the whole passage, and a zero-expert run still scored 0.35. The fix was a strict bias-free, no-residual modulation invariant. **Lesson: verify your bottleneck empirically; don't assume an architectural constraint holds.**)

**(b) The sequencing: 1a → 1b → 1c.** One component at a time:

- **1a:** do fine-grained operation primitives emerge *at all*, in a *flat* mixture (no chaining)?
- **1b:** do *composed*, multi-step operation programs emerge, and do their motifs stay consistent (S1)?
- **1c:** does tree/DAG composition help? *(deferred.)*

The PASS bar was deliberately **absolute**: operation selectivity had to beat **four** controls (random label, topic, token-type/length, geometry shuffle), not just a relaxed ceiling.

### Experiments, results & interpretation

- **1a (flat, logic-QA: LogiQA 2.0 + ReClor) → weak-ceiling negative.** Selectivity sat barely above chance (~0.05). The diagnostic that mattered: the **question stem alone** predicts the answer-type at a 0.98 ceiling. **Every problem announces its own single operation.** There is *no composition to discover*. The corpus is a **single-operation substrate**. This reframed the next step away from the *model* and toward the *data*.
- **1b (sequential chain over MuSiQue) → negative on sequential composition.** MuSiQue is multi-hop QA (2–4 hops) where intermediate answers are **hard distractors** and operations are *latent* (not announced). The **substrate is healthy** (the flat model reaches **~0.598** validation accuracy), but the sequential chain **discovers no adaptive depth** (effective breadth stays flat, `{2,2,2}`) and performs no better than the flat mixture.

**★ Reproduction & self-audit (2026-06-04).** Before building on these negatives, I audited my own evidence and found a gap: most numbers were recorded as prose, and most notebooks were unexecuted skeletons. So I built a one-click reproduction notebook ([`REPRODUCE_ALL.ipynb`](REPRODUCE_ALL.ipynb)) and re-ran every experiment from scratch on an A100. **All three negatives reproduced, and 1b got *stronger*:**

| claim | documented | measured (re-run) | verdict |
|---|---|---|---|
| Phase 1 (Engine-A) F3 selectivity (adj_op) | 0.176 | **0.176** (verdict FAIL) | ✅ reproduced exactly |
| 1a operation gate | weak, < ceiling | all σ-gates **FAIL**, op_gate < ceiling | ✅ weak-ceiling reproduced |
| 1b sequential depth | `{2,2,2}` non-monotone | `chain_steps=4` × **3 seeds** → all `{1,1,1}` non-monotone | ✅✅ confound rebutted, negative **strengthened** |
| SimBench base-rate (was it just a ceiling?) | — | random 0.30 / majority 0.38 / model 0.70 | ✅ **+32pp**, not a ceiling |
| "+1pp over no-MoE on reconstruction" | superiority | all runs converge ~0.867 → **+0.0** | ⚠️ **did not reproduce** (claim weakened) |

The 1b row is the important one: the earlier audit's top worry was that testing only `chain_steps=3` on 4-hop problems *structurally biased* the result toward negative. Re-running at `chain_steps=4` across three seeds **rebutted that confound directly**. The negative held, and is now better-supported.

### Remaining tasks

**Direction 1: parallel co-activation.** A negative on *sequential* composition is **not** a negative on *parallel* composition. The flat mixture already activates ~5 experts *simultaneously*, and that simultaneous co-activation **is itself a distribution over operations**: exactly the `G_u` the project set out to model. The sequential chain was, in hindsight, a detour. The plan is to test parallel co-activation directly via **causal lesioning** (turn off one putative operation; does accuracy drop *only* on problems that need it?) plus motif-consistency, which connects forward to the **S1** falsifier. The intervention/lesion/swap harness is already built and tested in [`experiments/phase1_5/`](experiments/phase1_5).

**Honest residual limitations:** the tested chain depth (`L=3`) is shallower than the deepest 4-hop problems; the **frozen** `e5` encoder caps MC accuracy around ~0.55, leaving little headroom, so the model may be *memorizing* artifacts rather than feeling genuine compositional pressure. The frozen-encoder rule (which keeps results attributable) may itself be the deeper bottleneck.

---

## 5. Phase 2 — a genuine per-user distribution *(planned)*

Everything above shares a **single** `G` across all "users." This tests the *architecture's* value, not personalization. Phase 2 adds the per-user axis: a true per-user activation distribution `G_u` plus context-conditional modulation `λ_u`, isolated as a minimal `use_user=True` ablation over the same architecture.

This is framed as **persona / cognitive user-modeling, not preference prediction** — the project models *how* a person reasons about the same facts, not *what* they prefer. So Phase 2 is not pitched on preference-accuracy leaderboards (that is the supervised-signal field's turf); its intended value is on axes the field actually cares about:

- **Headline = sample-efficiency / cold-start.** The mechanism is learned once from the whole corpus; per-user cost is only the activation distribution `G_u`. If `G_u` matches a per-user adapter (OPPU/Per-Pcs) at a *smaller* per-user data budget, that is the top-tier contribution — to be shown with a controlled per-user data-budget head-to-head, not assumed.
- **Interpretable + comparable.** A `K`-dim activation distribution is comparable, auditable, and controllable across users; an opaque adapter or embedding is not.
- **Evaluated by user-simulation / persona-faithfulness**, with preference-accuracy reported only as a secondary baseline.
- **Entry criterion:** Phase 2 only begins once the architectural value (Phase 1/1.5) is established, since otherwise per-user gains can't be attributed.
- **The falsifier (S1) lives here:** the same operation expert, active across different users, must produce a similar sub-KG signature. If it doesn't, the "universal mechanism" reduces to `K` user-specific subnetworks and the design is falsified. This is what makes the framing load-bearing rather than decorative.

---

## Summary — the `f(task, person)` map

Every experiment here is one cell in a single verification grid: *which subgraph
activates* is modeled as `f(task, person)`, and the **task axis (anonymous) is
validated before the person axis (per-user)**. The full frame, including the fixed
PASS bar, is in [`docs/verification-framework.md`](docs/verification-framework.md).

| Axis | Differentiates? | Manner ≠ topic? | Structural / stable? |
|---|---|---|---|
| **Task** (anonymous — done / active) | T1 — **weak** (splits by topic, tends to collapse) | T2 — ★ **open blank** (the target; cleanly unmeasured) | T3 — *sequential* **negative**; *parallel* unmeasured |
| **Person** (per-user — Phase 2, deferred) | P1 — deferred | P2 — deferred (the **S1** falsifier lives here) | P3 — deferred (intended headline: sample-efficiency) |

Substrate (L0) is the precondition for the whole task row, and it **passes**: MuSiQue
forces genuine multi-hop reasoning (~0.598 flat-model accuracy, ≫ chance). What each
negative bought, read through that grid:

- **Phase 1 (reconstruction)** falsified "reconstruction yields operation specialization" → bought the pivot to answer-prediction + the information bottleneck.
- **1a (flat, logic-QA)** falsified "operations emerge on any reasoning corpus" → located the failure in the **corpus** (single-op substrate), moving the work to multi-hop.
- **1b (sequential chain, MuSiQue)** falsified "*sequential* chaining is the right inductive bias" → pointed back to **parallel co-activation**, which is a *hypothesis shift*, not a claim that parallel is the answer.

All three negatives were **independently reproduced from scratch on 2026-06-04** (`REPRODUCE_ALL.ipynb`). None produced a triumphant positive result, and that is an honest description of a real research process. The value is in *which* hypotheses got cleanly ruled out, and why.

---

## Tech stack

- **Language / framework:** Python, PyTorch
- **Encoder:** frozen sentence encoder (1024-dim), with **no LLM in the training loop** (everything is embedding-level + multiple-choice scoring). Phase 1 uses `BGE-large-en-v1.5`; Phase 1.5 switches to `e5-large-v2` (chosen for a higher raw operation-alignment ceiling in diagnostics), with `BGE-large-en-v1.5` retained as an encoder-swap ablation
- **Router:** ReMoE-style gate (ReLU experts + an adaptive L1 controller targeting a small active set, `K_active ≈ 4`)
- **Decoder:** gated low-rank hypernet modulation with a strict **no-bypass** invariant (if the operation activation is zero, the output is zero, so the passage can't leak around the bottleneck)
- **Objective:** multiple-choice cross-entropy (contrastive over 4 candidates)
- **Datasets:** Phase 1 uses 7 diverse text sources (Reddit, Pennebaker, PANDORA, PersonaChat, ROCStories, αNLI, SocialIQA); Phase 1.5 uses LogiQA 2.0 + ReClor (single-op control) and **MuSiQue** (multi-hop / compositional)
- **Engineering hardening:** OOM-safe per-token expert batching, Herfindahl load-balancing, chance-normalized selectivity probes with control baselines, causal lesion/swap intervention harness, all test-driven (≈280 tests across `core/` + `experiments/`)

---

## Repository structure

```
personal-cognitive-routing/
├── README.md                     ← you are here
├── REPRODUCE_ALL.ipynb           ★ one-click from-scratch reproduction of every experiment
├── docs/
│   ├── research-journey.md       ★ the full "what I tried / why it failed" story
│   ├── verification-framework.md ★ the f(task, person) grid every experiment is read through
│   ├── vision.md                 the paradigm & architecture in depth
│   ├── glossary.md               domain terms + the project's invariants ("IRON rules")
│   ├── literature-review.md      annotated positioning against ~38 papers
│   └── reading-list.md           curated paper list by role
├── core/                         shared building blocks (single definition, no duplication)
│   ├── encoders.py               frozen per-token encoder (OOM-safe caching)
│   ├── routers.py                ReMoE-style router (ReLU gate + adaptive L1)
│   ├── shared_heads.py · loss_primitives.py · load_balance.py · attn_mask.py
│   └── eval_core.py              selectivity / geometry-control / chance-rate probes
└── experiments/
    ├── _archive/                 archived experiments (kept for provenance, not active)
    │   ├── 2026-06-08_recon_cycle/    Phase 1 reconstruction cycle (closed negative)
    │   └── 2026-06-08_seq_chain_1b/   1b sequential orchestrator (setup-failure, layout only)
    └── phase1_5/                 current work — flat operation router + multi-hop substrate (live)
        ├── README.md · *.py      model, MuSiQue loader, intervention harness, ablations
        ├── notebooks/            Colab run drivers (01–04; 1b notebook archived)
        └── tests/
```

Code is split along **shared vs. experiment-specific**: `core/` holds what every
experiment reuses (one definition each — encoder, router, loss, eval probes);
`experiments/phase1{,_5}/` keep their own model/training/data, which genuinely differ
(reconstruction vs. multiple-choice). Imports are prefix-free (`from core.…`,
`from experiments.…`) so the tree runs identically locally, on Colab, and here.

**Where to start reading:** this page → [`docs/research-journey.md`](docs/research-journey.md) → [`docs/verification-framework.md`](docs/verification-framework.md) for the `f(task, person)` frame → [`docs/vision.md`](docs/vision.md) for the formal paradigm → `experiments/` for the code.

## Running the code

The experiments were run on Google Colab (GPU) with embedding caches and checkpoints kept off-repo (they are large and regenerable). The code is organized as importable packages with a test suite you can run locally:

```bash
pip install torch numpy datasets sentence-transformers scikit-learn pytest
# run from the repo root so `core` and `experiments` resolve as packages:
python -m pytest experiments/phase1_5 -q   # ~200 unit tests, no GPU (pytest.ini ignores _archive)
```

**Full reproduction.** [`REPRODUCE_ALL.ipynb`](REPRODUCE_ALL.ipynb) re-runs every experiment from scratch and writes every number to a single `out/VERIFICATION.json` (documented value vs. freshly-measured value). Upload this repo to Google Drive, open the notebook in Colab, set the `BASE` variable at the top to the repo's root folder (the directory containing `core/` and `experiments/`), and Run-All. A GPU is required; sections are independently guarded so a partial run still produces partial results.

Heavy artifacts (`*.npy`, `*.parquet`, checkpoints, the generated `out/` tree, reference PDFs) are intentionally git-ignored, since they rebuild from the data loaders and training code.

---

*This is a personal research project, shared as a portfolio piece. It documents an ongoing line of inquiry, negative results included.*
