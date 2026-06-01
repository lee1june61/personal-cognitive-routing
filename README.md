# Cognitive Routing over a Universal Knowledge Graph

> Can we model *who a user is* not as a single vector, but as a **distribution over shared interpretive operations** — the reusable ways of reasoning that everyone draws from, but that each person activates differently depending on context?

This repository is a research log of that question. It is an **honest research project**: most of what is recorded here is a sequence of carefully designed experiments that **did not work the way I hoped**, what each negative result ruled out, and how it reshaped the next attempt. The negative results are the point — they are where the actual learning happened.

If you only read one thing after this page, read **[`docs/research-journey.md`](docs/research-journey.md)** — the full "what I tried, what broke, and why" narrative.

---

## The problem

Most LLM personalization treats a user as a **whole, monolithic object**:

- a single flat **embedding** (a point in space), or
- an **isolated per-user module** (one fine-tuned adapter per person).

Both miss something basic about people. The *same* person reasons differently in different contexts — analytically about a contract, narratively about a memory, socially about a friend. And *different* people often share reasoning patterns. A flat vector can't express "this user, in this context, leans on causal reasoning"; an isolated per-user adapter can't express what two users have *in common*.

**The hypothesis of this project:** user identity is better described as a **distribution over which shared interpretive operations a person activates**, conditioned on the situation.

## The core idea

Decompose personalization into two parts:

1. **A shared pool of interpretive operations** — `K` learnable "experts," each ideally capturing a reusable reasoning primitive (comparison, causal inference, narrative reconstruction, …). These are *universal* and learned from data, shared across all users.
2. **A per-user, context-conditional activation distribution `G_u`** — given an input, which operations fire, and how strongly. The *same* operation pool, routed differently per user and per context.

The knowledge graph (KG) is treated as a **cognitive output, not an input**: it is *generated* by combining the input with the activated operation pattern. This keeps the whole thing **unsupervised** — no external KG, no operation labels — so any operation structure has to **emerge** from the learning pressure, not be handed to the model.

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

Architecturally this is a **sparse Mixture-of-Experts** whose gate learns both input-driven routing and (eventually) user-conditional modulation. Same input, different routing → different operation path → different output.

> **A note on claims.** This is exploratory work. I am *not* claiming a novel method or state-of-the-art result. The contribution here is a clearly-stated hypothesis, a forced experimental design, and **honestly reported outcomes** — including the failures.

---

## What I built, and what happened

A compressed timeline. Full details, hypotheses, setups, and result tables are in **[`docs/research-journey.md`](docs/research-journey.md)**.

| Stage | Idea | Outcome |
|---|---|---|
| **Phase 1** — reconstruction cycle | Train an MoE to reconstruct its own input; hope experts specialize into *operations*. | ❌ **Negative.** Experts split by **topic / text source**, not by operation. Reconstruction rewards *content*, so operation structure never had to emerge. |
| **Phase 1.5 / 1a** — flat operation router | Swap the objective: input → KG → **predict the answer** (multiple-choice logic QA), with a 3-way information bottleneck so the answer *must* flow through the routed operations. | ❌ **Weak-ceiling negative.** Diagnosis: the logic-QA corpus is **single-operation** — every question announces its own operation. No *composition* to discover. |
| **Phase 1.5 / 1b** — sequential chain | Move to **multi-hop** QA (MuSiQue) where a problem needs 2–4 chained reasoning steps; let experts chain sequentially. | ❌ **Negative on sequential composition.** The substrate is healthy (flat model reaches ~0.60 val accuracy) but the chain **does not discover adaptive depth** — it performs no better than the flat mixture. |
| **Direction 1** — parallel co-activation *(current)* | Reframe: the flat mixture's **simultaneously-active experts** (~5 at a time) already *are* a co-activation distribution — i.e. exactly the `G_u` from the vision. Test parallel/simultaneous composition directly, via causal lesioning. | 🔬 **In progress.** |

The throughline: each negative result was **informative**. Phase 1 falsified "reconstruction yields operations." 1a localized the failure to the **corpus** (no composition substrate). 1b falsified "*sequential* chaining is the right inductive bias" — and, in doing so, pointed back at the *parallel* co-activation structure the project originally described.

---

## Tech stack

- **Language / framework:** Python, PyTorch
- **Encoder:** frozen sentence encoder (`e5-large-v2`, 1024-dim) — **no LLM in the training loop**; everything is embedding-level + multiple-choice scoring
- **Router:** ReMoE-style gate (ReLU experts + an adaptive L1 controller targeting a small active set, `K_active ≈ 4`)
- **Decoder:** gated low-rank hypernet modulation with a strict **no-bypass** invariant (if the operation activation is zero, the output is zero — so the passage can't leak around the bottleneck)
- **Objective:** multiple-choice cross-entropy (contrastive over 4 candidates)
- **Datasets:** Phase 1 — 7 diverse text sources (Reddit, Pennebaker, PANDORA, PersonaChat, ROCStories, αNLI, SocialIQA); Phase 1.5 — LogiQA 2.0 + ReClor (single-op control) and **MuSiQue** (multi-hop / compositional)
- **Engineering hardening:** OOM-safe per-token expert batching, Herfindahl load-balancing, chance-normalized selectivity probes with control baselines, causal lesion/swap intervention harness — all test-driven (≈190 tests in `phase1_5/`)

---

## Repository structure

```
personal-cognitive-routing/
├── README.md                     ← you are here
├── docs/
│   ├── research-journey.md       ★ the full "what I tried / why it failed" story
│   ├── vision.md                 the paradigm & architecture in depth
│   ├── glossary.md               domain terms + the project's invariants ("IRON rules")
│   ├── literature-review.md      annotated positioning against ~38 papers
│   └── reading-list.md           curated paper list by role
├── phase1/                       closed experiment — reconstruction cycle (negative result)
│   ├── README.md
│   ├── *.py                      model, training, evaluation, baselines
│   ├── notebooks/                Colab run drivers (04–07)
│   └── tests/
└── phase1_5/                     current work — operation router + multi-hop composition
    ├── README.md
    ├── *.py                      model, MuSiQue loader, intervention harness, ablations
    ├── notebooks/                Colab run drivers (01–05)
    └── tests/
```

**Where to start reading:** this page → [`docs/research-journey.md`](docs/research-journey.md) → [`docs/vision.md`](docs/vision.md) for the formal framing → `phase1/` and `phase1_5/` for the code.

## Running the code

The experiments were run on Google Colab (GPU) with embedding caches and checkpoints kept off-repo (they are large and regenerable). The code is organized as importable packages with a test suite you can run locally:

```bash
pip install torch numpy datasets sentence-transformers pytest
pytest phase1_5/tests -q        # unit tests (no GPU required)
```

Heavy artifacts (`*.npy`, `*.parquet`, checkpoints, reference PDFs) are intentionally git-ignored — they rebuild from the data loaders and training code.

---

*This is a personal research project, shared as a portfolio piece. It documents an ongoing line of inquiry, negative results included.*
