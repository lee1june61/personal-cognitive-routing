# Verification framework — `f(task, person)`

This document states *how* the project decides whether its central claim is true,
**before** any result is in. The point is that the evaluation criteria are fixed
in advance, so a negative result is informative rather than a moved goalpost.

It is the logical spine the rest of the repository is organized around. The
experiments in [`../README.md`](../README.md) are read *through* this frame, not the
other way around.

---

## The factorization

Which subgraph activates — which experts fire, which path is taken — is modeled as
a function of **two** independent factors:

```
subgraph activation  =  f(task, person)
                          │      │
                          │      └── manner / style overlay  →  G_u   (the personalization locus)
                          │          how each step is carried out
                          │          (causal vs. comparative, detail vs. gestalt, …)
                          │
                          └── structure / plan skeleton
                              which steps the problem demands
                              (2-hop bridge, comparison, …) — shared, not personal
```

- The **task** axis governs *structure*: what steps the problem requires. It is shared across people.
- The **person** axis governs *manner*: how those steps are carried out. This is `G_u`, the per-user activation distribution the project sets out to model.

**Validation is staged along this factorization: the task axis first (anonymous),
the person axis second (per-user).** The reason is nuisance control — task-driven
variation has to be characterized and held constant before person-driven variation
can be attributed cleanly. Otherwise the two confound in the activation signal. This
generalizes the existing Phase 1/1.5 (user-agnostic) → Phase 2 (per-user) split into
the organizing principle for the whole sequence.

---

## Task axis (anonymous — Phase 1 / 1.5)

| Cell | Claim to establish | Status |
|---|---|---|
| **L0** | The task forces reasoning (it is not solvable by a surface shortcut) | **Pass** — on MuSiQue the flat model reaches ~0.598 val (≫ chance), single-hop F1 −30. (Logic-QA was disqualified: a 0.98 stem-announce ceiling.) |
| **T1** | The subgraph differentiates by task (does not collapse to one expert) | **Weak** — differentiation seen, but at the source/format (topic) level, and the active set tends to collapse under regularization. |
| **T2** ★ | That differentiation is **content-free** (manner ≠ topic) | **Open blank — cleanly unmeasured.** This is the load-bearing cell, and no clean measurement exists yet (see below). |
| **T3** | That differentiation is **structural** (composition, not flat) | **Negative** for *sequential* chaining (effective breadth stayed flat, `{2,2,2}`; rebutted the depth-cap confound at `chain_steps=4` × 3 seeds). |

### T2 is the open blank, and why it is *blank* rather than *failed*

The experiments run so far (1a flat router; the Phase-1 Stage-1 selectivity probe)
varied **reasoning-type and topic together**. So when selectivity came out weak
(`col_spec ~0.05`; `adj_op 0.176` against a topic alignment of 0.58–0.97), that does
**not** decide manner-vs-topic — the two were confounded by construction. The cell is
unmeasured, not failed.

Filling it requires a **manner-controlled task design**: vary the *required
content-free manner* on **fixed** topic/content — e.g. the same passage answered
"via a causal chain" vs. "via a comparison" — so that routing is forced to
differentiate on manner rather than on topic or on a stem-announced reasoning-type.
Constructing that corpus and its metric is the open task.

### A scope guard on the T3 negative

The T3 negative is a negative on **sequential** composition. It is **not** a negative
on **parallel / simultaneous** composition. The flat mixture already co-activates
~5 experts at once, and that simultaneous distribution is itself a candidate `G_u`.
"Test parallel co-activation directly" (Direction 1 in the README) is therefore a
**hypothesis shift, not a result** — phrasing it as "parallel is the answer" or
"the flat mixture *is* `G_u`" would be a claim the data does not yet support.

---

## Person axis (per-user — Phase 2, deferred; no evidence yet)

| Cell | Claim | Status |
|---|---|---|
| **P1** | With task controlled, the subgraph differentiates by person | Phase 2 — deferred, no evidence |
| **P2** | That person-signature is stable across topic (= `G_u`); the **S1** falsifier lives here | Phase 2 — deferred, no evidence |
| **P3** | `G_u` predicts observables — persona-faithfulness, sample-efficiency (the intended headline) | Phase 2 — deferred, no evidence |

The person axis is *intentionally frozen* until the task axis is established — an
absence by design, not an untested gap.

---

## The measurement tool (a constant, not a variable) — the B1 gate

The PASS bar for T2/T3 is itself fixed in advance. To count as genuine *operation*
structure rather than a topical shortcut, an activation must clear, jointly:

1. **Causal selective lesion** — turning off a putative operation drops accuracy
   *only* on the problems that need it (`> 1σ`, diagonal-dominant by required step), not uniformly.
2. **Operation-vs-topic disentanglement** — operation purity beats topic under a
   geometry-control baseline.
3. **Motif-consistency** — problems sharing a decomposition structure activate
   similar sub-paths. This connects forward to **S1** (the Phase-2 falsifier).

Because this bar is defined *before* the results, having it at all is a result-independent
contribution (the methodology), regardless of which way the experiments fall.

---

## A confound that crosses every task-axis cell

The encoder is **frozen** (e5-large-v2, ~0.55 MC ceiling). This is an unverified
assumption that has never been raised as an experimental variable: every task-axis
negative could reflect "no structure" *or* "no synthesis pressure (the ceiling)",
and the two are not yet separated. The encoder-swap diagnostic is unrun. The
frozen-encoder rule keeps results attributable, but it may itself be the deeper cap.

---

## Where this stands (honestly)

Task axis: **L0 pass, T1 weak, T2 the open blank (the target), T3 sequential
negative (parallel unmeasured).** Person axis: **entirely deferred, zero evidence.**
The current mode is deliberate course-correction — set each criterion before running
the next experiment — with the governing question being *convergence vs. drift*,
not another reframe.
