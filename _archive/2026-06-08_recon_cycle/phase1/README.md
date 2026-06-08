# Phase 1 — Reconstruction cycle (closed, informative negative)

This is the **first experiment** in the project: an unsupervised Mixture-of-Experts trained on a *reconstruction cycle*, testing whether experts would specialize into **interpretive operations** on their own.

**Result: negative — but informative.** Experts specialized by **topic / text source**, not by operation, because a reconstruction objective only rewards *content*. This directly motivated the Phase 1.5 pivot to answer-prediction. The full reasoning is in [`../docs/research-journey.md`](../../docs/research-journey.md#phase-1--the-reconstruction-cycle-closed-informative-negative).

## Final ablation (the evidence)

| run | epochs | K_active | recon (cos) | downstream (SimBench) | note |
|---|---|---|---|---|---|
| v3_minimal | 30 | 13.88 | 0.8864 | **0.7120** | dense routing; best downstream (narrow corpus) |
| v4_diverse | 30 | 6.56 | 0.8786 | 0.7029 | 7-source; clusters form at source level |
| v5_arch | 30 | **1.00** | 0.8788 | 0.7067 | orthogonality penalty → degenerate K=1 collapse |
| **v6_long** | 79 | 5.66 | 0.8789 | **0.6901** | paradigm-faithful, sharpest clusters, lowest downstream |
| b0 (no MoE) | 30 | — | 0.8687 | 0.7082 | baseline |

Key findings: reconstruction beats baseline (+1pp) but clusters form at **source/format level not operation level**; sharper clustering *lowers* downstream performance (pretext/downstream mismatch); a raw-encoder diagnostic confirmed topic alignment (0.58–0.97) dominates operation alignment (~+0.18) — i.e. the objective structurally rewards topic.

## Layout

```
phase1/
├── model.py / cycle.py          MoE + reconstruction-cycle training core
├── train.py                     training loop
├── eval.py / eval_opcycle.py    selectivity probes + control baselines
├── eval_simbench_classifier.py  downstream evaluation
├── cluster_analysis.py          expert-cluster inspection
├── data.py                      7-source corpus loaders + OOM-safe frozen-encoder caching
├── model_opcycle.py / engine_a.py   operation-cycle variant ("Engine-A") + its driver
├── ENGINE_A_DESIGN.md           design notes for the operation-cycle variant
├── baselines/                   no-MoE and standard-MoE baselines
├── notebooks/                   Colab run drivers (00–07)
└── tests/                       unit tests
```

## Reusable hardening (carried into `phase1_5/`)

`FrozenEncoder` (per-token encode, fp16 + int8 mask, OOM-safe batching) · chance-normalized `selectivity_report` with four control baselines · adaptive L1 sparsity controller · Herfindahl load-balancing · `meanmax` aggregation. These were code-reviewed and test-hardened here, then adapted in Phase 1.5.

## Running

Experiments ran on Colab GPU; embedding caches and checkpoints are git-ignored (regenerable from `data.py` + training code). Unit tests run locally:

```bash
pytest phase1/tests -q
```
