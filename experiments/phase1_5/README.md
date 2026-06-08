# Phase 1.5 — Operation router & multi-hop composition (current work)

This folder holds the **current** line of work, after Phase 1's reconstruction cycle was closed as a negative result. The objective pivots from *reconstruction* to **answer prediction**: input → routed operations → predict a multiple-choice answer, behind a strict information bottleneck so the answer can only be reached *through* the routed operations.

See [`../docs/research-journey.md`](../../docs/research-journey.md#phase-15--the-objective-pivot) for the full narrative. Short version of where this has gone:

- **1a (flat operation router)** — `K=128` fine-grained experts, `K_active≈4`, on logic QA (LogiQA 2.0 + ReClor). **Weak-ceiling negative**: the corpus turned out to be *single-operation* (each question announces its own operation), so there was no composition to discover.
- **1b (sequential chain over MuSiQue)** — pivoted to multi-hop QA (2–4 reasoning hops, intermediate answers as hard distractors), converted to 4-choice MC, with logic-QA kept as a single-op control arm. A sequential chain-of-experts. **Negative on sequential composition**: substrate healthy (flat ≈0.598 val accuracy) but the chain discovers no adaptive depth.
- **Direction 1 (current)** — reframe the flat mixture's *simultaneously*-active experts as the parallel co-activation distribution `G_u`, and test it directly with causal lesioning.

## The 1a architecture (information bottleneck)

```
Q (question text)
  └► frozen e5-large-v2 ─► SharedEncoderHead (1024→256) ─► masked mean-pool ─► z_q
        └► ReMoERouter (K=128, ReLU + adaptive L1 → K_active≈4) ─► α   (operation activation = G_u)
              └► kg_vec = Σ_k α_k · op_token_k        (op_token_k: learnable, K=128)

P (passage text)
  └► frozen e5-large-v2 ─► Linear (1024→256) ─► P_repr      (side-channel only)

KGHypernetModulation:  out = U( s(kg_vec) ⊙ V( Attn(query=kg_vec, key/val=P_repr) ) )
  ── no-bypass invariant:  kg_vec = 0  ⟹  out = 0   (bias-free, no residual)
        └► candidate scoring over 4 MC options ─► logits ─► cross-entropy
```

**The 3-way bottleneck** (so the answer must flow through the operations):
1. **Question-only encoding** — the passage never reaches the encoder/router.
2. **Passage as side-channel** — it enters only as key/value in the decoder.
3. **No-bypass modulation** — zero operation activation ⟹ zero output. (This invariant came from a *real bug*: a zero attention query leaks the whole passage as uniform attention. See the journey doc.)

## Key hyperparameters

| item | value | rationale |
|---|---|---|
| encoder | `e5-large-v2` (frozen) | higher raw operation-alignment ceiling in diagnostics |
| `K_routed` | 128 | fine-grained sub-skill emergence |
| `K_active` target | ≈4 | matches sparse-MoE active ratios (e.g. DeepSeek-V3) |
| router | ReMoE (ReLU + adaptive L1) | validated in Phase 1 |
| modulation | gated low-rank hypernet, no-bypass | replaces cross-attn/FiLM baselines |
| corpora | LogiQA 2.0 + ReClor (single-op control); **MuSiQue** (multi-hop) | hard distractors |
| objective | 4-choice MC cross-entropy | contrastive |

## Layout

```
phase1_5/
├── model.py             Phase15MoE — flat router (1a) + sequential chain (1b, forward_chain)
├── encoders.py          frozen e5 encoding + caching
├── data.py              LogiQA/ReClor MC loaders
├── data_musique.py      MuSiQue → 4-choice MC conversion (intermediate-answer distractors)
├── kg_hypernet.py       no-bypass gated low-rank modulation
├── modulation.py / cross_attention.py / attn_mask.py   decoder components
├── load_balance.py      Herfindahl load-balancing + adaptive L1 controller
├── train.py / run_1b.py training loops (1a flat, 1b chain)
├── eval.py              operation-selectivity gate + four control baselines
├── intervention.py      causal lesion / swap harness (for Direction 1)
├── ablations.py         ablation drivers
├── notebooks/           Colab run drivers (01–05)
└── tests/               ~190 unit tests (test-driven)
```

## Running

Experiments run on Colab GPU; embedding caches and checkpoints are git-ignored (regenerable from the loaders + training code). The unit tests run locally without a GPU:

```bash
pytest phase1_5/tests -q
```
