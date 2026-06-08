# ARCHIVED 2026-06-08 — sequential chain-of-experts (1b orchestrator)

`run_1b.py` + `_diag_composition.py` + `test_run_1b.py` + `05_musique_1b.ipynb`.

**1b sequential chain = setup-failure (LAYOUT only).** breadth {2,2,2} non-monotone; the cap-confound is closed (L=4 × 3 seed → {1,1,1}); the **frozen-e5 ceiling is still OPEN**.

This does **NOT**:
- discard the operation-emergence **hypothesis** — it is lit-protected (operation experts are shown to exist in prior work; feasibility there is *existence* only, not our-setup emergence);
- discard **parallel / co-activation** composition (direction-1, unmeasured = the live direction).

The sequential chain was a **detour**. `forward_chain()` stays in `experiments/phase1_5/model.py` (called by `train.py` / `intervention.py` / `ablations.py`) with a `DEPRECATED` marker — removing it is invasive. The notebook that drove 1b is archived here and is stale → rewrite if reproduction is needed.

See the repo README for how this fits the broader journey (sequential negative → direction-1 parallel).
