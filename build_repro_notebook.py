"""Builder for REPRODUCE_ALL.ipynb — emits a single Colab notebook that re-runs
every Phase 1 / 1.5 experiment from scratch and saves committed JSON artifacts
(closes the 2026-06-03 audit "results are prose-only, not reproducible" gap).

Run:  python build_repro_notebook.py   →  writes REPRODUCE_ALL.ipynb next to this file.

Cell sources are kept as plain strings here (easy to edit); this script only
assembles the ipynb v4 JSON. No nbformat dependency.
"""
import json
from pathlib import Path

CELLS = []  # list of (kind, source_str)


def md(src):
    CELLS.append(("markdown", src.strip("\n")))


def code(src):
    CELLS.append(("code", src.strip("\n")))


# =====================================================================================
md(r"""
# REPRODUCE_ALL — Phase 1.5 task-axis live experiments with committed JSON artifacts

Upload the whole `sideproject` folder to Google Drive, open this notebook in Colab,
pick a GPU runtime, and **Run All** (or run section-by-section).

**What it reproduces.** The *live* task-axis line of evidence — three experiments:
- **flat 1a operation router** (ablation grid + seed/top-k robustness),
- **MuSiQue flat substrate check** (does the multi-hop corpus support an operation router at all),
- **causal intervention battery** (SWAP + LESION → `col_spec`) on the flat 1a router.

Each writes a JSON metric file so every documented number is independently verifiable. The final
**Section 10** prints a documented-vs-measured table and saves `out/VERIFICATION.json`.

**Sections are independent** (each wrapped in try/except) — one failure won't block the rest.
Paste any traceback back and we'll fix it. Heaviest cost = the 1a robustness grid; toggle it off
in the config cell if you want a faster first pass.

**Archived experiments.** The recon-cycle (Phase 1) and the 1b sequential chain-of-experts run
have been git-moved to `research/_archive/` (their package paths are no longer importable) and are
intentionally **out of scope** for this notebook. This reproduces only the live `phase1_5`
task-axis experiments.
""")

# ---- Cell 1: master config ----------------------------------------------------------
code(r"""
# ============================== MASTER CONFIG ==============================
BASE = '/content/drive/MyDrive/sideproject'   # <-- change if your Drive path differs

# --- section toggles (live phase1_5 task-axis experiments only) ---
RUN_1A             = True   # Section 5 — flat 1a operation-router ablations
RUN_1A_ROBUSTNESS  = True   # Section 6 — seed x use_best_val grid + top-k arm
RUN_INTERVENTION   = True   # Section 8 — causal SWAP/LESION battery -> col_spec
RUN_MUSIQUE_FLAT   = True   # Section 7 — MuSiQue flat substrate check (M3)

# --- confound-fix cells (Section 9) ---
INCLUDE_CONFOUND_FIXES  = True
FIX_1A_SAME_AXIS        = True   # 9.2 operation_gate adj vs operation_ceiling_raw adj, same axis

# --- reproducibility ---
REUSE_EXISTING = True             # if out/phase1_5/.../model.pt already exists on Drive, SKIP training and just eval.
FRESH       = False               # only used when actually training: True wipes the run dir first (defeats ckpt resume).
RUN_SUFFIX  = ''                  # '' = use the ORIGINAL run_ids so existing Drive checkpoints are found.
                                  #   set to e.g. '_repro' to force brand-new dirs (ignores your existing checkpoints).
SEEDS       = [0, 1, 2]           # 1a robustness multi-seed
print('config loaded.')
""")

# ---- Section 0: preflight -----------------------------------------------------------
md(r"""
## Section 0 — Preflight (mount, GPU caps, smoke tests, harness)
""")

code(r"""
# 0.0 mount + chdir + sys.path  (keep cwd = BASE so out/phase1 and out/phase1_5 share one tree)
import os, sys, shutil, json, traceback
from pathlib import Path
if not os.path.exists('/content/drive/MyDrive'):
    try:
        from google.colab import drive; drive.mount('/content/drive')
    except Exception as e:
        print('drive mount skipped (not on Colab?):', e)
os.chdir(BASE)
if BASE not in sys.path:
    sys.path.insert(0, BASE)
print('cwd =', os.getcwd())
""")

code(r"""
# 0.1 version print (no auto-pip; if an import fails below, install manually then re-run)
import importlib
for m in ['torch', 'transformers', 'datasets', 'numpy', 'sklearn', 'pandas']:
    try:
        mod = importlib.import_module(m); print(f'{m:14s} {getattr(mod, "__version__", "?")}')
    except Exception as e:
        print(f'{m:14s} MISSING  ->  pip install {m}   ({e})')
""")

code(r"""
# 0.2 GPU detect -> batch/sample caps
import torch
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
if 'A100' in gpu:
    TIER='A100'; BATCH_PHASE15=128; SAMPLE_CAP_15=20000
elif any(x in gpu for x in ['L4','A10','V100','RTX','L40']):
    TIER='L4';  BATCH_PHASE15=64;  SAMPLE_CAP_15=20000
elif 'T4' in gpu:
    TIER='T4';  BATCH_PHASE15=32;  SAMPLE_CAP_15=8000
else:
    TIER='CPU'; BATCH_PHASE15=16;  SAMPLE_CAP_15=2000
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
P15 = Path('out/phase1_5'); PHASE15_CACHE = P15 / 'cache'
PHASE15_CACHE.mkdir(parents=True, exist_ok=True)
print(f'GPU={gpu} | TIER={TIER} | DEV={DEV}')
print(f'BATCH_PHASE15={BATCH_PHASE15} SAMPLE_CAP_15={SAMPLE_CAP_15}')
if TIER == 'CPU':
    print('WARNING: no GPU detected — training will be very slow; this is smoke-only.')
""")

code(r"""
# 0.3 HF dataset reachability smoke (no auth needed). Failures are warnings, not fatal.
def _smoke():
    import pandas as pd
    from datasets import load_dataset
    checks = [
        ('MuSiQue', lambda: load_dataset('dgslibisey/MuSiQue', split='train', streaming=True)),
        ('Super-NI', lambda: load_dataset('Muennighoff/natural-instructions', split='train',
                                          streaming=True, verification_mode='no_checks')),
        ('QuAIL',   lambda: load_dataset('textmachinelab/quail', split='train', streaming=True)),
    ]
    for name, fn in checks:
        try:
            next(iter(fn().take(1))); print(f'  {name:10s} OK')
        except Exception as e:
            print(f'  {name:10s} UNREACHABLE: {type(e).__name__}: {e}')
    try:
        url = 'https://huggingface.co/datasets/pitehu/SimBench/resolve/main/SimBenchPop.csv'
        pd.read_csv(url, nrows=3); print('  SimBench   OK')
    except Exception as e:
        print(f'  SimBench   UNREACHABLE: {type(e).__name__}: {e}')
_smoke()
""")

code(r"""
# 0.4 import smoke for every module the notebook touches (catches a stale Drive sync early)
mods = [
    'experiments.phase1_5.engine_1a', 'experiments.phase1_5.ablations', 'experiments.phase1_5.eval',
    'experiments.phase1_5.data', 'experiments.phase1_5.intervention', 'experiments.phase1_5.model', 'experiments.phase1_5.train',
]
ok = True
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        ok = False; print(f'IMPORT FAIL {m}: {type(e).__name__}: {e}')
print('all imports OK' if ok else 'FIX IMPORTS ABOVE before running sections')
""")

code(r"""
# 0.5 + 0.6 helpers: fresh() wipe + run_section() harness
RESULTS = {}

def fresh(run_dir):
    p = Path(run_dir)
    if FRESH and p.exists():
        shutil.rmtree(p)

def run_section(name, fn):
    print('\n' + '=' * 72 + f'\n[{name}] starting\n' + '=' * 72)
    try:
        fn(); RESULTS[name] = 'OK'; print(f'[{name}] DONE')
    except Exception as e:
        RESULTS[name] = f'FAIL: {type(e).__name__}: {e}'
        print(f'[{name}] FAILED — full traceback:'); traceback.print_exc()
    finally:
        # Free GPU between sections so a trained model from one section doesn't
        # starve the next (the chain_steps=4 run is memory-heavy).
        try:
            import gc, torch as _t
            gc.collect()
            if _t.cuda.is_available():
                _t.cuda.empty_cache()
        except Exception:
            pass

def rid(base):
    return base + RUN_SUFFIX

print('helpers ready.')
""")

# ---- Section 5: 1a ablations --------------------------------------------------------
md(r"""
## Section 5 — Phase 1.5 1a ablations (pivot 2)
`run_all_rows` auto-saves `row_A.{1,2,3}_*.json` + probe codes + `summary_seed0.csv`.
""")

code(r"""
def _ablations_1a():
    from experiments.phase1_5.data import MCCorpusConfig
    from experiments.phase1_5.ablations import PHASE1_5_INITIAL_ROWS, run_all_rows
    from experiments.phase1_5.train import TrainConfig
    cfg = MCCorpusConfig(max_train_samples=SAMPLE_CAP_15, max_val_samples=2000,
                         max_test_samples=2000, cache_root=str(PHASE15_CACHE))
    tc = TrainConfig(epochs=40, lr=1e-3, k_target=4.0, seed=0)
    df = run_all_rows(PHASE1_5_INITIAL_ROWS, corpus_cfg=cfg, train_cfg=tc,
                      out_dir=str(P15 / 'ablations'), device=DEV, seed=0,
                      batch_size=BATCH_PHASE15, skip_if_exists=not FRESH)
    cols = [c for c in ['op_adj_operation', 'passes_sigma_gate', 'op_ceiling_raw_adj',
                        'adj_topic', 'val_mc_acc_final'] if c in df.columns]
    print(df[cols].to_string())

if RUN_1A:
    run_section('S5 1a-ablations', _ablations_1a)
""")

# ---- Section 6: 1a robustness -------------------------------------------------------
md(r"""
## Section 6 — 1a robustness / top-k  (does the 1a negative survive seed variation?)
Replication grid: seed × use_best_val → `out/phase1_5/replication/`. Top-k arm → `out/phase1_5/topk/`.
""")

code(r"""
def _robustness():
    from experiments.phase1_5.data import MCCorpusConfig, MODE_Q_ONLY
    from experiments.phase1_5.ablations import AblationRow, run_ablation_row
    from experiments.phase1_5.model import MOD_KG_HYPERNET
    from experiments.phase1_5.train import TrainConfig
    cfg = MCCorpusConfig(max_train_samples=SAMPLE_CAP_15, max_val_samples=2000,
                         max_test_samples=2000, cache_root=str(PHASE15_CACHE))
    for seed in SEEDS:
        for ubv in (True, False):
            row = AblationRow(row_id=f'R_s{seed}_bv{int(ubv)}', name='replication',
                              encoding_mode=MODE_Q_ONLY, k_routed=64, lb_strategy='aux_free',
                              modulation=MOD_KG_HYPERNET)
            tc = TrainConfig(epochs=40, lr=1e-3, k_target=4.0, use_best_val=ubv, seed=seed)
            run_ablation_row(row, corpus_cfg=cfg, train_cfg=tc, out_dir=str(P15 / 'replication'),
                             device=DEV, seed=seed, batch_size=BATCH_PHASE15, skip_if_exists=not FRESH)
    row = AblationRow(row_id='T_topk4', name='topk', encoding_mode=MODE_Q_ONLY, k_routed=64,
                      lb_strategy='aux_free', routing='topk', modulation=MOD_KG_HYPERNET)
    tc = TrainConfig(epochs=40, lr=1e-3, k_target=4.0, lam_z=1e-3, use_best_val=False, seed=0)
    run_ablation_row(row, corpus_cfg=cfg, train_cfg=tc, out_dir=str(P15 / 'topk'),
                     device=DEV, seed=0, batch_size=BATCH_PHASE15, skip_if_exists=not FRESH)

if RUN_1A_ROBUSTNESS:
    run_section('S6 1a-robustness', _robustness)
""")

# ---- Section 7: MuSiQue flat substrate check ---------------------------------------
md(r"""
## Section 7 — Phase 1.5 MuSiQue flat substrate check (M3)
Trains the flat 1a operation router on the MuSiQue multi-hop corpus to confirm the substrate
supports a flat router at all. `run_engine_1a` saves under `out/phase1_5/musique/`; the M3
result is also written explicitly.

(The 1b sequential chain-of-experts run that previously lived here — M5 chain / M6 logic-MC
control — has been archived under `research/_archive/`.)
""")

code(r"""
def _musique_flat():
    from experiments.phase1_5.data import MCCorpusConfig
    from experiments.phase1_5.ablations import AblationRow, _json_default
    from experiments.phase1_5.engine_1a import run_engine_1a
    from experiments.phase1_5.train import TrainConfig
    mus = P15 / 'musique'; mus.mkdir(parents=True, exist_ok=True)
    cfg = MCCorpusConfig(corpus='musique', cache_root=str(PHASE15_CACHE),
                         max_train_samples=SAMPLE_CAP_15, max_val_samples=2000, max_test_samples=2000)

    print('--- M3 flat (chain_steps=1) ---')
    row = AblationRow(row_id='M3', name='MuSiQue flat-1a', k_routed=128, lb_strategy='aux_free',
                      corpus='musique', chain_steps=1, dropout=0.1)
    res1a = run_engine_1a(ablation_row=row, corpus_cfg=cfg,
                          train_cfg=TrainConfig(epochs=15, lr=1e-3, k_target=4.0, best_metric='acc', seed=0),
                          device=DEV, seed=0, batch_size=BATCH_PHASE15, out_dir=str(mus))
    (mus / 'M3_result.json').write_text(json.dumps(res1a, indent=2, default=_json_default))
    print(f'[M3] flat val_acc(last)={(res1a["history"] or [{}])[-1].get("val_mc_acc")}')

if RUN_MUSIQUE_FLAT:
    run_section('S7 musique-flat', _musique_flat)
""")

# ---- Section 8: intervention --------------------------------------------------------
md(r"""
## Section 8 — Causal operation-specialization battery (1a flat: SWAP + LESION → col_spec)
Mirrors `04_intervention.ipynb` and saves the swap/lesion matrices + `col_spec` to
`out/phase1_5/musique/intervention_battery.json` (the notebook only printed them).
""")

code(r"""
def _intervention():
    import numpy as np, torch
    from experiments.phase1_5.data import (MCCorpusConfig, build_mc_corpus, encode_or_load_mc,
                               make_mc_loaders, MODE_Q_ONLY, infer_reasoning_type)
    from experiments.phase1_5.model import Phase15MoE, MOD_KG_HYPERNET
    from experiments.phase1_5.train import TrainConfig, train_phase15
    from experiments.phase1_5.eval import build_operation_labels
    from experiments.phase1_5.intervention import operation_signature, lesion_specificity, operation_swap
    from experiments.phase1_5.ablations import _json_default

    # logic_mc (LogiQA/ReClor): the regex LSAT operation axis lives here. The default
    # corpus is MuSiQue-multihop, where build_operation_labels finds no LSAT stems ([]).
    cfg = MCCorpusConfig(corpus='logic_mc', cache_root=str(PHASE15_CACHE), max_train_samples=SAMPLE_CAP_15,
                         max_val_samples=2000, max_test_samples=2000)
    corpus = build_mc_corpus(cfg)
    data = encode_or_load_mc(corpus, cfg, encoding_mode=MODE_Q_ONLY, device=DEV)
    tr, va, te = make_mc_loaders(data, batch_size=BATCH_PHASE15)
    model = Phase15MoE(d_emb=data['q_tokens'].shape[-1], d_z=256, k_routed=64,
                       modulation=MOD_KG_HYPERNET, lb_strategy='aux_free',
                       lb_target_active=4.0, routing='topk')
    res = train_phase15(model, tr, val_loader=va,
                        cfg=TrainConfig(epochs=40, lr=1e-3, k_target=4.0, lam_z=1e-3,
                                        use_best_val=False, seed=0), device=DEV)
    model = res['model']

    # build_operation_labels expects an array of operation LABELS, not question text.
    # The LSAT regex axis (infer_reasoning_type) turns each question into a label first —
    # this is exactly what regate does internally. (04_intervention passed raw questions,
    # a latent bug that was never caught because that notebook was never executed.)
    split_arr = np.asarray(data['split'])
    probe_split = 'test' if (split_arr == 'test').any() else 'val'
    pmask = split_arr == probe_split
    questions = corpus[corpus['split'] == probe_split]['question'].tolist()
    op_raw = np.array([infer_reasoning_type(q) for q in questions], dtype=object)
    labels, keep = build_operation_labels(op_raw, min_count=20)
    sub = lambda a: a[pmask][keep]
    t = lambda x: torch.from_numpy(np.asarray(x)).float()
    batch = {'q_tokens': t(sub(data['q_tokens'])), 'q_mask': t(sub(data['q_mask'])),
             'p_tokens': t(sub(data['p_tokens'])), 'p_mask': t(sub(data['p_mask'])),
             'cand_pooled': t(sub(data['cand_pooled'])),
             'answer_idx': torch.from_numpy(sub(data['answer_idx']).astype('int64'))}
    op_labels = labels
    ops = sorted(set(op_labels.tolist()))

    sigs, tops = operation_signature(model, batch['q_tokens'], batch['q_mask'], op_labels, k_top=4, device=DEV)
    swap = operation_swap(model, batch, op_labels, sigs, device=DEV)
    les = lesion_specificity(model, batch, op_labels, tops, device=DEV)
    drop = les['drop']
    col_spec = float(np.mean([drop[y][y] - np.mean([drop[x][y] for x in ops if x != y]) for y in ops]))
    diag = float(np.mean([swap[x][x] for x in ops]))
    offd = float(np.mean([swap[x][y] for x in ops for y in ops if x != y]))
    payload = {'ops': ops, 'col_spec': col_spec, 'swap_diag': diag, 'swap_offdiag': offd,
               'chance': 1.0 / len(ops), 'baseline': les['baseline'], 'swap': swap, 'drop': drop,
               'val_acc': (res['history'] or [{}])[-1].get('val_mc_acc')}
    (P15 / 'musique').mkdir(parents=True, exist_ok=True)
    (P15 / 'musique' / 'intervention_battery.json').write_text(json.dumps(payload, indent=2, default=_json_default))
    print(f'col_spec={col_spec:+.3f} | swap diag={diag:.3f} offd={offd:.3f} | chance={1.0/len(ops):.2f}')

if RUN_INTERVENTION:
    run_section('S8 intervention', _intervention)
""")

# ---- Section 9: confound fixes ------------------------------------------------------
md(r"""
## Section 9 — Confound-fix cells
""")

code(r"""
def _fix_1a_same_axis():
    import glob
    rows = []
    paths = (glob.glob(str(P15 / 'ablations' / 'row_*.json')) +
             glob.glob(str(P15 / 'replication' / 'row_*.json')) +
             glob.glob(str(P15 / 'topk' / 'row_*.json')) +
             [str(P15 / 'musique' / 'M3_result.json')])
    for p in paths:
        try:
            r = json.loads(Path(p).read_text())
        except Exception:
            continue
        og = r.get('operation_gate') or {}
        oc = r.get('operation_ceiling_raw') or {}
        rows.append({'file': Path(p).name, 'op_gate_adj': og.get('adj_operation'),
                     'ceiling_raw_adj': oc.get('adj_operation'),
                     'n_examples': og.get('n_operation_examples'), 'gate_verdict': og.get('verdict'),
                     'ceiling_key_present': 'operation_ceiling_raw' in r})
    (P15 / 'same_axis_reconciliation.json').write_text(json.dumps(rows, indent=2, default=str))
    print(f'{"file":40s} {"gate_adj":>9} {"ceil_adj":>9} {"n":>5}')
    for r in rows:
        print(f'{r["file"][:40]:40s} {str(r["op_gate_adj"])[:9]:>9} '
              f'{str(r["ceiling_raw_adj"])[:9]:>9} {str(r["n_examples"]):>5}')

def _confound_fixes():
    if FIX_1A_SAME_AXIS:
        print('\n# 9.2 1a same-axis ceiling reconciliation'); _fix_1a_same_axis()

if INCLUDE_CONFOUND_FIXES:
    run_section('S9 confound-fixes', _confound_fixes)
""")

# ---- Section 10: verification -------------------------------------------------------
md(r"""
## Section 10 — VERIFICATION (documented vs measured)
Loads every saved JSON from the live task-axis sections (1a ablations, MuSiQue flat M3,
intervention battery) and prints a consolidated table, then writes `out/VERIFICATION.json`.
""")

code(r"""
def _verification():
    rows = []

    def add(experiment, metric, documented, measured, path, note=''):
        delta = None
        try:
            if measured is not None and documented is not None:
                delta = round(float(measured) - float(documented), 4)
        except (TypeError, ValueError):
            delta = None
        rows.append({'experiment': experiment, 'metric': metric, 'documented': documented,
                     'measured': measured, 'delta': delta, 'path': path, 'note': note})

    def load(p):
        p = Path(p)
        return json.loads(p.read_text()) if p.exists() else None

    import glob
    for p in sorted(glob.glob(str(P15 / 'ablations' / 'row_A.*_seed0_*.json'))):
        r = load(p); og = (r or {}).get('operation_gate') or {}; oc = (r or {}).get('operation_ceiling_raw') or {}
        add(f'1a {r.get("row_id")}', 'op_gate_adj vs ceiling', f'ceil~{oc.get("adj_operation")}',
            og.get('adj_operation'), p, f'sigma_gate={(r or {}).get("gate", {}).get("passes_sigma_gate")}')

    m3 = load(P15 / 'musique' / 'M3_result.json')
    if m3:
        meas = (m3.get('history') or [{}])[-1].get('val_mc_acc')
        add('M3 musique-flat', 'val_mc_acc(last)', None, meas,
            'out/phase1_5/musique/M3_result.json', 'flat substrate check')

    ib = load(P15 / 'musique' / 'intervention_battery.json')
    if ib:
        add('intervention', 'col_spec', '~0.05 (weak)', round(ib['col_spec'], 4),
            'out/phase1_5/musique/intervention_battery.json', f'chance={ib["chance"]:.2f}')

    print(f'{"experiment":18s} {"metric":24s} {"documented":>22} {"measured":>14} {"delta":>9}  note')
    print('-' * 120)
    for r in rows:
        print(f'{str(r["experiment"])[:18]:18s} {str(r["metric"])[:24]:24s} '
              f'{str(r["documented"])[:22]:>22} {str(r["measured"])[:14]:>14} '
              f'{str(r["delta"]):>9}  {r["note"]}')

    Path('out').mkdir(exist_ok=True)
    Path('out/VERIFICATION.json').write_text(json.dumps(
        {'section_status': RESULTS, 'tier': TIER, 'rows': rows},
        indent=2, default=str))
    print('\nsaved -> out/VERIFICATION.json')
    print('\nsection status:', RESULTS)

run_section('S10 verification', _verification)
""")


# =====================================================================================
def main():
    nb = {
        "cells": [
            {"cell_type": k, "metadata": {},
             **({"source": s.splitlines(keepends=True)} if k == "markdown"
                else {"source": s.splitlines(keepends=True), "outputs": [], "execution_count": None})}
            for (k, s) in CELLS
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "colab": {"provenance": []}, "accelerator": "GPU",
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = Path(__file__).parent / "REPRODUCE_ALL.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
