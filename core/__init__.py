"""Shared building blocks extracted from the phase1 / phase1_5 experiment packages.

These modules were COPY-DUPLICATED across ``research/experiments/phase1`` and
``research/experiments/phase1_5``; they now live here once and both packages import
them. Behaviour is byte-identical to the pre-extraction copies — the core classes are
backward-compatible supersets (defaults chosen so each original call site is unchanged).

Imported as ``core.X`` (namespace path; the repo root is on ``sys.path`` under
pytest, the same mechanism that resolves ``experiments.phase1_5``).
"""
