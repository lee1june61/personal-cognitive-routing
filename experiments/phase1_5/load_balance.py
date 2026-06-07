"""Aux-loss-free load balancing — moved to core.load_balance.

This module's contents were extracted verbatim to ``research/core/load_balance.py``
(single copy; phase1 had none). Kept here as a thin re-export so existing importers —
``phase1_5.model`` and ``phase1_5.tests.test_load_balance`` — continue to resolve
``experiments.phase1_5.load_balance.{AuxLossFreeLB,make_lb,LB_OFF,...}``
unchanged.
"""

from __future__ import annotations

from core.load_balance import (  # noqa: F401
    LB_AUX_FREE,
    LB_OFF,
    LB_STRATEGIES,
    AuxLossFreeLB,
    make_lb,
)
