"""Frozen sentence encoder — moved to core.encoders.

``FrozenEncoder``, ``DEFAULT_ENCODER_NAME``, ``default_q_prefix`` and ``default_p_prefix``
were extracted to ``research/core/encoders.py`` as the shared superset (it absorbed
phase1's pooled ``forward`` / ``encode_batched`` / no-prefix paths AND phase1_5's E5
``prefix`` kwarg + ``encode_pooled`` paths — the union is behaviour-identical to this
former copy because every phase1_5 call passes the encoder name + prefixes explicitly).

Kept as a thin re-export so ``phase1_5.data`` and ``phase1_5.tests.test_encoders``
(which import from ``experiments.phase1_5.encoders``) keep resolving unchanged.

E5 prefix protocol (HF model card, e5 family default):
- query side  → ``"query: "``
- passage side → ``"passage: "``
"""

from __future__ import annotations

from core.encoders import (  # noqa: F401
    DEFAULT_ENCODER_NAME,
    FrozenEncoder,
    default_p_prefix,
    default_q_prefix,
)
