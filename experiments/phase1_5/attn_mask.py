"""Shared attention-mask helper — moved to core.attn_mask.

The implementation was extracted verbatim to ``research/core/attn_mask.py`` (single
copy). Kept here as a thin re-export so the in-package importers
(``phase1_5.cross_attention`` and ``phase1_5.kg_hypernet``, which do
``from .attn_mask import safe_key_padding_mask``) keep resolving unchanged.
"""

from __future__ import annotations

from core.attn_mask import safe_key_padding_mask  # noqa: F401
