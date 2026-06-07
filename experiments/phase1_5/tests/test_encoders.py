"""Tests for `phase1_5.encoders`.

CPU-only fast tests use a tiny synthetic encoder (we mock `transformers` lookup
where possible). Real-encoder tests (HF download) are marked ``slow``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.phase1_5.encoders import (
    DEFAULT_ENCODER_NAME,
    FrozenEncoder,
    default_p_prefix,
    default_q_prefix,
)


# ---- fast: prefix convention --------------------------------------------------


def test_default_q_prefix_e5_returns_query():
    assert default_q_prefix("intfloat/e5-large-v2") == "query: "
    assert default_q_prefix("intfloat/e5-base") == "query: "


def test_default_q_prefix_bge_returns_empty():
    assert default_q_prefix("BAAI/bge-large-en-v1.5") == ""


def test_default_p_prefix_e5_returns_passage():
    assert default_p_prefix("intfloat/e5-large-v2") == "passage: "


def test_default_p_prefix_bge_returns_empty():
    assert default_p_prefix("BAAI/bge-large-en-v1.5") == ""


# ---- slow: real encoder load -------------------------------------------------


@pytest.mark.slow
def test_frozen_encoder_e5_freezes_weights():
    enc = FrozenEncoder(DEFAULT_ENCODER_NAME)
    assert all(not p.requires_grad for p in enc.encoder.parameters())
    assert not enc.encoder.training
    assert enc.d_model == 1024


@pytest.mark.slow
def test_encode_tokens_pad_to_max_returns_fixed_T():
    enc = FrozenEncoder(DEFAULT_ENCODER_NAME)
    texts = ["short q", "a much much longer query string with more tokens"]
    hidden, mask = enc.encode_tokens(
        texts, prefix="query: ", max_length=32, pad_to_max=True
    )
    assert hidden.shape == (2, 32, 1024)
    assert mask.shape == (2, 32)
    # short example masked beyond its tokens
    assert mask[0].sum() < mask[1].sum()


@pytest.mark.slow
def test_encode_tokens_prefix_affects_first_token_embedding():
    """Prefixed input should produce a different token sequence than unprefixed."""
    enc = FrozenEncoder(DEFAULT_ENCODER_NAME)
    texts = ["a logic question"]
    h_no_prefix, _ = enc.encode_tokens(texts, prefix="", max_length=16, pad_to_max=True)
    h_with_prefix, _ = enc.encode_tokens(
        texts, prefix="query: ", max_length=16, pad_to_max=True
    )
    # different first content token embeddings (after [CLS])
    assert not torch.allclose(h_no_prefix[0, 1], h_with_prefix[0, 1], atol=1e-3)


@pytest.mark.slow
def test_encode_tokens_batched_shape_and_dtype():
    enc = FrozenEncoder(DEFAULT_ENCODER_NAME)
    texts = ["q one", "q two", "q three"]
    tokens, mask = enc.encode_tokens_batched(
        texts, prefix="query: ", batch_size=2, t_cap=16
    )
    assert tokens.shape == (3, 16, 1024)
    assert tokens.dtype == np.float16
    assert mask.shape == (3, 16)
    assert mask.dtype == np.int8


@pytest.mark.slow
def test_encode_pooled_returns_l2_normalized():
    enc = FrozenEncoder(DEFAULT_ENCODER_NAME)
    pooled = enc.encode_pooled(["candidate one"], prefix="query: ", max_length=16)
    norms = pooled.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


@pytest.mark.slow
def test_encode_pooled_no_l2_norm_disabled():
    enc = FrozenEncoder(DEFAULT_ENCODER_NAME)
    pooled = enc.encode_pooled(
        ["candidate"], prefix="query: ", max_length=16, l2_normalize=False
    )
    norms = pooled.norm(dim=-1)
    assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-2)


@pytest.mark.slow
def test_encode_pooled_batched_shape():
    enc = FrozenEncoder(DEFAULT_ENCODER_NAME)
    cands = ["c1", "c2", "c3", "c4", "c5"]
    pooled = enc.encode_pooled_batched(cands, prefix="query: ", batch_size=2, max_length=16)
    assert pooled.shape == (5, 1024)
    # rows L2-normalized to ~1
    norms = np.linalg.norm(pooled, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-3)


@pytest.mark.slow
def test_encoder_swap_bge_works():
    """Row C ablation — BGE-large-en-v1.5 swap-in."""
    enc = FrozenEncoder("BAAI/bge-large-en-v1.5")
    assert enc.d_model == 1024
    hidden, mask = enc.encode_tokens(["test"], prefix="", max_length=8, pad_to_max=True)
    assert hidden.shape == (1, 8, 1024)
