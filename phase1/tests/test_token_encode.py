"""Tests for the per-token encode path (operation-cycle pivot, 2026-05-27).

ENGINE_A_DESIGN §1: the cycle moves from masked-mean-pool (B, 1024) to per-token
last_hidden_state (B, T, 1024) + attention mask, cached as (N, T, 1024) fp16 + (N, T)
mask with a fixed T cap. These tests exercise that contract with a hand-built fake
tokenizer/encoder (no transformers, no network, CPU-only) — the HF model load is the
boundary and is covered separately by a `slow` Colab test.

The real `FrozenEncoder.__init__` downloads a model, so we bypass it via
`object.__new__` and attach fakes — we are testing the pooling-free shaping logic
(`encode_tokens`, `encode_tokens_batched`), not HF.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phase1.cycle import FrozenEncoder


D_MODEL = 8  # tiny stand-in for BGE's 1024


class _FakeTokenizer:
    """Returns deterministic input_ids/attention_mask. Honors padding strategy:
    'max_length' → fixed T = max_length; True → pad to longest in batch. Truncates to
    max_length. Token count per text = number of whitespace words (capped)."""

    def __call__(self, texts, padding=True, truncation=True, max_length=128, return_tensors="pt"):
        lens = [min(len(t.split()), max_length) for t in texts]
        lens = [max(1, n) for n in lens]
        width = max_length if padding == "max_length" else max(lens)
        ids = torch.zeros(len(texts), width, dtype=torch.long)
        mask = torch.zeros(len(texts), width, dtype=torch.long)
        for i, n in enumerate(lens):
            ids[i, :n] = 1
            mask[i, :n] = 1
        return {"input_ids": ids, "attention_mask": mask}


class _FakeEncoder(torch.nn.Module):
    """Maps input_ids (B,T) → last_hidden_state (B,T,D_MODEL) deterministically."""

    def __init__(self):
        super().__init__()
        # Deterministic weights from an explicit tensor — does NOT draw the global torch
        # RNG (random nn.Embedding init would, perturbing later unseeded tests and making
        # the suite order-dependent).
        w = torch.arange(2 * D_MODEL, dtype=torch.float32).reshape(2, D_MODEL) / (2 * D_MODEL)
        self.embed = torch.nn.Embedding.from_pretrained(w, freeze=True)

    def forward(self, input_ids=None, attention_mask=None, **kw):
        hidden = self.embed(input_ids)  # (B,T,D)
        return types.SimpleNamespace(last_hidden_state=hidden)


def _make_fake_frozen_encoder() -> FrozenEncoder:
    enc = object.__new__(FrozenEncoder)
    torch.nn.Module.__init__(enc)
    enc.tokenizer = _FakeTokenizer()
    enc.encoder = _FakeEncoder()
    enc.d_model = D_MODEL
    return enc


# ---------------------------------------------------------------------------
# encode_tokens — per-token, no pooling
# ---------------------------------------------------------------------------


def test_encode_tokens_returns_per_token_and_mask():
    fe = _make_fake_frozen_encoder()
    emb, mask = fe.encode_tokens(["one two three", "single"], max_length=16)
    assert emb.dim() == 3 and emb.shape[0] == 2 and emb.shape[2] == D_MODEL  # (B, T, d)
    assert mask.shape == emb.shape[:2]                                       # (B, T)
    # second text is shorter → fewer active mask positions
    assert mask[0].sum() > mask[1].sum()


def test_encode_tokens_pad_to_max_gives_fixed_T():
    fe = _make_fake_frozen_encoder()
    emb, mask = fe.encode_tokens(["a b c", "x"], max_length=32, pad_to_max=True)
    assert emb.shape[1] == 32 and mask.shape[1] == 32  # fixed T = max_length


# ---------------------------------------------------------------------------
# encode_tokens_batched — fixed-T fp16 cache across batches
# ---------------------------------------------------------------------------


def test_encode_tokens_batched_fixed_T_and_fp16():
    fe = _make_fake_frozen_encoder()
    texts = [f"text number {i} with words" for i in range(5)]
    tokens, mask = fe.encode_tokens_batched(texts, batch_size=2, t_cap=16)
    # batches (sizes 2,2,1) concatenate cleanly because every batch is padded to t_cap
    assert tokens.shape == (5, 16, D_MODEL)
    assert mask.shape == (5, 16)
    assert tokens.dtype == np.float16
    assert mask.dtype == np.int8


# ---------------------------------------------------------------------------
# data.encode_or_load_tokens — per-token cache (N, T, d) fp16 + (N, T) mask
# ---------------------------------------------------------------------------

pd = pytest.importorskip("pandas")
from phase1 import cycle as cycle_mod
from phase1 import data


class _FakeFrozenForCache:
    """Stand-in for FrozenEncoder used by data.encode_or_load_tokens — only the methods
    that function calls. Records call count to prove caching skips re-encode."""

    n_constructed = 0

    def __init__(self, name):
        type(self).n_constructed += 1
        self.d_model = D_MODEL

    def to(self, device):
        return self

    def encode_tokens_batched(self, texts, batch_size=32, t_cap=128):
        n = len(texts)
        tokens = np.ones((n, t_cap, D_MODEL), dtype=np.float16)
        mask = np.ones((n, t_cap), dtype=np.int8)
        return tokens, mask


def test_encode_or_load_tokens_shape_and_dtype(monkeypatch, tmp_path):
    monkeypatch.setattr(cycle_mod, "FrozenEncoder", _FakeFrozenForCache)
    corpus = pd.DataFrame({"text": ["alpha beta", "gamma", "delta epsilon zeta"]})
    tokens, mask = data.encode_or_load_tokens(
        corpus, t_cap=16, cache_dir=tmp_path, batch_size=2,
    )
    assert tokens.shape == (3, 16, D_MODEL) and tokens.dtype == np.float16
    assert mask.shape == (3, 16) and mask.dtype == np.int8


def test_encode_or_load_tokens_uses_cache_on_second_call(monkeypatch, tmp_path):
    _FakeFrozenForCache.n_constructed = 0
    monkeypatch.setattr(cycle_mod, "FrozenEncoder", _FakeFrozenForCache)
    corpus = pd.DataFrame({"text": ["alpha beta", "gamma"]})
    data.encode_or_load_tokens(corpus, t_cap=16, cache_dir=tmp_path)
    data.encode_or_load_tokens(corpus, t_cap=16, cache_dir=tmp_path)
    # second call hits the (tokens, mask) npy cache → encoder constructed only once
    assert _FakeFrozenForCache.n_constructed == 1


# ---------------------------------------------------------------------------
# Real BGE per-token shape — Colab only (downloads the model)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_real_bge_encode_tokens_shape():
    fe = cycle_mod.FrozenEncoder("BAAI/bge-large-en-v1.5")
    emb, mask = fe.encode_tokens(["A short operation instance.", "Another one here."], max_length=32)
    assert emb.shape[0] == 2 and emb.shape[2] == 1024  # (B, T, 1024)
    assert mask.shape == emb.shape[:2]
