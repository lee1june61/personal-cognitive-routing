"""Frozen sentence encoder — superset of the phase1 / phase1_5 copies.

This unifies two genuinely-divergent originals:

- ``phase1/cycle.py:FrozenEncoder`` — default ``BAAI/bge-large-en-v1.5``; has a pooled
  ``forward`` (masked-mean) + ``encode_batched`` (numpy concat); ``encode_tokens`` /
  ``encode_tokens_batched`` with NO prefix.
- ``phase1_5/encoders.py:FrozenEncoder`` — default ``intfloat/e5-large-v2``; adds an
  optional E5 ``prefix`` kwarg on every encode path, ``encode_pooled`` /
  ``encode_pooled_batched`` (OOM-safe preallocated numpy), and ``self.model_name``.

The core version is the union: every method from both, ``prefix`` optional (default
``""`` → identical to phase1's no-prefix behaviour), default model name kept as
``intfloat/e5-large-v2`` (no production call site relies on the default — all pass the
name explicitly). ``self.model_name`` is set, and methods that reference it for a tqdm
label use ``getattr`` so a test-fabricated encoder (``object.__new__`` without
``model_name``) still works.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


DEFAULT_ENCODER_NAME = "intfloat/e5-large-v2"


class FrozenEncoder(nn.Module):
    """Frozen HuggingFace encoder with per-token + pooled output and an optional prefix.

    Args:
        model_name: HF model id. Defaults to ``intfloat/e5-large-v2``.

    Attributes:
        d_model: encoder hidden size (1024 for e5-large-v2 and BGE-large-en-v1.5).
        model_name: the HF model id.
        tokenizer: HF tokenizer.
        encoder: HF AutoModel (frozen, eval mode).
    """

    def __init__(self, model_name: str = DEFAULT_ENCODER_NAME):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        self.d_model: int = self.encoder.config.hidden_size

    # ---- pooled forward (phase1 masked-mean pool) ----------------------------

    @torch.no_grad()
    def forward(self, texts: list[str], max_length: int = 256) -> torch.Tensor:
        """Masked-mean pool over the frozen encoder (phase1 default path)."""
        device = next(self.encoder.parameters()).device
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = self.encoder(**enc).last_hidden_state                              # (B, T, d)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        fact_emb = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)        # (B, d)
        return fact_emb

    @torch.no_grad()
    def encode_batched(self, texts: list[str], batch_size: int = 32, max_length: int = 256):
        """Numpy-yielding batched pooled encoder for cache-building (phase1 data.py)."""
        from tqdm.auto import tqdm
        out: list[np.ndarray] = []
        for i in tqdm(range(0, len(texts), batch_size), desc="encode"):
            batch = texts[i:i + batch_size]
            emb = self(batch, max_length=max_length).cpu().numpy()
            out.append(emb)
        return np.concatenate(out, axis=0)

    # ---- core encoding paths -------------------------------------------------

    @torch.no_grad()
    def encode_tokens(
        self,
        texts: list[str],
        *,
        prefix: str = "",
        max_length: int = 128,
        pad_to_max: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token (NO pooling) encode. Used for cross-attention KV input.

        Args:
            texts: list of input strings (no prefix yet).
            prefix: string prepended to each input before tokenization.
                ``"query: "`` for e5 Q-side, ``"passage: "`` for e5 P-side, ``""`` else.
            max_length: tokenizer truncation length (T_cap).
            pad_to_max: if True, every batch padded to exactly max_length (fixed-T cache).

        Returns:
            tuple ``(hidden, mask)`` where:
              - hidden: ``(B, T, d_model)`` last_hidden_state (no pooling)
              - mask: ``(B, T)`` int attention mask (1=real, 0=pad)
        """
        device = next(self.encoder.parameters()).device
        prefixed = [prefix + t for t in texts] if prefix else list(texts)
        enc = self.tokenizer(
            prefixed,
            padding=("max_length" if pad_to_max else True),
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        hidden = self.encoder(**enc).last_hidden_state  # (B, T, d)
        return hidden, enc["attention_mask"]

    @torch.no_grad()
    def encode_tokens_batched(
        self,
        texts: list[str],
        *,
        prefix: str = "",
        batch_size: int = 32,
        t_cap: int = 128,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batched per-token encode → fixed-T fp16 cache.

        Pre-allocates the output (N, t_cap, d) fp16 and (N, t_cap) int8 to keep peak
        RAM at ~1x the cache (vs ~2x for list-and-concat). At N=40k, T=128, d=1024 the
        fp16 tokens are ~10 GB and concat would briefly hold the chunks *and* the
        output (~2x peak), overflowing standard Colab RAM.

        Returns:
            ``(tokens (N, t_cap, d_model) fp16, mask (N, t_cap) int8)``.
        """
        from tqdm.auto import tqdm

        desc = f"encode_tokens[{getattr(self, 'model_name', '')}]"
        n = len(texts)
        tokens = np.empty((n, t_cap, self.d_model), dtype=np.float16)
        mask = np.empty((n, t_cap), dtype=np.int8)
        for i in tqdm(range(0, n, batch_size), desc=desc):
            batch = texts[i : i + batch_size]
            emb_b, mask_b = self.encode_tokens(
                batch, prefix=prefix, max_length=t_cap, pad_to_max=True
            )
            tokens[i : i + len(batch)] = emb_b.cpu().to(torch.float16).numpy()
            mask[i : i + len(batch)] = mask_b.cpu().to(torch.int8).numpy()
        return tokens, mask

    @torch.no_grad()
    def encode_pooled(
        self,
        texts: list[str],
        *,
        prefix: str = "",
        max_length: int = 64,
        l2_normalize: bool = True,
    ) -> torch.Tensor:
        """Masked-mean pool, optional L2-normalize. Used for MC candidate embeddings.

        Args:
            texts: list of strings (e.g. MC candidate sentences).
            prefix: e5 prefix string (``"query: "`` for candidates per the paper §5.4
                convention — candidates treated as query-side for symmetry with retrieval).
            max_length: tokenizer truncation length.
            l2_normalize: if True, L2-normalize each pooled vector (e5 retrieval protocol).

        Returns:
            ``(B, d_model)`` pooled embedding.
        """
        device = next(self.encoder.parameters()).device
        prefixed = [prefix + t for t in texts] if prefix else list(texts)
        enc = self.tokenizer(
            prefixed,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        hidden = self.encoder(**enc).last_hidden_state  # (B, T, d)
        mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # (B, d)
        if l2_normalize:
            pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return pooled

    @torch.no_grad()
    def encode_pooled_batched(
        self,
        texts: list[str],
        *,
        prefix: str = "",
        batch_size: int = 64,
        max_length: int = 64,
        l2_normalize: bool = True,
    ) -> np.ndarray:
        """Batched pooled-emb numpy yield. Used for caching ``cand_pooled (N·4, d)``.

        Preallocates the output ``(n, d_model)`` fp32 and slice-assigns per batch —
        same OOM-safe pattern as ``encode_tokens_batched``. At cand_pooled scale
        (~80k × 1024 fp32 ≈ 312 MB) the list+concat anti-pattern would briefly
        double peak RAM.
        """
        from tqdm.auto import tqdm

        desc = f"encode_pooled[{getattr(self, 'model_name', '')}]"
        n = len(texts)
        out = np.empty((n, self.d_model), dtype=np.float32)
        for i in tqdm(range(0, n, batch_size), desc=desc):
            batch = texts[i : i + batch_size]
            emb = self.encode_pooled(
                batch, prefix=prefix, max_length=max_length, l2_normalize=l2_normalize
            )
            out[i : i + len(batch)] = emb.cpu().to(torch.float32).numpy()
        return out


def default_q_prefix(model_name: str) -> str:
    """Prefix convention by encoder family. e5 → 'query: '; bge → ''."""
    if "e5" in model_name.lower():
        return "query: "
    return ""


def default_p_prefix(model_name: str) -> str:
    """Prefix convention by encoder family. e5 → 'passage: '; bge → ''."""
    if "e5" in model_name.lower():
        return "passage: "
    return ""
