"""Frozen sentence encoder with per-token + pooled output and prefix support.

Default = `intfloat/e5-large-v2` (1024d). Paper §5.4 commit on the basis of the
Phase 1 Stage 1 in-house ceiling (raw e5 operation adjacency ≈ 0.60 at q-first
T=256). For Row C ablation, swap to `BAAI/bge-large-en-v1.5` (1024d as well —
matches d_emb without architecture change).

E5 prefix protocol (HF model card, e5 family default):
- query side  → ``"query: "``
- passage side → ``"passage: "``
This is the *documented* retrieval protocol; for sentence-similarity the prefix
also yields stronger geometry. We apply it explicitly per call rather than
baking into the encoder so A.3 (full-P-cross-encoded) can compose
``"query: <Q> [SEP] <P>"`` without double-prefixing P.

BGE prefix (Row C ablation): BGE-large-en-v1.5 documents a query prefix only
for retrieval scoring; for the operation-axis probe + cross-attention setup we
use no prefix (matching Phase 1's `cycle.py` default).

OOM-safe `encode_tokens_batched` pattern lifted verbatim from
`phase1/cycle.py:FrozenEncoder.encode_tokens_batched` — pre-allocated output
keeps peak RAM at ~1x the cache size at N=20k, T=256, d=1024 (~10 GB fp16).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


DEFAULT_ENCODER_NAME = "intfloat/e5-large-v2"


class FrozenEncoder(nn.Module):
    """Frozen HuggingFace encoder with per-token + pooled output and prefix kwarg.

    Args:
        model_name: HF model id. Defaults to ``intfloat/e5-large-v2``.

    Attributes:
        d_model: encoder hidden size (1024 for e5-large-v2 and BGE-large-en-v1.5).
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
        RAM at ~1x the cache (vs ~2x for list-and-concat). Lifted from
        ``phase1/cycle.py:encode_tokens_batched`` — see that docstring for the
        OOM rationale at N=40k.

        Returns:
            ``(tokens (N, t_cap, d_model) fp16, mask (N, t_cap) int8)``.
        """
        from tqdm.auto import tqdm

        n = len(texts)
        tokens = np.empty((n, t_cap, self.d_model), dtype=np.float16)
        mask = np.empty((n, t_cap), dtype=np.int8)
        for i in tqdm(range(0, n, batch_size), desc=f"encode_tokens[{self.model_name}]"):
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

        n = len(texts)
        out = np.empty((n, self.d_model), dtype=np.float32)
        for i in tqdm(range(0, n, batch_size), desc=f"encode_pooled[{self.model_name}]"):
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
