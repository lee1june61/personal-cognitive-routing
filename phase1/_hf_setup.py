"""HuggingFace token auto-loader + GPU compute-dtype detection.

Module import 시 자동 실행 — Colab Secret 또는 기존 env var 로부터 HF_TOKEN 을 환경변수에
주입. transformers / datasets / huggingface_hub 가 자동으로 read.

Also exposes `detect_compute_dtype()` — auto bf16 vs fp16 vs fp32 based on GPU capability
(A100/H100/RTX30/40 = bf16, T4/V100 = fp16, CPU = fp32). Used by bitsandbytes 4bit config.

Usage:
    from . import _hf_setup    # side-effect import in any phase1 module
or:
    from ._hf_setup import setup_hf_token, detect_compute_dtype
    setup_hf_token()
    dtype = detect_compute_dtype()

Idempotent — 이미 set 된 token 은 덮어쓰지 않음.
"""

from __future__ import annotations

import contextlib
import os

import torch


_DONE = False


def setup_hf_token(verbose: bool = True) -> bool:
    """Load HF token from Colab Secret or existing env. Returns True if token set."""
    global _DONE
    if _DONE:
        return bool(os.environ.get("HF_TOKEN"))

    # Already set (local shell, .bashrc, etc.)
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        # Mirror to both common env vars
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        if verbose:
            print("[hf_setup] HF_TOKEN found in env, mirrored.")
        _DONE = True
        return True

    # Try Colab Secret
    try:
        from google.colab import userdata  # type: ignore
        try:
            token = userdata.get("HF_TOKEN")
        except Exception:
            token = None
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token
            if verbose:
                print("[hf_setup] HF_TOKEN loaded from Colab Secret.")
            _DONE = True
            return True
    except ImportError:
        pass

    if verbose:
        print(
            "[hf_setup] note: no HF_TOKEN in env or Colab Secret. "
            "BGE-large-en (revision 4 encoder) is public — works without token. "
            "Gated models (Phase 3 future LLM rendering) would need a token. "
            "Set via Colab Secret (key icon → HF_TOKEN) or `export HF_TOKEN=hf_xxx`."
        )
    _DONE = True
    return False


def detect_compute_dtype() -> torch.dtype:
    """Auto-select compute dtype for bnb 4bit quant + cast operations.

    - GPU compute capability >= 8.0 (A100, H100, RTX 30/40): bf16 (preferred — wider range)
    - GPU compute capability < 8.0 (T4 sm_75, V100 sm_70, RTX 20 sm_75): fp16 fallback
    - CPU: fp32 (no quantization supported anyway)
    """
    if not torch.cuda.is_available():
        return torch.float32
    try:
        cap = torch.cuda.get_device_capability(0)
        if cap[0] >= 8:
            return torch.bfloat16
        return torch.float16
    except Exception:
        return torch.float16


@contextlib.contextmanager
def compute_autocast(device: torch.device):
    """torch.autocast wrapper with auto-detected dtype + CUDA/CPU device-type fallback."""
    device_type = "cuda" if device.type == "cuda" else "cpu"
    with torch.autocast(device_type=device_type, dtype=detect_compute_dtype()):
        yield


@contextlib.contextmanager
def generation_mode(llm):
    """Temporarily disable gradient_checkpointing + enable KV cache for `generate()`.

    Required: `prepare_model_for_kbit_training` enables gradient_checkpointing which
    forces use_cache=False; generate() under that combination breaks SDPA attention
    mask shape. Restores prior state on exit.
    """
    had_disable = hasattr(llm, "gradient_checkpointing_disable")
    had_enable = hasattr(llm, "gradient_checkpointing_enable")
    if had_disable:
        try:
            llm.gradient_checkpointing_disable()
        except AttributeError:
            pass
    prev_use_cache = getattr(llm.config, "use_cache", True)
    llm.config.use_cache = True
    try:
        yield
    finally:
        llm.config.use_cache = prev_use_cache
        if had_enable:
            try:
                llm.gradient_checkpointing_enable()
            except AttributeError:
                pass
