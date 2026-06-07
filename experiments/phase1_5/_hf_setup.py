"""HuggingFace token auto-loader + GPU compute-dtype detection.

Verbatim copy of `phase1/_hf_setup.py` (Phase 1.5 keeps the package self-contained
to avoid Colab path-resolution coupling to phase1).

Module import 시 자동 실행 — Colab Secret 또는 기존 env var 로부터 HF_TOKEN 을
환경변수에 주입. transformers / datasets / huggingface_hub 가 자동으로 read.

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

    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        if verbose:
            print("[hf_setup] HF_TOKEN found in env, mirrored.")
        _DONE = True
        return True

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
            "e5-large-v2 (default encoder) and BGE-large-en (Row C swap) are public — "
            "work without token. Set via Colab Secret (HF_TOKEN) only if a gated "
            "model is added later."
        )
    _DONE = True
    return False


def detect_compute_dtype() -> torch.dtype:
    """Auto-select compute dtype.

    - GPU compute capability >= 8.0 (A100, H100, RTX 30/40): bf16
    - GPU compute capability < 8.0 (T4, V100, RTX 20): fp16
    - CPU: fp32
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
    device_type = "cuda" if device.type == "cuda" else "cpu"
    with torch.autocast(device_type=device_type, dtype=detect_compute_dtype()):
        yield
