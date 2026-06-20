"""Shared helpers."""

from __future__ import annotations

import gc

import torch


def release_torch_memory(label: str) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    print(f"[memory] Released {label}")
