"""Qwen3-TTS — live commentary (Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from utils import release_torch_memory

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_SPEAKER = "Ryan"
DEFAULT_INSTRUCT = "Shout like a World Cup goal commentator!"


def pick_qwen_tts_device(verbose: bool = True) -> str:
    forced = os.environ.get("QWEN_TTS_DEVICE", "").strip().lower()
    if forced in ("cuda", "cuda:0", "mps", "cpu"):
        if forced.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("QWEN_TTS_DEVICE=cuda but CUDA is not available.")
        if forced == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise RuntimeError("QWEN_TTS_DEVICE=mps but MPS is not available.")
        if verbose:
            print(f"[device] QWEN_TTS_DEVICE={forced}")
        return forced if forced != "cuda" else "cuda:0"
    if torch.cuda.is_available():
        if verbose:
            print("[device] CUDA")
        return "cuda:0"
    if verbose:
        print("[device] CPU (Qwen3-TTS default on Mac — MPS can be unstable)")
    return "cpu"


def _dtype_for_device(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def pick_qwen_speaker() -> str:
    return os.environ.get("QWEN_TTS_SPEAKER", DEFAULT_SPEAKER).strip() or DEFAULT_SPEAKER


def pick_qwen_instruct() -> str:
    return os.environ.get("QWEN_TTS_INSTRUCT", DEFAULT_INSTRUCT).strip() or DEFAULT_INSTRUCT


def load_qwen3_tts():
    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError as e:
        raise ImportError(
            "qwen-tts is not installed. Run: pip install qwen-tts soundfile"
        ) from e
    device = pick_qwen_tts_device()
    dtype = _dtype_for_device(device)
    model_id = os.environ.get("QWEN_TTS_MODEL", MODEL_ID).strip() or MODEL_ID
    base_kw = dict(device_map=device, dtype=dtype)
    if device.startswith("cuda"):
        try:
            model = Qwen3TTSModel.from_pretrained(
                model_id, attn_implementation="flash_attention_2", **base_kw,
            )
        except Exception:
            model = Qwen3TTSModel.from_pretrained(model_id, **base_kw)
    else:
        model = Qwen3TTSModel.from_pretrained(model_id, **base_kw)
    print(
        f"loaded: {model_id} on {device} "
        f"speaker={pick_qwen_speaker()} instruct={pick_qwen_instruct()!r}"
    )
    return model


def release_qwen3_tts(model) -> None:
    del model
    release_torch_memory("Qwen3-TTS")


@torch.inference_mode()
def synthesize_commentary(model, text: str) -> tuple[np.ndarray, int]:
    text = text.strip()
    if not text:
        raise ValueError("Empty broadcast text.")
    wavs, sr = model.generate_custom_voice(
        text=text,
        language="English",
        speaker=pick_qwen_speaker(),
        instruct=pick_qwen_instruct(),
    )
    return np.asarray(wavs[0], dtype=np.float32), int(sr)


def save_commentary_wav(model, text: str, save_path: Path) -> Path:
    wav, sr = synthesize_commentary(model, text)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(save_path), wav, sr)
    print(f"Saved commentary audio: {save_path}")
    return save_path


def run_commentary(text: str, save_path: Path) -> Path:
    """Load Qwen3-TTS, synthesize broadcast line, save WAV, release model."""
    model = load_qwen3_tts()
    try:
        return save_commentary_wav(model, text, save_path)
    finally:
        release_qwen3_tts(model)
