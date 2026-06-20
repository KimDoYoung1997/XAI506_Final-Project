"""SmolVLM2 — VAR-style natural-language report for offside demo."""

from __future__ import annotations

import os
import re
from pathlib import Path

import torch
from PIL import Image

from utils import release_torch_memory

try:
    import num2words  # noqa: F401 — SmolVLM2 processor dependency
except ImportError as e:
    raise ImportError("pip install num2words") from e

from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


def pick_smolvlm_device(verbose: bool = True) -> torch.device:
    forced = os.environ.get("SMOLVLM_DEVICE", "").strip().lower()
    if forced in ("cuda", "mps", "cpu"):
        if forced == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("SMOLVLM_DEVICE=cuda but CUDA is not available.")
        if forced == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise RuntimeError("SMOLVLM_DEVICE=mps but MPS is not available.")
        if verbose:
            print(f"[device] SMOLVLM_DEVICE={forced}")
        return torch.device(forced)
    if torch.cuda.is_available():
        if verbose:
            print("[device] CUDA")
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if verbose:
            print("[device] MPS")
        return torch.device("mps")
    if verbose:
        print("[device] CPU")
    return torch.device("cpu")


def _dtype_for_device(dev: torch.device) -> torch.dtype:
    if dev.type == "cuda":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if dev.type == "mps":
        return torch.float16
    return torch.float32


def load_smolvlm2(device: torch.device | None = None, verbose: bool = True):
    device = device or pick_smolvlm_device(verbose=verbose)
    torch_dtype = _dtype_for_device(device)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    load_kw: dict = dict(torch_dtype=torch_dtype, trust_remote_code=True)
    if device.type == "cuda":
        load_kw["device_map"] = "auto"

    model = None
    last_err = None
    attn_order = ("flash_attention_2", "sdpa", "eager") if device.type == "cuda" else ("sdpa", "eager")
    for impl in attn_order:
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                MODEL_ID,
                attn_implementation=impl,
                **load_kw,
            )
            if verbose and impl != attn_order[0]:
                print(f"[SmolVLM2] attn fallback: {impl}")
            break
        except Exception as e:
            last_err = e
            model = None
    if model is None:
        raise RuntimeError(f"SmolVLM2 load failed: {last_err}")

    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    if verbose:
        print(f"loaded: {MODEL_ID} on {device} dtype={torch_dtype}")
    return processor, model, device, torch_dtype


def release_smolvlm2(processor, model) -> None:
    del processor, model
    release_torch_memory("SmolVLM2")


def _move_inputs_to_device(inputs, device: torch.device, torch_dtype: torch.dtype):
    if hasattr(inputs, "to"):
        try:
            return inputs.to(device=device, dtype=torch_dtype)
        except Exception:
            pass
    for k in list(inputs.keys()):
        v = inputs[k]
        if hasattr(v, "to"):
            if torch.is_floating_point(v):
                inputs[k] = v.to(device=device, dtype=torch_dtype)
            else:
                inputs[k] = v.to(device=device)
    return inputs


@torch.inference_mode()
def _generate_answer(
    processor,
    model,
    device: torch.device,
    torch_dtype: torch.dtype,
    messages: list,
    max_new_tokens: int = 160,
) -> str:
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    if isinstance(inputs, dict) and "token_type_ids" in inputs:
        inputs.pop("token_type_ids", None)
    inputs = _move_inputs_to_device(inputs, device, torch_dtype)
    out_ids = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        repetition_penalty=1.15,
    )
    in_ids = inputs["input_ids"]
    trimmed = [o[len(i):] for i, o in zip(in_ids, out_ids)]
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        return str(tok.decode(trimmed[0], skip_special_tokens=True))
    return str(processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0])


def build_analysis_prompt(*, is_offside: bool) -> str:
    if is_offside:
        hint = "공격수가 마지막 수비수보다 앞에 있는 것으로 보입니다 — 오프사이드 의심."
    else:
        hint = "공격수가 수비수와 같은 라인이거나 뒤에 있는 것으로 보입니다 — 온사이드."
    return f"""축구 VAR 정지 화면. 빨강=공격수, 파랑=수비수, 노랑=오프사이드 라인.
{hint}

반드시 한국어로만, 픽셀·숫자·거리 없이 문장 하나만 작성하세요. 영어 사용 금지.
시작: 분석:"""


def build_broadcast_line(is_offside: bool) -> str:
    """Fixed live-TV shout for TTS — never mentions pixels."""
    if is_offside:
        return "오! 깃발이 올라갔습니다! 공격수가 마지막 수비수보다 앞에 있습니다! 오프사이드!"
    return "플레이 계속! 공격수는 온사이드입니다! 깃발 없습니다!"


def _has_pixel_refs(text: str) -> bool:
    lower = text.lower()
    return "px" in lower or "pixel" in lower or "픽셀" in text


def _has_hangul(text: str) -> bool:
    return any("\uac00" <= c <= "\ud7a3" for c in text)


def extract_analysis_line(report: str, fallback: str) -> str:
    patterns = (
        r"^\s*분석\s*:\s*(.+)\s*$",
        r"^\s*Analysis\s*:\s*(.+)\s*$",
    )
    for line in report.splitlines():
        for pattern in patterns:
            m = re.match(pattern, line, flags=re.IGNORECASE)
            if m:
                text = m.group(1).strip()
                if text and _has_hangul(text) and not _has_pixel_refs(text):
                    return text
    return fallback


def fallback_analysis(is_offside: bool) -> str:
    if is_offside:
        return "공격수가 마지막 수비수보다 앞에 있어 오프사이드 포지션이 의심됩니다."
    return "이 프레임에서 공격수는 마지막 수비수보다 앞에 있지 않습니다."


def compose_var_report(analysis: str, broadcast: str) -> str:
    return f"분석: {analysis}\n중계: {broadcast}\n"


def generate_var_report(
    processor,
    model,
    device: torch.device,
    torch_dtype: torch.dtype,
    annotated_image: Image.Image,
    prompt: str,
    max_new_tokens: int = 80,
) -> str:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": annotated_image.convert("RGB")},
        {"type": "text", "text": prompt},
    ]}]
    return _generate_answer(
        processor, model, device, torch_dtype, messages, max_new_tokens=max_new_tokens,
    )


def save_var_report(report: str, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(report.strip() + "\n", encoding="utf-8")
    print(f"Saved VAR report: {save_path}")


def print_var_report(report: str) -> None:
    print("\n" + "=" * 50)
    print("SmolVLM2 — VAR REPORT")
    print("=" * 50)
    print(report.strip())
    print("=" * 50)


def make_var_report(annotated_image: Image.Image, is_offside: bool) -> str:
    """Load SmolVLM2, generate Analysis line, compose full VAR report, release model."""
    processor, model, device, torch_dtype = load_smolvlm2()
    try:
        raw = generate_var_report(
            processor, model, device, torch_dtype,
            annotated_image,
            build_analysis_prompt(is_offside=is_offside),
        )
    finally:
        release_smolvlm2(processor, model)
    analysis = extract_analysis_line(raw, fallback_analysis(is_offside))
    return compose_var_report(analysis, build_broadcast_line(is_offside))
