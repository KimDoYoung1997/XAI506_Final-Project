#!/usr/bin/env python3
"""
Offside detection demo — SAM2 + SmolVLM2 + Qwen3-TTS.

Usage:
    python demo.py
    python demo.py --no-explain
    python demo.py --image imgs/offside.png
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from offside_core import (
    OffsideResult,
    analyze_masks,
    build_pitch_frame,
    mask_centroid,
    pick_calibration_points,
    print_calibration,
    print_verdict,
    save_player_crop,
    visualize,
)
from sam2_helpers import load_sam2, release_sam2, segment_player
from smolvlm_helpers import build_broadcast_line, make_var_report, print_var_report, save_var_report
from tts_helpers import run_commentary

ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE = ROOT / "imgs" / "offside.png"
OUTPUT_DIR = ROOT / "outputs"
OFFSIDE_FLAG_IMAGE = ROOT / "imgs" / "offside_flag.png"


def _play_audio(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(["afplay", str(path)], check=False)
    else:
        print(f"[audio] Saved to {path} (auto-play is macOS only; open the file manually)")


def _play_audio_with_offside_flag(wav_path: Path, flag_path: Path) -> None:
    """Show assistant-referee flag image while commentary audio plays."""
    import time

    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QPixmap
    from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

    player_cmd = ["afplay", str(wav_path)] if sys.platform == "darwin" else None
    if player_cmd is None:
        _play_audio(wav_path)
        return

    player = subprocess.Popen(player_cmd)

    app = QApplication.instance()
    created_app = False
    if app is None:
        app = QApplication(sys.argv)
        created_app = True

    win = QWidget()
    win.setWindowTitle("OFFSIDE — Assistant Ref Flag")
    layout = QVBoxLayout(win)
    label = QLabel()
    pixmap = QPixmap(str(flag_path))
    max_w = 520
    if pixmap.width() > max_w:
        pixmap = pixmap.scaledToWidth(max_w, Qt.SmoothTransformation)
    label.setPixmap(pixmap)
    label.setAlignment(Qt.AlignCenter)
    layout.addWidget(label)
    win.adjustSize()
    win.show()

    while player.poll() is None:
        app.processEvents()
        time.sleep(0.03)

    win.close()
    if created_app:
        try:
            app.quit()
        except Exception:
            pass


def _play_commentary(wav_path: Path, is_offside: bool) -> None:
    if is_offside and OFFSIDE_FLAG_IMAGE.exists():
        print(f"Showing offside flag: {OFFSIDE_FLAG_IMAGE}")
        _play_audio_with_offside_flag(wav_path, OFFSIDE_FLAG_IMAGE)
    else:
        _play_audio(wav_path)


def run_sam2_phase(img_pil: Image.Image, img_np: np.ndarray) -> tuple[
    OffsideResult, np.ndarray, np.ndarray
]:
    """Interactive SAM2 segmentation + geometric offside verdict."""
    model, processor, device = load_sam2()
    try:
        s1, s2, goal_pt = pick_calibration_points(img_np)
        attacker_mask, _, _ = segment_player(
            model, processor, device, img_pil, img_np, "3) Attacking player (pass receiver)",
        )
        defender_mask, _, _ = segment_player(
            model, processor, device, img_pil, img_np, "4) Last defender",
        )
        frame = build_pitch_frame(s1, s2, goal_pt, mask_centroid(defender_mask))
        result = analyze_masks(attacker_mask, defender_mask, frame, s1, s2)
        print_calibration(s1, s2, frame)
        print_verdict(result)
        return result, attacker_mask, defender_mask
    finally:
        release_sam2(model, processor)


def run_explain_phase(result: OffsideResult, offside_path: Path, stem: str) -> None:
    """SmolVLM2 VAR report + Qwen3-TTS broadcast audio."""
    broadcast = build_broadcast_line(result.is_offside)
    report = make_var_report(Image.open(offside_path).convert("RGB"), result.is_offside)
    print_var_report(report)
    print(f"\n[Broadcast line for TTS]\n  {broadcast}")

    report_path = OUTPUT_DIR / f"{stem}_report.txt"
    save_var_report(report, report_path)

    wav_path = OUTPUT_DIR / f"{stem}_commentary.wav"
    run_commentary(broadcast, wav_path)
    _play_commentary(wav_path, result.is_offside)


def run(
    image_path: Path,
    show_plot: bool = True,
    with_explain: bool = True,
) -> OffsideResult:
    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.array(img_pil)
    stem = image_path.stem

    result, attacker_mask, defender_mask = run_sam2_phase(img_pil, img_np)

    save_player_crop(
        img_np, attacker_mask, defender_mask,
        OUTPUT_DIR / f"{stem}_player.png",
    )
    offside_path = OUTPUT_DIR / f"{stem}_offside.png"
    visualize(img_np, result, attacker_mask, defender_mask, offside_path, show=show_plot)

    if with_explain:
        run_explain_phase(result, offside_path, stem)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Offside demo: SAM2 + SmolVLM2 + Qwen3-TTS")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE,
                        help=f"Input image (default: {DEFAULT_IMAGE})")
    parser.add_argument("--no-explain", action="store_true",
                        help="Skip SmolVLM2 VAR report + Qwen3-TTS commentary audio")
    args = parser.parse_args()
    if not args.image.exists():
        raise FileNotFoundError(args.image)
    run(args.image, with_explain=not args.no_explain)


if __name__ == "__main__":
    main()
