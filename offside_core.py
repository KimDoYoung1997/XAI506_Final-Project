"""Pitch calibration, geometric offside logic, and visualization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from sam2_helpers import pick_pos_neg_points_qt

ATTACKER_COLOR = np.array([220, 40, 40])
DEFENDER_COLOR = np.array([30, 100, 255])
MASK_ALPHA = 0.50


@dataclass
class PitchFrame:
    pitch_dir: np.ndarray
    goal_dir: np.ndarray


@dataclass
class OffsideResult:
    is_offside: bool
    attacker_point: tuple[float, float]
    defender_point: tuple[float, float]
    attacker_goal_coord: float
    defender_goal_coord: float
    margin_att_vs_def: float
    frame: PitchFrame
    stripe_p1: tuple[float, float]
    stripe_p2: tuple[float, float]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        raise ValueError("Zero-length vector.")
    return v / n


def mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("Empty mask.")
    return float(np.mean(xs)), float(np.mean(ys))


def _mask_max_goal_point(mask: np.ndarray, frame: PitchFrame) -> tuple[tuple[float, float], float]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("Empty mask.")
    scores = xs.astype(np.float64) * frame.goal_dir[0] + ys.astype(np.float64) * frame.goal_dir[1]
    i = int(np.argmax(scores))
    return (float(xs[i]), float(ys[i])), float(scores[i])


def build_pitch_frame(
    stripe_p1: tuple[float, float],
    stripe_p2: tuple[float, float],
    goal_hint: list[int],
    reference_point: tuple[float, float],
) -> PitchFrame:
    pitch_dir = _unit(np.array(stripe_p2, dtype=np.float64) - np.array(stripe_p1, dtype=np.float64))
    perp_a = np.array([-pitch_dir[1], pitch_dir[0]])
    perp_b = -perp_a
    hint = np.array(goal_hint, dtype=np.float64)
    ref = np.array(reference_point, dtype=np.float64)
    to_hint = hint - ref

    goal_dir = _unit(perp_a if np.dot(to_hint, perp_a) >= np.dot(to_hint, perp_b) else perp_b)
    if np.dot(to_hint, goal_dir) < 0:
        goal_dir = -goal_dir
    return PitchFrame(pitch_dir=pitch_dir, goal_dir=goal_dir)


def analyze_masks(
    attacker_mask: np.ndarray,
    defender_mask: np.ndarray,
    frame: PitchFrame,
    stripe_p1: tuple[float, float],
    stripe_p2: tuple[float, float],
    tol_px: float = 2.0,
) -> OffsideResult:
    att_pt, s_att = _mask_max_goal_point(attacker_mask, frame)
    def_pt, s_def = _mask_max_goal_point(defender_mask, frame)
    margin = s_att - s_def
    return OffsideResult(
        is_offside=margin > tol_px,
        attacker_point=att_pt,
        defender_point=def_pt,
        attacker_goal_coord=s_att,
        defender_goal_coord=s_def,
        margin_att_vs_def=margin,
        frame=frame,
        stripe_p1=stripe_p1,
        stripe_p2=stripe_p2,
    )


def _offside_line_segment(
    anchor: tuple[float, float],
    frame: PitchFrame,
    half_len: float = 2000.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    c = np.array(anchor, dtype=np.float64)
    d = frame.pitch_dir * half_len
    return (tuple(c - d), tuple(c + d))


def _blend_masks(base: np.ndarray, masks: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    out = base.astype(np.float32)
    for mask, color in masks:
        out[mask] = (1.0 - MASK_ALPHA) * out[mask] + MASK_ALPHA * color
    return out.astype(np.uint8)


def pick_calibration_points(img_np: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], list[int]]:
    print("\n=== [1/4] Pitch parallel-line calibration ===")
    print("Left-click 2 points on the SAME grass-stripe edge (not across stripes), then Enter")
    stripe_pts, _ = pick_pos_neg_points_qt(img_np, title="1) Same stripe edge — 2 points")
    if len(stripe_pts) < 2:
        raise RuntimeError("Need 2 points on a grass stripe boundary.")

    print("\n=== [2/4] Goal direction ===")
    print("Left-click 1 point toward the goal, then Enter")
    goal_pts, _ = pick_pos_neg_points_qt(img_np, title="2) Goal direction — 1 point")
    if not goal_pts:
        raise RuntimeError("Need 1 point toward the goal.")

    p1 = (float(stripe_pts[0][0]), float(stripe_pts[0][1]))
    p2 = (float(stripe_pts[1][0]), float(stripe_pts[1][1]))
    return p1, p2, goal_pts[0]


def format_verdict_line(result: OffsideResult) -> str:
    px = abs(result.margin_att_vs_def)
    if result.margin_att_vs_def > 0:
        return (
            f"Attacker's foremost body point is {px:.0f} px ahead of "
            f"the defender's (toward goal)"
        )
    if result.margin_att_vs_def < 0:
        return (
            f"Attacker's foremost body point is {px:.0f} px behind "
            f"the offside line"
        )
    return "Foremost body points are level with the offside line"


def print_verdict(result: OffsideResult) -> None:
    att, defe = result.attacker_point, result.defender_point
    print("\n" + "=" * 50)
    print("VERDICT")
    print("=" * 50)
    print(f"  Attacker foremost point: ({att[0]:.0f}, {att[1]:.0f})  goal_coord={result.attacker_goal_coord:.1f}")
    print(f"  Defender foremost point: ({defe[0]:.0f}, {defe[1]:.0f})  goal_coord={result.defender_goal_coord:.1f}")
    print(f"  {format_verdict_line(result)}")
    print(f"\n  >> {'OFFSIDE POSITION (suspected)' if result.is_offside else 'NOT OFFSIDE'}")
    print("\n* Uses max goal-axis pixel over full SAM2 mask (any body part). Ball not used.")


def print_calibration(stripe_p1: tuple[float, float], stripe_p2: tuple[float, float], frame: PitchFrame) -> None:
    dx = stripe_p2[0] - stripe_p1[0]
    dy = stripe_p2[1] - stripe_p1[1]
    angle_deg = float(np.degrees(np.arctan2(dy, dx)))
    print(f"\n[calibration] stripe clicks: {stripe_p1} -> {stripe_p2}")
    print(f"[calibration] pitch_dir = ({frame.pitch_dir[0]:.4f}, {frame.pitch_dir[1]:.4f}), angle = {angle_deg:.1f} deg")
    print("[calibration] Cyan and yellow lines share this direction (parallel by construction).")
    print("[calibration] If yellow looks skewed vs OTHER stripes elsewhere, that is camera perspective.")


def visualize(
    img_np: np.ndarray,
    result: OffsideResult,
    attacker_mask: np.ndarray,
    defender_mask: np.ndarray,
    save_path: Path,
    show: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(_blend_masks(img_np, [
        (defender_mask, DEFENDER_COLOR),
        (attacker_mask, ATTACKER_COLOR),
    ]))

    (x0, y0), (x1, y1) = _offside_line_segment(result.defender_point, result.frame)
    ax.plot([x0, x1], [y0, y1], color="yellow", lw=3.0, ls="--",
            label="Offside line (through defender)", zorder=4)

    (cx0, cy0), (cx1, cy1) = _offside_line_segment(
        ((result.stripe_p1[0] + result.stripe_p2[0]) / 2,
         (result.stripe_p1[1] + result.stripe_p2[1]) / 2),
        result.frame,
    )
    ax.plot([cx0, cx1], [cy0, cy1], color="cyan", lw=2.0, ls="-",
            label="Your stripe direction (calibration)", zorder=3)
    ax.plot([result.stripe_p1[0], result.stripe_p2[0]],
            [result.stripe_p1[1], result.stripe_p2[1]],
            color="lime", lw=4.0, solid_capstyle="round", zorder=5)
    ax.scatter([result.stripe_p1[0], result.stripe_p2[0]],
               [result.stripe_p1[1], result.stripe_p2[1]],
               s=100, c="lime", edgecolors="black", linewidths=1.5, zorder=6,
               label="Your 2 stripe clicks")

    ax.scatter(*result.attacker_point, s=160, c="red", edgecolors="white",
               linewidths=1.5, zorder=6, marker="*",
               label="Attacker foremost point")
    ax.scatter(*result.defender_point, s=160, c="dodgerblue", edgecolors="white",
               linewidths=1.5, zorder=6, marker="*",
               label="Defender foremost point")

    verdict = "OFFSIDE POSITION" if result.is_offside else "NOT OFFSIDE"
    ax.set_title(
        f"{verdict}\n{format_verdict_line(result)}",
        color="red" if result.is_offside else "limegreen",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax.axis("off")
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved result: {save_path}")
    (plt.show() if show else plt.close(fig))


def save_player_crop(
    img_np: np.ndarray,
    attacker_mask: np.ndarray,
    defender_mask: np.ndarray,
    save_path: Path,
    pad_ratio: float = 0.15,
) -> None:
    union = attacker_mask | defender_mask
    ys, xs = np.where(union)
    if len(xs) == 0:
        raise ValueError("Empty player masks.")
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    h, w = img_np.shape[:2]
    pad_x = max(8, int((x1 - x0 + 1) * pad_ratio))
    pad_y = max(8, int((y1 - y0 + 1) * pad_ratio))
    y0, y1 = max(0, y0 - pad_y), min(h - 1, y1 + pad_y)
    x0, x1 = max(0, x0 - pad_x), min(w - 1, x1 + pad_x)
    crop = img_np[y0 : y1 + 1, x0 : x1 + 1]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(crop).save(save_path)
    print(f"Saved player crop: {save_path}  ({crop.shape[1]}x{crop.shape[0]})")
