#!/usr/bin/env python
"""Render a synthetic third-person clip of a person reaching for a box.

The repo ships no sample footage. This script draws a filled human figure
against a plain wall and a table, so the pipeline has an input that the real
backends can attempt. The figure is a drawing, not a photograph. Treat any
detection result on it as a smoke test, not as accuracy evidence.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

SKIN = (150, 180, 215)  # BGR
SHIRT = (110, 85, 60)
TROUSERS = (70, 60, 55)
WALL = (205, 205, 200)
FLOOR = (150, 150, 145)
TABLE = (95, 120, 155)
BOX = (60, 70, 190)


def _limb(canvas, p0, p1, color, thickness):
    """Draw one rounded limb segment."""
    cv2.line(canvas, p0, p1, color, thickness, cv2.LINE_AA)
    cv2.circle(canvas, p1, thickness // 2, color, -1, cv2.LINE_AA)


def draw_frame(width: int, height: int, phase: float) -> np.ndarray:
    """Draw one frame of the reaching motion.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.
        phase: Motion phase in [0, 1).

    Returns:
        A BGR frame.
    """
    canvas = np.full((height, width, 3), WALL, dtype=np.uint8)
    cv2.rectangle(canvas, (0, int(height * 0.78)), (width, height), FLOOR, -1)

    # Table with a box on it.
    table_top = int(height * 0.66)
    cv2.rectangle(
        canvas, (int(width * 0.55), table_top), (width, int(height * 0.80)), TABLE, -1
    )
    reach = 0.5 - 0.5 * math.cos(2 * math.pi * phase)
    box_x = int(width * 0.68)
    box_y = table_top - int(height * 0.06)
    cv2.rectangle(
        canvas,
        (box_x, box_y),
        (box_x + int(width * 0.09), table_top),
        BOX,
        -1,
    )

    # Body anchors.
    cx = int(width * 0.36)
    hip_y = int(height * 0.58)
    shoulder_y = int(height * 0.34)
    head_r = int(height * 0.065)
    sway = int(6 * math.sin(2 * math.pi * phase))

    l_hip = (cx - int(width * 0.045), hip_y)
    r_hip = (cx + int(width * 0.045), hip_y)
    l_sh = (cx - int(width * 0.075) + sway, shoulder_y)
    r_sh = (cx + int(width * 0.075) + sway, shoulder_y)

    # Legs.
    knee_y = int(height * 0.72)
    ankle_y = int(height * 0.86)
    _limb(canvas, l_hip, (l_hip[0] - 6, knee_y), TROUSERS, int(height * 0.055))
    _limb(canvas, (l_hip[0] - 6, knee_y), (l_hip[0] - 10, ankle_y), TROUSERS,
          int(height * 0.045))
    _limb(canvas, r_hip, (r_hip[0] + 6, knee_y), TROUSERS, int(height * 0.055))
    _limb(canvas, (r_hip[0] + 6, knee_y), (r_hip[0] + 10, ankle_y), TROUSERS,
          int(height * 0.045))

    # Torso.
    torso = np.array([l_sh, r_sh, r_hip, l_hip], dtype=np.int32)
    cv2.fillConvexPoly(canvas, torso, SHIRT, cv2.LINE_AA)

    # Right arm reaches for the box, left arm hangs.
    r_elbow = (
        r_sh[0] + int(width * (0.05 + 0.05 * reach)),
        shoulder_y + int(height * (0.13 - 0.04 * reach)),
    )
    r_wrist = (
        r_sh[0] + int(width * (0.09 + 0.17 * reach)),
        shoulder_y + int(height * (0.24 - 0.08 * reach)),
    )
    _limb(canvas, r_sh, r_elbow, SHIRT, int(height * 0.045))
    _limb(canvas, r_elbow, r_wrist, SKIN, int(height * 0.035))
    cv2.circle(canvas, r_wrist, int(height * 0.028), SKIN, -1, cv2.LINE_AA)

    l_elbow = (l_sh[0] - int(width * 0.02), shoulder_y + int(height * 0.14))
    l_wrist = (l_sh[0] - int(width * 0.03), shoulder_y + int(height * 0.26))
    _limb(canvas, l_sh, l_elbow, SHIRT, int(height * 0.045))
    _limb(canvas, l_elbow, l_wrist, SKIN, int(height * 0.035))
    cv2.circle(canvas, l_wrist, int(height * 0.028), SKIN, -1, cv2.LINE_AA)

    # Neck and head.
    neck = ((l_sh[0] + r_sh[0]) // 2, shoulder_y - int(height * 0.02))
    head_c = (neck[0], neck[1] - head_r)
    _limb(canvas, neck, head_c, SKIN, int(height * 0.04))
    cv2.circle(canvas, head_c, head_r, SKIN, -1, cv2.LINE_AA)

    # Face marks give the detector eye and ear cues.
    eye_dx = int(head_r * 0.38)
    eye_y = head_c[1] - int(head_r * 0.15)
    cv2.circle(canvas, (head_c[0] - eye_dx, eye_y), max(2, head_r // 8), (40, 40, 40), -1,
               cv2.LINE_AA)
    cv2.circle(canvas, (head_c[0] + eye_dx, eye_y), max(2, head_r // 8), (40, 40, 40), -1,
               cv2.LINE_AA)
    cv2.ellipse(canvas, (head_c[0], head_c[1] + int(head_r * 0.35)),
                (int(head_r * 0.35), int(head_r * 0.18)), 0, 0, 180, (60, 60, 80), 2,
                cv2.LINE_AA)
    cv2.ellipse(canvas, (head_c[0], head_c[1] - int(head_r * 0.35)),
                (head_r, int(head_r * 0.75)), 0, 180, 360, (55, 45, 40), -1, cv2.LINE_AA)

    return canvas


def main() -> int:
    """Write the clip and report the path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", "-o", default="data/samples/synthetic_reach.mp4")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", "-n", type=int, default=90)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open {out} for writing")

    try:
        for i in range(args.frames):
            writer.write(draw_frame(args.width, args.height, (i % 45) / 45.0))
    finally:
        writer.release()

    print(f"Wrote {args.frames} frames to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
