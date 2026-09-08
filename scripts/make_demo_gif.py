#!/usr/bin/env python
"""Build the README demo GIF from a finished run.

Puts the source video beside the egocentric render the pipeline wrote, so the
GIF shows real pipeline output rather than a mock-up.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import imageio.v2 as imageio
import numpy as np


def read_frames(path: str) -> List[np.ndarray]:
    """Read every frame of a video into memory."""
    capture = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def fit_height(image: np.ndarray, height: int) -> np.ndarray:
    """Scale an image to a fixed height, keeping the aspect ratio."""
    h, w = image.shape[:2]
    return cv2.resize(image, (int(w * height / h), height), interpolation=cv2.INTER_AREA)


def add_label(image: np.ndarray, text: str) -> np.ndarray:
    """Draw a caption bar across the top of an image."""
    cv2.rectangle(image, (0, 0), (image.shape[1], 22), (20, 20, 20), -1)
    cv2.putText(
        image, text, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1,
        cv2.LINE_AA,
    )
    return image


def main() -> int:
    """Write the GIF and report its size."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/samples/synthetic_reach.mp4")
    parser.add_argument("--ego", default="outputs/real/egocentric.mp4")
    parser.add_argument("--out", "-o", default="assets/demo.gif")
    parser.add_argument("--height", type=int, default=260)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    source = read_frames(args.source)
    ego = read_frames(args.ego)
    count = min(len(source), len(ego))
    if count == 0:
        raise SystemExit("No frames to combine")

    gap = np.full((args.height, 6, 3), 30, dtype=np.uint8)
    frames = []
    for i in range(0, count, args.stride):
        left = add_label(fit_height(source[i], args.height), "third-person input")
        right = add_label(fit_height(ego[i], args.height), "egocentric output")
        frames.append(cv2.cvtColor(np.hstack([left, gap, right]), cv2.COLOR_BGR2RGB))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(
        out, frames, format="GIF", duration=args.stride / args.fps, loop=0
    )

    print(f"Wrote {len(frames)} frames to {out} ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
