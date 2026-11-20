"""Measure exact per-pixel temporal stability across every decoded frame."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
VIDEO = Path.home() / "Desktop" / "参考.mp4"
OUT = ROOT / "analysis"


def read_video(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    return np.stack(frames)


def main() -> None:
    stack = read_video(VIDEO)
    frame_count, height, width, _ = stack.shape
    base = np.empty((height, width, 3), dtype=np.uint8)
    exact_range = np.empty((height, width), dtype=np.uint8)
    deviation = np.empty((height, width), dtype=np.uint8)
    strip = 32
    for top in range(0, height, strip):
        bottom = min(top + strip, height)
        chunk = stack[:, top:bottom]
        med = np.median(chunk, axis=0).astype(np.uint8)
        base[top:bottom] = med
        exact_range[top:bottom] = np.clip((chunk.max(axis=0).astype(np.int16) - chunk.min(axis=0).astype(np.int16)).max(axis=2), 0, 255).astype(np.uint8)
        deviation[top:bottom] = np.clip(np.abs(chunk.astype(np.int16) - med.astype(np.int16)).max(axis=(0, 3)), 0, 255).astype(np.uint8)
        print(f"exact rows {top:03d}-{bottom:03d}", flush=True)

    Image.fromarray(base).save(OUT / "exact-median.png")
    Image.fromarray(np.clip(exact_range.astype(np.float32) * 20, 0, 255).astype(np.uint8)).save(OUT / "exact-range.png")
    results = {}
    for threshold in [0, 1, 2, 3, 4, 6, 8, 12, 16, 24]:
        static = exact_range <= threshold
        dilated = cv2.dilate((~static).astype(np.uint8) * 255, np.ones((3, 3), np.uint8)) > 0
        locked = static & ~dilated
        dynamic_cells2 = (~locked).reshape(height // 2, 2, width // 2, 2).any(axis=(1, 3))
        results[str(threshold)] = {
            "staticFraction": float(static.mean()),
            "lockedFraction": float(locked.mean()),
            "dynamicFraction": float((~locked).mean()),
            "dynamicCell2Count": int(dynamic_cells2.sum()),
            "dynamicCell2Fraction": float(dynamic_cells2.mean()),
            "maxError": int(deviation[locked].max()) if locked.any() else 0,
        }
        if threshold in {1, 3, 8}:
            preview = base.copy()
            preview[~locked] = [10, 12, 18]
            Image.fromarray(preview).save(OUT / f"exact-static-preview-{threshold}.png")
            Image.fromarray(locked.astype(np.uint8) * 255).save(OUT / f"exact-lock-{threshold}.png")
        print(threshold, results[str(threshold)], flush=True)
    (OUT / "exact-stability.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
