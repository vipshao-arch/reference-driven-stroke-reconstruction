"""Find a pixel-accurate static/dynamic partition using temporal quantiles."""

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
    p05 = np.empty((height, width, 3), dtype=np.uint8)
    p95 = np.empty((height, width, 3), dtype=np.uint8)
    p98_abs = np.empty((height, width), dtype=np.float32)
    strip = 32
    for top in range(0, height, strip):
        bottom = min(top + strip, height)
        chunk = stack[:, top:bottom].astype(np.uint8)
        med = np.median(chunk, axis=0).astype(np.uint8)
        base[top:bottom] = med
        p05[top:bottom] = np.percentile(chunk, 5, axis=0).astype(np.uint8)
        p95[top:bottom] = np.percentile(chunk, 95, axis=0).astype(np.uint8)
        deviation = np.abs(chunk.astype(np.int16) - med.astype(np.int16)).max(axis=3)
        p98_abs[top:bottom] = np.percentile(deviation, 98, axis=0).astype(np.float32)
        print(f"quantiles rows {top:03d}-{bottom:03d}", flush=True)

    quantile_range = np.abs(p95.astype(np.int16) - p05.astype(np.int16)).max(axis=2)
    Image.fromarray(base).save(OUT / "segment-median.png")
    Image.fromarray(np.clip(quantile_range * 24, 0, 255).astype(np.uint8)).save(OUT / "segment-quantile-range.png")
    Image.fromarray(np.clip(p98_abs * 20, 0, 255).astype(np.uint8)).save(OUT / "segment-deviation-p98.png")

    results = {}
    for qrange, deviation, dilation in [(2, 3, 1), (2, 4, 2), (3, 5, 2), (4, 6, 2), (6, 8, 3)]:
        strict_static = (quantile_range <= qrange) & (p98_abs <= deviation)
        # Close small holes but never reclassify a changing pixel as static.
        closed = cv2.morphologyEx(
            strict_static.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        locked = (closed > 0) & strict_static
        dynamic = cv2.dilate((~locked).astype(np.uint8) * 255, np.ones((dilation, dilation), np.uint8)) > 0
        locked = ~dynamic
        static_diffs = np.abs(stack.astype(np.int16) - base.astype(np.int16)).max(axis=3)
        locked_errors = static_diffs[:, locked]
        key = f"q{qrange}_d{deviation}_x{dilation}"
        results[key] = {
            "strictStaticFraction": float(strict_static.mean()),
            "lockedFraction": float(locked.mean()),
            "dynamicFraction": float(dynamic.mean()),
            "lockedMaxChannelError": float(locked_errors.max()),
            "lockedP999ChannelError": float(np.percentile(locked_errors, 99.9)),
            "lockedMeanChannelError": float(locked_errors.mean()),
        }
        if key in {"q2_d3_x1", "q3_d5_x2", "q4_d6_x2"}:
            preview = base.copy()
            preview[dynamic] = [10, 12, 18]
            Image.fromarray(preview).save(OUT / f"segment-static-preview-{key}.png")
            Image.fromarray(locked.astype(np.uint8) * 255).save(OUT / f"segment-lock-{key}.png")
        print(key, results[key], flush=True)
    (OUT / "segmentation-probe.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
