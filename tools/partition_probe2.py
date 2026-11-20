"""Refine coherent static/dynamic partition with exact-change exclusion."""

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


def statistics(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = stack.shape[1:3]
    base = np.empty((height, width, 3), dtype=np.uint8)
    robust = np.empty((height, width), dtype=np.uint8)
    exact = np.empty((height, width), dtype=np.uint8)
    strip = 24
    for top in range(0, height, strip):
        bottom = min(top + strip, height)
        chunk = stack[:, top:bottom]
        median = np.median(chunk, axis=0).astype(np.uint8)
        base[top:bottom] = median
        low = np.quantile(chunk, 0.02, axis=0).astype(np.int16)
        high = np.quantile(chunk, 0.98, axis=0).astype(np.int16)
        robust[top:bottom] = np.clip((high - low).max(axis=2), 0, 255).astype(np.uint8)
        exact[top:bottom] = np.clip(
            (chunk.max(axis=0).astype(np.int16) - chunk.min(axis=0).astype(np.int16)).max(axis=2),
            0,
            255,
        ).astype(np.uint8)
    return base, robust, exact


def partition(candidate: np.ndarray, close_size: int) -> np.ndarray:
    dynamic = cv2.morphologyEx(
        candidate.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        np.ones((close_size, close_size), np.uint8),
    )
    contours, _ = cv2.findContours(dynamic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(dynamic)
    for contour in contours:
        if cv2.contourArea(contour) >= 2500:
            cv2.drawContours(filled, [contour], -1, 255, -1)
    dynamic = cv2.bitwise_or(dynamic, filled)
    dynamic = cv2.morphologyEx(dynamic, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    dynamic = cv2.dilate(dynamic, np.ones((3, 3), np.uint8), iterations=1) > 0
    for y in range(dynamic.shape[0]):
        padded = np.concatenate(([False], dynamic[y], [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        for start, end in zip(edges[::2], edges[1::2]):
            if end - start < 16:
                dynamic[y, start:end] = padded[start] if start > 0 else padded[end]
    return dynamic


def main() -> None:
    stack = read_video(VIDEO)
    frame_count, height, width, _ = stack.shape
    base, robust, exact = statistics(stack)
    results = {}
    for exact_cut in [16, 24, 32]:
        candidate = (robust > 12) | (exact > exact_cut)
        dynamic = partition(candidate, 9)
        locked = ~dynamic
        deviation = np.abs(stack.astype(np.int16) - base.astype(np.int16)).max(axis=3)
        cells2 = dynamic.reshape(height // 2, 2, width // 2, 2).any(axis=(1, 3))
        name = f"exact{exact_cut}"
        preview = base.copy()
        preview[dynamic] = [10, 12, 18]
        Image.fromarray(preview).save(OUT / f"partition-static-{name}.png")
        Image.fromarray(locked.astype(np.uint8) * 255).save(OUT / f"partition-lock-{name}.png")
        results[name] = {
            "lockedFraction": float(locked.mean()),
            "dynamicFraction": float(dynamic.mean()),
            "dynamicCell2Count": int(cells2.sum()),
            "lockedMaxChannelDeviation": int(deviation[:, locked].max()) if locked.any() else 255,
            "lockedMeanChannelDeviation": round(float(deviation[:, locked].mean()), 3) if locked.any() else 255,
        }
        print("RESULT", name, results[name], flush=True)
    (OUT / "partition-probe2.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
