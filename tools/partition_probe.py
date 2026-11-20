"""Measure coherent static/dynamic partitions from robust temporal statistics."""

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


def robust_statistics(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = stack.shape[1:3]
    base = np.empty((height, width, 3), dtype=np.uint8)
    robust_range = np.empty((height, width), dtype=np.uint8)
    deviation95 = np.empty((height, width), dtype=np.uint8)
    exact_range = np.empty((height, width), dtype=np.uint8)
    strip = 24
    for top in range(0, height, strip):
        bottom = min(height, top + strip)
        chunk = stack[:, top:bottom]
        median = np.median(chunk, axis=0).astype(np.uint8)
        base[top:bottom] = median
        low = np.quantile(chunk, 0.02, axis=0).astype(np.int16)
        high = np.quantile(chunk, 0.98, axis=0).astype(np.int16)
        robust_range[top:bottom] = np.clip((high - low).max(axis=2), 0, 255).astype(np.uint8)
        absolute = np.abs(chunk.astype(np.int16) - median.astype(np.int16)).max(axis=3)
        deviation95[top:bottom] = np.clip(np.quantile(absolute, 0.95, axis=0), 0, 255).astype(np.uint8)
        exact_range[top:bottom] = np.clip(
            (chunk.max(axis=0).astype(np.int16) - chunk.min(axis=0).astype(np.int16)).max(axis=2),
            0,
            255,
        ).astype(np.uint8)
        print(f"statistics rows {top:03d}-{bottom:03d}", flush=True)
    return base, robust_range, deviation95, exact_range


def coherent_partition(candidate_dynamic: np.ndarray, close_size: int, fill_area: int) -> np.ndarray:
    dynamic = cv2.morphologyEx(
        candidate_dynamic.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        np.ones((close_size, close_size), np.uint8),
    )
    contours, _ = cv2.findContours(dynamic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(dynamic)
    for contour in contours:
        if cv2.contourArea(contour) >= fill_area:
            cv2.drawContours(filled, [contour], -1, 255, -1)
    dynamic = cv2.bitwise_or(dynamic, filled)
    dynamic = cv2.dilate(dynamic, np.ones((3, 3), np.uint8), iterations=1)
    return dynamic > 0


def scanline_smooth(mask: np.ndarray, minimum_run: int = 12) -> np.ndarray:
    """Close short holes and remove isolated short speckles along each scanline."""
    result = mask.copy()
    for y in range(result.shape[0]):
        row = np.concatenate(([False], result[y], [False]))
        edges = np.flatnonzero(row[1:] != row[:-1])
        for start, end in zip(edges[::2], edges[1::2]):
            if end - start >= minimum_run:
                continue
            result[y, start:end] = row[start] if start > 0 else row[end]
    return result


def main() -> None:
    stack = read_video(VIDEO)
    frame_count, height, width, _ = stack.shape
    base, robust_range, deviation95, exact_range = robust_statistics(stack)
    Image.fromarray(base).save(OUT / "partition-median.png")
    Image.fromarray(np.clip(robust_range.astype(np.float32) * 12, 0, 255).astype(np.uint8)).save(OUT / "partition-robust-range.png")
    configs = [
        ("r8_close7", robust_range > 8, 7, 4000),
        ("r12_close9", robust_range > 12, 9, 4000),
        ("d6_close9", deviation95 > 6, 9, 4000),
        ("d8_close11", deviation95 > 8, 11, 4000),
    ]
    results = {}
    for name, candidate, close_size, fill_area in configs:
        dynamic = coherent_partition(candidate, close_size, fill_area)
        dynamic = scanline_smooth(dynamic, 12)
        locked = ~dynamic
        if locked.any():
            deviation = np.abs(stack.astype(np.int16) - base.astype(np.int16)).max(axis=3)
            locked_max = int(deviation[:, locked].max())
            locked_mean = float(deviation[:, locked].mean())
        else:
            locked_max = 255
            locked_mean = 255.0
        cells2 = dynamic.reshape(height // 2, 2, width // 2, 2).any(axis=(1, 3))
        preview = base.copy()
        preview[dynamic] = [10, 12, 18]
        Image.fromarray(preview).save(OUT / f"partition-static-{name}.png")
        Image.fromarray(locked.astype(np.uint8) * 255).save(OUT / f"partition-lock-{name}.png")
        results[name] = {
            "lockedFraction": float(locked.mean()),
            "dynamicFraction": float(dynamic.mean()),
            "dynamicCell2Count": int(cells2.sum()),
            "lockedMaxChannelDeviation": locked_max,
            "lockedMeanChannelDeviation": round(locked_mean, 3),
        }
        print("RESULT", name, results[name], flush=True)
    (OUT / "partition-probe.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
