"""Measure temporal stability and high-fidelity dynamic command budgets."""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
VIDEO = Path.home() / "Desktop" / "参考.mp4"
OUT = ROOT / "analysis"


def read_video(path: Path) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    return np.stack(frames), fps


def main() -> None:
    OUT.mkdir(exist_ok=True)
    stack, fps = read_video(VIDEO)
    frame_count, height, width, _ = stack.shape
    print(f"video={width}x{height} frames={frame_count} fps={fps}")
    base = np.empty((height, width, 3), dtype=np.uint8)
    temporal_range = np.empty((height, width), dtype=np.uint8)
    mad95 = np.empty((height, width), dtype=np.float32)
    unstable8 = np.empty((height, width), dtype=np.uint8)
    unstable16 = np.empty((height, width), dtype=np.uint8)
    strip = 32
    for top in range(0, height, strip):
        bottom = min(top + strip, height)
        chunk = stack[:, top:bottom]
        med = np.median(chunk, axis=0).astype(np.uint8)
        base[top:bottom] = med
        lo = np.min(chunk, axis=0).astype(np.int16)
        hi = np.max(chunk, axis=0).astype(np.int16)
        temporal_range[top:bottom] = np.clip((hi - lo).max(axis=2), 0, 255).astype(np.uint8)
        diff = np.abs(chunk.astype(np.int16) - med.astype(np.int16)).mean(axis=3)
        mad95[top:bottom] = np.percentile(diff, 95, axis=0).astype(np.float32)
        unstable8[top:bottom] = (diff > 8).sum(axis=0).astype(np.uint8)
        unstable16[top:bottom] = (diff > 16).sum(axis=0).astype(np.uint8)
        print(f"temporal rows {top:03d}-{bottom:03d}", flush=True)

    Image.fromarray(base).save(OUT / "probe-median.png")
    Image.fromarray(np.clip(mad95 * 10, 0, 255).astype(np.uint8)).save(OUT / "probe-mad95.png")
    Image.fromarray(np.clip(temporal_range * 3, 0, 255).astype(np.uint8)).save(OUT / "probe-range.png")

    thresholds = [2, 3, 4, 6, 8, 12, 16, 24]
    summary = {}
    for threshold in thresholds:
        seed = mad95 > threshold
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cleaned = cv2.morphologyEx(seed.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
        dynamic = cv2.dilate(cleaned, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        summary[threshold] = {
            "seedFraction": float(seed.mean()),
            "dynamicFraction": float((dynamic > 0).mean()),
            "staticFraction": float((dynamic == 0).mean()),
        }
        if threshold in {3, 6, 12}:
            Image.fromarray(dynamic).save(OUT / f"probe-dynamic-{threshold}.png")
            static_only = base.copy()
            static_only[dynamic > 0] = [12, 14, 20]
            Image.fromarray(static_only).save(OUT / f"probe-static-only-{threshold}.png")
        print(f"threshold={threshold} {summary[threshold]}", flush=True)

    # Pixel-level temporal deltas: raw dynamic keyframes and sparse per-frame changes.
    previous = stack[0]
    changed_counts = []
    sparse_payload = bytearray()
    dense_payload = bytearray()
    for frame_index, frame in enumerate(stack):
        difference = np.abs(frame.astype(np.int16) - previous.astype(np.int16)).max(axis=2)
        changed = difference > 1
        changed_counts.append(int(changed.sum()))
        ys, xs = np.nonzero(changed)
        sparse_payload.extend(np.uint32(frame_index).tobytes())
        sparse_payload.extend(np.uint32(len(xs)).tobytes())
        sparse_payload.extend(xs.astype(np.uint16).tobytes())
        sparse_payload.extend(ys.astype(np.uint16).tobytes())
        sparse_payload.extend(frame[ys, xs].tobytes())
        dense_payload.extend(frame.tobytes())
        previous = frame
    print("changed>1 per frame", np.percentile(changed_counts, [0, 25, 50, 75, 100]).tolist(), "mean", float(np.mean(changed_counts)))
    print("dense raw MB", len(dense_payload) / 1e6, "deflate MB", len(zlib.compress(bytes(dense_payload), 6)) / 1e6)
    print("sparse raw MB", len(sparse_payload) / 1e6, "deflate MB", len(zlib.compress(bytes(sparse_payload), 6)) / 1e6)
    summary["changed"] = {
        "mean": float(np.mean(changed_counts)),
        "p50": float(np.percentile(changed_counts, 50)),
        "p95": float(np.percentile(changed_counts, 95)),
        "sparseRawBytes": len(sparse_payload),
        "sparseDeflateBytes": len(zlib.compress(bytes(sparse_payload), 6)),
        "denseRawBytes": len(dense_payload),
        "denseDeflateBytes": len(zlib.compress(bytes(dense_payload), 6)),
    }
    (OUT / "precision-probe.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
