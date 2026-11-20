"""Probe exact keyframes plus sparse per-pixel temporal drawing commands."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import cv2
import numpy as np


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


def exact_lock(stack: np.ndarray, threshold: int = 8) -> tuple[np.ndarray, np.ndarray]:
    height, width = stack.shape[1:3]
    base = np.empty((height, width, 3), dtype=np.uint8)
    temporal_range = np.empty((height, width), dtype=np.uint8)
    strip = 32
    for top in range(0, height, strip):
        bottom = min(top + strip, height)
        chunk = stack[:, top:bottom]
        med = np.median(chunk, axis=0).astype(np.uint8)
        base[top:bottom] = med
        temporal_range[top:bottom] = np.clip(
            (chunk.max(axis=0).astype(np.int16) - chunk.min(axis=0).astype(np.int16)).max(axis=2),
            0,
            255,
        ).astype(np.uint8)
    static = temporal_range <= threshold
    dynamic = cv2.dilate((~static).astype(np.uint8) * 255, np.ones((3, 3), np.uint8)) > 0
    locked = ~dynamic
    return base, locked


def spans_for_frame(frame: np.ndarray, state: np.ndarray, dynamic: np.ndarray, step: int) -> bytes:
    delta = frame.astype(np.int16) - state.astype(np.int16)
    delta = np.clip(np.rint(delta / step), -127, 127).astype(np.int16) * step
    delta[~dynamic] = 0
    changed = np.any(delta != 0, axis=2)
    payload = bytearray()
    span_count = 0
    for y in range(frame.shape[0]):
        row_delta = delta[y]
        row_changed = changed[y]
        x = 0
        while x < frame.shape[1]:
            if not row_changed[x]:
                x += 1
                continue
            first = x
            value = row_delta[x]
            while x + 1 < frame.shape[1] and row_changed[x + 1] and np.array_equal(row_delta[x + 1], value):
                x += 1
            length = x - first + 1
            normalized = np.clip(value // step, -127, 127).astype(np.int8)
            payload.extend(struct.pack("<HHHbbb", first, y, length, *normalized))
            span_count += 1
            x += 1
    return struct.pack("<I", span_count) + bytes(payload)


def main() -> None:
    stack = read_video(VIDEO)
    frame_count, height, width, _ = stack.shape
    base, locked = exact_lock(stack)
    dynamic = ~locked
    print("locked", locked.mean(), "dynamic", dynamic.mean(), flush=True)
    results = {}
    keyframe_stride = 12
    for step in [2, 4, 8]:
        state = base.copy()
        compressor = zlib.compressobj(level=9)
        raw_bytes = 0
        compressed_bytes = 0
        errors = []
        span_total = 0
        for frame_index, frame in enumerate(stack):
            if frame_index % keyframe_stride == 0:
                payload = b"K" + struct.pack("<H", frame_index) + frame[dynamic].tobytes()
                state[dynamic] = frame[dynamic]
                delta = np.zeros_like(frame, dtype=np.int8)
            else:
                before = state.copy()
                delta = np.clip(np.rint((frame.astype(np.int16) - before.astype(np.int16)) / step), -127, 127).astype(np.int16) * step
                delta[~dynamic] = 0
                state = np.clip(before.astype(np.int16) + delta, 0, 255).astype(np.uint8)
                payload = b"D" + struct.pack("<H", frame_index) + spans_for_frame(frame, before, dynamic, step)
                span_total += struct.unpack_from("<I", payload, 3)[0]
            compressed_bytes += len(compressor.compress(payload))
            raw_bytes += len(payload)
            difference = np.abs(state.astype(np.int16) - frame.astype(np.int16))
            errors.append(
                {
                    "frame": frame_index,
                    "mae": float(difference.mean()),
                    "dynamicMae": float(difference[dynamic].mean()),
                    "staticMae": float(difference[locked].mean()),
                }
            )
            if frame_index % 20 == 0:
                print(f"step={step} frame={frame_index} compressed={compressed_bytes/1e6:.1f}MB", flush=True)
        compressed_bytes += len(compressor.flush())
        results[str(step)] = {
            "rawBytes": raw_bytes,
            "deflateBytes": compressed_bytes,
            "spanCount": span_total,
            "meanMae": round(float(np.mean([item["mae"] for item in errors])), 3),
            "maxMae": round(max(item["mae"] for item in errors), 3),
            "meanDynamicMae": round(float(np.mean([item["dynamicMae"] for item in errors])), 3),
            "meanStaticMae": round(float(np.mean([item["staticMae"] for item in errors])), 3),
            "sampleErrors": [errors[i] for i in [0, 40, 80, 123]],
        }
        print("RESULT", step, results[str(step)], flush=True)
    (OUT / "temporal-command-probe.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
