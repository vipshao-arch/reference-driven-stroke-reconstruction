"""Measure keyframe + sparse quantized drawing command streams."""

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


def temporal_partition(stack: np.ndarray, threshold: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = stack.shape[1:3]
    base = np.empty((height, width, 3), dtype=np.uint8)
    temporal_range = np.empty((height, width), dtype=np.uint8)
    strip = 32
    for top in range(0, height, strip):
        bottom = min(top + strip, height)
        chunk = stack[:, top:bottom]
        median = np.median(chunk, axis=0).astype(np.uint8)
        base[top:bottom] = median
        temporal_range[top:bottom] = np.clip(
            (chunk.max(axis=0).astype(np.int16) - chunk.min(axis=0).astype(np.int16)).max(axis=2),
            0,
            255,
        ).astype(np.uint8)
    stable = temporal_range <= threshold
    dynamic = cv2.dilate((~stable).astype(np.uint8) * 255, np.ones((3, 3), np.uint8)) > 0
    return base, ~dynamic


def lut555() -> tuple[np.ndarray, np.ndarray]:
    encode = np.rint(np.arange(256, dtype=np.float32) * 31 / 255).astype(np.uint8)
    decode = np.rint(np.arange(32, dtype=np.float32) * 255 / 31).astype(np.uint8)
    return encode, decode


def varint(values: np.ndarray) -> bytes:
    """Encode unsigned gaps in original order using LEB128."""
    output = bytearray()
    for value in values.tolist():
        while value >= 0x80:
            output.append((value & 0x7F) | 0x80)
            value >>= 7
        output.append(value)
    return bytes(output)


def main() -> None:
    stack = read_video(VIDEO)
    frame_count, height, width, _ = stack.shape
    encode, decode = lut555()
    results = {}
    for lock_threshold in [8, 12]:
        base, locked = temporal_partition(stack, lock_threshold)
        dynamic = ~locked
        dynamic_indices = np.flatnonzero(dynamic.ravel())
        print("threshold", lock_threshold, "locked", locked.mean(), "dynamic pixels", len(dynamic_indices), flush=True)
        for keyframe_stride in [8, 12]:
            for delta_threshold in [2, 4]:
                state = base.copy()
                compressor = zlib.compressobj(level=9)
                raw_bytes = 0
                compressed_bytes = 0
                errors = []
                changed_total = 0
                key = f"lock{lock_threshold}_key{keyframe_stride}_delta{delta_threshold}"
                for frame_index, frame in enumerate(stack):
                    target = frame.reshape(-1, 3)[dynamic_indices]
                    encoded = encode[target]
                    packed = (
                        encoded[:, 0].astype(np.uint16) << 10
                        | encoded[:, 1].astype(np.uint16) << 5
                        | encoded[:, 2].astype(np.uint16)
                    )
                    if frame_index % keyframe_stride == 0:
                        payload = b"K" + struct.pack("<H", frame_index) + packed.astype("<u2").tobytes()
                        state.reshape(-1, 3)[dynamic_indices] = decode[encoded]
                    else:
                        state_dynamic = state.reshape(-1, 3)[dynamic_indices].astype(np.int16)
                        difference = np.abs(frame.reshape(-1, 3)[dynamic_indices].astype(np.int16) - state_dynamic).max(axis=1)
                        changed = np.flatnonzero(difference > delta_threshold)
                        changed_total += len(changed)
                        positions = dynamic_indices[changed]
                        gaps = np.diff(np.concatenate(([0], changed + 1))).astype(np.uint32)
                        colors = packed[changed].astype("<u2").tobytes()
                        payload = b"D" + struct.pack("<HI", frame_index, len(changed)) + varint(gaps) + colors
                        state.reshape(-1, 3)[dynamic_indices[changed]] = decode[encoded[changed]]
                    compressed_bytes += len(compressor.compress(payload))
                    raw_bytes += len(payload)
                    difference = np.abs(state.astype(np.int16) - frame.astype(np.int16))
                    errors.append(float(difference.mean()))
                compressed_bytes += len(compressor.flush())
                results[key] = {
                    "lockedFraction": float(locked.mean()),
                    "dynamicFraction": float(dynamic.mean()),
                    "rawBytes": raw_bytes,
                    "deflateBytes": compressed_bytes,
                    "changedPerFrame": changed_total / frame_count,
                    "meanMae": round(float(np.mean(errors)), 3),
                    "maxMae": round(float(np.max(errors)), 3),
                }
                print("RESULT", key, results[key], flush=True)
    (OUT / "compact-command-probe.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
