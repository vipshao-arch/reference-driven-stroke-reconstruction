"""Measure coherent masks with block-adaptive RGB555 per-pixel drawing commands."""

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
ENCODE = np.rint(np.arange(256, dtype=np.float32) * 31 / 255).astype(np.uint8)
DECODE = np.rint(np.arange(32, dtype=np.float32) * 255 / 31).astype(np.uint8)


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


def robust_statistics(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = stack.shape[1:3]
    base = np.empty((height, width, 3), dtype=np.uint8)
    robust_range = np.empty((height, width), dtype=np.uint8)
    strip = 24
    for top in range(0, height, strip):
        bottom = min(height, top + strip)
        chunk = stack[:, top:bottom]
        median = np.median(chunk, axis=0).astype(np.uint8)
        base[top:bottom] = median
        low = np.quantile(chunk, 0.02, axis=0).astype(np.int16)
        high = np.quantile(chunk, 0.98, axis=0).astype(np.int16)
        robust_range[top:bottom] = np.clip((high - low).max(axis=2), 0, 255).astype(np.uint8)
    return base, robust_range


def coherent_partition(robust_range: np.ndarray, threshold: int = 12) -> np.ndarray:
    dynamic = cv2.morphologyEx(
        (robust_range > threshold).astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), np.uint8),
    )
    contours, _ = cv2.findContours(dynamic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(dynamic)
    for contour in contours:
        if cv2.contourArea(contour) >= 4000:
            cv2.drawContours(filled, [contour], -1, 255, -1)
    dynamic = cv2.bitwise_or(dynamic, filled)
    dynamic = cv2.dilate(dynamic, np.ones((3, 3), np.uint8), iterations=1) > 0
    for y in range(dynamic.shape[0]):
        row = np.concatenate(([False], dynamic[y], [False]))
        edges = np.flatnonzero(row[1:] != row[:-1])
        for start, end in zip(edges[::2], edges[1::2]):
            if end - start < 12:
                dynamic[y, start:end] = row[start] if start > 0 else row[end]
    return dynamic


def varint(values: np.ndarray) -> bytes:
    output = bytearray()
    for value in values.tolist():
        while value >= 0x80:
            output.append((value & 0x7F) | 0x80)
            value >>= 7
        output.append(value)
    return bytes(output)


def encode_frame(target: np.ndarray, state: np.ndarray, active: np.ndarray, keyframe: bool, threshold: int) -> tuple[bytes, np.ndarray]:
    flat_target = target.reshape(-1, 3)
    flat_state = state.reshape(-1, 3)
    flat_active = active.ravel()
    encoded = ENCODE[flat_target[flat_active]]
    packed = (encoded[:, 0].astype(np.uint16) << 10) | (encoded[:, 1].astype(np.uint16) << 5) | encoded[:, 2].astype(np.uint16)
    result = np.zeros_like(flat_target)
    result[~flat_active] = flat_state[~flat_active]
    result[flat_active] = DECODE[encoded]
    result = result.reshape(target.shape)
    if keyframe:
        return packed.astype("<u2").tobytes(), result
    difference = np.abs(flat_target[flat_active].astype(np.int16) - flat_state[flat_active].astype(np.int16)).max(axis=1)
    changed = np.flatnonzero(difference > threshold)
    colors = packed[changed].astype("<u2").tobytes()
    gaps = np.diff(np.concatenate(([0], changed + 1))).astype(np.uint32)
    result = flat_state.reshape(target.shape).copy()
    result.reshape(-1, 3)[np.flatnonzero(flat_active)[changed]] = DECODE[encoded[changed]]
    return struct.pack("<I", len(changed)) + varint(gaps) + colors, result


def main() -> None:
    stack = read_video(VIDEO)
    frame_count, height, width, _ = stack.shape
    base, robust_range = robust_statistics(stack)
    mask = coherent_partition(robust_range)
    dynamic_cells2 = mask.reshape(height // 2, 2, width // 2, 2).any(axis=(1, 3))
    print("dynamic", mask.mean(), "dynamic2cells", dynamic_cells2.sum(), flush=True)
    results = {}
    for block_threshold in [8, 16, 32, 64]:
        for delta_threshold in [3, 4, 5]:
            compressor = zlib.compressobj(level=9)
            state = np.zeros((height, width, 3), dtype=np.uint8)
            raw_bytes = 0
            compressed_bytes = 0
            errors = []
            command_counts = []
            for frame_index, frame in enumerate(stack):
                keyframe = frame_index % 8 == 0
                if keyframe:
                    active = dynamic_cells2
                    payload, state = encode_frame(frame, state, np.repeat(np.repeat(active, 2, 0), 2, 1), True, delta_threshold)
                    dense = np.repeat(np.repeat(DECODE[ENCODE[frame.reshape(-1, 3)]].reshape(height, width, 3)[::2, ::2], 2, 0), 2, 1)
                    state = np.where(np.repeat(np.repeat(active, 2, 0), 2, 1)[..., None], dense, base)
                    command_counts.append(int(active.sum()))
                else:
                    difference = np.abs(frame.astype(np.int16) - state.astype(np.int16)).max(axis=2)
                    cells = difference.reshape(height // 2, 2, width // 2, 2).max(axis=(1, 3))
                    active = dynamic_cells2 & (cells > block_threshold)
                    exact_active = np.repeat(np.repeat(active, 2, 0), 2, 1)
                    payload, state = encode_frame(frame, state, exact_active, False, delta_threshold)
                    command_counts.append(int(active.sum()))
                raw_bytes += len(payload)
                compressed_bytes += len(compressor.compress(payload))
                errors.append(float(np.abs(state.astype(np.int16) - frame.astype(np.int16)).mean()))
            compressed_bytes += len(compressor.flush())
            key = f"block{block_threshold}_delta{delta_threshold}"
            results[key] = {
                "compressedBytes": compressed_bytes,
                "rawBytes": raw_bytes,
                "meanMae": round(float(np.mean(errors)), 3),
                "maxMae": round(float(np.max(errors)), 3),
                "meanCommandsPerFrame": round(float(np.mean(command_counts))),
            }
            print("RESULT", key, results[key], flush=True)
    (OUT / "compact-command-probe2.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
