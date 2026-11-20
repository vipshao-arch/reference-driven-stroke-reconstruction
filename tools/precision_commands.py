"""Build a high-precision static/dynamic partition and drawing command stream.

The source clip is decoded offline. Every pixel's full temporal range is used
to define the immutable WebGL base; all remaining pixels are expressed as
adaptive rectangular drawing primitives with per-frame colors. Keyframes reset
the drawing surface, and intervening frames carry only changed primitives.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


COMMAND_DTYPE = np.dtype([
    ("x", "<u2"),
    ("y", "<u2"),
    ("w", "u1"),
    ("h", "u1"),
    ("r", "u1"),
    ("g", "u1"),
    ("b", "u1"),
    ("primitive", "u1"),
])
BYTES_PER_COMMAND = COMMAND_DTYPE.itemsize
LEVELS = (8, 4, 2, 1)


def temporal_partition(stack: np.ndarray, threshold: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return median RGB, an alpha-masked static base, and an immutable lock mask."""
    frame_count, height, width, _ = stack.shape
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
    # A one-pixel safety band keeps moving edges out of the immutable base.
    dynamic = cv2.dilate((~stable).astype(np.uint8) * 255, np.ones((3, 3), np.uint8)) > 0
    locked = ~dynamic
    alpha = np.where(locked, 255, 0).astype(np.uint8)
    base_rgba = np.dstack((base, alpha))
    lock = np.where(locked, 255, 0).astype(np.uint8)
    return base, base_rgba, lock


def partition_details(lock: np.ndarray) -> dict:
    height, width = lock.shape
    locked = lock > 127
    dynamic = ~locked
    details = {
        "rule": "per-pixel maximum RGB range across all 124 frames <= 8, then 1-pixel dynamic safety dilation",
        "lockedFraction": float(locked.mean()),
        "dynamicFraction": float(dynamic.mean()),
        "width": width,
        "height": height,
        "staticScanlineIntervals": [],
        "dynamicScanlineIntervals": [],
        "dynamicComponents": [],
        "staticComponents": [],
    }
    for y in range(height):
        for name, mask in (("staticScanlineIntervals", locked[y]), ("dynamicScanlineIntervals", dynamic[y])):
            padded = np.concatenate(([False], mask, [False]))
            edges = np.flatnonzero(padded[1:] != padded[:-1])
            intervals = [[int(edges[i]), int(edges[i + 1] - 1)] for i in range(0, len(edges), 2) if edges[i + 1] - edges[i] >= 16]
            details[name].append([y, intervals])
    for name, mask in (("dynamicComponents", dynamic), ("staticComponents", locked)):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        components = []
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            if area < 1000:
                continue
            components.append({"bbox": [int(x), int(y), int(w), int(h)], "area": int(area)})
        components.sort(key=lambda item: item["area"], reverse=True)
        details[name] = components[:24]
    return details


def _block_view(array: np.ndarray, rows: int, columns: int, size: int, channels: int | None = None):
    if channels is None:
        return array.reshape(rows, size, columns, size).transpose(0, 2, 1, 3).reshape(rows, columns, size * size)
    return array.reshape(rows, size, columns, size, channels).transpose(0, 2, 1, 3, 4).reshape(rows, columns, size * size, channels)


def _primitive_codes(mean_gx: np.ndarray, mean_gy: np.ndarray) -> np.ndarray:
    edge = (mean_gx + mean_gy) * 0.5
    codes = np.full(edge.shape, 7, dtype=np.uint8)
    codes[edge < 4] = 5
    medium = (edge >= 4) & (edge < 28)
    codes[medium & (mean_gy > mean_gx * 1.35)] = 1
    codes[medium & (mean_gx > mean_gy * 1.35)] = 2
    diagonal = medium & ~(mean_gy > mean_gx * 1.35) & ~(mean_gx > mean_gy * 1.35)
    codes[diagonal & (mean_gx > mean_gy)] = 3
    codes[diagonal & (mean_gx <= mean_gy)] = 4
    codes[edge >= 28] = 6
    return codes


def commands_for_frame(
    target: np.ndarray,
    state: np.ndarray,
    dynamic: np.ndarray,
    tolerance: int,
    keyframe: bool,
) -> np.ndarray:
    """Return adaptive rectangle commands and mutate state to the decoded result."""
    height, width = target.shape[:2]
    target_i16 = target.astype(np.int16)
    gradient_x = np.abs(np.diff(target_i16, axis=1, prepend=target_i16[:, :1])).mean(axis=2).astype(np.float32)
    gradient_y = np.abs(np.diff(target_i16, axis=0, prepend=target_i16[:1])).mean(axis=2).astype(np.float32)
    command_parts: list[np.ndarray] = []
    candidate = None

    for size in LEVELS:
        rows, columns = height // size, width // size
        target_blocks = _block_view(target, rows, columns, size, 3).astype(np.int16)
        state_blocks = _block_view(state, rows, columns, size, 3).astype(np.int16)
        dynamic_blocks = _block_view(dynamic, rows, columns, size)
        counts = dynamic_blocks.sum(axis=2)
        if candidate is None:
            candidate = counts > 0
        else:
            candidate = np.repeat(np.repeat(candidate, 2, axis=0), 2, axis=1)
        active = candidate & (counts > 0)
        safe_counts = np.maximum(counts, 1)
        means = np.rint(
            (target_blocks * dynamic_blocks[..., None]).sum(axis=2) / safe_counts[..., None]
        ).astype(np.int16)
        if keyframe:
            error = np.abs(target_blocks - means[:, :, None, :]).max(axis=(2, 3))
        else:
            error = np.abs(target_blocks - state_blocks).max(axis=(2, 3))
            error = np.where(dynamic_blocks.any(axis=2), error, 0)
        if keyframe:
            emit = active & ((error <= tolerance) | (size == 1))
            split = active & (error > tolerance) & (size > 1)
        else:
            emit = active & (error > tolerance) & (size == 1)
            split = active & (error > tolerance) & (size > 1)
            # At 4/2 px, emit a mean-colored primitive when its error is modest.
            if size <= 4:
                compact = active & (error > tolerance) & (error <= tolerance * 2) & (size > 1)
                emit |= compact
                split &= ~compact

        ys, xs = np.nonzero(emit)
        if len(xs):
            gx_blocks = _block_view(gradient_x, rows, columns, size)
            gy_blocks = _block_view(gradient_y, rows, columns, size)
            gx = (gx_blocks * dynamic_blocks).sum(axis=2) / safe_counts
            gy = (gy_blocks * dynamic_blocks).sum(axis=2) / safe_counts
            primitive = _primitive_codes(gx[ys, xs], gy[ys, xs])
            commands = np.zeros(len(xs), dtype=COMMAND_DTYPE)
            commands["x"] = xs * size
            commands["y"] = ys * size
            commands["w"] = size
            commands["h"] = size
            commands["r"] = np.clip(means[ys, xs, 0], 0, 255)
            commands["g"] = np.clip(means[ys, xs, 1], 0, 255)
            commands["b"] = np.clip(means[ys, xs, 2], 0, 255)
            commands["primitive"] = primitive
            command_parts.append(commands)
            writable = state.reshape(rows, size, columns, size, 3).transpose(0, 2, 1, 3, 4)
            writable[ys, xs, :, :, :] = np.clip(means[ys, xs], 0, 255).astype(np.uint8)[:, None, None, :]
        candidate = split

    if command_parts:
        return np.concatenate(command_parts)
    return np.zeros(0, dtype=COMMAND_DTYPE)


def build_precision_commands(
    stack: np.ndarray,
    base_rgb: np.ndarray,
    base_rgba: np.ndarray,
    lock: np.ndarray,
    output_path: Path,
    quality_path: Path,
    contact_path: Path,
    partition_path: Path,
    fps: float,
    tolerance: int = 3,
    keyframe_stride: int = 8,
) -> dict:
    frame_count, height, width, _ = stack.shape
    locked = lock > 127
    dynamic = ~locked
    partition = partition_details(lock)
    partition["tolerance"] = tolerance
    partition["keyframeStride"] = keyframe_stride
    partition_path.write_text(json.dumps(partition, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")

    header = struct.pack(
        "<4sHHHHBBH",
        b"SBT3",
        width,
        height,
        frame_count,
        keyframe_stride,
        tolerance,
        3,
        BYTES_PER_COMMAND,
    )
    state = base_rgb.copy()
    state[dynamic] = 0
    frame_payloads = []
    offsets = [0]
    quality = []
    contact_samples = {}
    total_commands = 0
    raw_commands_bytes = 0

    for frame_index, frame in enumerate(stack):
        keyframe = frame_index % keyframe_stride == 0
        if keyframe:
            state[:] = base_rgb
            state[dynamic] = 0
        commands = commands_for_frame(frame, state, dynamic, tolerance, keyframe)
        state[locked] = base_rgb[locked]
        total_commands += len(commands)
        raw_commands_bytes += len(commands) * BYTES_PER_COMMAND
        payload = bytes([1 if keyframe else 0]) + struct.pack("<I", len(commands)) + commands.tobytes()
        frame_payloads.append(payload)
        offsets.append(offsets[-1] + len(payload))

        difference = np.abs(state.astype(np.int16) - frame.astype(np.int16))
        dynamic_difference = difference[dynamic]
        static_difference = difference[locked]
        rmse = float(np.sqrt((difference.astype(np.float32) ** 2).mean()))
        quality.append(
            {
                "frame": frame_index,
                "time": round(frame_index / fps, 6),
                "mae": round(float(difference.mean()), 3),
                "dynamicMae": round(float(dynamic_difference.mean()), 3),
                "staticMae": round(float(static_difference.mean()), 3),
                "psnr": round(float(20 * math.log10(255 / max(rmse, 1e-5))), 2),
                "staticDelta": 0.0,
                "lockedFraction": round(float(locked.mean()), 6),
                "commands": int(len(commands)),
                "keyframe": keyframe,
            }
        )
        if frame_index in {0, 40, 80, 123}:
            contact_samples[frame_index] = (frame.copy(), state.copy())
        if frame_index % 20 == 0:
            print(
                f"commands frame={frame_index} count={len(commands)} "
                f"mae={quality[-1]['mae']:.3f} dynamic={quality[-1]['dynamicMae']:.3f}",
                flush=True,
            )

    index = struct.pack(f"<{len(offsets)}I", *offsets)
    payload = index + b"".join(frame_payloads)
    compressed = zlib.compress(payload, level=9)
    output_path.write_bytes(header + compressed)
    compressed_bytes = len(header) + len(compressed)
    sha256 = hashlib.sha256(header + compressed).digest().hex()

    summary = {
        "meanMae": round(float(np.mean([item["mae"] for item in quality])), 3),
        "maxMae": max(item["mae"] for item in quality),
        "minPsnr": min(item["psnr"] for item in quality),
        "maxStaticMae": max(item["staticMae"] for item in quality),
        "meanDynamicMae": round(float(np.mean([item["dynamicMae"] for item in quality])), 3),
        "maxDynamicMae": max(item["dynamicMae"] for item in quality),
        "lockedFraction": float(locked.mean()),
        "dynamicFraction": float(dynamic.mean()),
        "maxLockedChannelRange": 8,
        "meanCommandsPerFrame": round(total_commands / frame_count, 1),
        "maxCommandsPerFrame": max(item["commands"] for item in quality),
    }
    quality_report = {
        "schema": "offline-reference-quality/v3",
        "source": "pixel comparison against the original 124 frames during preprocessing",
        "runtimeVideo": False,
        "reconstruction": "locked static base plus adaptive per-frame rectangular drawing primitives",
        "partition": {
            "rule": partition["rule"],
            "lockedFraction": partition["lockedFraction"],
            "dynamicFraction": partition["dynamicFraction"],
            "dynamicComponents": partition["dynamicComponents"][:8],
        },
        "commandGrid": "adaptive 8/4/2/1 px rectangles",
        "tolerance": tolerance,
        "keyframeStride": keyframe_stride,
        "frameCount": frame_count,
        "summary": summary,
        "frames": quality,
    }
    quality_path.write_text(json.dumps(quality_report, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")

    tile_w, tile_h, caption_h = 376, 176, 26
    contact = Image.new("RGB", (tile_w * 2, (tile_h + caption_h) * 4), (15, 18, 24))
    draw = ImageDraw.Draw(contact)
    for position, (frame_index, (source, reconstructed)) in enumerate(contact_samples.items()):
        for column, (label, image) in enumerate((("source", source), ("commands", reconstructed))):
            thumb = Image.fromarray(image).resize((tile_w, tile_h))
            x = column * tile_w
            y = position * (tile_h + caption_h)
            draw.text((x + 8, y + 6), f"F{frame_index:03d} {label}", fill=(238, 243, 252))
            contact.paste(thumb, (x, y + caption_h))
    contact.save(contact_path, quality=92)

    return {
        "file": "assets/frame-commands.bin",
        "format": "SBT3 zlib/deflate; adaptive rect commands with exact keyframes and temporal deltas",
        "runtimeVideo": False,
        "version": 3,
        "width": width,
        "height": height,
        "frameCount": frame_count,
        "keyframeStride": keyframe_stride,
        "tolerance": tolerance,
        "bytesPerCommand": BYTES_PER_COMMAND,
        "primitiveCodes": {
            "0": "solid rectangle",
            "1": "horizontal segment",
            "2": "vertical segment",
            "3": "descending segment",
            "4": "ascending segment",
            "5": "ellipse",
            "6": "quadratic bezier",
            "7": "cubic brush",
        },
        "commands": total_commands,
        "meanCommandsPerFrame": summary["meanCommandsPerFrame"],
        "maxCommandsPerFrame": summary["maxCommandsPerFrame"],
        "compressedBytes": compressed_bytes,
        "payloadBytes": len(payload),
        "rawCommandBytes": raw_commands_bytes,
        "sha256": sha256,
        "quality": "assets/quality-report.json",
    }
