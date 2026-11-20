"""Coherent static partition and high-precision SBT4 drawing command stream.

The source video is decoded only in this offline preprocessing step. Stable
regions become an RGBA WebGL base with alpha 255; every other pixel belongs to
the dynamic drawing surface. Commands are exact 2x2-pixel rectangles (RGB555)
on periodic keyframes, followed by sparse per-pixel repaint commands when a
2x2 command region changes perceptibly. Each command carries a primitive code
describing the observed local edge: rectangle, line, ellipse or bezier brush.
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


COMMAND_STRIDE = 2
KEYFRAME_STRIDE = 8
BLOCK_THRESHOLD = 8
DELTA_THRESHOLD = 4
MAGIC = b"SBT4"
HEADER = struct.Struct("<4sHHHHBBBBI")
ENCODE = np.rint(np.arange(256, dtype=np.float32) * 31 / 255).astype(np.uint8)
DECODE = np.rint(np.arange(32, dtype=np.float32) * 255 / 31).astype(np.uint8)


def temporal_partition(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return median RGB, alpha static base, dynamic mask and exact interval data."""
    frame_count, height, width, _ = stack.shape
    base = np.empty((height, width, 3), dtype=np.uint8)
    robust_range = np.empty((height, width), dtype=np.uint8)
    exact_range = np.empty((height, width), dtype=np.uint8)
    strip = 24
    for top in range(0, height, strip):
        bottom = min(top + strip, height)
        chunk = stack[:, top:bottom]
        median = np.median(chunk, axis=0).astype(np.uint8)
        base[top:bottom] = median
        low = np.quantile(chunk, 0.02, axis=0).astype(np.int16)
        high = np.quantile(chunk, 0.98, axis=0).astype(np.int16)
        robust_range[top:bottom] = np.clip((high - low).max(axis=2), 0, 255).astype(np.uint8)
        exact_range[top:bottom] = np.clip(
            (chunk.max(axis=0).astype(np.int16) - chunk.min(axis=0).astype(np.int16)).max(axis=2),
            0,
            255,
        ).astype(np.uint8)

    candidate = (robust_range > 12) | (exact_range > 16)
    dynamic_image = cv2.morphologyEx(
        candidate.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), np.uint8),
    )
    contours, _ = cv2.findContours(dynamic_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(dynamic_image)
    for contour in contours:
        if cv2.contourArea(contour) >= 2500:
            cv2.drawContours(filled, [contour], -1, 255, -1)
    dynamic_image = cv2.bitwise_or(dynamic_image, filled)
    dynamic_image = cv2.morphologyEx(dynamic_image, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    dynamic_image = cv2.dilate(dynamic_image, np.ones((3, 3), np.uint8), iterations=1)
    dynamic = dynamic_image > 0
    for y in range(height):
        padded = np.concatenate(([False], dynamic[y], [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        for start, end in zip(edges[::2], edges[1::2]):
            if end - start < 16:
                dynamic[y, start:end] = padded[start] if start > 0 else padded[end]

    scale = COMMAND_STRIDE
    cells = dynamic.reshape(height // scale, scale, width // scale, scale).any(axis=(1, 3))
    dynamic = np.repeat(np.repeat(cells, scale, axis=0), scale, axis=1)
    locked = ~dynamic
    base_rgba = np.dstack((base, np.where(locked, 255, 0).astype(np.uint8)))
    lock = np.where(locked, 255, 0).astype(np.uint8)

    rule = (
        "2%-98% temporal channel range > 12 or exact range > 16; 9px closing, "
        "large-component fill, 11px closing, 1px dilation, >=16px scanline runs, "
        "then 2x2 command alignment"
    )
    details = {
        "rule": rule,
        "robustRangeThreshold": 12,
        "exactRangeThreshold": 16,
        "commandStride": scale,
        "width": width,
        "height": height,
        "lockedFraction": float(locked.mean()),
        "dynamicFraction": float(dynamic.mean()),
        "dynamicCommandCellCount": int(cells.sum()),
        "staticScanlineIntervals": [],
        "dynamicScanlineIntervals": [],
        "dynamicComponents": [],
        "staticComponents": [],
    }
    horizontal_kernel = np.ones((1, 16), np.uint8)
    cleaned_locked = cv2.morphologyEx(locked.astype(np.uint8) * 255, cv2.MORPH_OPEN, horizontal_kernel) > 0
    cleaned_dynamic = cv2.morphologyEx(dynamic.astype(np.uint8) * 255, cv2.MORPH_OPEN, horizontal_kernel) > 0
    for y in range(height):
        for name, row in (("staticScanlineIntervals", cleaned_locked[y]), ("dynamicScanlineIntervals", cleaned_dynamic[y])):
            padded = np.concatenate(([False], row, [False]))
            edges = np.flatnonzero(padded[1:] != padded[:-1])
            details[name].append([y, [[int(edges[i]), int(edges[i + 1] - 1)] for i in range(0, len(edges), 2)]])
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
    return base, base_rgba, lock, details


def rgb555(frame: np.ndarray) -> np.ndarray:
    encoded = ENCODE[frame]
    return (
        encoded[:, :, 0].astype(np.uint16) << 10
        | encoded[:, :, 1].astype(np.uint16) << 5
        | encoded[:, :, 2].astype(np.uint16)
    )


def decode555(packed: np.ndarray) -> np.ndarray:
    red = ((packed >> 10) & 31).astype(np.uint8)
    green = ((packed >> 5) & 31).astype(np.uint8)
    blue = (packed & 31).astype(np.uint8)
    return np.dstack((DECODE[red], DECODE[green], DECODE[blue]))


def primitive_codes(target: np.ndarray) -> np.ndarray:
    """Classify each 2x2 drawing command by its local horizontal/vertical edge."""
    signed = target.astype(np.int16)
    horizontal = np.abs(np.diff(signed, axis=1, prepend=signed[:, :1])).max(axis=2)
    vertical = np.abs(np.diff(signed, axis=0, prepend=signed[:1])).max(axis=2)
    cells_h = horizontal.reshape(target.shape[0] // 2, 2, target.shape[1] // 2, 2).max(axis=(1, 3))
    cells_v = vertical.reshape(target.shape[0] // 2, 2, target.shape[1] // 2, 2).max(axis=(1, 3))
    edge = np.maximum(cells_h, cells_v)
    codes = np.full(edge.shape, 4, dtype=np.uint8)  # cubic brush / complex edge
    codes[edge < 24] = 3  # ellipse / low-contrast patch
    flat = edge < 5
    codes[flat] = 0
    horizontal_edge = (edge >= 24) & (cells_v > cells_h * 1.4)
    vertical_edge = (edge >= 24) & (cells_h > cells_v * 1.4)
    codes[horizontal_edge] = 1
    codes[vertical_edge] = 2
    codes[(edge >= 6) & (edge < 24) & (cells_h > cells_v * 1.4)] = 5
    codes[(edge >= 6) & (edge < 24) & (cells_v > cells_h * 1.4)] = 6
    return codes


def varint(values: np.ndarray) -> bytes:
    output = bytearray()
    for value in values.tolist():
        while value >= 0x80:
            output.append((value & 0x7F) | 0x80)
            value >>= 7
        output.append(value)
    return bytes(output)


def frame_commands(
    target: np.ndarray,
    state: np.ndarray,
    dynamic_cells: np.ndarray,
    keyframe: bool,
) -> tuple[bytes, np.ndarray, int]:
    height, width = target.shape[:2]
    dynamic = np.repeat(np.repeat(dynamic_cells, 2, axis=0), 2, axis=1)
    packed = rgb555(target)
    codes = primitive_codes(target)
    next_state = state.copy()

    if keyframe:
        encoded_top_left = ENCODE[target[::2, ::2]]
        command_colors = (
            encoded_top_left[:, :, 0].astype(np.uint16) << 10
            | encoded_top_left[:, :, 1].astype(np.uint16) << 5
            | encoded_top_left[:, :, 2].astype(np.uint16)
        )[dynamic_cells]
        command_codes = codes[dynamic_cells]
        colors = decode555(command_colors.reshape(-1, 1)).reshape(-1, 3)
        writable = next_state.reshape(height // 2, 2, width // 2, 2, 3).transpose(0, 2, 1, 3, 4)
        ys, xs = np.nonzero(dynamic_cells)
        writable[ys, xs, :, :, :] = colors[:, None, None, :]
        payload = command_colors.astype("<u2").tobytes() + command_codes.astype(np.uint8).tobytes()
        return payload, next_state, int(dynamic_cells.sum())

    difference = np.abs(target.astype(np.int16) - state.astype(np.int16)).max(axis=2)
    cell_difference = difference.reshape(height // 2, 2, width // 2, 2).max(axis=(1, 3))
    active_cells = dynamic_cells & (cell_difference > BLOCK_THRESHOLD)
    active = np.repeat(np.repeat(active_cells, 2, axis=0), 2, axis=1)
    active_indices = np.flatnonzero(active.ravel())
    active_target = target.reshape(-1, 3)[active_indices]
    active_state = state.reshape(-1, 3)[active_indices]
    encoded_active = ENCODE[active_target]
    decoded_active = DECODE[encoded_active]
    packed_active = (
        encoded_active[:, 0].astype(np.uint16) << 10
        | encoded_active[:, 1].astype(np.uint16) << 5
        | encoded_active[:, 2].astype(np.uint16)
    )
    changed = np.flatnonzero(
        np.abs(decoded_active.astype(np.int16) - active_state.astype(np.int16)).max(axis=1) > DELTA_THRESHOLD
    )
    changed_colors = packed_active[changed].astype("<u2").tobytes()
    changed_positions = active_indices[changed]
    gaps = np.diff(np.concatenate(([0], changed_positions + 1))).astype(np.uint32)
    writable = next_state.reshape(-1, 3)
    writable[active_indices[changed]] = decoded_active[changed]
    payload = struct.pack("<I", len(changed)) + varint(gaps) + changed_colors
    return payload, next_state, int(active_cells.sum())


def build_compact_commands(
    stack: np.ndarray,
    base: np.ndarray,
    lock: np.ndarray,
    partition: dict,
    output_path: Path,
    quality_path: Path,
    contact_path: Path,
    partition_path: Path,
    fps: float,
) -> dict:
    frame_count, height, width, _ = stack.shape
    locked = lock > 127
    dynamic = ~locked
    dynamic_cells = dynamic.reshape(height // 2, 2, width // 2, 2).any(axis=(1, 3))
    columns = width // 2
    rows = height // 2
    header = HEADER.pack(
        MAGIC,
        width,
        height,
        frame_count,
        KEYFRAME_STRIDE,
        BLOCK_THRESHOLD,
        DELTA_THRESHOLD,
        4,
        COMMAND_STRIDE,
        int(dynamic_cells.sum()),
    )
    partition["blockThreshold"] = BLOCK_THRESHOLD
    partition["deltaThreshold"] = DELTA_THRESHOLD
    partition["keyframeStride"] = KEYFRAME_STRIDE
    partition["qualitySchema"] = "offline-reference-quality/v4"
    partition_path.write_text(json.dumps(partition, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    mask_payload = np.packbits(dynamic_cells.ravel(), bitorder="little").tobytes()
    state = base.copy()
    state[dynamic] = 0
    frame_payloads = []
    offsets = [0]
    quality = []
    contact_samples = {}
    total_commands = 0
    locked_deviation_sum = 0.0
    locked_deviation_max = 0

    for frame_index, frame in enumerate(stack):
        keyframe = frame_index % KEYFRAME_STRIDE == 0
        if keyframe:
            state[:] = base
            state[dynamic] = 0
        payload, state, command_count = frame_commands(frame, state, dynamic_cells, keyframe)
        state[locked] = base[locked]
        total_commands += command_count
        frame_payloads.append(bytes([1 if keyframe else 0]) + payload)
        offsets.append(offsets[-1] + len(payload) + 1)

        difference = np.abs(state.astype(np.int16) - frame.astype(np.int16))
        locked_deviation = difference.max(axis=2)[locked]
        locked_deviation_sum += float(locked_deviation.mean())
        locked_deviation_max = max(locked_deviation_max, int(locked_deviation.max()))
        dynamic_difference = difference[dynamic]
        static_difference = difference[locked]
        rmse = float(np.sqrt((difference.astype(np.float32) ** 2).mean()))
        quality.append({
            "frame": frame_index,
            "time": round(frame_index / fps, 6),
            "mae": round(float(difference.mean()), 3),
            "dynamicMae": round(float(dynamic_difference.mean()), 3),
            "staticMae": round(float(static_difference.mean()), 3),
            "psnr": round(float(20 * math.log10(255 / max(rmse, 1e-5))), 2),
            "staticDelta": 0.0,
            "lockedFraction": round(float(locked.mean()), 6),
            "commands": command_count,
            "keyframe": keyframe,
        })
        if frame_index in {0, 40, 80, 123}:
            contact_samples[frame_index] = (frame.copy(), state.copy())
        if frame_index % 20 == 0:
            print(f"frame={frame_index} commands={command_count} mae={quality[-1]['mae']:.3f}", flush=True)

    index = struct.pack(f"<{len(offsets)}I", *offsets)
    raw_payload = mask_payload + index + b"".join(frame_payloads)
    compressed = zlib.compress(raw_payload, level=9)
    output_path.write_bytes(header + compressed)
    compressed_bytes = len(header) + len(compressed)
    sha256 = hashlib.sha256(header + compressed).hexdigest()

    summary = {
        "meanMae": round(float(np.mean([item["mae"] for item in quality])), 3),
        "maxMae": max(item["mae"] for item in quality),
        "minPsnr": min(item["psnr"] for item in quality),
        "meanDynamicMae": round(float(np.mean([item["dynamicMae"] for item in quality])), 3),
        "maxDynamicMae": max(item["dynamicMae"] for item in quality),
        "maxStaticMae": max(item["staticMae"] for item in quality),
        "lockedFraction": float(locked.mean()),
        "dynamicFraction": float(dynamic.mean()),
        "lockedMeanChannelDeviation": round(locked_deviation_sum / frame_count, 3),
        "lockedMaxChannelDeviation": locked_deviation_max,
        "meanCommandsPerFrame": round(total_commands / frame_count, 1),
        "maxCommandsPerFrame": max(item["commands"] for item in quality),
    }
    quality_report = {
        "schema": "offline-reference-quality/v4",
        "source": "pixel comparison against the original 124 frames during preprocessing",
        "runtimeVideo": False,
        "reconstruction": "locked RGBA static base plus high-precision 2x2 RGB555 drawing commands and sparse per-pixel temporal repaints",
        "partition": {
            "rule": partition["rule"],
            "lockedFraction": partition["lockedFraction"],
            "dynamicFraction": partition["dynamicFraction"],
            "dynamicCommandCellCount": partition["dynamicCommandCellCount"],
            "dynamicComponents": partition["dynamicComponents"][:8],
            "staticComponents": partition["staticComponents"][:8],
        },
        "commandGrid": "2x2 command cells with RGB555 colors and sparse exact-pixel deltas",
        "blockThreshold": BLOCK_THRESHOLD,
        "deltaThreshold": DELTA_THRESHOLD,
        "keyframeStride": KEYFRAME_STRIDE,
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
        "format": "SBT4 zlib/deflate; coherent RGBA static base, RGB555 2x2 keyframe drawing commands and sparse absolute-position per-pixel drawing deltas",
        "runtimeVideo": False,
        "version": 4,
        "width": width,
        "height": height,
        "frameCount": frame_count,
        "commandStride": COMMAND_STRIDE,
        "keyframeStride": KEYFRAME_STRIDE,
        "blockThreshold": BLOCK_THRESHOLD,
        "deltaThreshold": DELTA_THRESHOLD,
        "dynamicCells": int(dynamic_cells.sum()),
        "headerBytes": HEADER.size,
        "maskPayloadBytes": len(mask_payload),
        "bytesPerKeyframeCommand": 3,
        "deltaCommand": "LEB128 absolute pixel gap plus 2-byte RGB555 color",
        "primitiveCodes": {
            "0": "solid rectangle",
            "1": "horizontal segment",
            "2": "vertical segment",
            "3": "ellipse",
            "4": "cubic brush",
            "5": "quadratic bezier horizontal",
            "6": "quadratic bezier vertical",
            "7": "reserved",
        },
        "commands": total_commands,
        "meanCommandsPerFrame": summary["meanCommandsPerFrame"],
        "maxCommandsPerFrame": summary["maxCommandsPerFrame"],
        "compressedBytes": compressed_bytes,
        "payloadBytes": len(raw_payload),
        "sha256": sha256,
        "quality": "assets/quality-report.json",
    }
