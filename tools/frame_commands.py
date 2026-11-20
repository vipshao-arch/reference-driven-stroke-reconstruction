"""Serialize source frames as compact per-cell drawing command deltas.

The static base is shared by every frame. Only unlocked cells carry a signed
RGB delta and a primitive code; the browser draws their colors with ellipses,
segments, bezier curves or brush strokes. The video clip is never packaged.
"""

from __future__ import annotations

import json
import math
import struct
import hashlib
import zlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


CELL_SIZE = 8


def build_frame_commands(
    stack: np.ndarray,
    base: np.ndarray,
    lock: np.ndarray,
    output_path: Path,
    quality_path: Path,
    contact_path: Path,
    fps: float,
) -> dict:
    frame_count, height, width, _ = stack.shape
    rows, columns = height // CELL_SIZE, width // CELL_SIZE
    unlocked = lock < 128
    mask_cells = unlocked.reshape(rows, CELL_SIZE, columns, CELL_SIZE)
    dynamic_cells = mask_cells.any(axis=(1, 3))
    counts = mask_cells.sum(axis=(1, 3)).astype(np.float32)

    base_pixels = base.reshape(rows, CELL_SIZE, columns, CELL_SIZE, 3).astype(np.float32)
    mask_float = mask_cells.astype(np.float32)
    base_all = base_pixels.mean(axis=(1, 3))
    base_sums = (base_pixels * mask_float[..., None]).sum(axis=(1, 3))
    base_cell = np.divide(
        base_sums,
        counts[..., None],
        out=base_all.copy(),
        where=counts[..., None] > 0,
    )
    base_cell = np.clip(np.rint(base_cell), 0, 255).astype(np.int16)
    palette = base_cell.astype(np.uint8)

    header = struct.pack(
        "<4sHHHHBBH",
        b"SBT2",
        columns,
        rows,
        frame_count,
        CELL_SIZE,
        4,  # signed RGB delta bytes + primitive byte
        2,
        palette.size,
    )
    payload = bytearray(header)
    payload.extend(palette.tobytes())

    quality = []
    contact_samples = {}
    previous_unlocked_static: np.ndarray | None = None

    for frame_index in range(frame_count):
        frame = stack[frame_index]
        frame_pixels = frame.reshape(rows, CELL_SIZE, columns, CELL_SIZE, 3).astype(np.float32)
        color_sums = (frame_pixels * mask_float[..., None]).sum(axis=(1, 3))
        color = np.divide(
            color_sums,
            counts[..., None],
            out=base_all.copy(),
            where=counts[..., None] > 0,
        )
        color = np.clip(np.rint(color), 0, 255).astype(np.int16)
        delta = np.clip(color - base_cell, -127, 127).astype(np.int8)
        delta[~dynamic_cells] = 0

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gradient_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        gradient_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        edge_x = gradient_x.reshape(rows, CELL_SIZE, columns, CELL_SIZE).mean(axis=(1, 3))
        edge_y = gradient_y.reshape(rows, CELL_SIZE, columns, CELL_SIZE).mean(axis=(1, 3))
        edge = (edge_x + edge_y) * 0.5
        residual = np.abs(frame.astype(np.int16) - base.astype(np.int16)).mean(axis=2)
        residual_cells = residual.reshape(rows, CELL_SIZE, columns, CELL_SIZE).mean(axis=(1, 3))

        active_edge = edge[dynamic_cells]
        active_x = edge_x[dynamic_cells]
        active_y = edge_y[dynamic_cells]
        active_residual = residual_cells[dynamic_cells]
        local_codes = np.full(int(dynamic_cells.sum()), 7, dtype=np.uint8)
        local_codes[active_edge < 4] = 5
        horizontal = (active_edge >= 4) & (active_edge < 28) & (active_y > active_x * 1.35)
        vertical = (active_edge >= 4) & (active_edge < 28) & (active_x > active_y * 1.35)
        diagonal = (active_edge >= 4) & (active_edge < 28) & ~horizontal & ~vertical
        local_codes[horizontal] = 1
        local_codes[vertical] = 2
        local_codes[diagonal & (active_x > active_y)] = 3
        local_codes[diagonal & (active_x <= active_y)] = 4
        local_codes[(active_edge >= 28) | (active_residual > 35)] = 6
        codes = np.zeros((rows, columns), dtype=np.uint8)
        codes[dynamic_cells] = local_codes

        packed = np.dstack((delta.astype(np.uint8), codes)).astype(np.uint8)
        payload.extend(packed.tobytes())

        signed_delta = delta.astype(np.uint8).astype(np.int16)
        signed_delta[signed_delta > 127] -= 256
        decoded_colors = np.clip(base_cell + signed_delta, 0, 255).astype(np.uint8)
        reconstructed_pixels = np.repeat(
            np.repeat(decoded_colors, CELL_SIZE, axis=0), CELL_SIZE, axis=1
        )
        reconstruction = base.copy()
        reconstruction[unlocked] = reconstructed_pixels[unlocked]
        difference = np.abs(reconstruction.astype(np.int16) - frame.astype(np.int16))
        rmse = float(np.sqrt((difference.astype(np.float32) ** 2).mean()))

        unlocked_reconstruction = reconstructed_pixels.copy()
        unlocked_difference = np.abs(unlocked_reconstruction.astype(np.int16) - frame.astype(np.int16))
        unlocked_rmse = float(np.sqrt((unlocked_difference.astype(np.float32) ** 2).mean()))
        unlocked_static = unlocked_reconstruction[~unlocked]
        unlocked_static_delta = 0.0
        if previous_unlocked_static is not None:
            unlocked_static_delta = float(
                np.abs(unlocked_static.astype(np.int16) - previous_unlocked_static.astype(np.int16)).mean()
            )
        previous_unlocked_static = unlocked_static.copy()

        quality.append(
            {
                "frame": frame_index,
                "time": round(frame_index / fps, 6),
                "mae": round(float(difference.mean()), 3),
                "dynamicMae": round(float(difference[unlocked].mean()), 3),
                "staticMae": round(float(difference[~unlocked].mean()), 3),
                "psnr": round(float(20 * math.log10(255 / max(rmse, 1e-5))), 2),
                "unlockedMae": round(float(unlocked_difference.mean()), 3),
                "unlockedDynamicMae": round(float(unlocked_difference[unlocked].mean()), 3),
                "unlockedStaticMae": round(float(unlocked_difference[~unlocked].mean()), 3),
                "unlockedPsnr": round(float(20 * math.log10(255 / max(unlocked_rmse, 1e-5))), 2),
                "staticDelta": 0.0,
                "unlockedStaticDelta": round(unlocked_static_delta, 3),
                "lockedFraction": round(float((~unlocked).mean()), 6),
                "commands": int(dynamic_cells.sum()),
            }
        )
        if frame_index in {0, 40, 80, 123}:
            contact_samples[frame_index] = reconstruction.copy()

    compressed = zlib.compress(bytes(payload), level=9)
    output_path.write_bytes(compressed)
    quality_path.write_text(
        json.dumps(
            {
                "schema": "offline-reference-quality/v2",
                "source": "pixel comparison against the original 124 frames during preprocessing",
                "runtimeVideo": False,
                "reconstruction": "static base plus opaque dynamic-cell command deltas; locked pixels immutable",
                "cellSize": CELL_SIZE,
                "grid": [columns, rows],
                "frameCount": frame_count,
                "commandsPerFrame": int(dynamic_cells.sum()),
                "summary": {
                    "meanMae": round(float(np.mean([item["mae"] for item in quality])), 3),
                    "maxMae": max(item["mae"] for item in quality),
                    "minPsnr": min(item["psnr"] for item in quality),
                    "maxStaticMae": max(item["staticMae"] for item in quality),
                    "maxUnlockedStaticDelta": max(item["unlockedStaticDelta"] for item in quality),
                    "lockedFraction": float((~unlocked).mean()),
                },
                "frames": quality,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    tile_w, tile_h, caption_h = 376, 176, 26
    contact = Image.new("RGB", (tile_w * 2, (tile_h + caption_h) * 2), (15, 18, 24))
    draw = ImageDraw.Draw(contact)
    for position, (frame_index, image) in enumerate(contact_samples.items()):
        thumb = Image.fromarray(image).resize((tile_w, tile_h))
        x = (position % 2) * tile_w
        y = (position // 2) * (tile_h + caption_h)
        draw.text((x + 8, y + 6), f"reconstruction F{frame_index:03d} / {frame_index / fps:.2f}s", fill=(238, 243, 252))
        contact.paste(thumb, (x, y + caption_h))
    contact.save(contact_path, quality=90)

    return {
        "file": "assets/frame-commands.bin",
        "format": "SBT2 zlib/deflate; static RGB palette plus signed RGB deltas and primitive codes",
        "cellSize": CELL_SIZE,
        "grid": [columns, rows],
        "frameCount": frame_count,
        "commandsPerFrame": int(dynamic_cells.sum()),
        "runtimeVideo": False,
        "primitiveCodes": {
            "1": "horizontal segment",
            "2": "vertical segment",
            "3": "descending segment",
            "4": "ascending segment",
            "5": "ellipse",
            "6": "quadratic bezier",
            "7": "cubic brush",
        },
        "compressedBytes": len(compressed),
        "rawBytes": len(payload),
        "sha256": hashlib.sha256(compressed).hexdigest(),
        "paletteBytes": int(palette.size),
        "quality": "assets/quality-report.json",
    }
