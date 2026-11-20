"""Compare exact and compact dynamic drawing-command stream sizes/errors."""

from __future__ import annotations

import json
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
    frame_count, height, width, _ = stack.shape
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


def main() -> None:
    stack = read_video(VIDEO)
    frame_count, height, width, _ = stack.shape
    base, locked = exact_lock(stack)
    dynamic = ~locked
    print("locked", locked.mean(), "dynamic", dynamic.mean(), flush=True)
    rows, columns = height // 2, width // 2
    dynamic_cells = dynamic.reshape(rows, 2, columns, 2).any(axis=(1, 3))
    cell_indices = np.flatnonzero(dynamic_cells.ravel())
    print("dynamic 2px cells", len(cell_indices), "fraction", dynamic_cells.mean(), flush=True)

    exact_compressor = zlib.compressobj(level=9)
    exact_bytes = 0
    raw_bytes = 0
    exact_file = OUT / "probe-exact-dynamic.deflate"
    cell_compressor = zlib.compressobj(level=9)
    cell_bytes = 0
    cell_file = OUT / "probe-cell-dynamic.deflate"
    sample_errors = []
    with exact_file.open("wb") as exact_out, cell_file.open("wb") as cell_out:
        for frame_index, frame in enumerate(stack):
            exact_pixels = frame[dynamic]
            exact_out.write(exact_compressor.compress(exact_pixels.tobytes()))
            raw_bytes += exact_pixels.size

            cells = frame.reshape(rows, 2, columns, 2, 3).transpose(0, 2, 1, 3, 4)
            active = cells[dynamic_cells].reshape(-1, 4, 3).astype(np.int16)
            mean = np.rint(active.mean(axis=1)).astype(np.uint8)
            cell_out.write(cell_compressor.compress(mean.tobytes()))

            if frame_index in {0, 40, 80, 123}:
                reconstructed = base.copy()
                cell_colors = np.zeros((rows, columns, 3), dtype=np.uint8)
                cell_colors[dynamic_cells] = mean
                expanded = np.repeat(np.repeat(cell_colors, 2, axis=0), 2, axis=1)
                reconstructed[dynamic] = expanded[dynamic]
                difference = np.abs(reconstructed.astype(np.int16) - frame.astype(np.int16))
                sample_errors.append(
                    {
                        "frame": frame_index,
                        "mae": round(float(difference.mean()), 3),
                        "dynamicMae": round(float(difference[dynamic].mean()), 3),
                        "psnr": round(float(20 * np.log10(255 / max(np.sqrt((difference.astype(np.float32) ** 2).mean()), 1e-5))), 2),
                    }
                )
            if frame_index % 20 == 0:
                print("encoded frames", frame_index, flush=True)
        exact_out.write(exact_compressor.flush())
        cell_out.write(cell_compressor.flush())
    exact_bytes = exact_file.stat().st_size
    cell_bytes = cell_file.stat().st_size
    result = {
        "lockedFraction": float(locked.mean()),
        "dynamicFraction": float(dynamic.mean()),
        "dynamicCellCount": int(dynamic_cells.sum()),
        "exactDynamicRawBytes": int(raw_bytes),
        "exactDynamicDeflateBytes": int(exact_bytes),
        "cellDynamicDeflateBytes": int(cell_bytes),
        "cellSampleErrors": sample_errors,
    }
    print(json.dumps(result, indent=2), flush=True)
    (OUT / "encoding-probe.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
