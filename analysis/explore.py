"""Inspect the supplied reference clip and produce a visual contact sheet."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis"
OUT.mkdir(exist_ok=True)
SOURCE = Path(r"C:\Users\vipshao\Desktop\参考.mp4")


def read_frames(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames, fps


def main() -> None:
    frames, fps = read_frames(SOURCE)
    height, width = frames[0].shape[:2]
    print(f"frames={len(frames)} fps={fps} size={width}x{height}")

    indices = [0, 8, 20, 32, 44, 56, 68, 80, 92, 104, 116, 123]
    tile_w, tile_h, caption_h = 376, 176, 26
    sheet = Image.new("RGB", (tile_w * 3, (tile_h + caption_h) * 4), (15, 18, 24))
    draw = ImageDraw.Draw(sheet)
    for position, frame_index in enumerate(indices):
        image = Image.fromarray(frames[frame_index]).resize((tile_w, tile_h))
        x = (position % 3) * tile_w
        y = (position // 3) * (tile_h + caption_h)
        draw.text((x + 8, y + 5), f"frame {frame_index:03d} / {frame_index / fps:.2f}s", fill=(240, 244, 250))
        sheet.paste(image, (x, y + caption_h))
    sheet.save(OUT / "contact-sheet.jpg", quality=90)

    stack = np.asarray(frames, dtype=np.int16)
    median = np.median(stack, axis=0)
    motion = np.median(np.abs(stack - median), axis=0).mean(axis=2)
    print("temporal MAD percentiles", np.percentile(motion, [0, 25, 50, 75, 90, 95, 99, 100]))
    dynamic = motion > 8
    print("dynamic fraction", dynamic.mean(), "bbox", cv2.boundingRect(dynamic.astype(np.uint8)))
    for frame_index in [0, 20, 40, 60, 80, 100, 123]:
        delta = np.abs(stack[frame_index] - median).mean(axis=2)
        print(
            frame_index,
            "median MAE",
            round(float(delta.mean()), 3),
            "p95",
            round(float(np.percentile(delta, 95)), 3),
            "changed>15",
            round(float((delta > 15).mean()), 5),
        )
    Image.fromarray(median.astype(np.uint8)).save(OUT / "median.png")
    Image.fromarray(np.clip(motion * 10, 0, 255).astype(np.uint8)).save(OUT / "motion-heatmap.png")


if __name__ == "__main__":
    main()
