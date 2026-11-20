"""Validate the offline static-base measurements without a runtime video."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "public" / "assets"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        help="optional offline source clip used to re-check the quality report",
    )
    args = parser.parse_args()
    base = cv2.imread(str(ASSETS / "static-base.png"), cv2.IMREAD_UNCHANGED)
    lock = cv2.imread(str(ASSETS / "static-lock.png"), cv2.IMREAD_GRAYSCALE)
    quality = json.loads((ASSETS / "quality-report.json").read_text(encoding="utf-8"))
    assert base is not None and lock is not None
    assert quality["runtimeVideo"] is False
    locked = lock > 127
    assert base.ndim == 3 and base.shape[2] == 4, "static base must have RGBA alpha"
    assert np.array_equal(base[:, :, 3] > 127, locked), "static base alpha and lock mask disagree"
    frame_count = quality["frameCount"]
    assert frame_count == 124, f"expected 124 frames, report has {frame_count}"
    assert len(quality["frames"]) == frame_count
    assert locked.mean() > 0.2, f"lock fraction too small: {locked.mean()}"
    assert all(frame["staticDelta"] == 0 for frame in quality["frames"]), "locked base has temporal changes"
    max_mean_error = max(frame["staticMae"] for frame in quality["frames"])
    assert max_mean_error < 3.0, f"locked pixels deviate too much: {max_mean_error}"

    if args.video is not None:
        capture = cv2.VideoCapture(str(args.video))
        assert capture.isOpened(), f"cannot open offline source: {args.video}"
        decoded = 0
        measured_max = 0.0
        base_rgb = cv2.cvtColor(base[:, :, :3], cv2.COLOR_BGRA2RGB)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            residual = float(np.abs(rgb.astype(np.int16) - base_rgb.astype(np.int16))[locked].mean())
            measured_max = max(measured_max, residual)
            decoded += 1
        capture.release()
        assert decoded == frame_count, f"expected {frame_count} frames, decoded {decoded}"
        assert measured_max < 3.0, f"offline video locked pixels changed too much: {measured_max}"
        max_mean_error = measured_max

    print(
        f"Locked-base check passed: {frame_count} frames, "
        f"{locked.mean() * 100:.1f}% locked, max mean residual {max_mean_error:.3f}/255"
    )


if __name__ == "__main__":
    main()
