"""Extract a locked WebGL base and a time-varying parametric stroke timeline.

The analysis is intentionally deterministic and reproducible: temporal robust
statistics lock the static canvas, residual contours become drawing commands,
optical flow measures their motion, and assignment keeps stroke identities
between keyframes. Every source frame is also serialized as cell drawing
commands. The browser uses only those commands and static textures to draw
the requested frame; the source clip is never loaded at runtime.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

from compact_commands import (
    build_compact_commands,
    temporal_partition,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = Path.home() / "Desktop" / "参考.mp4"
DEFAULT_ASSETS = ROOT / "public" / "assets"
DEFAULT_ANALYSIS = ROOT / "analysis"
KEYFRAME_STRIDE = 4
POINT_COUNT = 12


def read_video(path: Path) -> tuple[np.ndarray, float, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"Video contains no frames: {path}")
    return np.stack(frames), fps, len(frames)


def temporal_base_and_motion(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute a temporal median and a robust per-pixel change score in strips."""
    frame_count, height, width, _ = stack.shape
    base = np.empty((height, width, 3), dtype=np.uint8)
    motion = np.empty((height, width), dtype=np.float32)
    strip = 48
    for top in range(0, height, strip):
        bottom = min(height, top + strip)
        chunk = stack[:, top:bottom, :, :]
        tile_base = np.median(chunk, axis=0).astype(np.uint8)
        base[top:bottom] = tile_base
        delta = np.abs(chunk.astype(np.int16) - tile_base.astype(np.int16)).mean(axis=3)
        motion[top:bottom] = np.median(delta, axis=0)
    return base, motion


def create_lock_mask(motion: np.ndarray) -> np.ndarray:
    """255 marks pixels whose WebGL base is immutable across the entire clip."""
    smoothed = cv2.GaussianBlur(motion, (0, 0), 1.1)
    static = (smoothed < 5.0).astype(np.uint8) * 255
    static = cv2.morphologyEx(static, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    static = cv2.morphologyEx(static, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dynamic = (static == 0).astype(np.uint8) * 255
    dynamic = cv2.dilate(dynamic, np.ones((3, 3), np.uint8), iterations=1)
    return (dynamic == 0).astype(np.uint8) * 255


def resample_closed(points: np.ndarray, count: int) -> np.ndarray:
    """Sample a closed contour uniformly by arc length."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) < 2:
        return np.repeat(points[:1], count, axis=0)
    closed = np.vstack([points, points[:1]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = cumulative[-1]
    if total < 1e-5:
        return np.repeat(points[:1], count, axis=0)
    targets = np.linspace(0.0, total, count, endpoint=False)
    result = np.empty((count, 2), dtype=np.float32)
    segment = 0
    for index, target in enumerate(targets):
        while segment < len(lengths) - 1 and cumulative[segment + 1] < target:
            segment += 1
        length = max(lengths[segment], 1e-5)
        ratio = (target - cumulative[segment]) / length
        result[index] = closed[segment] + ratio * (closed[segment + 1] - closed[segment])
    return result


def contour_detection(
    frame: np.ndarray,
    base: np.ndarray,
    lock: np.ndarray,
    contour: np.ndarray,
    diff_gray: np.ndarray,
) -> dict | None:
    area = float(cv2.contourArea(contour))
    if area < 90:
        return None
    half_h, half_w = diff_gray.shape
    local_mask = np.zeros((half_h, half_w), dtype=np.uint8)
    cv2.drawContours(local_mask, [contour], -1, 255, -1)

    data = contour[:, 0, :].astype(np.float64)
    mean = data.mean(axis=0)
    centered = data - mean
    if len(data) < 3:
        return None
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = int(np.argmax(eigenvalues))
    major = eigenvectors[:, order]
    minor = np.array([-major[1], major[0]], dtype=np.float64)
    angle = math.degrees(math.atan2(major[1], major[0]))

    projected = centered @ major
    perpendicular = centered @ minor
    radius_x = max(float(np.percentile(np.abs(projected), 96)), 2.0) * 2.0
    radius_y = max(float(np.percentile(np.abs(perpendicular), 96)), 1.5) * 2.0
    if radius_y > radius_x:
        radius_x, radius_y = radius_y, radius_x
        angle += 90.0
    radius_x = max(radius_x, 6.0)
    radius_y = max(radius_y, 3.0)

    hull = cv2.convexHull(contour)
    sampled = resample_closed(hull[:, 0, :], POINT_COUNT) * 2.0
    center = mean * 2.0
    radians = math.radians(angle)
    direction = np.array([math.cos(radians), math.sin(radians)])
    normal = np.array([-direction[1], direction[0]])
    local_points = []
    for point in sampled:
        offset = point - center
        local_points.append(
            [
                round(float(np.clip(offset @ direction / radius_x, -2.5, 2.5)), 4),
                round(float(np.clip(offset @ normal / radius_y, -2.5, 2.5)), 4),
            ]
        )

    curvature = float(np.mean((perpendicular / (max(np.std(perpendicular), 1.0))) ** 3))
    curvature = float(np.clip(curvature / 3.0, -1.0, 1.0))
    mean_color = cv2.mean(frame, mask=local_mask)[:3]
    mean_residual = float(cv2.mean(diff_gray, mask=local_mask)[0])
    elongation = radius_x / max(radius_y, 1.0)
    if elongation > 3.1:
        primitive = "line" if area < 280 else "bezier"
    elif elongation > 1.65:
        primitive = "bezier"
    else:
        primitive = "ellipse"

    return {
        "center": [round(float(center[0]), 2), round(float(center[1]), 2)],
        "radii": [round(radius_x, 2), round(radius_y, 2)],
        "angle": round(angle, 2),
        "points": local_points,
        "curvature": round(curvature, 4),
        "color": [round(float(channel), 1) for channel in mean_color],
        "width": round(float(np.clip(radius_y * 0.32, 2.0, 22.0)), 2),
        "alpha": round(float(np.clip(mean_residual / 55.0, 0.18, 0.82)), 3),
        "area": round(area * 4.0, 1),
        "residual": round(mean_residual, 2),
        "primitive": primitive,
        "elongation": round(float(elongation), 3),
    }


def detect_keyframe(frame: np.ndarray, base: np.ndarray, lock: np.ndarray) -> list[dict]:
    height, width = frame.shape[:2]
    half = (width // 2, height // 2)
    small_frame = cv2.resize(frame, half, interpolation=cv2.INTER_AREA)
    small_base = cv2.resize(base, half, interpolation=cv2.INTER_AREA)
    small_lock = cv2.resize(lock, half, interpolation=cv2.INTER_NEAREST)
    difference = cv2.absdiff(small_frame, small_base)
    diff_gray = cv2.cvtColor(difference, cv2.COLOR_RGB2GRAY)
    diff_gray = cv2.GaussianBlur(diff_gray, (5, 5), 0)
    changed = (diff_gray > 7).astype(np.uint8) * 255
    unlocked = (small_lock < 128).astype(np.uint8) * 255
    mask = cv2.bitwise_and(changed, unlocked)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for contour in contours:
        result = contour_detection(small_frame, small_base, small_lock, contour, diff_gray)
        if result is not None:
            detections.append(result)
    detections.sort(key=lambda item: item["area"] * (0.5 + item["residual"] / 40.0), reverse=True)
    return detections[:12]


def detection_cost(previous: dict, current: dict) -> float:
    center_distance = math.dist(previous["center"], current["center"])
    area_ratio = abs(math.log(max(previous["area"], 1.0) / max(current["area"], 1.0)))
    color_distance = math.dist(previous["color"], current["color"]) / 70.0
    angle_gap = abs(((current["angle"] - previous["angle"] + 90.0) % 180.0) - 90.0) / 80.0
    shape_gap = abs(math.log(max(previous["elongation"], 0.1) / max(current["elongation"], 0.1))) * 0.35
    return center_distance / 260.0 + area_ratio * 0.75 + color_distance + angle_gap + shape_gap


def optical_flow(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    height, width = previous.shape[:2]
    half = (width // 2, height // 2)
    previous_gray = cv2.cvtColor(cv2.resize(previous, half), cv2.COLOR_RGB2GRAY)
    current_gray = cv2.cvtColor(cv2.resize(current, half), cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        previous_gray, current_gray, None, 0.5, 3, 21, 3, 5, 1.2, 0
    )
    return flow * 2.0


def sample_flow_for_detection(flow: np.ndarray, detection: dict) -> tuple[float, float, float]:
    radians = math.radians(detection["angle"])
    direction = (math.cos(radians), math.sin(radians))
    normal = (-direction[1], direction[0])
    map_x = []
    map_y = []
    for local_x, local_y in detection["points"]:
        offset_x = (local_x * detection["radii"][0] * direction[0] + local_y * detection["radii"][1] * normal[0]) / 2.0
        offset_y = (local_x * detection["radii"][0] * direction[1] + local_y * detection["radii"][1] * normal[1]) / 2.0
        map_x.append(detection["center"][0] / 2.0 + offset_x)
        map_y.append(detection["center"][1] / 2.0 + offset_y)
    coordinates = np.array([map_x, map_y], dtype=np.float32).reshape(2, 1, -1)
    dx = cv2.remap(flow[:, :, 0], coordinates[0], coordinates[1], cv2.INTER_LINEAR).ravel()
    dy = cv2.remap(flow[:, :, 1], coordinates[0], coordinates[1], cv2.INTER_LINEAR).ravel()
    mean_x, mean_y = float(dx.mean()), float(dy.mean())
    deformation = float(np.sqrt(((dx - mean_x) ** 2 + (dy - mean_y) ** 2)).mean())
    deformation /= max(min(detection["radii"]), 8.0)
    return mean_x, mean_y, deformation


def build_tracks(
    stack: np.ndarray,
    base: np.ndarray,
    lock: np.ndarray,
    keyframes: list[int],
    fps: float,
) -> tuple[list[dict], list[dict]]:
    detections_by_frame: dict[int, list[dict]] = {}
    for frame_index in keyframes:
        detections_by_frame[frame_index] = detect_keyframe(stack[frame_index], base, lock)

    tracks: dict[int, dict] = {}
    previous: list[tuple[int, dict]] = []
    next_track_id = 1
    samples: list[dict] = []

    for position, frame_index in enumerate(keyframes):
        current_detections = detections_by_frame[frame_index]
        current_pairs: list[tuple[int, dict]] = []
        if position == 0 or not previous:
            current_pairs = [(next_track_id + offset, detection) for offset, detection in enumerate(current_detections)]
            next_track_id += len(current_detections)
            flow = np.zeros((stack.shape[1] // 2, stack.shape[2] // 2, 2), dtype=np.float32)
        else:
            previous_frame = keyframes[position - 1]
            flow = optical_flow(stack[previous_frame], stack[frame_index])
            costs = np.array(
                [[detection_cost(previous_detection, current_detection) for current_detection in current_detections]
                 for _, previous_detection in previous],
                dtype=np.float32,
            )
            if costs.size:
                rows, columns = linear_sum_assignment(costs)
                claimed: set[int] = set()
                for row, column in zip(rows, columns):
                    if costs[row, column] > 1.75:
                        continue
                    track_id = previous[row][0]
                    current_pairs.append((track_id, current_detections[column]))
                    claimed.add(int(column))
                for column, detection in enumerate(current_detections):
                    if column not in claimed:
                        current_pairs.append((next_track_id, detection))
                        next_track_id += 1
            else:
                current_pairs = [(next_track_id + offset, detection) for offset, detection in enumerate(current_detections)]
                next_track_id += len(current_detections)

        frame_motions: list[dict] = []
        for track_id, detection in current_pairs:
            track = tracks.setdefault(
                track_id,
                {
                    "id": f"S{track_id:03d}",
                    "primitive": detection["primitive"],
                    "birth": frame_index,
                    "death": frame_index,
                    "states": [],
                    "totalDisplacement": 0.0,
                    "meanDeformation": 0.0,
                },
            )
            flow_dx, flow_dy, flow_deformation = sample_flow_for_detection(flow, detection)
            previous_state = track["states"][-1] if track["states"] else None
            if previous_state:
                dx = detection["center"][0] - previous_state["center"][0]
                dy = detection["center"][1] - previous_state["center"][1]
                displacement = math.hypot(dx, dy)
                scale_delta = abs(math.log(max(detection["radii"][0], 1) / max(previous_state["radii"][0], 1)))
                scale_delta += abs(math.log(max(detection["radii"][1], 1) / max(previous_state["radii"][1], 1)))
                rotation_delta = abs(((detection["angle"] - previous_state["angle"] + 90) % 180) - 90) / 90.0
                deformation = float(np.clip(flow_deformation + scale_delta * 0.4 + rotation_delta * 0.3, 0.0, 4.0))
            else:
                dx = dy = displacement = deformation = 0.0

            state = {
                "frame": frame_index,
                "center": detection["center"],
                "radii": detection["radii"],
                "angle": detection["angle"],
                "points": detection["points"],
                "curvature": detection["curvature"],
                "color": detection["color"],
                "width": detection["width"],
                "alpha": detection["alpha"],
                "area": detection["area"],
                "primitive": detection["primitive"],
                "motion": {
                    "dx": round(dx, 2),
                    "dy": round(dy, 2),
                    "displacement": round(displacement, 2),
                    "flowDx": round(flow_dx, 2),
                    "flowDy": round(flow_dy, 2),
                    "deformation": round(deformation, 4),
                },
            }
            track["states"].append(state)
            track["death"] = frame_index
            track["primitive"] = detection["primitive"]
            track["totalDisplacement"] += displacement
            frame_motions.append(state["motion"])

        if frame_motions:
            samples.append(
                {
                    "frame": frame_index,
                    "time": round(frame_index / fps, 5),
                    "activeStrokes": len(frame_motions),
                    "meanDisplacement": round(float(np.mean([item["displacement"] for item in frame_motions])), 3),
                    "maxDisplacement": round(float(np.max([item["displacement"] for item in frame_motions])), 3),
                    "meanDeformation": round(float(np.mean([item["deformation"] for item in frame_motions])), 4),
                    "maxDeformation": round(float(np.max([item["deformation"] for item in frame_motions])), 4),
                }
            )
        previous = current_pairs

    ordered = []
    for track in tracks.values():
        count = len(track["states"])
        track["totalDisplacement"] = round(track["totalDisplacement"], 2)
        track["meanDeformation"] = round(
            float(np.mean([state["motion"]["deformation"] for state in track["states"][1:]])), 4
        ) if count > 1 else 0.0
        track["length"] = count
        if count >= 2 or track["states"][0]["area"] > 1200:
            ordered.append(track)
    ordered.sort(key=lambda item: (-item["length"], -item["states"][0]["area"]))
    return ordered, samples


def write_motion_svg(samples: list[dict], path: Path, last_frame: int) -> None:
    width, height = 960, 280
    left, top, plot_w, plot_h = 60, 24, 860, 210
    max_displacement = max(max(item["maxDisplacement"] for item in samples), 1.0)
    max_deformation = max(max(item["maxDeformation"] for item in samples), 0.1)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#10131c"/>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#171c28" stroke="#343b4c"/>',
    ]
    for tick in range(5):
        y = top + plot_h * tick / 4
        lines.append(f'<line x1="{left}" y1="{y}" x2="{left + plot_w}" y2="{y}" stroke="#2b3241"/>')
    for name, values, maximum, color in [
        ("displacement", [item["meanDisplacement"] for item in samples], max_displacement, "#69e2ff"),
        ("deformation", [item["meanDeformation"] for item in samples], max_deformation, "#ff9bd2"),
    ]:
        points = []
        for index, value in enumerate(values):
            x = left + plot_w * index / max(len(values) - 1, 1)
            y = top + plot_h * (1.0 - min(value / maximum, 1.0))
            points.append(f"{x:.1f},{y:.1f}")
        lines.append(f'<polyline data-series="{name}" points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>')
    lines.extend([
        f'<text x="{left}" y="16" fill="#69e2ff" font-size="12">displacement / deformation by reference keyframe</text>',
        f'<text x="{left}" y="{height - 12}" fill="#8d96a9" font-size="12">frame 0</text>',
        f'<text x="{left + plot_w - 50}" y="{height - 12}" fill="#8d96a9" font-size="12">frame {last_frame}</text>',
        '</svg>',
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_contact_sheet(stack: np.ndarray, fps: float, path: Path) -> None:
    indices = [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, len(stack) - 1]
    tile_w, tile_h, caption_h = 376, 176, 26
    sheet = Image.new("RGB", (tile_w * 3, (tile_h + caption_h) * 4), (15, 18, 24))
    draw = ImageDraw.Draw(sheet)
    for position, frame_index in enumerate(indices):
        thumb = Image.fromarray(stack[frame_index]).resize((tile_w, tile_h))
        x = (position % 3) * tile_w
        y = (position // 3) * (tile_h + caption_h)
        draw.text((x + 8, y + 6), f"frame {frame_index:03d} / {frame_index / fps:.2f}s", fill=(238, 243, 252))
        sheet.paste(thumb, (x, y + caption_h))
    sheet.save(path, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    args = parser.parse_args()
    args.assets.mkdir(parents=True, exist_ok=True)
    args.analysis.mkdir(parents=True, exist_ok=True)

    stack, fps, frame_count = read_video(args.video)
    height, width = stack.shape[1:3]
    base, base_rgba, lock, partition = temporal_partition(stack)
    static_fraction = float((lock > 127).mean())
    print(f"video={width}x{height} frames={frame_count} fps={fps}")
    print(f"locked static fraction={static_fraction:.4f}")

    Image.fromarray(base_rgba, mode="RGBA").save(args.assets / "static-base.png", optimize=True)
    Image.fromarray(lock, mode="L").save(args.assets / "static-lock.png", optimize=True)
    Image.fromarray(stack[0]).save(args.analysis / "poster.jpg", quality=92)
    Image.fromarray(lock, mode="L").save(args.analysis / "static-lock-preview.png")
    Image.fromarray(base).save(args.analysis / "median.png")
    write_contact_sheet(stack, fps, args.analysis / "reference-contact.jpg")

    keyframes = list(range(0, frame_count, KEYFRAME_STRIDE))
    tracks, samples = build_tracks(stack, base, lock, keyframes, fps)
    state_count = sum(len(track["states"]) for track in tracks)
    print(f"tracks={len(tracks)} keyframe states={state_count} keyframes={len(samples)}")

    frame_commands = build_compact_commands(
        stack,
        base,
        lock,
        partition,
        args.assets / "frame-commands.bin",
        args.assets / "quality-report.json",
        args.analysis / "reconstruction-contact.jpg",
        args.assets / "static-partition.json",
        fps,
    )
    print(
        f"frame commands={frame_commands['frameCount']} format={frame_commands['format']} "
        f"commands={frame_commands['commands']} compressed={frame_commands['compressedBytes']} bytes"
    )

    instructions = {
        "schema": "reference-driven-stroke-timeline/v3",
        "source": {
            "width": width,
            "height": height,
            "fps": fps,
            "frameCount": frame_count,
            "duration": round(frame_count / fps, 6),
            "runtimeSource": frame_commands["file"],
        },
        "base": {
            "color": "assets/static-base.png",
            "lockMask": "assets/static-lock.png",
            "lockedPixelFraction": round(static_fraction, 6),
            "rule": partition["rule"],
            "mode": "RGBA; transparent pixels are excluded from the immutable WebGL base",
            "staticPartition": "assets/static-partition.json",
        },
        "coordinateSystem": "original video pixels, top-left origin",
        "keyframeStride": KEYFRAME_STRIDE,
        "primitives": ["ellipse", "line", "bezier", "brush"],
        "referenceDriven": {
            "position": "residual centroid within each stroke search region",
            "shape": "reference covariance drives radii and rotation",
            "color": "reference-frame mean color in residual region",
            "detail": "offline frames are serialized as 2x2 RGB555 primitive commands and sparse per-pixel repaints; runtime stroke geometry is corrected against the decoded dynamic canvas",
        },
        "tracks": tracks,
        "frameCommands": frame_commands,
        "staticPartition": "assets/static-partition.json",
    }
    (args.assets / "drawing-instructions.json").write_text(
        json.dumps(instructions, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )

    report = {
        "fps": fps,
        "frameCount": frame_count,
        "keyframeStride": KEYFRAME_STRIDE,
        "tracks": len(tracks),
        "states": state_count,
        "primitiveCommandsAtRuntime": state_count * 4,
        "frameCommands": frame_commands,
        "samples": samples,
        "trackSummary": [
            {
                "id": track["id"],
                "primitive": track["primitive"],
                "frames": [track["birth"], track["death"]],
                "length": track["length"],
                "totalDisplacement": track["totalDisplacement"],
                "meanDeformation": track["meanDeformation"],
            }
            for track in tracks[:20]
        ],
    }
    (args.assets / "motion-report.json").write_text(
        json.dumps(report, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    write_motion_svg(samples, args.analysis / "motion-chart.svg", keyframes[-1])
    print(f"assets written to {args.assets}")


if __name__ == "__main__":
    main()
