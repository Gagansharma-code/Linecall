"""Extract still frames from amateur match video for later manual annotation.

Reads AMATEUR_FOOTAGE_DIR from the environment (same pattern as
ROBOFLOW_API_KEY) and writes JPEGs to a sibling candidate_frames directory
outside the repo. No annotation, upload, or training happens here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

VIDEO_SUFFIXES = {".mp4", ".mov"}
DEFAULT_INTERVAL_SECONDS = 2.0
DEFAULT_MAX_FRAMES = 200
# Laplacian-variance below this is treated as blurry. Starting value in the
# commonly cited 100–150 range; not yet validated against this project's footage
# and may need adjusting once someone looks at what gets rejected.
DEFAULT_BLUR_THRESHOLD = 100.0


class MissingFootageDirError(Exception):
    """Raised when AMATEUR_FOOTAGE_DIR is unset or not a directory."""


class VideoOpenError(Exception):
    """Raised when a video file cannot be opened or has unreadable metadata."""


class FrameRecord(TypedDict):
    source: str
    frame_index: int
    timestamp_seconds: float
    output_filename: str


class VideoStats(TypedDict):
    source: str
    sampled: int
    kept: int
    skipped_blurry: int


def require_footage_dir() -> Path:
    raw = os.environ.get("AMATEUR_FOOTAGE_DIR", "").strip()
    if not raw:
        raise MissingFootageDirError(
            "AMATEUR_FOOTAGE_DIR is not set. Copy .env.example to .env and set "
            "AMATEUR_FOOTAGE_DIR to the local footage directory."
        )
    path = Path(raw)
    if not path.is_dir():
        raise MissingFootageDirError(
            "AMATEUR_FOOTAGE_DIR does not exist or is not a directory. Copy "
            ".env.example to .env and set AMATEUR_FOOTAGE_DIR to a valid path."
        )
    return path


def list_video_files(footage_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in footage_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def iter_candidate_frame_indices(
    fps: float, frame_count: int, interval_seconds: float, max_frames: int
) -> list[int]:
    """Return wall-clock-spaced frame indices, capped at max_frames.

    Step is derived from the clip's actual fps so 2.0 seconds is 2.0 seconds
    whether the phone recorded 24, 30, or 60 fps. Does not pad or repeat.
    """
    if fps <= 0 or frame_count <= 0 or interval_seconds <= 0 or max_frames <= 0:
        return []
    duration_seconds = frame_count / fps
    indices: list[int] = []
    seen: set[int] = set()
    t = 0.0
    while t < duration_seconds and len(indices) < max_frames:
        idx = int(round(t * fps))
        if idx >= frame_count:
            break
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
        t += interval_seconds
    return indices


def is_blurry(frame: np.ndarray, threshold: float) -> bool:
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return variance < threshold


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    interval_seconds: float,
    blur_threshold: float,
    max_frames: int,
) -> tuple[list[FrameRecord], VideoStats]:
    """Seek-sample a video, skip blurry frames, write kept JPEGs.

    Blurry skips do not count against max_frames: sampling continues along the
    wall-clock grid until max_frames usable frames are kept or the clip ends.
    The second return value is per-video counts for the combined manifest.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoOpenError(f"Failed to open video: {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            raise VideoOpenError(f"Failed to read fps/frame count from video: {video_path}")

        # Full grid, then stop once we have max_frames usable (not merely sampled).
        candidates = iter_candidate_frame_indices(
            fps, frame_count, interval_seconds, max_frames=frame_count
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        records: list[FrameRecord] = []
        sampled = 0
        skipped_blurry = 0
        stem = video_path.stem

        for idx in candidates:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            sampled += 1
            if is_blurry(frame, blur_threshold):
                skipped_blurry += 1
                continue
            filename = f"{stem}_{idx:06d}.jpg"
            written = cv2.imwrite(str(output_dir / filename), frame)
            if not written:
                raise VideoOpenError(f"Failed to write frame {idx} from video: {video_path}")
            records.append(
                {
                    "source": video_path.name,
                    "frame_index": idx,
                    "timestamp_seconds": idx / fps,
                    "output_filename": filename,
                }
            )
            if len(records) >= max_frames:
                break
    finally:
        cap.release()

    stats: VideoStats = {
        "source": video_path.name,
        "sampled": sampled,
        "kept": len(records),
        "skipped_blurry": skipped_blurry,
    }
    return records, stats


def write_manifest(
    records: list[FrameRecord],
    output_dir: Path,
    video_stats: list[VideoStats],
) -> Path:
    """Write candidate_frames_manifest.json (per-frame rows + per-video counts)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "summary": {
            "videos": video_stats,
            "totals": {
                "videos": len(video_stats),
                "sampled": sum(item["sampled"] for item in video_stats),
                "kept": sum(item["kept"] for item in video_stats),
                "skipped_blurry": sum(item["skipped_blurry"] for item in video_stats),
            },
        },
        "frames": records,
    }
    path = output_dir / "candidate_frames_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract candidate stills from amateur tennis footage."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: sibling candidate_frames next to footage).",
    )
    args = parser.parse_args(argv)

    load_dotenv(REPO_ROOT / ".env")
    try:
        footage_dir = require_footage_dir()
    except MissingFootageDirError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    output_dir = args.output if args.output is not None else footage_dir.parent / "candidate_frames"
    videos = list_video_files(footage_dir)
    all_records: list[FrameRecord] = []
    all_stats: list[VideoStats] = []

    try:
        for video_path in videos:
            records, stats = extract_frames_from_video(
                video_path,
                output_dir,
                interval_seconds=DEFAULT_INTERVAL_SECONDS,
                blur_threshold=DEFAULT_BLUR_THRESHOLD,
                max_frames=DEFAULT_MAX_FRAMES,
            )
            all_records.extend(records)
            all_stats.append(stats)
            print(
                f"{stats['source']}: sampled={stats['sampled']} "
                f"kept={stats['kept']} skipped_blurry={stats['skipped_blurry']}"
            )
    except VideoOpenError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    write_manifest(all_records, output_dir, all_stats)
    total_kept = sum(item["kept"] for item in all_stats)
    total_skipped = sum(item["skipped_blurry"] for item in all_stats)
    print(
        f"{len(all_stats)} videos processed, {total_kept} frames kept, "
        f"{total_skipped} skipped as blurry"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
