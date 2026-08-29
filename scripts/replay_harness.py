"""Replay recorded footage through the detector and log frames to a Rerun .rrd file."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rerun as rr
from ultralytics import YOLO  # type: ignore[attr-defined]

from check_ncnn_parity import detections_from_result
from download_dataset import REPO_ROOT
from export_ncnn import find_latest_checkpoint, ncnn_dir_for_checkpoint
from extract_amateur_frames import VideoOpenError

DEFAULT_REPLAYS_DIR = REPO_ROOT / "replays"
APPLICATION_ID = "linecall-replay"


def iter_video_frames(video_path: Path, stride: int = 1) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, timestamp_seconds, frame_bgr) every stride-th frame."""
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoOpenError(f"Failed to open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            raise VideoOpenError(f"Failed to read fps from video: {video_path}")
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_index % stride == 0:
                yield frame_index, frame_index / fps, frame
            frame_index += 1
    finally:
        cap.release()


def detect_frame(model: Any, frame: np.ndarray) -> list[dict[str, Any]]:
    """Run the model on one BGR frame; returns A5 detection dicts."""
    results = list(model.predict(source=frame, verbose=False, device="cpu"))
    if not results:
        return []
    return detections_from_result(results[0])


def log_frame(
    frame_index: int,
    timestamp: float,
    frame_bgr: np.ndarray,
    detections: list[dict[str, Any]],
) -> None:
    """Log image, boxes, and confidences on Rerun timelines for this frame."""
    rr.set_time("frame", sequence=frame_index)
    rr.set_time("time", duration=timestamp)
    # OpenCV is BGR; Rerun Image defaults to RGB for 3-channel arrays.
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rr.log("video", rr.Image(frame_rgb))

    xyxy = [list(det["xyxy"]) for det in detections]
    labels = [f"{float(det['confidence']):.2f}" for det in detections]
    confidences = [float(det["confidence"]) for det in detections]
    rr.log(
        "detections",
        rr.Boxes2D(array=xyxy, array_format=rr.Box2DFormat.XYXY, labels=labels),
    )
    rr.log("detections/confidence", rr.Scalars(confidences))


def default_model_path() -> Path:
    checkpoint = find_latest_checkpoint()
    ncnn_dir = ncnn_dir_for_checkpoint(checkpoint)
    if ncnn_dir.exists():
        return ncnn_dir
    return checkpoint


def default_output_path(video_path: Path) -> Path:
    return DEFAULT_REPLAYS_DIR / f"{video_path.stem}.rrd"


def run_replay(
    video_path: Path,
    model_path: Path,
    output_rrd: Path,
    stride: int = 1,
    max_frames: int | None = None,
) -> dict[str, Any]:
    output_rrd.parent.mkdir(parents=True, exist_ok=True)
    rr.init(APPLICATION_ID, spawn=False)
    rr.save(str(output_rrd))
    model = YOLO(str(model_path))
    processed = 0
    frames_with_detections = 0
    confidences: list[float] = []
    started = time.perf_counter()
    try:
        for frame_index, timestamp, frame in iter_video_frames(video_path, stride=stride):
            if max_frames is not None and processed >= max_frames:
                break
            detections = detect_frame(model, frame)
            log_frame(frame_index, timestamp, frame, detections)
            processed += 1
            if detections:
                frames_with_detections += 1
                confidences.extend(float(det["confidence"]) for det in detections)
    finally:
        rr.disconnect()
    elapsed = time.perf_counter() - started
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return {
        "frame_count": processed,
        "frames_with_detections": frames_with_detections,
        "mean_confidence": mean_confidence,
        "wall_clock_seconds": round(elapsed, 2),
        "video": str(video_path),
        "model": str(model_path),
        "output": str(output_rrd),
        "stride": stride,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a video through the tennis-ball detector and save a Rerun .rrd."
    )
    parser.add_argument("--video", type=Path, required=True, help="Path to a recorded clip.")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="YOLO .pt or NCNN directory (default: latest NCNN export from A5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination .rrd path (default: replays/<video-stem>.rrd).",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many logged frames. Omit to process the whole clip.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    video_path = args.video if args.video.is_absolute() else REPO_ROOT / args.video
    try:
        model_path = args.model.resolve() if args.model else default_model_path()
        output_rrd = args.output.resolve() if args.output else default_output_path(video_path)
        summary = run_replay(
            video_path=video_path,
            model_path=model_path,
            output_rrd=output_rrd,
            stride=args.stride,
            max_frames=args.max_frames,
        )
    except (VideoOpenError, FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
