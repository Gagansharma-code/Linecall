"""Tests for scripts/replay_harness.py. No real footage or trained weights."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

import replay_harness as replay
from extract_amateur_frames import VideoOpenError


def _checkerboard(size: int = 48, cell: int = 4) -> np.ndarray:
    rows, cols = np.indices((size, size))
    tiles = ((rows // cell) + (cols // cell)) % 2
    gray = (tiles * 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter.fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened(), f"VideoWriter failed to open {path}"
    for frame in frames:
        writer.write(frame)
    writer.release()


class _FakeBoxes:
    def __init__(self, xyxy: list[float], confidence: float) -> None:
        self.xyxy = [xyxy]
        self.conf = [confidence]

    def __len__(self) -> int:
        return 1


class _FakeResult:
    def __init__(self, xyxy: list[float], confidence: float) -> None:
        self.boxes = _FakeBoxes(xyxy, confidence)


def _fake_model(xyxy: list[float], confidence: float) -> MagicMock:
    model = MagicMock()
    model.predict.return_value = [_FakeResult(xyxy, confidence)]
    return model


def test_iter_video_frames_indices_timestamps_and_stride(tmp_path: Path) -> None:
    fps = 10.0
    frames = [_checkerboard() for _ in range(10)]
    video_path = tmp_path / "clip.avi"
    _write_video(video_path, frames, fps)

    all_frames = list(replay.iter_video_frames(video_path, stride=1))
    assert [item[0] for item in all_frames] == list(range(len(all_frames)))
    assert all(item[1] == item[0] / fps for item in all_frames)
    assert all(item[2].shape[2] == 3 for item in all_frames)
    assert len(all_frames) == 10

    strided = list(replay.iter_video_frames(video_path, stride=2))
    assert [item[0] for item in strided] == [0, 2, 4, 6, 8]
    assert [item[1] for item in strided] == [0.0, 0.2, 0.4, 0.6, 0.8]


def test_iter_video_frames_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(VideoOpenError, match="Failed to open video"):
        list(replay.iter_video_frames(tmp_path / "missing.avi"))


def test_run_replay_writes_nonempty_rrd_with_mocked_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fps = 10.0
    frames = [_checkerboard() for _ in range(6)]
    video_path = tmp_path / "clip.avi"
    _write_video(video_path, frames, fps)
    output_rrd = tmp_path / "out.rrd"
    fake = _fake_model([10.0, 12.0, 30.0, 40.0], 0.87)
    monkeypatch.setattr(replay, "YOLO", lambda _path: fake)

    summary = replay.run_replay(
        video_path=video_path,
        model_path=tmp_path / "unused.pt",
        output_rrd=output_rrd,
        stride=2,
        max_frames=None,
    )

    assert summary["frame_count"] == 3
    assert summary["frames_with_detections"] == 3
    assert summary["mean_confidence"] == pytest.approx(0.87)
    assert output_rrd.is_file()
    assert output_rrd.stat().st_size > 0
    assert fake.predict.call_count == 3
