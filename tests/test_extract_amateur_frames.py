"""Tests for scripts/extract_amateur_frames.py. No real footage required."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import extract_amateur_frames as extract


def _checkerboard(size: int = 48, cell: int = 4) -> np.ndarray:
    rows, cols = np.indices((size, size))
    tiles = ((rows // cell) + (cols // cell)) % 2
    gray = (tiles * 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _flat(size: int = 48, value: int = 128) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def _write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter.fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened(), f"VideoWriter failed to open {path}"
    for frame in frames:
        writer.write(frame)
    writer.release()


def test_iter_candidate_indices_uses_wall_clock_not_fixed_stride() -> None:
    thirty = extract.iter_candidate_frame_indices(
        fps=30.0, frame_count=300, interval_seconds=2.0, max_frames=200
    )
    sixty = extract.iter_candidate_frame_indices(
        fps=60.0, frame_count=600, interval_seconds=2.0, max_frames=200
    )
    assert thirty == [0, 60, 120, 180, 240]
    assert sixty == [0, 120, 240, 360, 480]


def test_iter_candidate_indices_caps_and_does_not_pad() -> None:
    capped = extract.iter_candidate_frame_indices(
        fps=30.0, frame_count=30_000, interval_seconds=2.0, max_frames=5
    )
    assert capped == [0, 60, 120, 180, 240]
    short = extract.iter_candidate_frame_indices(
        fps=30.0, frame_count=90, interval_seconds=1.0, max_frames=200
    )
    assert short == [0, 30, 60]
    assert (
        extract.iter_candidate_frame_indices(
            fps=0.0, frame_count=100, interval_seconds=2.0, max_frames=10
        )
        == []
    )


def test_is_blurry_distinguishes_sharp_checkerboard_from_flat() -> None:
    threshold = extract.DEFAULT_BLUR_THRESHOLD
    assert extract.is_blurry(_checkerboard(), threshold) is False
    assert extract.is_blurry(_flat(), threshold) is True


def test_extract_skips_blurry_and_keeps_sampling_to_fill_max_frames(
    tmp_path: Path,
) -> None:
    fps = 10.0
    frames = [_flat() if i < 40 else _checkerboard() for i in range(80)]
    video_path = tmp_path / "clip.avi"
    _write_video(video_path, frames, fps)
    output_dir = tmp_path / "out"

    records, stats = extract.extract_frames_from_video(
        video_path,
        output_dir,
        interval_seconds=1.0,
        blur_threshold=extract.DEFAULT_BLUR_THRESHOLD,
        max_frames=3,
    )

    assert stats["skipped_blurry"] >= 1
    assert stats["kept"] == 3
    assert [row["frame_index"] for row in records] == [40, 50, 60]
    for row in records:
        assert (output_dir / row["output_filename"]).is_file()
        assert row["source"] == "clip.avi"
        assert row["timestamp_seconds"] == row["frame_index"] / fps


def test_extract_raises_on_unreadable_video(tmp_path: Path) -> None:
    fake = tmp_path / "broken.mp4"
    fake.write_bytes(b"not a video")
    with pytest.raises(extract.VideoOpenError) as exc_info:
        extract.extract_frames_from_video(
            fake,
            tmp_path / "out",
            interval_seconds=1.0,
            blur_threshold=extract.DEFAULT_BLUR_THRESHOLD,
            max_frames=5,
        )
    assert "broken.mp4" in str(exc_info.value)


def test_write_manifest_includes_frames_and_per_video_summary(tmp_path: Path) -> None:
    records: list[extract.FrameRecord] = [
        {
            "source": "a.mp4",
            "frame_index": 0,
            "timestamp_seconds": 0.0,
            "output_filename": "a_000000.jpg",
        }
    ]
    stats: list[extract.VideoStats] = [
        {"source": "a.mp4", "sampled": 4, "kept": 1, "skipped_blurry": 3}
    ]
    path = extract.write_manifest(records, tmp_path, stats)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["frames"] == records
    assert payload["summary"]["videos"] == stats
    assert payload["summary"]["totals"]["kept"] == 1
    assert payload["summary"]["totals"]["skipped_blurry"] == 3


def test_missing_footage_dir_raises_catchable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AMATEUR_FOOTAGE_DIR", raising=False)
    with pytest.raises(extract.MissingFootageDirError) as exc_info:
        extract.require_footage_dir()
    message = str(exc_info.value)
    assert "AMATEUR_FOOTAGE_DIR" in message
    assert ".env.example" in message


def test_cli_missing_footage_dir_prints_one_line_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("AMATEUR_FOOTAGE_DIR", raising=False)
    monkeypatch.setattr(extract, "load_dotenv", lambda *args, **kwargs: False)

    returncode = extract.main([])

    assert returncode == 1
    captured = capsys.readouterr()
    err = captured.err.strip()
    assert captured.out == ""
    assert err.count("\n") == 0
    assert "AMATEUR_FOOTAGE_DIR" in err
    assert ".env.example" in err
    assert "Traceback" not in captured.err


def test_cli_missing_directory_prints_one_line_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("AMATEUR_FOOTAGE_DIR", str(tmp_path / "does-not-exist"))
    monkeypatch.setattr(extract, "load_dotenv", lambda *args, **kwargs: False)

    returncode = extract.main([])

    assert returncode == 1
    err = capsys.readouterr().err.strip()
    assert err.count("\n") == 0
    assert "AMATEUR_FOOTAGE_DIR" in err
    assert ".env.example" in err


def test_main_processes_synthetic_videos_via_output_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    footage = tmp_path / "amateur_footage_raw"
    footage.mkdir()
    frames = [_checkerboard() for _ in range(30)]
    _write_video(footage / "match.mp4", frames, fps=10.0)
    output_dir = tmp_path / "candidate_frames"
    monkeypatch.setenv("AMATEUR_FOOTAGE_DIR", str(footage))
    monkeypatch.setattr(extract, "load_dotenv", lambda *args, **kwargs: False)

    returncode = extract.main(["--output", str(output_dir)])

    assert returncode == 0
    out = capsys.readouterr().out
    assert "1 videos processed" in out
    assert "skipped as blurry" in out
    manifest = json.loads(
        (output_dir / "candidate_frames_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["summary"]["totals"]["videos"] == 1
    assert manifest["summary"]["totals"]["kept"] >= 1
    assert list(output_dir.glob("*.jpg"))
