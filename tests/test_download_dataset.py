"""Offline tests for scripts/download_dataset.py. No Roboflow network calls."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

import download_dataset


def _write_config(path: Path, **overrides: object) -> dict:
    config = {
        "workspace": "viren-dhanwani",
        "project": "tennis-ball-detection",
        "version": 6,
        "format": "yolov8",
        **overrides,
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return {
        "workspace": str(config["workspace"]),
        "project": str(config["project"]),
        "version": int(config["version"]),
        "format": str(config["format"]),
    }


def _fake_download(model_format: str, location: str, overwrite: bool = False) -> None:
    dest = Path(location)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "data.yaml").write_text("names: [tennis-ball]\n", encoding="utf-8")
    images = dest / "train" / "images"
    images.mkdir(parents=True, exist_ok=True)
    (images / "frame_001.jpg").write_bytes(b"fake-jpeg")


def test_missing_api_key_raises_catchable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    with pytest.raises(download_dataset.MissingAPIKeyError) as exc_info:
        download_dataset.require_api_key()
    message = str(exc_info.value)
    assert "ROBOFLOW_API_KEY" in message
    assert ".env.example" in message


def test_cli_missing_api_key_prints_one_line_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    monkeypatch.setattr(download_dataset, "load_dotenv", lambda *args, **kwargs: False)

    returncode = download_dataset.main([])

    assert returncode == 1
    captured = capsys.readouterr()
    err = captured.err.strip()
    assert captured.out == ""
    assert err.count("\n") == 0
    assert "ROBOFLOW_API_KEY" in err
    assert ".env.example" in err
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out


@patch("download_dataset.Roboflow")
def test_writes_manifest_with_expected_fields(
    mock_roboflow_cls: MagicMock, tmp_path: Path
) -> None:
    mock_roboflow_cls.return_value.workspace.return_value.project.return_value.version.return_value.download.side_effect = _fake_download

    config = _write_config(tmp_path / "dataset.yaml")
    dest = download_dataset.destination_dir(tmp_path / "raw", config["project"], config["version"])

    result = download_dataset.download_dataset(config, dest, api_key="test-key")

    assert result == "downloaded"
    manifest_path = dest / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["workspace"] == "viren-dhanwani"
    assert manifest["project"] == "tennis-ball-detection"
    assert manifest["version"] == 6
    assert manifest["format"] == "yolov8"
    assert manifest["downloaded_at"].endswith("Z")
    assert len(manifest["sha256"]) == 64
    assert manifest["sha256"] == download_dataset.compute_fingerprint(dest)
    mock_roboflow_cls.assert_called_once_with(api_key="test-key")


@patch("download_dataset.Roboflow")
def test_redownloads_when_manifest_format_differs(
    mock_roboflow_cls: MagicMock, tmp_path: Path
) -> None:
    """A manifest matching on workspace/project/version but not format must
    not be treated as a match — the format is part of what was asked for."""
    mock_roboflow_cls.return_value.workspace.return_value.project.return_value.version.return_value.download.side_effect = _fake_download

    config = _write_config(tmp_path / "dataset.yaml")
    dest = download_dataset.destination_dir(tmp_path / "raw", config["project"], config["version"])
    dest.mkdir(parents=True)
    stale_config = {**config, "format": "coco"}
    download_dataset.write_manifest(dest, stale_config)

    result = download_dataset.download_dataset(config, dest, api_key="test-key")

    assert result == "downloaded"
    mock_roboflow_cls.assert_called_once_with(api_key="test-key")
    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "yolov8"


@patch("download_dataset.Roboflow")
def test_skips_download_when_matching_manifest_exists(
    mock_roboflow_cls: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _write_config(tmp_path / "dataset.yaml")
    dest = download_dataset.destination_dir(tmp_path / "raw", config["project"], config["version"])
    dest.mkdir(parents=True)
    (dest / "already_there.txt").write_text("keep me", encoding="utf-8")
    download_dataset.write_manifest(dest, config)

    result = download_dataset.download_dataset(config, dest, api_key="test-key")

    assert result == "skipped"
    mock_roboflow_cls.assert_not_called()
    output = capsys.readouterr().out
    assert "already present" in output.lower()
    assert "Skipping download" in output
    assert (dest / "already_there.txt").read_text(encoding="utf-8") == "keep me"
