"""Unit tests for scripts/benchmark.py. No real dataset or model required."""

from __future__ import annotations

from pathlib import Path

import yaml

import benchmark as bench


def test_filter_image_list_splits_by_filename_prefix(tmp_path: Path) -> None:
    listing = tmp_path / "test.txt"
    listing.write_text(
        "\n".join(
            [
                str(tmp_path / "clay12_jpg.rf.abc.jpg"),
                str(tmp_path / "synframe99_jpg.rf.def.jpg"),
                str(tmp_path / "20260613_172544_000001.jpg"),
                str(tmp_path / "June15a_000240.jpg"),
                str(tmp_path / "mystery.jpg"),
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    broadcast = bench.filter_image_list(listing, "broadcast")
    amateur = bench.filter_image_list(listing, "amateur")
    other = bench.filter_image_list(listing, "other")

    assert [path.name for path in broadcast] == [
        "clay12_jpg.rf.abc.jpg",
        "synframe99_jpg.rf.def.jpg",
    ]
    assert [path.name for path in amateur] == [
        "20260613_172544_000001.jpg",
        "June15a_000240.jpg",
    ]
    assert [path.name for path in other] == ["mystery.jpg"]


def test_write_source_data_yaml_points_val_and_test_at_same_list(tmp_path: Path) -> None:
    images = [
        tmp_path / "fed1.jpg",
        tmp_path / "IMG_9593_1.jpg",
    ]
    for path in images:
        path.write_bytes(b"x")

    yaml_path = bench.write_source_data_yaml(tmp_path / "eval", images)
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert payload["nc"] == 1
    assert payload["names"] == ["tennis ball"]
    assert payload["train"] == "images.txt"
    assert payload["val"] == "images.txt"
    assert payload["test"] == "images.txt"

    lines = (yaml_path.parent / "images.txt").read_text(encoding="utf-8").splitlines()
    assert lines == [path.resolve().as_posix() for path in images]
