"""Unit tests for scripts/merge_datasets.py. No real dataset required."""

from __future__ import annotations

from pathlib import Path

import yaml

import merge_datasets as merge


def _touch_image(root: Path, split: str, name: str) -> Path:
    images = root / split / "images"
    images.mkdir(parents=True, exist_ok=True)
    path = images / name
    path.write_bytes(b"fake-jpeg")
    return path


def test_collect_image_paths_returns_absolute_images_only(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    kept = _touch_image(root, "train", "clay1.jpg")
    _touch_image(root, "train", "notes.txt")
    (root / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (root / "train" / "labels" / "clay1.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (root / "train" / "images" / "subdir").mkdir()
    (root / "valid" / "images").mkdir(parents=True)
    (root / "valid" / "images" / "fed1.png").write_bytes(b"png")

    train_paths = merge.collect_image_paths(root, "train")
    assert train_paths == [kept.resolve()]
    assert train_paths[0].is_absolute()

    valid_paths = merge.collect_image_paths(root, "valid")
    assert [path.name for path in valid_paths] == ["fed1.png"]
    assert merge.collect_image_paths(root, "test") == []
    assert merge.collect_image_paths(tmp_path / "missing", "train") == []


def test_write_split_file_one_absolute_path_per_line(tmp_path: Path) -> None:
    a = tmp_path / "a.jpg"
    b = tmp_path / "nested" / "b.jpg"
    a.write_bytes(b"a")
    b.parent.mkdir()
    b.write_bytes(b"b")
    out = tmp_path / "lists" / "train.txt"

    merge.write_split_file([a, b], out)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines == [a.resolve().as_posix(), b.resolve().as_posix()]
    assert all(Path(line).is_absolute() for line in lines)


def test_write_split_file_empty_list(tmp_path: Path) -> None:
    out = tmp_path / "empty.txt"
    merge.write_split_file([], out)
    assert out.read_text(encoding="utf-8") == ""


def test_write_merged_data_yaml(tmp_path: Path) -> None:
    train_txt = tmp_path / "train.txt"
    val_txt = tmp_path / "val.txt"
    test_txt = tmp_path / "test.txt"
    yaml_path = tmp_path / "data.yaml"
    train_txt.write_text("x\n", encoding="utf-8")
    val_txt.write_text("y\n", encoding="utf-8")
    test_txt.write_text("z\n", encoding="utf-8")

    merge.write_merged_data_yaml(yaml_path, train_txt, val_txt, test_txt)

    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert payload["nc"] == 1
    assert payload["names"] == ["tennis ball"]
    assert payload["train"] == "train.txt"
    assert payload["val"] == "val.txt"
    assert payload["test"] == "test.txt"


def test_classify_source_uses_existing_filename_prefixes() -> None:
    assert merge.classify_source(Path("clay0_jpg.rf.abc.jpg")) == "broadcast"
    assert merge.classify_source(Path("fed12_jpg.rf.abc.jpg")) == "broadcast"
    assert merge.classify_source(Path("synframe101_jpg.rf.abc.jpg")) == "broadcast"
    assert merge.classify_source(Path("synthetic1_jpg.rf.abc.jpg")) == "broadcast"
    assert merge.classify_source(Path("20260613_172544_000123.jpg")) == "amateur"
    assert merge.classify_source(Path("20260629_204129_000001.jpg")) == "amateur"
    assert merge.classify_source(Path("IMG_9593_000010.jpg")) == "amateur"
    assert merge.classify_source(Path("June15a_000002.jpg")) == "amateur"
    assert merge.classify_source(Path("June15b_000003.jpg")) == "amateur"
    assert merge.classify_source(Path("mystery.jpg")) == "other"


def test_merge_datasets_writes_lists_and_counts_by_source(tmp_path: Path) -> None:
    broadcast = tmp_path / "broadcast"
    amateur = tmp_path / "amateur"
    output = tmp_path / "merged"
    _touch_image(broadcast, "train", "clay1.jpg")
    _touch_image(broadcast, "valid", "fed1.jpg")
    _touch_image(broadcast, "test", "synframe1.jpg")
    _touch_image(amateur, "train", "20260613_1.jpg")
    _touch_image(amateur, "train", "IMG_9593_1.jpg")
    _touch_image(amateur, "valid", "June15a_1.jpg")
    _touch_image(amateur, "test", "June15b_1.jpg")

    summary = merge.merge_datasets(broadcast, amateur, output)

    assert summary["train"] == {"broadcast": 1, "amateur": 2, "other": 0, "total": 3}
    assert summary["valid"] == {"broadcast": 1, "amateur": 1, "other": 0, "total": 2}
    assert summary["test"] == {"broadcast": 1, "amateur": 1, "other": 0, "total": 2}

    train_lines = (output / "train.txt").read_text(encoding="utf-8").splitlines()
    assert len(train_lines) == 3
    payload = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert payload["names"] == ["tennis ball"]
    assert payload["nc"] == 1
