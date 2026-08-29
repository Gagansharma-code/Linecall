"""Merge broadcast and amateur YOLOv8 datasets into Ultralytics image-list files.

Does not copy images. Writes train.txt / val.txt / test.txt (absolute paths)
and a data.yaml under data/merged/ that Ultralytics can train from.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import yaml

from download_dataset import DEFAULT_DATA_ROOT, REPO_ROOT, destination_dir, load_dataset_config

BROADCAST_CONFIG = REPO_ROOT / "configs" / "dataset.yaml"
AMATEUR_CONFIG = REPO_ROOT / "configs" / "amateur_dataset.yaml"
MERGED_DIR = REPO_ROOT / "data" / "merged"

# Roboflow YOLO exports use "valid"; Ultralytics data.yaml uses "val".
SOURCE_SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

AMATEUR_PREFIXES = (
    "20260613_",
    "20260629_",
    "IMG_9593_",
    "June15a",
    "June15b",
)
BROADCAST_PREFIXES = ("clay", "fed", "synframe", "synthetic")


def collect_image_paths(dataset_root: Path, split: str) -> list[Path]:
    """Return absolute paths to images in <dataset_root>/<split>/images/."""
    images_dir = dataset_root / split / "images"
    if not images_dir.is_dir():
        return []
    return sorted(
        path.resolve()
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def write_split_file(paths: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [path.resolve().as_posix() for path in paths]
    text = "\n".join(lines)
    if lines:
        text += "\n"
    output_path.write_text(text, encoding="utf-8")


def write_merged_data_yaml(
    output_path: Path,
    train_txt: Path,
    val_txt: Path,
    test_txt: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "nc": 1,
        "names": ["tennis ball"],
        "train": _ref_for_yaml(train_txt, output_path),
        "val": _ref_for_yaml(val_txt, output_path),
        "test": _ref_for_yaml(test_txt, output_path),
    }
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def classify_source(path: Path) -> str:
    """Identify broadcast vs amateur from existing filename prefixes."""
    name = path.name
    if name.startswith(AMATEUR_PREFIXES):
        return "amateur"
    lower = name.lower()
    if lower.startswith(BROADCAST_PREFIXES):
        return "broadcast"
    return "other"


def count_by_source(paths: list[Path]) -> dict[str, int]:
    counts: Counter[str] = Counter(classify_source(path) for path in paths)
    return {
        "broadcast": int(counts.get("broadcast", 0)),
        "amateur": int(counts.get("amateur", 0)),
        "other": int(counts.get("other", 0)),
        "total": len(paths),
    }


def dataset_root_from_config(config_path: Path, data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    config = load_dataset_config(config_path)
    return destination_dir(data_root, str(config["project"]), int(config["version"]))


def merge_datasets(
    broadcast_root: Path,
    amateur_root: Path,
    output_dir: Path,
) -> dict[str, dict[str, int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_paths: dict[str, list[Path]] = {}
    summary: dict[str, dict[str, int]] = {}

    for split in SOURCE_SPLITS:
        combined = collect_image_paths(broadcast_root, split) + collect_image_paths(
            amateur_root, split
        )
        combined = sorted(combined)
        split_paths[split] = combined
        summary[split] = count_by_source(combined)

    train_txt = output_dir / "train.txt"
    val_txt = output_dir / "val.txt"
    test_txt = output_dir / "test.txt"
    write_split_file(split_paths["train"], train_txt)
    write_split_file(split_paths["valid"], val_txt)
    write_split_file(split_paths["test"], test_txt)
    write_merged_data_yaml(output_dir / "data.yaml", train_txt, val_txt, test_txt)
    return summary


def _ref_for_yaml(target: Path, yaml_path: Path) -> str:
    try:
        return target.resolve().relative_to(yaml_path.resolve().parent).as_posix()
    except ValueError:
        return target.resolve().as_posix()


def print_summary(summary: dict[str, dict[str, int]]) -> None:
    for split in SOURCE_SPLITS:
        counts = summary[split]
        print(
            f"{split}: broadcast={counts['broadcast']} amateur={counts['amateur']} "
            f"other={counts['other']} total={counts['total']}"
        )
    totals = {
        key: sum(summary[split][key] for split in SOURCE_SPLITS)
        for key in ("broadcast", "amateur", "other", "total")
    }
    print(
        f"all: broadcast={totals['broadcast']} amateur={totals['amateur']} "
        f"other={totals['other']} total={totals['total']}"
    )


def main() -> int:
    broadcast_root = dataset_root_from_config(BROADCAST_CONFIG)
    amateur_root = dataset_root_from_config(AMATEUR_CONFIG)
    if not broadcast_root.is_dir():
        print(f"Broadcast dataset not found at {broadcast_root}", file=sys.stderr)
        return 1
    if not amateur_root.is_dir():
        print(f"Amateur dataset not found at {amateur_root}", file=sys.stderr)
        return 1

    summary = merge_datasets(broadcast_root, amateur_root, MERGED_DIR)
    print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
