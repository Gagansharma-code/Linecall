"""Held-out detection metrics broken out by broadcast vs amateur source."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from download_dataset import REPO_ROOT
from export_ncnn import CheckpointNotFoundError, find_latest_checkpoint
from merge_datasets import classify_source, write_split_file

DEFAULT_TEST_LIST = REPO_ROOT / "data" / "merged" / "test.txt"
SMALL_SAMPLE_THRESHOLD = 30
SOURCES = ("combined", "broadcast", "amateur")
GAP_SIGN_CONVENTION = "amateur minus broadcast; negative means amateur is worse than broadcast"


def filter_image_list(list_path: Path, source: str) -> list[Path]:
    """Return image paths from an Ultralytics .txt list matching one source."""
    if not list_path.is_file():
        raise FileNotFoundError(f"Image list not found: {list_path}")
    paths: list[Path] = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        path = Path(stripped)
        if classify_source(path) == source:
            paths.append(path)
    return paths


def write_source_data_yaml(output_dir: Path, image_paths: list[Path]) -> Path:
    """Write a single-split YOLO data.yaml with train, val, and test pointing at the list.

    Ultralytics 8.4.133's check_det_dataset requires both 'train' and 'val'.
    model.val() defaults to split='val'; pointing all three keys at the same
    .txt means either split= works.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images_txt = output_dir / "images.txt"
    write_split_file(image_paths, images_txt)
    yaml_path = output_dir / "data.yaml"
    payload = {
        "nc": 1,
        "names": ["tennis ball"],
        # Ultralytics 8.4.133's check_det_dataset requires 'train' and 'val'.
        # model.val() defaults to split='val'; 'test' is here so split='test' works too.
        "train": images_txt.name,
        "val": images_txt.name,
        "test": images_txt.name,
    }
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return yaml_path


def run_source_benchmark(
    checkpoint_path: Path,
    image_paths: list[Path],
    workdir: Path,
    *,
    model: Any | None = None,
) -> dict[str, Any]:
    from ultralytics import YOLO  # type: ignore[attr-defined]

    from train import metrics_from_results

    if not image_paths:
        raise ValueError(f"No images to evaluate under {workdir}")
    yaml_path = write_source_data_yaml(workdir, image_paths)
    if model is None:
        model = YOLO(str(checkpoint_path))
    val_kwargs: dict[str, Any] = {
        "data": str(yaml_path),
        "split": "val",
        "imgsz": 640,
        "plots": False,
        "project": str(workdir),
        "name": "val",
        "exist_ok": True,
        "verbose": False,
    }
    if sys.platform == "win32":
        # Windows DataLoader multiprocessing frequently hangs; keep workers in-process.
        val_kwargs["workers"] = 0
    results = model.val(**val_kwargs)
    save_dir = Path(str(getattr(results, "save_dir", workdir / "val")))
    metrics = metrics_from_results(results, save_dir)
    return {
        "image_count": len(image_paths),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "mAP50": metrics["mAP50"],
        "mAP50-95": metrics["mAP50-95"],
    }


def run_benchmark(
    checkpoint_path: Path,
    test_list: Path,
    output_dir: Path,
) -> dict[str, Any]:
    from ultralytics import YOLO  # type: ignore[attr-defined]

    combined = [
        Path(line.strip())
        for line in test_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_source = {
        "combined": combined,
        "broadcast": filter_image_list(test_list, "broadcast"),
        "amateur": filter_image_list(test_list, "amateur"),
    }
    model = YOLO(str(checkpoint_path))
    per_source: dict[str, dict[str, Any]] = {}
    for source in SOURCES:
        per_source[source] = run_source_benchmark(
            checkpoint_path,
            by_source[source],
            output_dir / source,
            model=model,
        )

    amateur_count = int(per_source["amateur"]["image_count"])
    small_sample = amateur_count < SMALL_SAMPLE_THRESHOLD
    report: dict[str, Any] = {
        "checkpoint": str(checkpoint_path.resolve()),
        "test_list": str(test_list.resolve()),
        "per_source": per_source,
        "gap_amateur_minus_broadcast": {
            "mAP50": _subtract(per_source["amateur"]["mAP50"], per_source["broadcast"]["mAP50"]),
            "mAP50-95": _subtract(
                per_source["amateur"]["mAP50-95"], per_source["broadcast"]["mAP50-95"]
            ),
            "sign_convention": GAP_SIGN_CONVENTION,
        },
        "small_sample": {
            "amateur": small_sample,
            "amateur_image_count": amateur_count,
            "threshold": SMALL_SAMPLE_THRESHOLD,
            "note": (
                "Amateur test set has fewer than "
                f"{SMALL_SAMPLE_THRESHOLD} images; treat that source's metrics "
                "as a weak estimate, not with the same confidence as a large split."
                if small_sample
                else (
                    f"Amateur test set has {amateur_count} images "
                    f"(threshold {SMALL_SAMPLE_THRESHOLD}); small-sample flag is off."
                )
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def _subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def default_output_dir(checkpoint_path: Path) -> Path:
    # runs/detect/<name>/weights/best.pt -> runs/detect/<name>/
    if checkpoint_path.parent.name == "weights":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Held-out YOLO metrics broken out by broadcast vs amateur source."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="PyTorch best.pt (default: latest run under runs/detect).",
    )
    parser.add_argument("--test-list", type=Path, default=DEFAULT_TEST_LIST)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory for benchmark_report.json (default: the checkpoint's run dir).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        checkpoint = args.checkpoint.resolve() if args.checkpoint else find_latest_checkpoint()
        test_list = args.test_list if args.test_list.is_absolute() else REPO_ROOT / args.test_list
        if not test_list.is_file():
            raise FileNotFoundError(
                f"Test list not found at {test_list}. Run scripts/merge_datasets.py first."
            )
        output_dir = args.output.resolve() if args.output else default_output_dir(checkpoint)
        report = run_benchmark(checkpoint, test_list, output_dir)
    except (CheckpointNotFoundError, FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
