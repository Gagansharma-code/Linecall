"""Fine-tune YOLOv8n on the merged broadcast + amateur tennis-ball dataset."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import albumentations as A  # noqa: N812
import yaml
from ultralytics import YOLO  # type: ignore[attr-defined]

from download_dataset import REPO_ROOT, load_dataset_config
from merge_datasets import (
    AMATEUR_CONFIG,
    BROADCAST_CONFIG,
    count_by_source,
)

DEFAULT_DATA = REPO_ROOT / "data" / "merged" / "data.yaml"
DEFAULT_EPOCHS = 30
DEFAULT_IMGSZ = 640
CHECKPOINT = "yolov8n.pt"
# Ultralytics default hsv_v is 0.4. Wider range for lighting/exposure robustness
# called out in docs/DECISIONS.md (broadcast set is uniformly well-lit).
HSV_V = 0.8
RUNS_DETECT = REPO_ROOT / "runs" / "detect"


def git_commit_hash() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def build_albumentations() -> list[Any]:
    """Custom Albumentations set passed through Ultralytics' augmentations= API.

    Replaces Ultralytics' default Albumentations transforms (blur/CLAHE at p=0.01)
    while leaving native YOLO augs (hsv_*, mosaic, etc.) in place.
    """
    return [
        A.RandomShadow(p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.3, p=0.5),
        A.MotionBlur(blur_limit=7, p=0.3),
    ]


def albumentations_manifest(transforms: list[Any]) -> list[dict[str, Any]]:
    recorded: list[dict[str, Any]] = []
    for transform in transforms:
        if hasattr(A, "to_dict"):
            recorded.append(A.to_dict(transform))
        else:
            recorded.append({"name": type(transform).__name__, "repr": repr(transform)})
    return recorded


def load_dataset_pins() -> dict[str, dict[str, Any]]:
    broadcast = load_dataset_config(BROADCAST_CONFIG)
    amateur = load_dataset_config(AMATEUR_CONFIG)
    return {
        "broadcast": {
            "workspace": broadcast["workspace"],
            "project": broadcast["project"],
            "version": broadcast["version"],
        },
        "amateur": {
            "workspace": amateur["workspace"],
            "project": amateur["project"],
            "version": amateur["version"],
        },
    }


def image_counts_from_data_yaml(data_yaml: Path) -> dict[str, dict[str, int]]:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    yaml_dir = data_yaml.resolve().parent
    split_files = {
        "train": payload.get("train"),
        "valid": payload.get("val"),
        "test": payload.get("test"),
    }
    counts: dict[str, dict[str, int]] = {}
    for split, ref in split_files.items():
        if not ref:
            counts[split] = {"broadcast": 0, "amateur": 0, "other": 0, "total": 0}
            continue
        txt = Path(str(ref))
        if not txt.is_absolute():
            txt = yaml_dir / txt
        paths = [
            Path(line) for line in txt.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        counts[split] = count_by_source(paths)
    return counts


def metrics_from_results(results: Any, save_dir: Path) -> dict[str, float | None]:
    metrics = _metrics_from_box(getattr(results, "box", None))
    if any(value is None for value in metrics.values()):
        csv_metrics = _metrics_from_results_csv(save_dir / "results.csv")
        for key, value in csv_metrics.items():
            if metrics.get(key) is None and value is not None:
                metrics[key] = value
    return metrics


def _metrics_from_box(box: Any) -> dict[str, float | None]:
    if box is None:
        return {"precision": None, "recall": None, "mAP50": None, "mAP50-95": None}
    return {
        "precision": _as_float(getattr(box, "mp", None)),
        "recall": _as_float(getattr(box, "mr", None)),
        "mAP50": _as_float(getattr(box, "map50", None)),
        "mAP50-95": _as_float(getattr(box, "map", None)),
    }


def _metrics_from_results_csv(csv_path: Path) -> dict[str, float | None]:
    empty: dict[str, float | None] = {
        "precision": None,
        "recall": None,
        "mAP50": None,
        "mAP50-95": None,
    }
    if not csv_path.is_file():
        return empty
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return empty
    last = {key.strip(): value for key, value in rows[-1].items()}
    aliases = {
        "precision": ("metrics/precision(B)", "metrics/precision", "precision"),
        "recall": ("metrics/recall(B)", "metrics/recall", "recall"),
        "mAP50": ("metrics/mAP50(B)", "metrics/mAP50", "mAP50"),
        "mAP50-95": ("metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95"),
    }
    out = dict(empty)
    for dest, keys in aliases.items():
        for key in keys:
            if key in last and last[key] not in ("", None):
                out[dest] = _as_float(last[key])
                break
    return out


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8n on the merged tennis-ball dataset."
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size. Omit to let Ultralytics choose its default.",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_yaml = args.data if args.data.is_absolute() else REPO_ROOT / args.data
    if not data_yaml.is_file():
        print(f"Merged data yaml not found at {data_yaml}", file=sys.stderr)
        return 1

    transforms = build_albumentations()
    run_name = f"yolov8n-merged-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    train_kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "hsv_v": HSV_V,
        "augmentations": transforms,
        "project": str(RUNS_DETECT),
        "name": run_name,
        "exist_ok": False,
    }
    if args.batch is not None:
        train_kwargs["batch"] = args.batch
    if sys.platform == "win32":
        # Windows DataLoader multiprocessing frequently hangs; keep workers in-process.
        train_kwargs["workers"] = 0

    print(f"Starting run {run_name} (checkpoint={CHECKPOINT}, hsv_v={HSV_V})")
    model = YOLO(CHECKPOINT)
    started = time.perf_counter()
    results = model.train(**train_kwargs)
    elapsed = time.perf_counter() - started

    save_dir = Path(getattr(model.trainer, "save_dir", RUNS_DETECT / run_name))
    trainer_args = getattr(model.trainer, "args", None)
    used_batch = getattr(trainer_args, "batch", args.batch)
    used_epochs = getattr(trainer_args, "epochs", args.epochs)
    used_imgsz = getattr(trainer_args, "imgsz", args.imgsz)
    used_hsv_v = getattr(trainer_args, "hsv_v", HSV_V)

    manifest = {
        "git_commit": git_commit_hash(),
        "run_name": run_name,
        "checkpoint": CHECKPOINT,
        "datasets": load_dataset_pins(),
        "image_counts": image_counts_from_data_yaml(data_yaml),
        "hyperparameters": {
            "epochs": used_epochs,
            "imgsz": used_imgsz,
            "batch": used_batch,
            "hsv_v": used_hsv_v,
            "albumentations": albumentations_manifest(transforms),
            "workers": train_kwargs.get("workers"),
        },
        "training_duration_seconds": round(elapsed, 2),
        "metrics": metrics_from_results(results, save_dir),
    }
    write_manifest(save_dir / "manifest.json", manifest)
    print(f"Wrote {save_dir / 'manifest.json'}")
    print(f"Wall-clock: {elapsed:.1f}s")
    print(f"Metrics: {json.dumps(manifest['metrics'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
