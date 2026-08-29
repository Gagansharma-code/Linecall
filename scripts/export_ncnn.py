"""Export a trained YOLOv8 checkpoint to NCNN via Ultralytics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from download_dataset import REPO_ROOT

DEFAULT_RUNS_DIR = REPO_ROOT / "runs" / "detect"


class CheckpointNotFoundError(FileNotFoundError):
    """Raised when no trained best.pt exists under runs/detect."""


def find_latest_checkpoint(runs_dir: Path = DEFAULT_RUNS_DIR) -> Path:
    """Return best.pt from the most recently modified run directory."""
    candidates = [path for path in runs_dir.glob("*/weights/best.pt") if path.is_file()]
    if not candidates:
        raise CheckpointNotFoundError(
            f"No checkpoint found at {runs_dir}/*/weights/best.pt. "
            "Train a model first with scripts/train.py."
        )
    latest = max(candidates, key=lambda path: path.parent.parent.stat().st_mtime)
    return latest.resolve()


def ncnn_dir_for_checkpoint(checkpoint_path: Path) -> Path:
    """Ultralytics writes <stem>_ncnn_model/ next to the source .pt file."""
    return checkpoint_path.resolve().with_name(f"{checkpoint_path.stem}_ncnn_model")


def export_to_ncnn(checkpoint_path: Path) -> Path:
    """Export via Ultralytics' first-party NCNN path; keep its default output location."""
    from ultralytics import YOLO  # type: ignore[attr-defined]

    checkpoint = checkpoint_path.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model = YOLO(str(checkpoint))
    exported = model.export(format="ncnn")
    return Path(str(exported)).resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a YOLOv8 checkpoint to NCNN.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to best.pt (default: latest run under runs/detect).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        checkpoint = args.checkpoint.resolve() if args.checkpoint else find_latest_checkpoint()
        exported = export_to_ncnn(checkpoint)
    except CheckpointNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Exported {checkpoint} -> {exported}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
