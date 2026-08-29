"""Compare PyTorch and NCNN detections from the same YOLOv8 checkpoint."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from download_dataset import REPO_ROOT
from export_ncnn import (
    CheckpointNotFoundError,
    find_latest_checkpoint,
    ncnn_dir_for_checkpoint,
)

DEFAULT_TEST_LIST = REPO_ROOT / "data" / "merged" / "test.txt"
SAMPLE_SIZE = 25
SAMPLE_SEED = 0
IOU_THRESHOLD = 0.5

# Export-fidelity bars, not detector-quality bars. A correct NCNN conversion
# should keep boxes nearly pixel-aligned: 0.95 IoU is a few pixels of drift on
# a typical 640-scale box. Confidence heads can move a little more than boxes;
# 0.05 is five percentage points. Unmatched boxes across a 25-image sample
# should be near zero — more than two total (either side) is a conversion
# problem, not residual model noise.
MIN_MEAN_MATCHED_IOU = 0.95
MAX_MEAN_CONFIDENCE_DIFF = 0.05
MAX_UNMATCHED_BOXES = 2

Box = tuple[float, float, float, float]
Detection = dict[str, Any]


def iou(box_a: Box, box_b: Box) -> float:
    """Intersection-over-union for axis-aligned (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def match_detections(
    pytorch_boxes: list[Detection],
    ncnn_boxes: list[Detection],
    iou_threshold: float = IOU_THRESHOLD,
) -> dict[str, list[dict[str, Any]]]:
    """Greedy-match detections by highest IoU. One-to-one, no reuse."""
    pairs: list[tuple[float, int, int]] = []
    for i, pytorch in enumerate(pytorch_boxes):
        for j, ncnn in enumerate(ncnn_boxes):
            score = iou(_xyxy(pytorch), _xyxy(ncnn))
            if score >= iou_threshold:
                pairs.append((score, i, j))
    pairs.sort(key=lambda item: item[0], reverse=True)

    used_pytorch: set[int] = set()
    used_ncnn: set[int] = set()
    matches: list[dict[str, Any]] = []
    for score, i, j in pairs:
        if i in used_pytorch or j in used_ncnn:
            continue
        used_pytorch.add(i)
        used_ncnn.add(j)
        pytorch_conf = _confidence(pytorch_boxes[i])
        ncnn_conf = _confidence(ncnn_boxes[j])
        matches.append(
            {
                "iou": score,
                "confidence_pytorch": pytorch_conf,
                "confidence_ncnn": ncnn_conf,
                "confidence_diff": abs(pytorch_conf - ncnn_conf),
                "pytorch_index": i,
                "ncnn_index": j,
            }
        )

    unmatched_pytorch = [
        {"index": i, **pytorch_boxes[i]} for i in range(len(pytorch_boxes)) if i not in used_pytorch
    ]
    unmatched_ncnn = [
        {"index": j, **ncnn_boxes[j]} for j in range(len(ncnn_boxes)) if j not in used_ncnn
    ]
    return {
        "matches": matches,
        "unmatched_pytorch": unmatched_pytorch,
        "unmatched_ncnn": unmatched_ncnn,
    }


def load_test_image_paths(test_list: Path) -> list[Path]:
    if not test_list.is_file():
        raise FileNotFoundError(
            f"Image list not found at {test_list}. Run scripts/merge_datasets.py first."
        )
    paths: list[Path] = []
    for line in test_list.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            paths.append(Path(stripped))
    return paths


def sample_image_paths(
    paths: list[Path],
    sample_size: int = SAMPLE_SIZE,
    seed: int = SAMPLE_SEED,
) -> list[Path]:
    existing = [path for path in paths if path.is_file()]
    if len(existing) <= sample_size:
        return existing
    rng = random.Random(seed)
    return rng.sample(existing, sample_size)


def detections_from_result(result: Any) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = _to_list(boxes.xyxy)
    confs = _to_list(boxes.conf)
    detections: list[Detection] = []
    for box, conf in zip(xyxy, confs, strict=True):
        detections.append(
            {
                "xyxy": (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                "confidence": float(conf),
            }
        )
    return detections


def run_parity_check(
    pytorch_model_path: Path,
    ncnn_model_path: Path,
    image_paths: list[Path],
    iou_threshold: float = IOU_THRESHOLD,
) -> dict[str, Any]:
    from ultralytics import YOLO  # type: ignore[attr-defined]

    pytorch_model = YOLO(str(pytorch_model_path))
    ncnn_model = YOLO(str(ncnn_model_path))
    predict_kwargs: dict[str, Any] = {"verbose": False, "device": "cpu"}

    per_image: list[dict[str, Any]] = []
    matched_ious: list[float] = []
    conf_diffs: list[float] = []
    unmatched_pytorch_total = 0
    unmatched_ncnn_total = 0
    count_disagree_images = 0

    for image_path in image_paths:
        pytorch_results = list(pytorch_model.predict(source=str(image_path), **predict_kwargs))
        ncnn_results = list(ncnn_model.predict(source=str(image_path), **predict_kwargs))
        pytorch_dets = detections_from_result(pytorch_results[0])
        ncnn_dets = detections_from_result(ncnn_results[0])
        matched = match_detections(pytorch_dets, ncnn_dets, iou_threshold=iou_threshold)
        unmatched_p = len(matched["unmatched_pytorch"])
        unmatched_n = len(matched["unmatched_ncnn"])
        unmatched_pytorch_total += unmatched_p
        unmatched_ncnn_total += unmatched_n
        if len(pytorch_dets) != len(ncnn_dets):
            count_disagree_images += 1
        for pair in matched["matches"]:
            matched_ious.append(float(pair["iou"]))
            conf_diffs.append(float(pair["confidence_diff"]))
        per_image.append(
            {
                "image": str(image_path),
                "pytorch_count": len(pytorch_dets),
                "ncnn_count": len(ncnn_dets),
                "matches": matched["matches"],
                "unmatched_pytorch": unmatched_p,
                "unmatched_ncnn": unmatched_n,
            }
        )

    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 1.0
    mean_conf_diff = sum(conf_diffs) / len(conf_diffs) if conf_diffs else 0.0
    unmatched_total = unmatched_pytorch_total + unmatched_ncnn_total
    # No matches and no unmatched boxes means both backends produced empty
    # detections on every image — still faithful, so mean IoU stays at 1.0.
    if not matched_ious and unmatched_total > 0:
        mean_iou = 0.0

    passed = (
        mean_iou >= MIN_MEAN_MATCHED_IOU
        and mean_conf_diff <= MAX_MEAN_CONFIDENCE_DIFF
        and unmatched_total <= MAX_UNMATCHED_BOXES
    )
    failures: list[str] = []
    if mean_iou < MIN_MEAN_MATCHED_IOU:
        failures.append(f"mean matched IoU {mean_iou:.4f} < {MIN_MEAN_MATCHED_IOU}")
    if mean_conf_diff > MAX_MEAN_CONFIDENCE_DIFF:
        failures.append(f"mean confidence diff {mean_conf_diff:.4f} > {MAX_MEAN_CONFIDENCE_DIFF}")
    if unmatched_total > MAX_UNMATCHED_BOXES:
        failures.append(f"unmatched boxes {unmatched_total} > {MAX_UNMATCHED_BOXES}")

    return {
        "passed": passed,
        "failures": failures,
        "thresholds": {
            "iou_match": iou_threshold,
            "min_mean_matched_iou": MIN_MEAN_MATCHED_IOU,
            "max_mean_confidence_diff": MAX_MEAN_CONFIDENCE_DIFF,
            "max_unmatched_boxes": MAX_UNMATCHED_BOXES,
        },
        "aggregate": {
            "images": len(image_paths),
            "matched_pairs": len(matched_ious),
            "mean_matched_iou": mean_iou,
            "mean_confidence_diff": mean_conf_diff,
            "unmatched_pytorch": unmatched_pytorch_total,
            "unmatched_ncnn": unmatched_ncnn_total,
            "unmatched_total": unmatched_total,
            "images_with_count_disagreement": count_disagree_images,
        },
        "per_image": per_image,
    }


def write_parity_report(
    report: dict[str, Any],
    ncnn_model_path: Path,
    pytorch_model_path: Path,
    image_paths: list[Path],
) -> Path:
    run_dir = _run_dir_for_ncnn(ncnn_model_path)
    payload = {
        "pytorch_checkpoint": str(pytorch_model_path.resolve()),
        "ncnn_model": str(ncnn_model_path.resolve()),
        "images": [str(path) for path in image_paths],
        "sample_size": SAMPLE_SIZE,
        "sample_seed": SAMPLE_SEED,
        **report,
    }
    output = run_dir / "parity_report.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def _run_dir_for_ncnn(ncnn_model_path: Path) -> Path:
    resolved = ncnn_model_path.resolve()
    # Ultralytics layout: <run>/weights/<stem>_ncnn_model
    if resolved.parent.name == "weights":
        return resolved.parent.parent
    return resolved.parent


def _xyxy(detection: Detection) -> Box:
    box = detection["xyxy"]
    return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))


def _confidence(detection: Detection) -> float:
    return float(detection["confidence"])


def _to_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return list(value.tolist())
    return list(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check PyTorch vs NCNN detection parity for a YOLOv8 export."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="PyTorch best.pt (default: latest run under runs/detect).",
    )
    parser.add_argument(
        "--ncnn",
        type=Path,
        default=None,
        help="NCNN model directory (default: <checkpoint-stem>_ncnn_model next to the .pt).",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=DEFAULT_TEST_LIST,
        help="Ultralytics image-list .txt (default: data/merged/test.txt).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        checkpoint = args.checkpoint.resolve() if args.checkpoint else find_latest_checkpoint()
        ncnn_path = args.ncnn.resolve() if args.ncnn else ncnn_dir_for_checkpoint(checkpoint)
        if not ncnn_path.exists():
            print(
                f"NCNN export not found at {ncnn_path}. Run scripts/export_ncnn.py first.",
                file=sys.stderr,
            )
            return 1
        sampled = sample_image_paths(load_test_image_paths(args.images))
        if not sampled:
            print(f"No readable images in {args.images}.", file=sys.stderr)
            return 1
        report = run_parity_check(checkpoint, ncnn_path, sampled)
        report_path = write_parity_report(report, ncnn_path, checkpoint, sampled)
    except (CheckpointNotFoundError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    aggregate = report["aggregate"]
    print(f"Wrote {report_path}")
    print(
        f"mean matched IoU={aggregate['mean_matched_iou']:.4f} "
        f"mean confidence diff={aggregate['mean_confidence_diff']:.4f} "
        f"unmatched pytorch={aggregate['unmatched_pytorch']} "
        f"unmatched ncnn={aggregate['unmatched_ncnn']} "
        f"count-disagree images={aggregate['images_with_count_disagreement']}"
    )
    if report["passed"]:
        print("parity: PASS")
        return 0
    print("parity: FAIL", file=sys.stderr)
    for reason in report["failures"]:
        print(reason, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
