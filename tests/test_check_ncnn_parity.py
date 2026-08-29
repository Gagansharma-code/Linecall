"""Unit tests for scripts/check_ncnn_parity.py. No real model or images."""

from __future__ import annotations

import math

import check_ncnn_parity as parity


def test_iou_identical_boxes_is_one() -> None:
    box = (10.0, 20.0, 40.0, 80.0)
    assert parity.iou(box, box) == 1.0


def test_iou_non_overlapping_boxes_is_zero() -> None:
    assert parity.iou((0.0, 0.0, 1.0, 1.0), (2.0, 2.0, 3.0, 3.0)) == 0.0
    assert parity.iou((0.0, 0.0, 2.0, 2.0), (2.0, 0.0, 4.0, 2.0)) == 0.0


def test_iou_partial_overlap_hand_computed() -> None:
    # A=(0,0,2,2) area 4; B=(1,0,3,2) area 4; intersection=(1,0,2,2) area 2;
    # union=6; IoU=2/6=1/3.
    assert parity.iou((0.0, 0.0, 2.0, 2.0), (1.0, 0.0, 3.0, 2.0)) == 1.0 / 3.0
    # A=(0,0,4,4) area 16; B=(2,2,6,6) area 16; intersection=(2,2,4,4) area 4;
    # union=28; IoU=4/28=1/7.
    assert parity.iou((0.0, 0.0, 4.0, 4.0), (2.0, 2.0, 6.0, 6.0)) == 1.0 / 7.0


def test_match_detections_pairs_identical_boxes() -> None:
    pytorch = [{"xyxy": (0.0, 0.0, 10.0, 10.0), "confidence": 0.9}]
    ncnn = [{"xyxy": (0.0, 0.0, 10.0, 10.0), "confidence": 0.88}]
    result = parity.match_detections(pytorch, ncnn, iou_threshold=0.5)
    assert len(result["matches"]) == 1
    assert result["matches"][0]["iou"] == 1.0
    assert result["matches"][0]["confidence_pytorch"] == 0.9
    assert result["matches"][0]["confidence_ncnn"] == 0.88
    assert math.isclose(result["matches"][0]["confidence_diff"], 0.02)
    assert result["unmatched_pytorch"] == []
    assert result["unmatched_ncnn"] == []


def test_match_detections_leaves_non_overlapping_unmatched() -> None:
    pytorch = [{"xyxy": (0.0, 0.0, 1.0, 1.0), "confidence": 0.7}]
    ncnn = [{"xyxy": (5.0, 5.0, 6.0, 6.0), "confidence": 0.6}]
    result = parity.match_detections(pytorch, ncnn, iou_threshold=0.5)
    assert result["matches"] == []
    assert len(result["unmatched_pytorch"]) == 1
    assert len(result["unmatched_ncnn"]) == 1


def test_match_detections_greedy_highest_iou_first() -> None:
    pytorch = [
        {"xyxy": (0.0, 0.0, 10.0, 10.0), "confidence": 0.9},
        {"xyxy": (20.0, 20.0, 30.0, 30.0), "confidence": 0.4},
    ]
    ncnn = [
        {"xyxy": (1.0, 0.0, 11.0, 10.0), "confidence": 0.85},  # IoU with first: 9/11
        {"xyxy": (0.0, 0.0, 10.0, 10.0), "confidence": 0.8},  # IoU with first: 1.0
    ]
    result = parity.match_detections(pytorch, ncnn, iou_threshold=0.5)
    assert len(result["matches"]) == 1
    match = result["matches"][0]
    assert match["pytorch_index"] == 0
    assert match["ncnn_index"] == 1
    assert match["iou"] == 1.0
    assert len(result["unmatched_pytorch"]) == 1
    assert result["unmatched_pytorch"][0]["index"] == 1
    assert len(result["unmatched_ncnn"]) == 1
    assert result["unmatched_ncnn"][0]["index"] == 0


def test_match_detections_empty_lists() -> None:
    result = parity.match_detections([], [], iou_threshold=0.5)
    assert result["matches"] == []
    assert result["unmatched_pytorch"] == []
    assert result["unmatched_ncnn"] == []
