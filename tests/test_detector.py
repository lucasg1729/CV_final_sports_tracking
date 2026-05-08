"""Unit tests for the detector module.

These cover the parts that don't require running YOLO itself 
(FrameDetections.filter, save_detections / load_detections round-trip,
load_detections's confidence threshold and dense-mode behavior)

The actual YoloDetector class needs PyTorch + downloaded weights, so
its real-inference behavior is verified by running the runner script
against a real clip rather than in unit tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sports_tracker.detector import (
    FrameDetections,
    load_detections,
    save_detections,
)


# FrameDetections


def test_frame_detections_filter():
    fd = FrameDetections(
        frame_idx=5,
        boxes=np.array(
            [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], dtype=np.float32
        ),
        scores=np.array([0.9, 0.4, 0.1], dtype=np.float32),
    )
    filtered = fd.filter(0.5)
    assert filtered.frame_idx == 5
    assert filtered.boxes.shape == (1, 4)
    assert filtered.scores.shape == (1,)
    assert filtered.scores[0] == pytest.approx(0.9)


def test_filter_keeps_at_threshold_boundary():
    """A score exactly equal to min_conf should be kept (>=, not >)."""
    fd = FrameDetections(
        frame_idx=1,
        boxes=np.array([[0, 0, 10, 10]], dtype=np.float32),
        scores=np.array([0.5], dtype=np.float32),
    )
    assert len(fd.filter(0.5).boxes) == 1
    assert len(fd.filter(0.50001).boxes) == 0


def test_filter_returns_empty_when_all_below_threshold():
    fd = FrameDetections(
        frame_idx=1,
        boxes=np.array([[0, 0, 10, 10]], dtype=np.float32),
        scores=np.array([0.1], dtype=np.float32),
    )
    out = fd.filter(0.9)
    assert out.boxes.shape == (0, 4)
    assert out.scores.shape == (0,)


# Cache I/O


def test_cache_roundtrip_basic(tmp_path: Path):
    """Save and load a small clip's worth of detections; data should be
    bit-identical (modulo float32 storage) on the way back out.
    """
    frame_dets = [
        FrameDetections(
            frame_idx=1,
            boxes=np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.float32),
            scores=np.array([0.9, 0.8], dtype=np.float32),
        ),
        FrameDetections(
            frame_idx=2,
            boxes=np.array([[15, 25, 35, 45]], dtype=np.float32),
            scores=np.array([0.7], dtype=np.float32),
        ),
    ]
    meta_in = {"clip": "test_clip", "model": "yolov8n.pt", "n_frames": 2}

    cache_path = tmp_path / "test.npz"
    save_detections(cache_path, frame_dets, meta_in)
    assert cache_path.exists()

    loaded, meta_out = load_detections(cache_path)
    assert meta_out == meta_in
    assert len(loaded) == 2

    # Frame 1: two boxes
    assert loaded[0].frame_idx == 1
    np.testing.assert_array_equal(loaded[0].boxes, frame_dets[0].boxes)
    np.testing.assert_array_equal(loaded[0].scores, frame_dets[0].scores)

    # Frame 2: one box
    assert loaded[1].frame_idx == 2
    np.testing.assert_array_equal(loaded[1].boxes, frame_dets[1].boxes)


def test_cache_roundtrip_empty(tmp_path: Path):
    """A clip where no frames had detections should round-trip cleanly,
    not crash on empty arrays.
    """
    cache_path = tmp_path / "empty.npz"
    save_detections(cache_path, [], {"clip": "empty"})
    loaded, meta = load_detections(cache_path)
    assert loaded == []
    assert meta == {"clip": "empty"}


def test_cache_roundtrip_some_empty_frames(tmp_path: Path):
    """Some frames have detections, others don't. Sparse mode (n_frames=None)
    should return only frames with detections.
    """
    frame_dets = [
        FrameDetections(
            frame_idx=1,
            boxes=np.array([[0, 0, 10, 10]], dtype=np.float32),
            scores=np.array([0.9], dtype=np.float32),
        ),
        FrameDetections(
            frame_idx=2,
            boxes=np.empty((0, 4), dtype=np.float32),
            scores=np.empty(0, dtype=np.float32),
        ),
        FrameDetections(
            frame_idx=3,
            boxes=np.array([[5, 5, 15, 15]], dtype=np.float32),
            scores=np.array([0.8], dtype=np.float32),
        ),
    ]
    cache_path = tmp_path / "sparse.npz"
    save_detections(cache_path, frame_dets, {})

    loaded, _ = load_detections(cache_path)
    # Sparse mode: empty frame 2 isn't returned
    assert [fd.frame_idx for fd in loaded] == [1, 3]


def test_cache_dense_mode_fills_missing_frames(tmp_path: Path):
    """Dense mode (n_frames=N) returns one entry per frame from 1...N,
    with empty FrameDetections for frames that had nothing.
    """
    frame_dets = [
        FrameDetections(
            frame_idx=1,
            boxes=np.array([[0, 0, 10, 10]], dtype=np.float32),
            scores=np.array([0.9], dtype=np.float32),
        ),
        FrameDetections(
            frame_idx=3,
            boxes=np.array([[5, 5, 15, 15]], dtype=np.float32),
            scores=np.array([0.8], dtype=np.float32),
        ),
    ]
    cache_path = tmp_path / "dense.npz"
    save_detections(cache_path, frame_dets, {})

    loaded, _ = load_detections(cache_path, n_frames=4)
    assert len(loaded) == 4
    assert loaded[0].frame_idx == 1 and len(loaded[0].boxes) == 1
    assert loaded[1].frame_idx == 2 and len(loaded[1].boxes) == 0
    assert loaded[2].frame_idx == 3 and len(loaded[2].boxes) == 1
    assert loaded[3].frame_idx == 4 and len(loaded[3].boxes) == 0


def test_cache_load_applies_min_conf(tmp_path: Path):
    """Loading with min_conf > 0 should drop low-confidence detections"""
    frame_dets = [
        FrameDetections(
            frame_idx=1,
            boxes=np.array(
                [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]],
                dtype=np.float32,
            ),
            scores=np.array([0.9, 0.4, 0.1], dtype=np.float32),
        ),
    ]
    cache_path = tmp_path / "filtered.npz"
    save_detections(cache_path, frame_dets, {})

    loaded_all, _ = load_detections(cache_path, min_conf=0.0)
    assert len(loaded_all[0].boxes) == 3

    loaded_strict, _ = load_detections(cache_path, min_conf=0.5)
    assert len(loaded_strict[0].boxes) == 1
    assert loaded_strict[0].scores[0] == pytest.approx(0.9)


def test_cache_load_min_conf_can_empty_a_frame(tmp_path: Path):
    """Sparse mode skips frames whose detections all get filtered out"""
    frame_dets = [
        FrameDetections(
            frame_idx=1,
            boxes=np.array([[0, 0, 10, 10]], dtype=np.float32),
            scores=np.array([0.2], dtype=np.float32),
        ),
        FrameDetections(
            frame_idx=2,
            boxes=np.array([[5, 5, 15, 15]], dtype=np.float32),
            scores=np.array([0.9], dtype=np.float32),
        ),
    ]
    cache_path = tmp_path / "filter-out.npz"
    save_detections(cache_path, frame_dets, {})

    loaded, _ = load_detections(cache_path, min_conf=0.5)
    assert [fd.frame_idx for fd in loaded] == [2]


def test_cache_save_sorts_unsorted_input(tmp_path: Path):
    """save_detections should sort frames by index even if given out of order"""
    frame_dets = [
        FrameDetections(
            frame_idx=3,
            boxes=np.array([[3, 3, 13, 13]], dtype=np.float32),
            scores=np.array([0.5], dtype=np.float32),
        ),
        FrameDetections(
            frame_idx=1,
            boxes=np.array([[1, 1, 11, 11]], dtype=np.float32),
            scores=np.array([0.5], dtype=np.float32),
        ),
        FrameDetections(
            frame_idx=2,
            boxes=np.array([[2, 2, 12, 12]], dtype=np.float32),
            scores=np.array([0.5], dtype=np.float32),
        ),
    ]
    cache_path = tmp_path / "unsorted.npz"
    save_detections(cache_path, frame_dets, {})

    loaded, _ = load_detections(cache_path)
    assert [fd.frame_idx for fd in loaded] == [1, 2, 3]
