"""Unit tests for the association module."""

from __future__ import annotations

import numpy as np
import pytest

from sports_tracker.association import (
    DEFAULT_IOU_THRESHOLD,
    associate,
    iou,
    iou_matrix,
)


# --- Scalar IOU --------------------------------------------------------------


def test_iou_identical_boxes_is_one():
    box = np.array([10.0, 20.0, 110.0, 120.0])
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    a = np.array([0.0, 0.0, 10.0, 10.0])
    b = np.array([100.0, 100.0, 110.0, 110.0])
    assert iou(a, b) == 0.0


def test_iou_touching_boxes_is_zero():
    """Boxes that touch on an edge but don't overlap have IOU 0, not negative."""
    a = np.array([0.0, 0.0, 10.0, 10.0])
    b = np.array([10.0, 0.0, 20.0, 10.0])
    assert iou(a, b) == 0.0


def test_iou_known_value():
    """Two 10x10 boxes offset by (5, 5) overlap on a 5x5 region.
    Intersection = 25, union = 100 + 100 - 25 = 175, IOU = 25/175 = 1/7.
    """
    a = np.array([0.0, 0.0, 10.0, 10.0])
    b = np.array([5.0, 5.0, 15.0, 15.0])
    assert iou(a, b) == pytest.approx(1.0 / 7.0)


def test_iou_one_box_inside_other():
    """A 5x5 box fully inside a 10x10 box.
    Intersection = 25, union = 100, IOU = 0.25.
    """
    outer = np.array([0.0, 0.0, 10.0, 10.0])
    inner = np.array([2.0, 2.0, 7.0, 7.0])
    assert iou(outer, inner) == pytest.approx(0.25)


def test_iou_degenerate_box_returns_zero():
    """Boxes with zero or negative area should give 0, not NaN or crash."""
    good = np.array([0.0, 0.0, 10.0, 10.0])
    zero_width = np.array([5.0, 5.0, 5.0, 10.0])
    zero_height = np.array([5.0, 5.0, 10.0, 5.0])
    inverted = np.array([10.0, 10.0, 0.0, 0.0])
    assert iou(good, zero_width) == 0.0
    assert iou(good, zero_height) == 0.0
    assert iou(good, inverted) == 0.0


# --- Vectorized matrix version ----------------------------------------------


def test_iou_matrix_matches_scalar():
    """The vectorized matrix version must produce the same numbers as
    calling iou() in a double loop. This catches any broadcasting
    mistakes in the matrix implementation.
    """
    rng = np.random.default_rng(seed=42)
    n, m = 5, 7
    # Random boxes with positive area.
    tracks = rng.uniform(0, 100, size=(n, 2))
    sizes_t = rng.uniform(10, 50, size=(n, 2))
    tracks = np.hstack([tracks, tracks + sizes_t])

    dets = rng.uniform(0, 100, size=(m, 2))
    sizes_d = rng.uniform(10, 50, size=(m, 2))
    dets = np.hstack([dets, dets + sizes_d])

    matrix = iou_matrix(tracks, dets)
    expected = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            expected[i, j] = iou(tracks[i], dets[j])

    np.testing.assert_allclose(matrix, expected, atol=1e-9)


def test_iou_matrix_empty_inputs():
    """Empty track or detection lists should produce empty matrices, not crash."""
    boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
    empty = np.empty((0, 4))
    assert iou_matrix(empty, boxes).shape == (0, 1)
    assert iou_matrix(boxes, empty).shape == (1, 0)
    assert iou_matrix(empty, empty).shape == (0, 0)


# --- Association --------------------------------------------------------------


def test_associate_perfect_match():
    """If detections exactly match tracks, every pair should be matched."""
    boxes = np.array(
        [[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0], [100.0, 0.0, 110.0, 10.0]]
    )
    result = associate(boxes, boxes)
    assert len(result.matches) == 3
    assert len(result.unmatched_tracks) == 0
    assert len(result.unmatched_detections) == 0
    # Each track should be matched to the detection at the same index.
    matches_dict = dict(result.matches)
    assert matches_dict == {0: 0, 1: 1, 2: 2}


def test_associate_disjoint_boxes_no_matches():
    """Tracks and detections that don't overlap should produce no matches."""
    tracks = np.array([[0.0, 0.0, 10.0, 10.0]])
    dets = np.array([[100.0, 100.0, 110.0, 110.0]])
    result = associate(tracks, dets)
    assert len(result.matches) == 0
    assert list(result.unmatched_tracks) == [0]
    assert list(result.unmatched_detections) == [0]


def test_associate_empty_tracks():
    """No existing tracks: every detection becomes unmatched (i.e., a new track)."""
    dets = np.array([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]])
    result = associate(np.empty((0, 4)), dets)
    assert len(result.matches) == 0
    assert len(result.unmatched_tracks) == 0
    assert list(result.unmatched_detections) == [0, 1]


def test_associate_empty_detections():
    """No detections this frame: every track is unmatched (occluded/missed)."""
    tracks = np.array([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]])
    result = associate(tracks, np.empty((0, 4)))
    assert len(result.matches) == 0
    assert list(result.unmatched_tracks) == [0, 1]
    assert len(result.unmatched_detections) == 0


def test_associate_gating_rejects_low_iou():
    """A pair with IOU below threshold should be rejected as no-match,
    even if Hungarian's optimum assigned them. Both fall through to
    unmatched lists.
    """
    # Slight overlap (IOU ~= 0.04, well below 0.3 default).
    tracks = np.array([[0.0, 0.0, 10.0, 10.0]])
    dets = np.array([[8.0, 8.0, 18.0, 18.0]])
    overlap = iou(tracks[0], dets[0])
    assert overlap < DEFAULT_IOU_THRESHOLD, (
        f"Test setup: overlap should be below threshold, got {overlap}"
    )
    result = associate(tracks, dets)
    assert len(result.matches) == 0
    assert list(result.unmatched_tracks) == [0]
    assert list(result.unmatched_detections) == [0]


def test_associate_picks_best_when_two_tracks_compete():
    """Two tracks both overlap the same detection. Hungarian should pick
    the higher-IOU pairing. The other track ends up unmatched.
    """
    # Track 0 is far from the detection (low overlap).
    # Track 1 is exactly aligned with the detection.
    tracks = np.array(
        [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]
    )
    dets = np.array([[20.0, 20.0, 30.0, 30.0]])
    result = associate(tracks, dets)
    # Track 1 should be the one matched (perfect IOU).
    assert len(result.matches) == 1
    assert tuple(result.matches[0]) == (1, 0)
    assert list(result.unmatched_tracks) == [0]
    assert len(result.unmatched_detections) == 0


def test_associate_resolves_crossing_paths():
    """Two tracks moving toward each other, with detections at their
    predicted next positions. The Hungarian algorithm must keep their
    identities straight rather than swapping them.

    Setup:
        Track 0 was at x=0, predicted to be at x=30 (moved right).
        Track 1 was at x=40, predicted to be at x=10 (moved left).
        Detections appear at x=30 and x=10.

    The correct assignment is (track 0 -> det 30, track 1 -> det 10),
    not the swap. Hungarian guarantees this because total IOU is higher
    when each track gets its own predicted position.
    """
    tracks = np.array(
        [
            [25.0, 0.0, 35.0, 20.0],   # predicted box centered at x=30
            [5.0, 0.0, 15.0, 20.0],    # predicted box centered at x=10
        ]
    )
    dets = np.array(
        [
            [28.0, 0.0, 38.0, 20.0],   # actual det near x=33 (matches track 0)
            [3.0, 0.0, 13.0, 20.0],    # actual det near x=8 (matches track 1)
        ]
    )
    result = associate(tracks, dets)
    matches_dict = dict(result.matches)
    assert matches_dict == {0: 0, 1: 1}, (
        f"Expected track 0 -> det 0 and track 1 -> det 1, got {matches_dict}"
    )


def test_associate_one_new_one_lost():
    """A common real-world frame: one track persists, one disappears,
    one new object enters the scene.
    """
    tracks = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],     # persists (matches det 0)
            [50.0, 50.0, 60.0, 60.0],   # disappears (no matching det)
        ]
    )
    dets = np.array(
        [
            [1.0, 1.0, 11.0, 11.0],         # matches track 0
            [200.0, 200.0, 210.0, 210.0],   # brand new
        ]
    )
    result = associate(tracks, dets)
    matches_dict = dict(result.matches)
    assert matches_dict == {0: 0}
    assert list(result.unmatched_tracks) == [1]
    assert list(result.unmatched_detections) == [1]


def test_associate_threshold_is_tunable():
    """Setting iou_threshold below the actual overlap should accept the
    pair; setting it above should reject. This confirms the gating
    parameter actually works.
    """
    tracks = np.array([[0.0, 0.0, 10.0, 10.0]])
    dets = np.array([[5.0, 5.0, 15.0, 15.0]])
    overlap = iou(tracks[0], dets[0])  # = 1/7 ~= 0.143

    # Threshold below the overlap: matched.
    result = associate(tracks, dets, iou_threshold=overlap - 0.01)
    assert len(result.matches) == 1

    # Threshold above the overlap: rejected.
    result = associate(tracks, dets, iou_threshold=overlap + 0.01)
    assert len(result.matches) == 0
    assert list(result.unmatched_tracks) == [0]
    assert list(result.unmatched_detections) == [0]
