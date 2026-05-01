"""Run the association module checks without needing pytest.

A standalone smoke test for the association module. The full pytest
suite lives in tests/test_association.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sports_tracker.association import (
    DEFAULT_IOU_THRESHOLD,
    associate,
    iou,
    iou_matrix,
)


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        sys.exit(1)


print("Scalar IOU")
print("-" * 60)

box = np.array([10.0, 20.0, 110.0, 120.0])
check("identical boxes -> IOU 1", abs(iou(box, box) - 1.0) < 1e-9)

a = np.array([0.0, 0.0, 10.0, 10.0])
b = np.array([100.0, 100.0, 110.0, 110.0])
check("disjoint boxes -> IOU 0", iou(a, b) == 0.0)

a = np.array([0.0, 0.0, 10.0, 10.0])
b = np.array([10.0, 0.0, 20.0, 10.0])
check("touching boxes -> IOU 0", iou(a, b) == 0.0)

a = np.array([0.0, 0.0, 10.0, 10.0])
b = np.array([5.0, 5.0, 15.0, 15.0])
check(
    "10x10 boxes offset (5,5) -> IOU 1/7",
    abs(iou(a, b) - 1.0 / 7.0) < 1e-9,
    f"got {iou(a, b)}",
)

outer = np.array([0.0, 0.0, 10.0, 10.0])
inner = np.array([2.0, 2.0, 7.0, 7.0])
check("box-inside-box -> IOU 0.25", abs(iou(outer, inner) - 0.25) < 1e-9)

good = np.array([0.0, 0.0, 10.0, 10.0])
zero_w = np.array([5.0, 5.0, 5.0, 10.0])
inverted = np.array([10.0, 10.0, 0.0, 0.0])
check("degenerate zero-width box -> IOU 0", iou(good, zero_w) == 0.0)
check("inverted box -> IOU 0", iou(good, inverted) == 0.0)

print()
print("Vectorized IOU matrix")
print("-" * 60)

rng = np.random.default_rng(seed=42)
n, m = 5, 7
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
check(
    "iou_matrix matches scalar iou",
    np.allclose(matrix, expected, atol=1e-9),
    f"max diff = {np.max(np.abs(matrix - expected))}",
)

empty = np.empty((0, 4))
boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
check("empty tracks: matrix shape (0, M)", iou_matrix(empty, boxes).shape == (0, 1))
check("empty detections: matrix shape (N, 0)", iou_matrix(boxes, empty).shape == (1, 0))
check("both empty: matrix shape (0, 0)", iou_matrix(empty, empty).shape == (0, 0))

print()
print("Association via Hungarian")
print("-" * 60)

# Perfect match.
boxes = np.array(
    [[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0], [100.0, 0.0, 110.0, 10.0]]
)
result = associate(boxes, boxes)
check("perfect match: 3 matches", len(result.matches) == 3)
check("perfect match: no unmatched tracks", len(result.unmatched_tracks) == 0)
check("perfect match: no unmatched detections", len(result.unmatched_detections) == 0)
check("perfect match: identity assignment", dict(result.matches) == {0: 0, 1: 1, 2: 2})

# Disjoint.
tracks = np.array([[0.0, 0.0, 10.0, 10.0]])
dets = np.array([[100.0, 100.0, 110.0, 110.0]])
result = associate(tracks, dets)
check("disjoint boxes: no matches", len(result.matches) == 0)
check("disjoint boxes: track unmatched", list(result.unmatched_tracks) == [0])
check("disjoint boxes: det unmatched", list(result.unmatched_detections) == [0])

# Empty inputs.
result = associate(np.empty((0, 4)), boxes)
check("empty tracks: all dets unmatched", list(result.unmatched_detections) == [0, 1, 2])

result = associate(boxes, np.empty((0, 4)))
check("empty dets: all tracks unmatched", list(result.unmatched_tracks) == [0, 1, 2])

# Gating.
tracks = np.array([[0.0, 0.0, 10.0, 10.0]])
dets = np.array([[8.0, 8.0, 18.0, 18.0]])
overlap = iou(tracks[0], dets[0])
check(
    "gating setup: overlap < threshold",
    overlap < DEFAULT_IOU_THRESHOLD,
    f"overlap = {overlap}",
)
result = associate(tracks, dets)
check(
    "gating rejects low IOU pair",
    len(result.matches) == 0
    and list(result.unmatched_tracks) == [0]
    and list(result.unmatched_detections) == [0],
)

# Two tracks competing for one detection.
tracks = np.array(
    [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]
)
dets = np.array([[20.0, 20.0, 30.0, 30.0]])
result = associate(tracks, dets)
check(
    "two tracks compete: better one wins",
    len(result.matches) == 1 and tuple(result.matches[0]) == (1, 0),
)
check(
    "two tracks compete: loser unmatched",
    list(result.unmatched_tracks) == [0],
)

# Crossing paths.
tracks = np.array(
    [[25.0, 0.0, 35.0, 20.0], [5.0, 0.0, 15.0, 20.0]]
)
dets = np.array(
    [[28.0, 0.0, 38.0, 20.0], [3.0, 0.0, 13.0, 20.0]]
)
result = associate(tracks, dets)
check(
    "crossing paths: identities preserved",
    dict(result.matches) == {0: 0, 1: 1},
    f"got {dict(result.matches)}",
)

# Realistic frame: one persists, one disappears, one is new.
tracks = np.array(
    [[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]]
)
dets = np.array(
    [[1.0, 1.0, 11.0, 11.0], [200.0, 200.0, 210.0, 210.0]]
)
result = associate(tracks, dets)
check(
    "mixed scene: one match, one lost, one new",
    dict(result.matches) == {0: 0}
    and list(result.unmatched_tracks) == [1]
    and list(result.unmatched_detections) == [1],
)

# Tunable threshold.
tracks = np.array([[0.0, 0.0, 10.0, 10.0]])
dets = np.array([[5.0, 5.0, 15.0, 15.0]])
overlap = iou(tracks[0], dets[0])

result_loose = associate(tracks, dets, iou_threshold=overlap - 0.01)
check("threshold below overlap: pair accepted", len(result_loose.matches) == 1)

result_strict = associate(tracks, dets, iou_threshold=overlap + 0.01)
check("threshold above overlap: pair rejected", len(result_strict.matches) == 0)

print()
print("All checks passed.")
