"""Detection-to-track association via IOU cost and the Hungarian algorithm.

Each frame, the tracker has a set of existing tracks (each with a predicted
bounding box from its Kalman filter) and a set of new detections from the
detector. We need to decide which detection corresponds to which track,
which detections start new tracks, and which tracks have no matching
detection.

This module follows SORT (Bewley et al., 2016):

  1. Build a cost matrix where cost[i, j] = 1 - IOU(track_i, detection_j)
  2. Solve the assignment problem with the Hungarian algorithm
  3. Reject any matched pair whose IOU is below `iou_threshold`. These
     fall through to "unmatched" on both sides

The IOU-only cost has known limitations. It cannot distinguish two
players who briefly swap positions, since boxes overlap purely on
geometry. Adding an appearance descriptor (DeepSORT, Wojke et al. 2017)
addresses this. Whether that's necessary for sports footage is one of
the questions this project investigates.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# Default IOU threshold below which a Hungarian-assigned pair is
# rejected. The SORT paper uses 0.3, justified by ablation against
# the MOT15 training set. We leave it tunable because sports footage
# may benefit from a different value
DEFAULT_IOU_THRESHOLD = 0.3


class AssociationResult(NamedTuple):
    """The output of associating detections to tracks for one frame

    Attributes:
        matches: shape (M, 2) int array. Each row is (track_idx, detection_idx)
            for a successful match (IOU >= threshold)
        unmatched_tracks: shape (U,) int array. Indices of tracks that
            had no matching detection. The tracker will mark these as
            "missed this frame" and may delete them after enough misses
        unmatched_detections: shape (V,) int array. Indices of detections
            that had no matching track. The tracker will spawn new tracks
            for these
    """

    matches: np.ndarray
    unmatched_tracks: np.ndarray
    unmatched_detections: np.ndarray


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IOU of two boxes in [x1, y1, x2, y2] format

    Returns 0.0 if the boxes don't overlap, or if either box is degenerate
    (zero or negative area). The latter is a defensive guard since a buggy
    Kalman filter could in principle produce a box with x2 < x1
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Intersection rectangle
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = ix2 - ix1
    ih = iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0

    return float(inter / union)


def iou_matrix(tracks: np.ndarray, detections: np.ndarray) -> np.ndarray:
    """Compute the full IOU matrix between two sets of boxes

    Vectorized for speed where this gets called once per frame per clip

    Args:
        tracks:     shape (N, 4), boxes in [x1, y1, x2, y2] format
        detections: shape (M, 4), boxes in [x1, y1, x2, y2] format

    Returns:
        shape (N, M), where entry (i, j) is IOU(tracks[i], detections[j])
        Returns shape (0, 0) if either input is empty
    """
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)), dtype=np.float64)

    t = np.asarray(tracks, dtype=np.float64)
    d = np.asarray(detections, dtype=np.float64)

    # Pairwise intersection corners using broadcasting:
    #   t[:, None, :]  ->  shape (N, 1, 4)
    #   d[None, :, :]  ->  shape (1, M, 4)
    # which gives us a (N, M, 4) view we can max/min over per axis
    ix1 = np.maximum(t[:, None, 0], d[None, :, 0])
    iy1 = np.maximum(t[:, None, 1], d[None, :, 1])
    ix2 = np.minimum(t[:, None, 2], d[None, :, 2])
    iy2 = np.minimum(t[:, None, 3], d[None, :, 3])

    iw = np.clip(ix2 - ix1, a_min=0.0, a_max=None)
    ih = np.clip(iy2 - iy1, a_min=0.0, a_max=None)
    inter = iw * ih

    area_t = np.clip(t[:, 2] - t[:, 0], 0, None) * np.clip(t[:, 3] - t[:, 1], 0, None)
    area_d = np.clip(d[:, 2] - d[:, 0], 0, None) * np.clip(d[:, 3] - d[:, 1], 0, None)

    union = area_t[:, None] + area_d[None, :] - inter
    # Avoid division warnings by zeroing the IOU wherever union is 0
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(union > 0, inter / union, 0.0)
    return result


def associate(
    tracks: np.ndarray,
    detections: np.ndarray,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> AssociationResult:
    """Match detections to tracks using IOU cost + Hungarian assignment

    Args:
        tracks: shape (N, 4) array of predicted track boxes (may be empty)
        detections: shape (M, 4) array of new detection boxes (may be empty)
        iou_threshold: pairs with IOU below this are rejected even if
            Hungarian assigned them

    Returns:
        AssociationResult with matched indices and unmatched indices
    """
    n_tracks = len(tracks)
    n_dets = len(detections)

    # Edge cases: empty input on either side. linear_sum_assignment
    # handles empty matrices but it's cleaner to short-circuit here so
    # the empty-array shapes are explicit
    if n_tracks == 0:
        return AssociationResult(
            matches=np.empty((0, 2), dtype=np.int64),
            unmatched_tracks=np.empty(0, dtype=np.int64),
            unmatched_detections=np.arange(n_dets, dtype=np.int64),
        )
    if n_dets == 0:
        return AssociationResult(
            matches=np.empty((0, 2), dtype=np.int64),
            unmatched_tracks=np.arange(n_tracks, dtype=np.int64),
            unmatched_detections=np.empty(0, dtype=np.int64),
        )

    iou_mat = iou_matrix(tracks, detections)
    # Hungarian minimizes cost; we want to maximize IOU, so cost = -IOU
    cost = -iou_mat

    track_idx, det_idx = linear_sum_assignment(cost)

    matches: list[tuple[int, int]] = []
    matched_tracks: set[int] = set()
    matched_dets: set[int] = set()

    for ti, di in zip(track_idx, det_idx):
        if iou_mat[ti, di] >= iou_threshold:
            matches.append((int(ti), int(di)))
            matched_tracks.add(int(ti))
            matched_dets.add(int(di))
        # else: rejected by gating. Both fall through to unmatched

    unmatched_tracks = np.array(
        [i for i in range(n_tracks) if i not in matched_tracks], dtype=np.int64
    )
    unmatched_detections = np.array(
        [j for j in range(n_dets) if j not in matched_dets], dtype=np.int64
    )

    if matches:
        matches_arr = np.array(matches, dtype=np.int64)
    else:
        matches_arr = np.empty((0, 2), dtype=np.int64)

    return AssociationResult(
        matches=matches_arr,
        unmatched_tracks=unmatched_tracks,
        unmatched_detections=unmatched_detections,
    )
