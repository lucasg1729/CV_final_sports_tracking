"""End-to-end tests for the Tracker class.

These run synthetic detection sequences through the tracker and assert
that the per-frame outputs match what a SORT-style tracker should
produce: stable IDs over time, ID promotion after min_hits, ID
preservation through brief occlusions, deletion after max_age.
"""

from __future__ import annotations

import numpy as np
import pytest

from sports_tracker.tracker import Tracker, TrackOutput


def box(cx: float, cy: float, w: float = 50.0, h: float = 100.0) -> np.ndarray:
    """Build a [x1, y1, x2, y2] box centered at (cx, cy)"""
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


# Initialization and configuration


def test_init_validates_max_age():
    with pytest.raises(ValueError):
        Tracker(max_age=0)


def test_init_validates_min_hits():
    with pytest.raises(ValueError):
        Tracker(min_hits=0)


def test_empty_input_yields_empty_output():
    """No detections, no tracks means nothing to report"""
    tr = Tracker()
    outputs = tr.step(np.empty((0, 4)))
    assert outputs == []


# Track promotion


def test_new_track_held_back_for_min_hits_frames():
    """A new track should not appear in output until it has accumulated
    min_hits consecutive matches. With min_hits=3, the track should be
    silent on frames 1 and 2, and first appear on frame 3
    """
    tr = Tracker(min_hits=3, max_age=30)
    detections = [box(100, 100), box(105, 100), box(110, 100)]

    out1 = tr.step(np.array([detections[0]]))
    out2 = tr.step(np.array([detections[1]]))
    out3 = tr.step(np.array([detections[2]]))

    assert out1 == [], "Track should be tentative on frame 1"
    assert out2 == [], "Track should still be tentative on frame 2"
    assert len(out3) == 1, "Track should be confirmed and reported on frame 3"
    assert out3[0].id == 1
    assert out3[0].time_since_update == 0


def test_track_id_is_stable_across_frames():
    """Once confirmed, a track's ID should not change frame to frame
    even though its position is moving
    """
    tr = Tracker(min_hits=3, max_age=30)
    ids_seen = []
    for t in range(10):
        cx = 100 + 10 * t
        out = tr.step(np.array([box(cx, 100)]))
        if out:
            ids_seen.append(out[0].id)

    # Frames 1 and 2 produce no output, so we have 8 outputs
    assert len(ids_seen) == 8
    assert all(i == ids_seen[0] for i in ids_seen), (
        f"Track ID changed during the clip: {ids_seen}"
    )


# Multiple tracks


def test_two_separate_objects_get_different_ids():
    """Two well-separated objects, both moving, should be tracked with
    distinct IDs that never swap.
    """
    tr = Tracker(min_hits=3, max_age=30)
    last_assignment = None
    # Run for 8 frames so both tracks are confirmed and reported
    for t in range(8):
        dets = np.array([box(100 + 5 * t, 100), box(500 - 5 * t, 100)])
        out = tr.step(dets)
        if not out:
            continue
        # Sort by x position so we can compare frame to frame
        out_sorted = sorted(out, key=lambda o: o.bbox[0])
        ids_now = (out_sorted[0].id, out_sorted[1].id)
        if last_assignment is not None:
            assert ids_now == last_assignment, (
                f"IDs swapped: was {last_assignment}, now {ids_now}"
            )
        last_assignment = ids_now

    # Confirm we have two distinct IDs total
    assert last_assignment is not None
    assert last_assignment[0] != last_assignment[1]


# Occlusion handling


def test_id_preserved_through_brief_occlusion():
    """A confirmed track that briefly disappears 
    should keep its ID when it reappears, as long as we
    stay within max_age
    """
    tr = Tracker(min_hits=3, max_age=10)

    # Establish a confirmed track over 5 frames
    for t in range(5):
        tr.step(np.array([box(100 + 5 * t, 100)]))
    out_before = tr.step(np.array([box(125, 100)]))
    assert len(out_before) == 1
    id_before = out_before[0].id

    # Simulate 3 frames of occlusion (no detections)
    for _ in range(3):
        out = tr.step(np.empty((0, 4)))
        # Confirmed track should still be reported using its predicted box
        assert len(out) == 1
        assert out[0].id == id_before
        assert out[0].time_since_update > 0

    # Player reappears and the velocity model has been extrapolating, so
    # the predicted position is now around x = 125 + 4*5 = 145
    out_after = tr.step(np.array([box(150, 100)]))
    assert len(out_after) == 1
    assert out_after[0].id == id_before, (
        f"Track ID changed across occlusion: {id_before} -> {out_after[0].id}"
    )


def test_track_deleted_after_max_age_misses():
    """If a confirmed track goes unmatched for more than max_age frames,
    it should be deleted, and a new detection in the same area should
    receive a new ID rather than the old one
    """
    tr = Tracker(min_hits=3, max_age=5)

    # Establish a confirmed track
    for t in range(5):
        tr.step(np.array([box(100, 100)]))
    out = tr.step(np.array([box(100, 100)]))
    original_id = out[0].id

    # Now feed empty detections for max_age + 2 = 7 frames. The track
    # should disappear from output (and from the internal track list)
    # somewhere in there
    for _ in range(tr.max_age + 2):
        tr.step(np.empty((0, 4)))

    # Internal state: the track list should be empty
    assert len(tr.tracks) == 0, (
        f"Track should have been deleted, but {len(tr.tracks)} remain"
    )

    # A new detection in the same place should get a new, larger ID.
    # We need min_hits frames before it shows up in output
    new_id = None
    for t in range(5):
        out = tr.step(np.array([box(100, 100)]))
        if out:
            new_id = out[0].id
            break
    assert new_id is not None
    assert new_id > original_id


# Spurious detections


def test_spurious_single_frame_detection_never_reported():
    """A detector blip that appears for one frame should not produce any
    output, since min_hits=3 requires three consecutive matches
    """
    tr = Tracker(min_hits=3, max_age=30)

    out = tr.step(np.array([box(100, 100)]))
    assert out == []
    # No more detections
    for _ in range(20):
        out = tr.step(np.empty((0, 4)))
        assert out == [], "Tentative track was reported despite never confirming"


# Reset behavior


def test_reset_clears_state_between_clips():
    """After reset(), IDs should restart at 1 and the frame counter at 0"""
    tr = Tracker(min_hits=3, max_age=30)
    for t in range(5):
        tr.step(np.array([box(100, 100)]))
    assert len(tr.tracks) >= 1
    assert tr._next_id > 1

    tr.reset()
    assert tr.tracks == []
    assert tr._next_id == 1
    assert tr._frame_count == 0

    # Run again from scratch so new track should get ID 1
    for t in range(3):
        out = tr.step(np.array([box(200, 200)]))
    assert out[0].id == 1
