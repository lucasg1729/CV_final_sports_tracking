"""Unit tests for the Kalman filter module.

These run on synthetic trajectories where the ground truth is known
exactly. If any of these fail, nothing downstream can be trusted.
"""

from __future__ import annotations

import numpy as np
import pytest

from sports_tracker.kalman import (
    BoxKalmanFilter,
    bbox_to_z,
    x_to_bbox,
)


# --- Bounding box <-> measurement conversion --------------------------------


def test_bbox_to_z_centers_correctly():
    """A 100x200 box at origin has center (50, 100), area 20000, ratio 0.5."""
    bbox = np.array([0.0, 0.0, 100.0, 200.0])
    z = bbox_to_z(bbox)
    assert z[0] == pytest.approx(50.0)
    assert z[1] == pytest.approx(100.0)
    assert z[2] == pytest.approx(20000.0)
    assert z[3] == pytest.approx(0.5)


def test_bbox_z_roundtrip():
    """Going box -> z -> state -> box should recover the original box exactly."""
    bbox = np.array([10.0, 20.0, 110.0, 220.0])
    z = bbox_to_z(bbox)
    # Embed z into a state vector with zero velocities.
    x = np.zeros(7)
    x[:4] = z
    recovered = x_to_bbox(x)
    np.testing.assert_allclose(recovered, bbox, atol=1e-9)


def test_x_to_bbox_handles_degenerate_scale():
    """A negative or zero scale should not produce NaN. We clip and continue."""
    x = np.array([100.0, 100.0, -5.0, 0.5, 0.0, 0.0, 0.0])
    bbox = x_to_bbox(x)
    assert np.all(np.isfinite(bbox)), "Degenerate scale produced NaN/inf"


# --- Kalman filter behavior --------------------------------------------------


def test_init_sets_position_from_bbox():
    """After init, the predicted box should match the input box (no motion yet)."""
    bbox = np.array([100.0, 200.0, 200.0, 400.0])
    kf = BoxKalmanFilter(bbox)
    np.testing.assert_allclose(kf.get_state(), bbox, atol=1e-6)


def test_predict_with_zero_velocity_does_not_move():
    """First predict() with no updates should leave the box where it started."""
    bbox = np.array([100.0, 200.0, 200.0, 400.0])
    kf = BoxKalmanFilter(bbox)
    predicted = kf.predict()
    np.testing.assert_allclose(predicted, bbox, atol=1e-6)


def test_filter_learns_constant_velocity():
    """Feed the filter a box moving at constant velocity. After several updates,
    the predicted velocity should be close to the true velocity, and predictions
    should anticipate where the box is going next.
    """
    # Ground truth: a 50x100 box moving right at 10 px/frame, down at 5 px/frame.
    vx, vy = 10.0, 5.0
    boxes = []
    for t in range(20):
        x1 = 100.0 + vx * t
        y1 = 200.0 + vy * t
        boxes.append(np.array([x1, y1, x1 + 50.0, y1 + 100.0]))

    kf = BoxKalmanFilter(boxes[0])
    # Run predict/update for the first 15 frames.
    for t in range(1, 15):
        kf.predict()
        kf.update(boxes[t])

    # After ~15 observations of constant motion, the velocity estimates
    # should have converged. Use a generous tolerance because the filter
    # is biased toward measurement noise early on.
    assert kf.x[4] == pytest.approx(vx, abs=1.0), (
        f"Expected u_dot ≈ {vx}, got {kf.x[4]:.3f}"
    )
    assert kf.x[5] == pytest.approx(vy, abs=1.0), (
        f"Expected v_dot ≈ {vy}, got {kf.x[5]:.3f}"
    )


def test_predict_extrapolates_during_occlusion():
    """During an occlusion (consecutive predicts with no updates), the filter
    should still extrapolate forward at the learned velocity. This is the
    whole point of having a motion model.
    """
    vx = 10.0
    boxes = [
        np.array([100.0 + vx * t, 200.0, 150.0 + vx * t, 300.0]) for t in range(15)
    ]
    kf = BoxKalmanFilter(boxes[0])
    # Run normally for 10 frames so the filter learns velocity.
    for t in range(1, 10):
        kf.predict()
        kf.update(boxes[t])

    # Now occlude: predict 3 frames with no updates.
    for _ in range(3):
        kf.predict()

    # Center should have advanced by roughly 3 * vx pixels from the last update.
    expected_u = bbox_to_z(boxes[9])[0] + 3 * vx
    actual_u = kf.x[0]
    assert actual_u == pytest.approx(expected_u, abs=2.0), (
        f"Expected extrapolated center near {expected_u}, got {actual_u:.3f}"
    )

    # The lost-frame counter should reflect three missed updates.
    assert kf.time_since_update == 3
    # Hit streak is broken once a frame goes by without an update.
    assert kf.hit_streak == 0


def test_update_reduces_position_uncertainty():
    """The diagonal of P (state covariance) should shrink for observed
    components after an update. This is a basic sanity check that the
    Kalman gain is being applied with the right sign.
    """
    bbox = np.array([100.0, 200.0, 200.0, 400.0])
    kf = BoxKalmanFilter(bbox)
    p_position_before = kf.P[0, 0] + kf.P[1, 1]
    kf.predict()
    kf.update(bbox)
    p_position_after = kf.P[0, 0] + kf.P[1, 1]
    assert p_position_after < p_position_before, (
        "Position covariance should decrease after an update"
    )


def test_track_bookkeeping_counters():
    """hits, age, time_since_update, hit_streak should evolve correctly."""
    bbox = np.array([100.0, 200.0, 200.0, 400.0])
    kf = BoxKalmanFilter(bbox)
    assert kf.hits == 1
    assert kf.age == 0
    assert kf.time_since_update == 0
    assert kf.hit_streak == 1

    kf.predict()
    # After a predict with no following update, time_since_update is 1.
    assert kf.age == 1
    assert kf.time_since_update == 1

    kf.update(bbox)
    # update() resets time_since_update and increments hits.
    assert kf.time_since_update == 0
    assert kf.hits == 2
    assert kf.hit_streak == 2
