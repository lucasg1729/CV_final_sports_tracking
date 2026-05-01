"""
Constant-velocity Kalman filter for bounding box tracking.

Following Bewley et al. 2016 (SORT), each tracked object has the state vector

    x = [u, v, s, r, u_dot, v_dot, s_dot]^T

where:
    u, v       = bounding box center coordinates (pixels)
    s          = scale (area = w * h)
    r          = aspect ratio (w / h), assumed constant
    u_dot      = horizontal velocity
    v_dot      = vertical velocity
    s_dot      = rate of change of scale

The aspect ratio r has no velocity component because SORT assumes it is
constant over the track's lifetime. This is a known limitation for sports
(e.g., a player diving for a ball), and a candidate for ablation later.

The measurement vector is z = [u, v, s, r]^T -- we observe position, scale,
and aspect ratio directly from each detection's bounding box, but never
observe velocity (which is what makes a Kalman filter useful here).

References:
    Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016.
    Kalman, "A New Approach to Linear Filtering and Prediction Problems", 1960.
"""

from __future__ import annotations

import numpy as np

# State and measurement dimensions.
STATE_DIM = 7
MEAS_DIM = 4


def bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """Convert a bounding box [x1, y1, x2, y2] into a measurement [u, v, s, r].

    Args:
        bbox: shape (4,), top-left and bottom-right corners.

    Returns:
        z: shape (4,), measurement vector.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    u = x1 + w / 2.0
    v = y1 + h / 2.0
    s = w * h
    # Guard against zero-height boxes that would blow up r.
    r = w / h if h > 0 else 0.0
    return np.array([u, v, s, r], dtype=np.float64)


def x_to_bbox(x: np.ndarray) -> np.ndarray:
    """Convert a state vector x back to a bounding box [x1, y1, x2, y2].

    Recovers w = sqrt(s * r) and h = s / w from scale and aspect ratio.
    Negative or zero scale (which can happen if the filter goes unstable)
    is clipped before the sqrt to avoid NaNs in downstream code.
    """
    u, v, s, r = x[0], x[1], x[2], x[3]
    s = max(float(s), 1e-6)
    r = max(float(r), 1e-6)
    w = np.sqrt(s * r)
    h = s / w
    return np.array([u - w / 2.0, v - h / 2.0, u + w / 2.0, v + h / 2.0], dtype=np.float64)


class BoxKalmanFilter:
    """A constant-velocity Kalman filter for one bounding box.

    The state-transition model assumes constant velocity over one frame:

        u_{t+1}     = u_t + u_dot_t
        v_{t+1}     = v_t + v_dot_t
        s_{t+1}     = s_t + s_dot_t
        r_{t+1}     = r_t                  (aspect ratio constant)
        u_dot_{t+1} = u_dot_t
        v_dot_{t+1} = v_dot_t
        s_dot_{t+1} = s_dot_t

    which gives the F matrix below. The measurement model H pulls out the
    first four state components (u, v, s, r) since we only observe the box.

    Noise covariances Q (process) and R (measurement) follow the values
    used in the SORT reference implementation. They are intentionally
    biased toward trusting detections more than the motion prior, because
    a good detector is more reliable than constant-velocity assumptions
    for non-rigid moving athletes.
    """

    def __init__(self, bbox: np.ndarray):
        """Initialize the filter with an initial bounding box detection.

        The velocity components are initialized to zero (we have no motion
        information yet) and given a large covariance so the first few
        updates can move them quickly.
        """
        # State transition F: 7x7. Identity plus velocity contribution to position.
        self.F = np.eye(STATE_DIM, dtype=np.float64)
        self.F[0, 4] = 1.0  # u += u_dot
        self.F[1, 5] = 1.0  # v += v_dot
        self.F[2, 6] = 1.0  # s += s_dot

        # Measurement matrix H: 4x7. Observe (u, v, s, r) directly.
        self.H = np.zeros((MEAS_DIM, STATE_DIM), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # Measurement noise R. Larger for scale than position because
        # detector boxes are noisier in size than in center location.
        self.R = np.eye(MEAS_DIM, dtype=np.float64)
        self.R[2:, 2:] *= 10.0

        # Initial state covariance P. Velocities are highly uncertain at
        # init (we've only seen one frame), so they get inflated covariance.
        self.P = np.eye(STATE_DIM, dtype=np.float64) * 10.0
        self.P[4:, 4:] *= 1000.0

        # Process noise Q. Small for position/scale, larger for the velocity
        # components and the scale-velocity in particular (athletes change
        # apparent size rapidly as they move toward/away from the camera).
        self.Q = np.eye(STATE_DIM, dtype=np.float64)
        self.Q[4:, 4:] *= 0.01
        self.Q[6, 6] *= 0.01

        # Initial state. Position from the detection; velocities at zero.
        z = bbox_to_z(bbox)
        self.x = np.zeros(STATE_DIM, dtype=np.float64)
        self.x[:MEAS_DIM] = z

        # Track how long this filter has been running and whether it
        # currently has a fresh observation. The tracker uses these to
        # decide track lifecycle (confirmed / tentative / dead).
        self.age = 0
        self.time_since_update = 0
        self.hits = 1  # We've seen at least one detection (the init one).
        self.hit_streak = 1

    def predict(self) -> np.ndarray:
        """Advance the state by one frame.

        x_pred = F @ x
        P_pred = F @ P @ F^T + Q

        Returns the predicted bounding box, which is what the tracker
        uses to compute IOU costs against incoming detections.
        """
        # If scale would go negative after the predict step, zero out the
        # scale velocity to keep the box from inverting. This is the same
        # safeguard used in the SORT reference code.
        if (self.x[2] + self.x[6]) <= 0:
            self.x[6] = 0.0

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        # If we've gone a frame without an update, break the hit streak.
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return x_to_bbox(self.x)

    def update(self, bbox: np.ndarray) -> None:
        """Incorporate a new detection.

        Standard Kalman update:
            y = z - H @ x          (innovation)
            S = H @ P @ H^T + R    (innovation covariance)
            K = P @ H^T @ S^-1     (Kalman gain)
            x = x + K @ y
            P = (I - K @ H) @ P
        """
        z = bbox_to_z(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(STATE_DIM, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P

        # Track bookkeeping: a successful update resets the lost-frame
        # counter and extends the hit streak.
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def get_state(self) -> np.ndarray:
        """Return the current bounding box estimate [x1, y1, x2, y2]."""
        return x_to_bbox(self.x)
