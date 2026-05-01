"""Run the Kalman filter unit tests without needing pytest.

This is a one-off verification script. The real test suite lives in
tests/test_kalman.py and runs via pytest in your local environment.
"""

from __future__ import annotations

import sys
import traceback

import numpy as np

sys.path.insert(0, ".")
from sports_tracker.kalman import BoxKalmanFilter, bbox_to_z, x_to_bbox


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        sys.exit(1)


print("Conversion functions")
print("-" * 60)

bbox = np.array([0.0, 0.0, 100.0, 200.0])
z = bbox_to_z(bbox)
check("bbox_to_z center u", abs(z[0] - 50.0) < 1e-9, f"got {z[0]}")
check("bbox_to_z center v", abs(z[1] - 100.0) < 1e-9, f"got {z[1]}")
check("bbox_to_z scale", abs(z[2] - 20000.0) < 1e-9, f"got {z[2]}")
check("bbox_to_z aspect ratio", abs(z[3] - 0.5) < 1e-9, f"got {z[3]}")

bbox = np.array([10.0, 20.0, 110.0, 220.0])
x = np.zeros(7)
x[:4] = bbox_to_z(bbox)
recovered = x_to_bbox(x)
check(
    "bbox -> z -> bbox roundtrip",
    np.allclose(recovered, bbox, atol=1e-9),
    f"got {recovered}",
)

x_bad = np.array([100.0, 100.0, -5.0, 0.5, 0.0, 0.0, 0.0])
recovered_bad = x_to_bbox(x_bad)
check(
    "x_to_bbox handles negative scale",
    np.all(np.isfinite(recovered_bad)),
    f"got {recovered_bad}",
)

print()
print("Kalman filter behavior")
print("-" * 60)

bbox = np.array([100.0, 200.0, 200.0, 400.0])
kf = BoxKalmanFilter(bbox)
check(
    "init: state matches input bbox",
    np.allclose(kf.get_state(), bbox, atol=1e-6),
)

predicted = kf.predict()
check(
    "predict with zero velocity does not move",
    np.allclose(predicted, bbox, atol=1e-6),
    f"moved to {predicted}",
)

# Constant velocity learning test.
vx, vy = 10.0, 5.0
boxes = []
for t in range(20):
    x1 = 100.0 + vx * t
    y1 = 200.0 + vy * t
    boxes.append(np.array([x1, y1, x1 + 50.0, y1 + 100.0]))

kf = BoxKalmanFilter(boxes[0])
for t in range(1, 15):
    kf.predict()
    kf.update(boxes[t])

check(
    "filter learns u_dot",
    abs(kf.x[4] - vx) < 1.0,
    f"expected ~{vx}, got {kf.x[4]:.3f}",
)
check(
    "filter learns v_dot",
    abs(kf.x[5] - vy) < 1.0,
    f"expected ~{vy}, got {kf.x[5]:.3f}",
)

# Occlusion extrapolation test.
boxes_occ = [
    np.array([100.0 + vx * t, 200.0, 150.0 + vx * t, 300.0]) for t in range(15)
]
kf = BoxKalmanFilter(boxes_occ[0])
for t in range(1, 10):
    kf.predict()
    kf.update(boxes_occ[t])

for _ in range(3):
    kf.predict()

expected_u = bbox_to_z(boxes_occ[9])[0] + 3 * vx
actual_u = kf.x[0]
check(
    "predict extrapolates during occlusion",
    abs(actual_u - expected_u) < 2.0,
    f"expected ~{expected_u:.2f}, got {actual_u:.3f}",
)
check(
    "occlusion increments time_since_update",
    kf.time_since_update == 3,
    f"got {kf.time_since_update}",
)
check(
    "occlusion breaks hit streak",
    kf.hit_streak == 0,
    f"got {kf.hit_streak}",
)

# Covariance shrinks after update.
bbox = np.array([100.0, 200.0, 200.0, 400.0])
kf = BoxKalmanFilter(bbox)
p_before = kf.P[0, 0] + kf.P[1, 1]
kf.predict()
kf.update(bbox)
p_after = kf.P[0, 0] + kf.P[1, 1]
check(
    "update reduces position covariance",
    p_after < p_before,
    f"before {p_before:.3f}, after {p_after:.3f}",
)

# Bookkeeping counters.
bbox = np.array([100.0, 200.0, 200.0, 400.0])
kf = BoxKalmanFilter(bbox)
check("init hits", kf.hits == 1)
check("init age", kf.age == 0)
check("init time_since_update", kf.time_since_update == 0)
check("init hit_streak", kf.hit_streak == 1)

kf.predict()
check("after predict: age", kf.age == 1)
check("after predict: time_since_update", kf.time_since_update == 1)

kf.update(bbox)
check("after update: time_since_update reset", kf.time_since_update == 0)
check("after update: hits", kf.hits == 2)
check("after update: hit_streak", kf.hit_streak == 2)

print()
print("All checks passed.")
