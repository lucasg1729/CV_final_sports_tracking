"""Run the tracker class checks without needing pytest"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sports_tracker.tracker import Tracker


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        sys.exit(1)


def box(cx, cy, w=50.0, h=100.0):
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


print("Initialization and configuration")
print("-" * 60)

try:
    Tracker(max_age=0)
    raised = False
except ValueError:
    raised = True
check("max_age=0 raises ValueError", raised)

try:
    Tracker(min_hits=0)
    raised = False
except ValueError:
    raised = True
check("min_hits=0 raises ValueError", raised)

tr = Tracker()
check("empty input -> empty output", tr.step(np.empty((0, 4))) == [])

print()
print("Track promotion")
print("-" * 60)

tr = Tracker(min_hits=3, max_age=30)
out1 = tr.step(np.array([box(100, 100)]))
out2 = tr.step(np.array([box(105, 100)]))
out3 = tr.step(np.array([box(110, 100)]))

check("frame 1: track tentative, not reported", out1 == [])
check("frame 2: track still tentative", out2 == [])
check(
    "frame 3: track confirmed and reported",
    len(out3) == 1 and out3[0].id == 1 and out3[0].time_since_update == 0,
    f"got {out3}",
)

# Stable IDs
tr = Tracker(min_hits=3, max_age=30)
ids_seen = []
for t in range(10):
    cx = 100 + 10 * t
    out = tr.step(np.array([box(cx, 100)]))
    if out:
        ids_seen.append(out[0].id)
check("stable ID across 10 frames", len(ids_seen) == 8 and all(i == ids_seen[0] for i in ids_seen),
      f"ids = {ids_seen}")

print()
print("Multiple tracks")
print("-" * 60)

tr = Tracker(min_hits=3, max_age=30)
last_assignment = None
swap_detected = False
for t in range(8):
    dets = np.array([box(100 + 5 * t, 100), box(500 - 5 * t, 100)])
    out = tr.step(dets)
    if not out:
        continue
    out_sorted = sorted(out, key=lambda o: o.bbox[0])
    ids_now = (out_sorted[0].id, out_sorted[1].id)
    if last_assignment is not None and ids_now != last_assignment:
        swap_detected = True
    last_assignment = ids_now
check("two separate objects: no ID swap", not swap_detected)
check(
    "two separate objects: distinct IDs",
    last_assignment is not None and last_assignment[0] != last_assignment[1],
    f"got {last_assignment}",
)

print()
print("Occlusion handling")
print("-" * 60)

tr = Tracker(min_hits=3, max_age=10)
for t in range(5):
    tr.step(np.array([box(100 + 5 * t, 100)]))
out_before = tr.step(np.array([box(125, 100)]))
check("track established before occlusion", len(out_before) == 1)
id_before = out_before[0].id

# 3 frames of occlusion
all_kept_id = True
for _ in range(3):
    out = tr.step(np.empty((0, 4)))
    if not (len(out) == 1 and out[0].id == id_before and out[0].time_since_update > 0):
        all_kept_id = False
check("ID preserved during 3-frame occlusion", all_kept_id)

# Reappear
out_after = tr.step(np.array([box(150, 100)]))
check(
    "ID matches across occlusion",
    len(out_after) == 1 and out_after[0].id == id_before,
    f"id_before={id_before}, after={out_after}",
)

# Track deleted after max_age
tr = Tracker(min_hits=3, max_age=5)
for t in range(5):
    tr.step(np.array([box(100, 100)]))
tr.step(np.array([box(100, 100)]))
for _ in range(tr.max_age + 2):
    tr.step(np.empty((0, 4)))
check("track deleted after max_age + 2 misses", len(tr.tracks) == 0)

# New ID on reappearance after deletion
new_id = None
for t in range(5):
    out = tr.step(np.array([box(100, 100)]))
    if out:
        new_id = out[0].id
        break
check("new ID assigned after deletion", new_id is not None and new_id > 1)

print()
print("Spurious detections")
print("-" * 60)

tr = Tracker(min_hits=3, max_age=30)
out = tr.step(np.array([box(100, 100)]))
check("frame-1 spurious: no output", out == [])
ever_reported = False
for _ in range(20):
    out = tr.step(np.empty((0, 4)))
    if out:
        ever_reported = True
        break
check("never-confirmed track stays silent", not ever_reported)

print()
print("Reset behavior")
print("-" * 60)

tr = Tracker(min_hits=3, max_age=30)
for t in range(5):
    tr.step(np.array([box(100, 100)]))
check("tracks before reset", len(tr.tracks) >= 1)
check("next id incremented before reset", tr._next_id > 1)

tr.reset()
check("reset clears tracks", tr.tracks == [])
check("reset resets ID counter", tr._next_id == 1)
check("reset resets frame counter", tr._frame_count == 0)

last_out = None
for t in range(3):
    last_out = tr.step(np.array([box(200, 200)]))
check("post-reset: new track starts at ID 1", len(last_out) == 1 and last_out[0].id == 1)

print()
print("All checks passed.")
