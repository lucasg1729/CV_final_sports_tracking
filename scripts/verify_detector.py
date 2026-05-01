"""Run the detector module's cache I/O checks without pytest."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sports_tracker.detector import (
    FrameDetections,
    load_detections,
    save_detections,
)


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        sys.exit(1)


def make_fd(frame_idx, *box_score_pairs):
    if not box_score_pairs:
        return FrameDetections(
            frame_idx=frame_idx,
            boxes=np.empty((0, 4), dtype=np.float32),
            scores=np.empty(0, dtype=np.float32),
        )
    boxes = np.array([bs[0] for bs in box_score_pairs], dtype=np.float32)
    scores = np.array([bs[1] for bs in box_score_pairs], dtype=np.float32)
    return FrameDetections(frame_idx=frame_idx, boxes=boxes, scores=scores)


with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)

    print("FrameDetections.filter")
    print("-" * 60)

    fd = make_fd(5, ([0, 0, 10, 10], 0.9), ([20, 20, 30, 30], 0.4), ([40, 40, 50, 50], 0.1))
    f = fd.filter(0.5)
    check("filter keeps frame_idx", f.frame_idx == 5)
    check("filter drops below threshold", f.boxes.shape == (1, 4))
    check("filter score correct", abs(f.scores[0] - 0.9) < 1e-6)

    fd = make_fd(1, ([0, 0, 10, 10], 0.5))
    check("filter at threshold (>=) keeps", len(fd.filter(0.5).boxes) == 1)
    check("filter just above threshold drops", len(fd.filter(0.50001).boxes) == 0)

    fd = make_fd(1, ([0, 0, 10, 10], 0.1))
    out = fd.filter(0.9)
    check("filter all below: empty boxes shape (0, 4)", out.boxes.shape == (0, 4))
    check("filter all below: empty scores shape (0,)", out.scores.shape == (0,))

    print()
    print("Cache I/O round-trip")
    print("-" * 60)

    # Basic round-trip.
    fds = [
        make_fd(1, ([10, 20, 30, 40], 0.9), ([50, 60, 70, 80], 0.8)),
        make_fd(2, ([15, 25, 35, 45], 0.7)),
    ]
    meta_in = {"clip": "test_clip", "model": "yolov8n.pt", "n_frames": 2}
    cache = tmp_path / "basic.npz"
    save_detections(cache, fds, meta_in)
    check("basic save: file written", cache.exists())

    loaded, meta_out = load_detections(cache)
    check("basic load: meta preserved", meta_out == meta_in, f"got {meta_out}")
    check("basic load: 2 frames", len(loaded) == 2)
    check("basic load: frame 1 has 2 boxes", loaded[0].boxes.shape == (2, 4))
    check(
        "basic load: frame 1 boxes equal",
        np.array_equal(loaded[0].boxes, fds[0].boxes),
    )
    check("basic load: frame 2 has 1 box", loaded[1].boxes.shape == (1, 4))

    # Empty round-trip.
    cache = tmp_path / "empty.npz"
    save_detections(cache, [], {"clip": "empty"})
    loaded, meta = load_detections(cache)
    check("empty save/load: no frames", loaded == [])
    check("empty save/load: meta preserved", meta == {"clip": "empty"})

    # Sparse mode skips frames with no detections.
    fds = [
        make_fd(1, ([0, 0, 10, 10], 0.9)),
        make_fd(2),  # empty
        make_fd(3, ([5, 5, 15, 15], 0.8)),
    ]
    cache = tmp_path / "sparse.npz"
    save_detections(cache, fds, {})
    loaded, _ = load_detections(cache)
    check(
        "sparse mode: empty frame skipped",
        [fd.frame_idx for fd in loaded] == [1, 3],
        f"got {[fd.frame_idx for fd in loaded]}",
    )

    # Dense mode fills in missing frames.
    loaded, _ = load_detections(cache, n_frames=4)
    check("dense mode: all 4 frames present", len(loaded) == 4)
    check("dense mode: frame 1 has det", loaded[0].frame_idx == 1 and len(loaded[0].boxes) == 1)
    check("dense mode: frame 2 empty", loaded[1].frame_idx == 2 and len(loaded[1].boxes) == 0)
    check("dense mode: frame 3 has det", loaded[2].frame_idx == 3 and len(loaded[2].boxes) == 1)
    check("dense mode: frame 4 empty", loaded[3].frame_idx == 4 and len(loaded[3].boxes) == 0)

    # min_conf at load time.
    fds = [
        make_fd(
            1,
            ([0, 0, 10, 10], 0.9),
            ([20, 20, 30, 30], 0.4),
            ([40, 40, 50, 50], 0.1),
        )
    ]
    cache = tmp_path / "filtered.npz"
    save_detections(cache, fds, {})
    loaded_all, _ = load_detections(cache, min_conf=0.0)
    check("load min_conf=0: all 3 kept", len(loaded_all[0].boxes) == 3)
    loaded_strict, _ = load_detections(cache, min_conf=0.5)
    check("load min_conf=0.5: 1 kept", len(loaded_strict[0].boxes) == 1)
    check(
        "load min_conf=0.5: kept the high-score one",
        abs(loaded_strict[0].scores[0] - 0.9) < 1e-6,
    )

    # min_conf can empty a frame in sparse mode.
    fds = [
        make_fd(1, ([0, 0, 10, 10], 0.2)),
        make_fd(2, ([5, 5, 15, 15], 0.9)),
    ]
    cache = tmp_path / "filter-out.npz"
    save_detections(cache, fds, {})
    loaded, _ = load_detections(cache, min_conf=0.5)
    check(
        "min_conf empties frame 1 in sparse mode",
        [fd.frame_idx for fd in loaded] == [2],
    )

    # Save sorts unsorted input.
    fds = [
        make_fd(3, ([3, 3, 13, 13], 0.5)),
        make_fd(1, ([1, 1, 11, 11], 0.5)),
        make_fd(2, ([2, 2, 12, 12], 0.5)),
    ]
    cache = tmp_path / "unsorted.npz"
    save_detections(cache, fds, {})
    loaded, _ = load_detections(cache)
    check(
        "save sorts unsorted frames",
        [fd.frame_idx for fd in loaded] == [1, 2, 3],
    )

print()
print("All checks passed.")
