"""Pytest-free end-to-end check of the tracking runner.

Builds a synthetic SportsMOT-style fixture and runs the tracker through
scripts.run_tracker.run_clip. Verifies that output files are well-formed
and that the runner correctly iterates ALL frames (not just frames with
detections), which is essential for correct track aging.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

# Make package + scripts importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sports_tracker.detector import FrameDetections, save_detections
from sports_tracker.mot_io import read_mot_file
from run_tracker import run_clip  # type: ignore


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        sys.exit(1)


def make_clip(base_dir: Path, clip: str, n_frames: int) -> None:
    cd = base_dir / "val" / clip
    (cd / "img1").mkdir(parents=True, exist_ok=True)
    (cd / "gt").mkdir(parents=True, exist_ok=True)
    (cd / "seqinfo.ini").write_text(
        "[Sequence]\n"
        f"name={clip}\nimDir=img1\nframeRate=25\n"
        f"seqLength={n_frames}\nimWidth=1280\nimHeight=720\nimExt=.jpg\n"
    )


def make_cache(cache_path: Path, clip: str, n_frames: int, box_funcs):
    """box_funcs: list of callables, each takes t and returns (x1,y1,x2,y2)."""
    fds = []
    for t in range(1, n_frames + 1):
        boxes = np.array([f(t) for f in box_funcs], dtype=np.float32)
        scores = np.full(len(box_funcs), 0.9, dtype=np.float32)
        fds.append(FrameDetections(frame_idx=t, boxes=boxes, scores=scores))
    save_detections(cache_path, fds, {"clip": clip})


with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)

    print("Two-object end-to-end run")
    print("-" * 60)

    sportsmot_root = tmp_path / "dataset"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    results_dir = tmp_path / "results"

    clip = "v_fake_001"
    n_frames = 15
    make_clip(sportsmot_root, clip, n_frames)
    cache_path = cache_dir / f"{clip}.npz"
    make_cache(
        cache_path,
        clip,
        n_frames,
        box_funcs=[
            lambda t: (100 + 5 * t, 100, 150 + 5 * t, 200),
            lambda t: (800 - 5 * t, 100, 850 - 5 * t, 200),
        ],
    )

    out_path = results_dir / f"{clip}.txt"
    stats = run_clip(
        clip, cache_path, sportsmot_root,
        split="val", out_path=out_path,
        max_age=30, min_hits=3, iou_threshold=0.3, det_conf=0.3,
    )

    check("output file exists", out_path.exists())
    rows = read_mot_file(out_path)
    check("output has rows", len(rows) > 0, f"got {len(rows)}")
    frames_seen = {r.frame for r in rows}
    check(
        "frame indices within [1, n_frames]",
        frames_seen.issubset(set(range(1, n_frames + 1))),
    )
    unique_ids = {r.track_id for r in rows}
    check("exactly 2 unique track IDs", len(unique_ids) == 2, f"got {unique_ids}")
    check("stats: n_frames", stats["n_frames"] == n_frames)
    check("stats: n_unique_tracks", stats["n_unique_tracks"] == 2)

    print()
    print("Empty-cache run")
    print("-" * 60)

    clip = "v_empty_001"
    n_frames = 5
    make_clip(sportsmot_root, clip, n_frames)
    cache_path = cache_dir / f"{clip}.npz"
    save_detections(cache_path, [], {"clip": clip})

    out_path = results_dir / f"{clip}.txt"
    stats = run_clip(
        clip, cache_path, sportsmot_root,
        split="val", out_path=out_path,
        max_age=30, min_hits=3, iou_threshold=0.3, det_conf=0.3,
    )

    check("empty cache: output file exists", out_path.exists())
    check("empty cache: output is empty", read_mot_file(out_path) == [])
    check("empty cache: 0 unique tracks", stats["n_unique_tracks"] == 0)
    check("empty cache: 0 detections", stats["total_detections"] == 0)

    print()
    print("Aging through detection gap")
    print("-" * 60)

    clip = "v_gappy_001"
    n_frames = 8
    make_clip(sportsmot_root, clip, n_frames)
    cache_path = cache_dir / f"{clip}.npz"
    make_cache(
        cache_path,
        clip,
        n_frames=4,  # detections only on frames 1-4
        box_funcs=[lambda t: (100, 100, 150, 200)],
    )

    out_path = results_dir / f"{clip}.txt"
    stats = run_clip(
        clip, cache_path, sportsmot_root,
        split="val", out_path=out_path,
        max_age=2, min_hits=3, iou_threshold=0.3, det_conf=0.3,
    )

    check("aging: stats reports all frames in clip", stats["n_frames"] == 8)
    rows = read_mot_file(out_path)
    frames_seen = sorted({r.frame for r in rows})
    check(
        "aging: track stops being reported by frame 7",
        max(frames_seen) <= 6,
        f"max frame seen = {max(frames_seen)}",
    )
    check(
        "aging: track was reported at least through frame 4",
        max(frames_seen) >= 4,
        f"max frame seen = {max(frames_seen)}",
    )

print()
print("All checks passed.")
