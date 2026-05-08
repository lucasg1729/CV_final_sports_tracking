"""End-to-end test for the tracking pipeline.

Builds a minimal SportsMOT-style fixture on disk: a fake clip directory
with a seqinfo.ini, and a cached detection .npz file. Runs the tracker
through scripts.run_tracker.run_clip, and verifies that the output MOT
file has the structure we expect.

This catches integration bugs like wrong frame indexing, wrong coordinate
conversion, the runner not actually writing output, etc.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sports_tracker.detector import FrameDetections, save_detections
from sports_tracker.mot_io import read_mot_file


def make_fake_clip(
    base_dir: Path,
    clip_name: str = "v_fake_001",
    n_frames: int = 10,
    width: int = 1280,
    height: int = 720,
) -> Path:
    """Create a minimal SportsMOT-style clip directory

    Only includes seqinfo.ini and gt we don't actually need the JPG
    frames for the runner to work, since detections come from the cache
    """
    clip_dir = base_dir / "val" / clip_name
    (clip_dir / "img1").mkdir(parents=True, exist_ok=True)
    (clip_dir / "gt").mkdir(parents=True, exist_ok=True)

    seqinfo = clip_dir / "seqinfo.ini"
    seqinfo.write_text(
        "[Sequence]\n"
        f"name={clip_name}\n"
        "imDir=img1\n"
        "frameRate=25\n"
        f"seqLength={n_frames}\n"
        f"imWidth={width}\n"
        f"imHeight={height}\n"
        "imExt=.jpg\n"
    )
    return clip_dir


def make_fake_cache(
    cache_path: Path,
    clip_name: str,
    n_frames: int,
    *,
    box_a_at_t,  # callable: t -> (x1, y1, x2, y2)
    box_b_at_t=None,
) -> None:
    """Write a synthetic detection cache for two moving objects"""
    frame_dets = []
    for t in range(1, n_frames + 1):
        boxes = [box_a_at_t(t)]
        scores = [0.9]
        if box_b_at_t is not None:
            boxes.append(box_b_at_t(t))
            scores.append(0.9)
        frame_dets.append(
            FrameDetections(
                frame_idx=t,
                boxes=np.array(boxes, dtype=np.float32),
                scores=np.array(scores, dtype=np.float32),
            )
        )
    save_detections(cache_path, frame_dets, {"clip": clip_name, "model": "synthetic"})


def _import_run_clip():
    """run_tracker.py is in scripts/, so we have to add scripts to sys.path"""
    import sys

    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from run_tracker import run_clip  # type: ignore

    return run_clip


def test_runner_produces_well_formed_mot_output(tmp_path: Path):
    """Run the runner end-to-end on a synthetic clip with two well-separated
    moving objects. Confirm the output file exists, has rows, and the rows
    parse as valid MOT format
    """
    sportsmot_root = tmp_path / "dataset"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    results_dir = tmp_path / "results"

    clip = "v_fake_001"
    n_frames = 15
    make_fake_clip(sportsmot_root, clip, n_frames=n_frames)

    # Two objects moving in opposite directions across the frame
    cache_path = cache_dir / f"{clip}.npz"
    make_fake_cache(
        cache_path,
        clip,
        n_frames,
        box_a_at_t=lambda t: (100 + 5 * t, 100, 150 + 5 * t, 200),
        box_b_at_t=lambda t: (800 - 5 * t, 100, 850 - 5 * t, 200),
    )

    run_clip = _import_run_clip()
    out_path = results_dir / f"{clip}.txt"
    stats = run_clip(
        clip,
        cache_path,
        sportsmot_root,
        split="val",
        out_path=out_path,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        det_conf=0.3,
    )

    # The output file should exist and contain rows
    assert out_path.exists(), "Tracker did not produce an output file"
    rows = read_mot_file(out_path)
    assert len(rows) > 0, "Output file is empty"

    # Frame indices in the output should fall within [1, n_frames]
    frames_seen = {r.frame for r in rows}
    assert frames_seen.issubset(set(range(1, n_frames + 1)))

    # We have 2 distinct moving objects, so we should see exactly 2 unique IDs
    unique_ids = {r.track_id for r in rows}
    assert len(unique_ids) == 2, f"Expected 2 unique IDs, got {unique_ids}"

    # Each ID should track its own object across multiple frames
    assert stats["n_unique_tracks"] == 2
    assert stats["n_frames"] == n_frames


def test_runner_handles_no_detections(tmp_path: Path):
    """A clip where the cache has zero detections in every frame should
    still produce a (possibly empty) output file without crashing
    """
    sportsmot_root = tmp_path / "dataset"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    results_dir = tmp_path / "results"

    clip = "v_empty_001"
    n_frames = 5
    make_fake_clip(sportsmot_root, clip, n_frames=n_frames)

    # Cache with no detections at all
    cache_path = cache_dir / f"{clip}.npz"
    save_detections(cache_path, [], {"clip": clip})

    run_clip = _import_run_clip()
    out_path = results_dir / f"{clip}.txt"
    stats = run_clip(
        clip,
        cache_path,
        sportsmot_root,
        split="val",
        out_path=out_path,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        det_conf=0.3,
    )

    assert out_path.exists()
    assert read_mot_file(out_path) == []
    assert stats["n_unique_tracks"] == 0
    assert stats["total_detections"] == 0


def test_runner_iterates_all_frames_even_when_some_have_no_detections(
    tmp_path: Path,
):
    """The runner must use the clip's seqLength for the loop, not the
    set of frames with detections. Otherwise a track that's matched on
    frame 1 and again on frame 10 (with detection-free frames between)
    wouldn't age correctly.
    """
    sportsmot_root = tmp_path / "dataset"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    results_dir = tmp_path / "results"

    clip = "v_gappy_001"
    n_frames = 8
    make_fake_clip(sportsmot_root, clip, n_frames=n_frames)

    # Detection only on the first 4 frames then nothing
    cache_path = cache_dir / f"{clip}.npz"
    make_fake_cache(
        cache_path,
        clip,
        n_frames=4,  # only writes 4 frames of detections
        box_a_at_t=lambda t: (100, 100, 150, 200),
    )

    run_clip = _import_run_clip()
    out_path = results_dir / f"{clip}.txt"
    stats = run_clip(
        clip,
        cache_path,
        sportsmot_root,
        split="val",
        out_path=out_path,
        max_age=2,  # tight max_age so we can verify aging happened
        min_hits=3,
        iou_threshold=0.3,
        det_conf=0.3,
    )

    # Report covers all 8 frames (the run iterated all of them)
    assert stats["n_frames"] == n_frames

    # The track was confirmed by frame 3, then ran out of detections
    # after frame 4. With max_age=2, it should be reported with a
    # predicted box for at most 2 more frames (5 and 6), then disappear
    rows = read_mot_file(out_path)
    frames_seen = sorted({r.frame for r in rows})
    assert max(frames_seen) <= 6, (
        f"Track was still being reported on frame {max(frames_seen)}, "
        f"but max_age=2 should have killed it by frame 7"
    )
    assert max(frames_seen) >= 4, "Track should have been reported at least through frame 4"
