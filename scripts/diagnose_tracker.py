"""Diagnose why a tracker run produced too many unique IDs.

Re-runs the tracker on a clip's cached detections (same parameters as
run_tracker.py) but instruments every track to record its full lifetime.
Prints a summary that helps distinguish association failures from
detector noise.

Usage:
    python scripts/diagnose_tracker.py --clip v_xxx
    python scripts/diagnose_tracker.py --clip v_xxx --det-conf 0.2 --iou-threshold 0.2
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sports_tracker.association import (
    DEFAULT_IOU_THRESHOLD,
    associate,
    iou_matrix,
)
from sports_tracker.config import REPO_ROOT, load_config, resolve_path
from sports_tracker.detector import load_detections
from sports_tracker.kalman import BoxKalmanFilter
from sports_tracker.tracker import (
    DEFAULT_MAX_AGE,
    DEFAULT_MIN_HITS,
)


def get_clip_n_frames(sportsmot_root: Path, split: str, clip_name: str) -> int:
    import configparser

    seqinfo_path = sportsmot_root / split / clip_name / "seqinfo.ini"
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(seqinfo_path)
    return int(parser["Sequence"]["seqLength"])


def diagnose(
    clip_name: str,
    cache_path: Path,
    sportsmot_root: Path,
    split: str,
    *,
    max_age: int,
    min_hits: int,
    iou_threshold: float,
    det_conf: float,
) -> None:
    """Run a tracker on a clip but record every track's lifecycle"""
    n_frames = get_clip_n_frames(sportsmot_root, split, clip_name)
    frame_dets, _ = load_detections(cache_path, min_conf=det_conf, n_frames=n_frames)

    # We'll re-implement a stripped-down version of the tracker loop here
    # so that we can capture per-track histories without modifying the
    # main Tracker class. Each track is a (id, kf, history) tuple.
    next_id = 1
    tracks: list[dict] = []
    histories: dict[int, dict] = {}  # finished tracks for analysis

    # Stats we want to capture across all frames
    total_dets = 0
    total_unmatched_dets = 0
    total_matched_pairs = 0

    # Distribution of IOU values for matched pairs vs near-misses
    matched_ious: list[float] = []
    near_miss_ious: list[float] = []  # best-IOU per unmatched-det, when nonzero

    for fd in frame_dets:
        total_dets += len(fd.boxes)

        predicted_boxes = np.array(
            [t["kf"].predict() for t in tracks], dtype=np.float64
        ).reshape(-1, 4)

        result = associate(
            predicted_boxes, fd.boxes, iou_threshold=iou_threshold
        )

        # Capture IOU stats
        if len(predicted_boxes) and len(fd.boxes):
            iou_mat = iou_matrix(predicted_boxes, fd.boxes)
            for ti, di in result.matches:
                matched_ious.append(float(iou_mat[ti, di]))
                total_matched_pairs += 1
            for di in result.unmatched_detections:
                if iou_mat.shape[0] > 0:
                    best = float(iou_mat[:, di].max())
                    if best > 0:
                        near_miss_ious.append(best)
        total_unmatched_dets += len(result.unmatched_detections)

        for track_idx, det_idx in result.matches:
            tracks[track_idx]["kf"].update(fd.boxes[det_idx])
            tracks[track_idx]["history"].append((fd.frame_idx, "match"))

        # Spawn new tracks
        for det_idx in result.unmatched_detections:
            kf = BoxKalmanFilter(fd.boxes[det_idx])
            tid = next_id
            next_id += 1
            tracks.append(
                {
                    "id": tid,
                    "kf": kf,
                    "history": [(fd.frame_idx, "born")],
                    "born_frame": fd.frame_idx,
                }
            )

        # Record misses for unmatched tracks
        for ti in result.unmatched_tracks:
            tracks[ti]["history"].append((fd.frame_idx, "miss"))

        survivors = []
        for t in tracks:
            if t["kf"].time_since_update > max_age:
                t["died_frame"] = fd.frame_idx
                histories[t["id"]] = t
            else:
                survivors.append(t)
        tracks = survivors

    # Add still-alive tracks at end
    for t in tracks:
        t["died_frame"] = None
        histories[t["id"]] = t

    # Print diagnostic summary
    n_total_tracks = len(histories)
    n_died = sum(1 for t in histories.values() if t["died_frame"] is not None)
    n_alive_at_end = n_total_tracks - n_died

    # Track lifetimes
    lifetimes = []
    for t in histories.values():
        end = t["died_frame"] if t["died_frame"] is not None else n_frames
        lifetimes.append(end - t["born_frame"] + 1)

    # How many of those tracks ever reached confirmation (min_hits matches)?
    n_ever_confirmed = sum(
        1
        for t in histories.values()
        if sum(1 for _, ev in t["history"] if ev == "match")
        >= min_hits  # total matches >= min_hits
    )

    # Distribution of matches per track
    matches_per_track = Counter()
    for t in histories.values():
        n_match = sum(1 for _, ev in t["history"] if ev == "match")
        matches_per_track[n_match] += 1

    def bucket(n):
        if n <= 1:
            return "1"
        if n <= 3:
            return "2-3"
        if n <= 5:
            return "4-5"
        if n <= 10:
            return "6-10"
        if n <= 30:
            return "11-30"
        if n <= 100:
            return "31-100"
        return "100+"

    lifetime_buckets = Counter(bucket(l) for l in lifetimes)
    bucket_order = ["1", "2-3", "4-5", "6-10", "11-30", "31-100", "100+"]

    print(f"Clip                    : {clip_name}")
    print(f"Frames                  : {n_frames}")
    print(f"Total detections        : {total_dets}")
    print(f"  Avg detections/frame  : {total_dets / max(n_frames, 1):.2f}")
    print(f"Matched pairs           : {total_matched_pairs}")
    print(f"Unmatched detections    : {total_unmatched_dets}")
    print(f"  (these spawn new IDs)")
    print()
    print(f"Total track IDs created : {n_total_tracks}")
    print(f"  Died during clip      : {n_died}")
    print(f"  Alive at end          : {n_alive_at_end}")
    print(
        f"  Likely confirmed      : {n_ever_confirmed} "
        f"(>={min_hits} total matches)"
    )
    print()
    print("Track lifetime distribution (frames lived, including birth):")
    for b in bucket_order:
        if lifetime_buckets[b]:
            bar = "#" * min(50, lifetime_buckets[b])
            print(f"  {b:>7}  {lifetime_buckets[b]:>4}  {bar}")
    print()
    print("Matches per track (how many frames they were actually associated):")
    for k in sorted(matches_per_track):
        if k > 20:
            break
        bar = "#" * min(50, matches_per_track[k])
        print(f"  {k:>3}  {matches_per_track[k]:>4}  {bar}")
    if max(matches_per_track) > 20:
        n_long = sum(v for k, v in matches_per_track.items() if k > 20)
        print(f"  >20  {n_long:>4}  (long-lived tracks)")
    print()

    # IOU distributions
    if matched_ious:
        m = np.array(matched_ious)
        print(
            f"Matched-pair IOUs       : "
            f"min={m.min():.2f}  med={np.median(m):.2f}  "
            f"mean={m.mean():.2f}  max={m.max():.2f}"
        )
    if near_miss_ious:
        nm = np.array(near_miss_ious)
        # How many unmatched detections had a track box overlapping
        # with IOU just below the threshold? Those are association failures
        n_close = sum(
            1 for v in nm if v < iou_threshold and v >= iou_threshold * 0.5
        )
        print(
            f"Unmatched-det best IOUs : "
            f"med={np.median(nm):.2f}  mean={nm.mean():.2f}  max={nm.max():.2f}"
        )
        print(
            f"  Unmatched dets with best IOU in "
            f"[{iou_threshold * 0.5:.2f}, {iou_threshold:.2f}): "
            f"{n_close}  (these are 'almost matched' -- "
            f"lowering iou_threshold may help)"
        )

    print()
    print("Diagnosis hints:")
    if n_total_tracks > 50:
        if matches_per_track[1] > n_total_tracks * 0.3:
            print(
                "  - Many tracks lived only 1 frame. This usually means the "
                "detector is noisy at low-confidence detections. Consider "
                "raising --det-conf to filter out single-frame ghosts"
            )
        if (
            near_miss_ious
            and sum(1 for v in near_miss_ious if v < iou_threshold) > total_unmatched_dets * 0.3
        ):
            print(
                "  - Many unmatched detections had a nearby track. This usually "
                "means iou_threshold is too strict for the motion in this clip. "
                "Try lowering --iou-threshold to 0.2 or even 0.1"
            )
        median_life = float(np.median(lifetimes))
        if median_life < 10:
            print(
                f"  - Median track lifetime is only {median_life:.0f} frames. "
                "Tracks aren't surviving. Could be (a) too-strict gating, "
                "(b) detector misses, or (c) Kalman motion model unstable. "
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--max-age", type=int, default=DEFAULT_MAX_AGE)
    ap.add_argument("--min-hits", type=int, default=DEFAULT_MIN_HITS)
    ap.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    ap.add_argument("--det-conf", type=float, default=0.3)
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    cfg = load_config()
    sportsmot_root = resolve_path(cfg["sportsmot_root"])
    split = args.split or cfg["split"]
    cache_dir = (
        Path(args.cache_dir) if args.cache_dir else REPO_ROOT / "cache" / "detections"
    )
    cache_path = cache_dir / f"{args.clip}.npz"
    if not cache_path.exists():
        print(f"No cache at {cache_path}. Run scripts/run_detector.py first.")
        return 1

    diagnose(
        args.clip,
        cache_path,
        sportsmot_root,
        split,
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=args.iou_threshold,
        det_conf=args.det_conf,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
