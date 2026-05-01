"""Run the tracker on a clip's cached detections and save MOT-format output.

Usage:
    python scripts/run_tracker.py --clip v_xxx
    python scripts/run_tracker.py --clip v_xxx --max-age 30 --det-conf 0.4
    python scripts/run_tracker.py --all
    python scripts/run_tracker.py --clip v_xxx --tag t-lost-ablation/max_age_5

The --tag option puts results under results/<tag>/<clip>.txt instead of
results/<clip>.txt, which is the cleanest way to organize ablation runs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sports_tracker.config import REPO_ROOT, load_config, resolve_path
from sports_tracker.detector import load_detections
from sports_tracker.mot_io import MotRow, write_mot_file
from sports_tracker.tracker import (
    DEFAULT_MAX_AGE,
    DEFAULT_MIN_HITS,
    Tracker,
)
from sports_tracker.association import DEFAULT_IOU_THRESHOLD


# Confidence below which we drop YOLO detections before they reach the
# tracker. Lower than YOLO's typical operating threshold because we want
# to catch occluded players, but high enough to filter obvious noise.
DEFAULT_DET_CONF = 0.3


def get_clip_n_frames(sportsmot_root: Path, split: str, clip_name: str) -> int:
    """Read seqLength from the clip's seqinfo.ini.

    We need this to run the tracker over EVERY frame, including ones
    YOLO didn't detect anything in. If we only iterated over frames
    that had detections, the tracker would be stepped fewer times than
    it should be, causing tracks to age incorrectly.
    """
    import configparser

    seqinfo_path = sportsmot_root / split / clip_name / "seqinfo.ini"
    parser = configparser.ConfigParser()
    parser.optionxform = str  # preserve camelCase keys
    parser.read(seqinfo_path)
    return int(parser["Sequence"]["seqLength"])


def run_clip(
    clip_name: str,
    cache_path: Path,
    sportsmot_root: Path,
    split: str,
    out_path: Path,
    *,
    max_age: int,
    min_hits: int,
    iou_threshold: float,
    det_conf: float,
) -> dict:
    """Run the tracker over one clip and save MOT-format output.

    Returns a dict of summary statistics for the run: frame count,
    average detections per frame, total track outputs, runtime.
    """
    n_frames = get_clip_n_frames(sportsmot_root, split, clip_name)

    # Load cached detections in DENSE mode so we get one entry per
    # frame (with empty boxes on frames YOLO found nothing in). This
    # is essential for correct tracker bookkeeping.
    frame_dets, meta = load_detections(
        cache_path, min_conf=det_conf, n_frames=n_frames
    )

    tracker = Tracker(
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_threshold,
    )

    rows: list[MotRow] = []
    total_dets = 0
    total_outputs = 0
    t0 = time.time()

    for fd in frame_dets:
        total_dets += len(fd.boxes)
        outputs = tracker.step(fd.boxes)
        total_outputs += len(outputs)
        for o in outputs:
            rows.append(
                MotRow(
                    frame=fd.frame_idx,
                    track_id=o.id,
                    x1=float(o.bbox[0]),
                    y1=float(o.bbox[1]),
                    x2=float(o.bbox[2]),
                    y2=float(o.bbox[3]),
                    # Tracker doesn't carry detection confidence through, so
                    # we report 1.0. Could be wired through later if needed.
                    confidence=1.0,
                )
            )

    elapsed = time.time() - t0
    write_mot_file(out_path, rows)

    return {
        "clip": clip_name,
        "n_frames": n_frames,
        "total_detections": total_dets,
        "mean_detections_per_frame": total_dets / max(n_frames, 1),
        "total_track_outputs": total_outputs,
        "n_unique_tracks": len({r.track_id for r in rows}),
        "elapsed_seconds": elapsed,
        "tracking_fps": n_frames / elapsed if elapsed > 0 else float("inf"),
        "params": {
            "max_age": max_age,
            "min_hits": min_hits,
            "iou_threshold": iou_threshold,
            "det_conf": det_conf,
        },
        "detection_meta": meta,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", help="A specific clip name to process.")
    ap.add_argument(
        "--all",
        action="store_true",
        help="Process every clip that has a cached .npz under cache/detections/.",
    )
    ap.add_argument("--split", default=None, help="Override split from config.")
    ap.add_argument(
        "--max-age",
        type=int,
        default=DEFAULT_MAX_AGE,
        help=f"Max frames a track can be unmatched before deletion. "
        f"Default {DEFAULT_MAX_AGE} (vs SORT's reference value of 1).",
    )
    ap.add_argument(
        "--min-hits",
        type=int,
        default=DEFAULT_MIN_HITS,
        help=f"Consecutive hits before a new track is reported. "
        f"Default {DEFAULT_MIN_HITS}.",
    )
    ap.add_argument(
        "--iou-threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help=f"Min IOU for matching. Default {DEFAULT_IOU_THRESHOLD}.",
    )
    ap.add_argument(
        "--det-conf",
        type=float,
        default=DEFAULT_DET_CONF,
        help=f"Drop detections with confidence below this before tracking. "
        f"Default {DEFAULT_DET_CONF}.",
    )
    ap.add_argument(
        "--tag",
        default=None,
        help="Subdirectory under results/ for organizing ablation runs. "
        "E.g. --tag max_age_sweep/30 puts output at "
        "results/max_age_sweep/30/<clip>.txt.",
    )
    ap.add_argument(
        "--cache-dir",
        default=None,
        help="Where cached detections live. Default: cache/detections/.",
    )
    ap.add_argument(
        "--results-dir",
        default=None,
        help="Where to write tracking results. Default: results/.",
    )
    args = ap.parse_args()

    if not args.clip and not args.all:
        ap.error("must specify either --clip CLIP_NAME or --all")

    cfg = load_config()
    sportsmot_root = resolve_path(cfg["sportsmot_root"])
    split = args.split or cfg["split"]

    cache_dir = (
        Path(args.cache_dir) if args.cache_dir else REPO_ROOT / "cache" / "detections"
    )
    results_dir = (
        Path(args.results_dir) if args.results_dir else REPO_ROOT / "results"
    )
    if args.tag:
        results_dir = results_dir / args.tag

    # Resolve which clips to process.
    if args.all:
        cache_files = sorted(cache_dir.glob("*.npz"))
        if not cache_files:
            print(f"No cached detection files in {cache_dir}.")
            print("Run scripts/run_detector.py first.")
            return 1
        clip_names = [p.stem for p in cache_files]
        print(f"Will process {len(clip_names)} clips.")
    else:
        clip_names = [args.clip]

    for clip_name in clip_names:
        cache_path = cache_dir / f"{clip_name}.npz"
        if not cache_path.exists():
            print(f"[skip] {clip_name}: no cache at {cache_path}")
            continue

        out_path = results_dir / f"{clip_name}.txt"

        try:
            stats = run_clip(
                clip_name,
                cache_path,
                sportsmot_root,
                split,
                out_path,
                max_age=args.max_age,
                min_hits=args.min_hits,
                iou_threshold=args.iou_threshold,
                det_conf=args.det_conf,
            )
        except Exception as exc:
            print(f"[fail] {clip_name}: {exc}")
            continue

        print(
            f"[done] {clip_name}: "
            f"{stats['n_frames']} frames, "
            f"{stats['mean_detections_per_frame']:.1f} dets/frame, "
            f"{stats['n_unique_tracks']} unique track IDs, "
            f"{stats['tracking_fps']:.0f} tracking fps -> {out_path}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
