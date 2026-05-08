"""Run YOLO over a SportsMOT clip and cache the detections

Usage:
    python scripts/run_detector.py --clip v_-6Os86HzwCs_c001
    python scripts/run_detector.py --clip v_xxx --conf 0.05 --model yolov8s.pt
    python scripts/run_detector.py --all          # process every basketball clip in val

The cache is written to cache/detections/<clip>.npz relative to the repo root
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sports_tracker.config import REPO_ROOT, load_config, resolve_path
from sports_tracker.detector import (
    DEFAULT_CACHE_CONF,
    FrameDetections,
    YoloDetector,
    save_detections,
)


def list_clips_for_sport(splits_dir: Path, sport: str) -> set[str]:
    if sport == "all":
        return set()  # caller handles "all" as "no filter"
    sport_file = splits_dir / f"{sport}.txt"
    with open(sport_file) as f:
        return {line.strip() for line in f if line.strip()}


def find_clip_dir(sportsmot_root: Path, split: str, clip_name: str) -> Path:
    """Locate a clip directory by name."""
    candidate = sportsmot_root / split / clip_name
    if not candidate.exists():
        raise FileNotFoundError(
            f"Clip not found: {candidate}. "
            f"Check that the clip name is correct and the dataset is downloaded."
        )
    return candidate


def detect_clip(
    clip_dir: Path,
    detector: YoloDetector,
    conf: float,
    progress_every: int = 50,
) -> list[FrameDetections]:
    """Run YOLO over every frame in a clip's img1/ directory

    Returns one FrameDetections per processed frame. Frames with zero 
    detections are still included so the caller can recover the full 
    sequence length
    """
    import cv2

    img_dir = clip_dir / "img1"
    frame_paths = sorted(img_dir.glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No JPG frames found in {img_dir}")

    print(f"Processing {len(frame_paths)} frames from {clip_dir.name}")
    t0 = time.time()
    results: list[FrameDetections] = []

    for i, frame_path in enumerate(frame_paths, start=1):
        # Frame index follows the file name's stem
        frame_idx = int(frame_path.stem)

        img = cv2.imread(str(frame_path))
        if img is None:
            raise IOError(f"Failed to read {frame_path}")

        boxes, scores = detector.detect_frame(img, conf=conf)
        results.append(
            FrameDetections(frame_idx=frame_idx, boxes=boxes, scores=scores)
        )

        if i % progress_every == 0 or i == len(frame_paths):
            elapsed = time.time() - t0
            fps = i / elapsed if elapsed > 0 else 0.0
            remaining = (len(frame_paths) - i) / fps if fps > 0 else 0.0
            print(
                f"  {i:>4}/{len(frame_paths)}  "
                f"({fps:.1f} fps, ~{remaining:.0f}s remaining)"
            )

    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", help="A specific clip name to process.")
    ap.add_argument(
        "--all",
        action="store_true",
        help="Process every clip in the configured split that matches the "
        "configured sport filter.",
    )
    ap.add_argument(
        "--split",
        default=None,
        help="Override the split from config (train/val/test).",
    )
    ap.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CACHE_CONF,
        help=f"Confidence threshold for caching. Default {DEFAULT_CACHE_CONF}. "
        "Lower values cache more boxes (you can filter at load time); "
        "higher values save disk and detection time.",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="YOLO model name. Overrides the config value.",
    )
    ap.add_argument(
        "--cache-dir",
        default=None,
        help="Where to write cache .npz files. Default: cache/detections/.",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="cpu / cuda / mps. Default: ultralytics auto-detect.",
    )
    args = ap.parse_args()

    if not args.clip and not args.all:
        ap.error("must specify either --clip CLIP_NAME or --all")

    cfg = load_config()
    sportsmot_root = resolve_path(cfg["sportsmot_root"])
    split = args.split or cfg["split"]
    sport = cfg.get("sport", "all")
    model_name = args.model or cfg.get("yolo_model", "yolov8n.pt")
    cache_dir = (
        Path(args.cache_dir)
        if args.cache_dir
        else REPO_ROOT / "cache" / "detections"
    )

    if args.all:
        # splits_dir is only needed when filtering by sport across the full split
        splits_dir = resolve_path(cfg["splits_dir"])
        whitelist = list_clips_for_sport(splits_dir, sport) if sport != "all" else None
        all_on_disk = sorted(
            p.name for p in (sportsmot_root / split).iterdir() if p.is_dir()
        )
        if whitelist is None:
            clip_names = all_on_disk
        else:
            clip_names = [n for n in all_on_disk if n in whitelist]
        if not clip_names:
            print(f"No clips matched sport={sport} in {sportsmot_root}/{split}")
            return 1
        print(f"Will process {len(clip_names)} clips.")
    else:
        clip_names = [args.clip]

    detector = YoloDetector(model_name=model_name, device=args.device)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for clip_name in clip_names:
        out_path = cache_dir / f"{clip_name}.npz"
        if out_path.exists():
            print(f"[skip] {clip_name}: cache already exists at {out_path}")
            continue

        clip_dir = find_clip_dir(sportsmot_root, split, clip_name)
        frame_dets = detect_clip(clip_dir, detector, conf=args.conf)

        meta = {
            "clip": clip_name,
            "split": split,
            "model": model_name,
            "conf_threshold": args.conf,
            "n_frames": len(frame_dets),
        }
        save_detections(out_path, frame_dets, meta)
        n_dets = sum(len(fd.boxes) for fd in frame_dets)
        print(
            f"[done] {clip_name}: cached {n_dets} detections "
            f"across {len(frame_dets)} frames -> {out_path}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
