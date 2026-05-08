"""Run YOLO's built-in tracker and produce MOT-format predictions

YOLO ships two trackers through model.track() with bytetrack and
botsort. Both apply Kalman + Hungarian on top of the YOLO detector,
similar to our hand-crafted SORT, but with various refinements.

We want to compare our from-scratch SORT against an established, similar 
tracker. By running both through the same evaluation script 
(scripts/evaluate.py) we get a clean comparison

Usage:
    python scripts/run_yolo_tracker.py --clip v_xxx --tracker bytetrack
    python scripts/run_yolo_tracker.py --all --tracker botsort
    python scripts/run_yolo_tracker.py --all --tracker bytetrack --conf 0.55

Output: results/yolo-<tracker>/<clip>.txt in MOT-15 format
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sports_tracker.config import REPO_ROOT, load_config, resolve_path
from sports_tracker.detector import PERSON_CLASS_ID
from sports_tracker.mot_io import MotRow, write_mot_file


# Confidence threshold matching what our hand-crafted tracker uses by default
DEFAULT_CONF = 0.55


def list_clips_for_sport(splits_dir: Path, sport: str) -> set[str]:
    if sport == "all":
        return set()
    sport_file = splits_dir / f"{sport}.txt"
    with open(sport_file) as f:
        return {line.strip() for line in f if line.strip()}


def find_clip_dir(sportsmot_root: Path, split: str, clip_name: str) -> Path:
    candidate = sportsmot_root / split / clip_name
    if not candidate.exists():
        raise FileNotFoundError(f"Clip not found: {candidate}")
    return candidate


def track_clip(
    clip_dir: Path,
    out_path: Path,
    *,
    model_name: str,
    tracker_name: str,
    conf: float,
    device: str | None,
) -> dict:
    """Run YOLO's tracker over a clip and save MOT-format predictions

    YOLO's model.track() returns one Results object per frame, with
    a .boxes attribute that includes .id which is the assigned track
    ID. We pull (frame, id, box, conf) from each frame and append rows
    """
    import cv2
    from ultralytics import YOLO

    img_dir = clip_dir / "img1"
    frame_paths = sorted(img_dir.glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No JPG frames in {img_dir}")

    print(f"Running {tracker_name} on {len(frame_paths)} frames from {clip_dir.name}")
    model = YOLO(model_name)
    t0 = time.time()
    rows: list[MotRow] = []
    n_with_id = 0

    for i, frame_path in enumerate(frame_paths, start=1):
        frame_idx = int(frame_path.stem)
        img = cv2.imread(str(frame_path))
        if img is None:
            raise IOError(f"Failed to read {frame_path}")

        # persist=True maintains tracking state across frames
        results = model.track(
            img,
            classes=[PERSON_CLASS_ID],
            conf=conf,
            tracker=f"{tracker_name}.yaml",
            persist=True,
            device=device,
            verbose=False,
        )
        r = results[0]

        if r.boxes is None or r.boxes.id is None:
            continue  # frame had no tracked boxes

        # Pull tensors to numpy so we can iterate
        ids = r.boxes.id.cpu().numpy().astype(np.int64)
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        for tid, box, score in zip(ids, xyxy, confs):
            rows.append(
                MotRow(
                    frame=frame_idx,
                    track_id=int(tid),
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                    confidence=float(score),
                )
            )
            n_with_id += 1

        if i % 100 == 0 or i == len(frame_paths):
            elapsed = time.time() - t0
            fps = i / elapsed if elapsed else 0.0
            remaining = (len(frame_paths) - i) / fps if fps else 0.0
            print(
                f"  {i:>4}/{len(frame_paths)}  "
                f"({fps:.1f} fps, ~{remaining:.0f}s remaining)"
            )

    write_mot_file(out_path, rows)
    elapsed = time.time() - t0

    return {
        "clip": clip_dir.name,
        "n_frames": len(frame_paths),
        "n_track_outputs": n_with_id,
        "n_unique_tracks": len({r.track_id for r in rows}),
        "elapsed_seconds": elapsed,
        "fps": len(frame_paths) / elapsed if elapsed else float("inf"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", help="A specific clip name to process.")
    ap.add_argument("--all", action="store_true", help="Process every basketball val clip.")
    ap.add_argument("--split", default=None)
    ap.add_argument(
        "--tracker",
        default="bytetrack",
        choices=["bytetrack", "botsort"],
        help="Which YOLO tracker to use. Default: bytetrack (motion-only). "
        "botsort additionally uses appearance features.",
    )
    ap.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help=f"Detection confidence threshold. Default {DEFAULT_CONF} "
        "(matches our hand-crafted tracker's default).",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="YOLO model name. Overrides the config value.",
    )
    ap.add_argument("--device", default=None, help="cpu / cuda / mps.")
    ap.add_argument(
        "--results-dir",
        default=None,
        help="Where to write MOT-format outputs. Default: results/.",
    )
    args = ap.parse_args()

    if not args.clip and not args.all:
        ap.error("must specify either --clip CLIP_NAME or --all")

    cfg = load_config()
    sportsmot_root = resolve_path(cfg["sportsmot_root"])
    split = args.split or cfg["split"]
    sport = cfg.get("sport", "all")
    model_name = args.model or cfg.get("yolo_model", "yolov8n.pt")

    results_dir = (
        Path(args.results_dir) if args.results_dir else REPO_ROOT / "results"
    )
    out_subdir = results_dir / f"yolo-{args.tracker}"

    # Clips to process resolved
    if args.all:
        splits_dir = resolve_path(cfg["splits_dir"])
        whitelist = list_clips_for_sport(splits_dir, sport) if sport != "all" else None
        all_on_disk = sorted(
            p.name for p in (sportsmot_root / split).iterdir() if p.is_dir()
        )
        clip_names = (
            all_on_disk if whitelist is None
            else [n for n in all_on_disk if n in whitelist]
        )
        # Only process clips we have detections for
        cache_dir = REPO_ROOT / "cache" / "detections"
        cached = {p.stem for p in cache_dir.glob("*.npz")}
        if cached:
            clip_names = [n for n in clip_names if n in cached]
        if not clip_names:
            print("No clips to process.")
            return 1
        print(f"Will process {len(clip_names)} clips with {args.tracker}.")
    else:
        clip_names = [args.clip]

    out_subdir.mkdir(parents=True, exist_ok=True)

    for clip_name in clip_names:
        out_path = out_subdir / f"{clip_name}.txt"
        if out_path.exists():
            print(f"[skip] {clip_name}: results already exist at {out_path}")
            continue

        clip_dir = find_clip_dir(sportsmot_root, split, clip_name)
        try:
            stats = track_clip(
                clip_dir,
                out_path,
                model_name=model_name,
                tracker_name=args.tracker,
                conf=args.conf,
                device=args.device,
            )
        except Exception as exc:
            print(f"[fail] {clip_name}: {exc}")
            continue

        print(
            f"[done] {clip_name}: {stats['n_frames']} frames, "
            f"{stats['n_unique_tracks']} unique track IDs, "
            f"{stats['fps']:.1f} fps -> {out_path}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
