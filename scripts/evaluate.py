"""Evaluate tracker predictions against SportsMOT ground truth.

Usage:
    python scripts/evaluate.py --clip v_xxx
    python scripts/evaluate.py --all
    python scripts/evaluate.py --all --tag max_age_sweep/30
    python scripts/evaluate.py --all --save-csv results/summary.csv

Reads ground-truth files from data/sportsmot_publish/dataset/<split>/<clip>/gt/gt.txt
and predictions from results/<clip>.txt (or results/<tag>/<clip>.txt with --tag)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sports_tracker.config import REPO_ROOT, load_config, resolve_path
from sports_tracker.evaluation import (
    DEFAULT_METRICS,
    METRIC_LABELS,
    evaluate_clips,
    format_summary_table,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", help="A specific clip to evaluate.")
    ap.add_argument(
        "--all",
        action="store_true",
        help="Evaluate every clip with both a results file and a ground-truth file.",
    )
    ap.add_argument("--split", default=None, help="Override split from config.")
    ap.add_argument(
        "--tag",
        default=None,
        help="Subdirectory under results/. Use this to evaluate a specific "
        "ablation run, matching the same --tag passed to run_tracker.py.",
    )
    ap.add_argument(
        "--results-dir",
        default=None,
        help="Where tracking results live. Default: results/.",
    )
    ap.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="Min IOU for a prediction to count as matching a GT box. "
        "MOT Challenge standard is 0.5. (Note: this is independent of the "
        "iou_threshold passed to the tracker, which controls track-to-detection "
        "matching during tracking.)",
    )
    ap.add_argument(
        "--save-csv",
        default=None,
        help="If given, also write a CSV with per-clip and overall metrics.",
    )
    args = ap.parse_args()

    if not args.clip and not args.all:
        ap.error("must specify either --clip CLIP_NAME or --all")

    cfg = load_config()
    sportsmot_root = resolve_path(cfg["sportsmot_root"])
    split = args.split or cfg["split"]

    results_dir = (
        Path(args.results_dir) if args.results_dir else REPO_ROOT / "results"
    )
    if args.tag:
        results_dir = results_dir / args.tag

    # Build the (clip_name, gt_path, pred_path) list
    if args.all:
        pred_files = sorted(results_dir.glob("*.txt"))
        if not pred_files:
            print(f"No prediction files in {results_dir}.")
            print("Run scripts/run_tracker.py first.")
            return 1
        clip_names = [p.stem for p in pred_files]
    else:
        clip_names = [args.clip]

    pairs = []
    for clip_name in clip_names:
        gt_path = sportsmot_root / split / clip_name / "gt" / "gt.txt"
        pred_path = results_dir / f"{clip_name}.txt"
        if not gt_path.exists():
            print(f"[skip] {clip_name}: no ground truth at {gt_path}")
            continue
        if not pred_path.exists():
            print(f"[skip] {clip_name}: no predictions at {pred_path}")
            continue
        pairs.append((clip_name, gt_path, pred_path))

    if not pairs:
        print("No clips have both ground truth and predictions; nothing to evaluate.")
        return 1

    print(f"Evaluating {len(pairs)} clip(s) at IOU threshold {args.iou_threshold}\n")
    per_clip, combined = evaluate_clips(pairs, iou_threshold=args.iou_threshold)
    print(format_summary_table(per_clip, combined))

    if args.save_csv:
        csv_path = Path(args.save_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w") as f:
            f.write("clip," + ",".join(METRIC_LABELS[m] for m in DEFAULT_METRICS) + "\n")
            for c in per_clip:
                row = [c.clip] + [str(c.metrics[m]) for m in DEFAULT_METRICS]
                f.write(",".join(row) + "\n")
            row = ["OVERALL"] + [str(combined[m]) for m in DEFAULT_METRICS]
            f.write(",".join(row) + "\n")
        print(f"\nSaved CSV to {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
