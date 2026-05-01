"""Verify the evaluation module without pytest.

If py-motmetrics isn't installed, the metric-computation checks are
skipped with a notice but the rest of the script (formatting helpers)
still runs.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sports_tracker.evaluation import (
    DEFAULT_METRICS,
    METRIC_LABELS,
    ClipEvalResult,
    _fmt_value,
    format_summary_table,
)


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        sys.exit(1)


def write_mot(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for frame, tid, x, y, w, h in rows:
            f.write(f"{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")


def write_gt(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for frame, tid, x, y, w, h in rows:
            f.write(f"{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1.0\n")


print("Formatting helpers (no motmetrics needed)")
print("-" * 60)

check("fmt int metric", _fmt_value("num_misses", 5) == "5")
check("fmt int metric from float", _fmt_value("num_misses", 5.0) == "5")
check("fmt rate metric", _fmt_value("mota", 0.5) == "0.500")
check("fmt rate metric truncates", _fmt_value("idf1", 0.123456) == "0.123")
check("fmt None as dash", _fmt_value("mota", None) == "-")

per_clip = [
    ClipEvalResult(
        clip="clip_a",
        metrics={
            m: 0.5 if m in {"mota", "motp", "idf1", "idp", "idr"} else 1
            for m in DEFAULT_METRICS
        },
    )
]
combined = {
    m: 0.5 if m in {"mota", "motp", "idf1", "idp", "idr"} else 2
    for m in DEFAULT_METRICS
}
table = format_summary_table(per_clip, combined)

all_labels = all(METRIC_LABELS[m] in table for m in DEFAULT_METRICS)
check("summary table includes all metric labels", all_labels)
check("summary table includes OVERALL row", "OVERALL" in table)
check("summary table includes clip name", "clip_a" in table)

print()
print("Metric computation (requires motmetrics)")
print("-" * 60)

try:
    import motmetrics  # noqa
except ImportError:
    print("  SKIP  py-motmetrics not installed; skipping computation checks.")
    print("        Install with: pip install motmetrics")
    print()
    print("All available checks passed.")
    sys.exit(0)

from sports_tracker.evaluation import evaluate_clip, evaluate_clips

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)

    # Perfect predictions.
    gt_rows = [
        (1, 1, 100, 100, 50, 100),
        (1, 2, 300, 200, 50, 100),
        (2, 1, 105, 100, 50, 100),
        (2, 2, 305, 200, 50, 100),
        (3, 1, 110, 100, 50, 100),
        (3, 2, 310, 200, 50, 100),
    ]
    gt_path = tmp_path / "gt.txt"
    pred_path = tmp_path / "pred.txt"
    write_gt(gt_path, gt_rows)
    write_mot(pred_path, gt_rows)

    metrics = evaluate_clip(gt_path, pred_path)
    check("perfect predictions: MOTA = 1.0", abs(metrics["mota"] - 1.0) < 1e-9,
          f"got {metrics['mota']}")
    check("perfect predictions: 0 misses", metrics["num_misses"] == 0)
    check("perfect predictions: 0 false positives", metrics["num_false_positives"] == 0)
    check("perfect predictions: 0 ID switches", metrics["num_switches"] == 0)
    check("perfect predictions: 2 unique GT objects", metrics["num_unique_objects"] == 2)

    # Half-coverage predictions.
    pred_rows = [(1, 1, 100, 100, 50, 100), (2, 1, 105, 100, 50, 100)]
    pred_path = tmp_path / "half.txt"
    write_mot(pred_path, pred_rows)

    gt_rows = [
        (1, 1, 100, 100, 50, 100),
        (1, 2, 300, 200, 50, 100),
        (2, 1, 105, 100, 50, 100),
        (2, 2, 305, 200, 50, 100),
    ]
    gt_path = tmp_path / "half_gt.txt"
    write_gt(gt_path, gt_rows)

    metrics = evaluate_clip(gt_path, pred_path)
    check("half-coverage: MOTA = 0.5", abs(metrics["mota"] - 0.5) < 1e-9,
          f"got {metrics['mota']}")
    check("half-coverage: 2 misses", metrics["num_misses"] == 2)
    check("half-coverage: 0 FP", metrics["num_false_positives"] == 0)

    # ID switch.
    gt_rows = []
    for frame in range(1, 5):
        gt_rows.append((frame, 1, 100, 100, 50, 100))
        gt_rows.append((frame, 2, 300, 200, 50, 100))
    pred_rows = [
        (1, 1, 100, 100, 50, 100),
        (1, 2, 300, 200, 50, 100),
        (2, 1, 100, 100, 50, 100),
        (2, 2, 300, 200, 50, 100),
        (3, 99, 100, 100, 50, 100),
        (3, 2, 300, 200, 50, 100),
        (4, 99, 100, 100, 50, 100),
        (4, 2, 300, 200, 50, 100),
    ]
    gt_path = tmp_path / "swap_gt.txt"
    pred_path = tmp_path / "swap.txt"
    write_gt(gt_path, gt_rows)
    write_mot(pred_path, pred_rows)
    metrics = evaluate_clip(gt_path, pred_path)
    check("ID swap: at least 1 ID switch counted",
          metrics["num_switches"] >= 1, f"got {metrics['num_switches']}")

    # Ignored GT rows.
    gt_path = tmp_path / "ignored_gt.txt"
    with open(gt_path, "w") as f:
        f.write("1,1,100,100,50,100,1,1,1.0\n")
        f.write("1,2,300,200,50,100,1,1,1.0\n")
        f.write("1,99,500,500,50,100,0,1,1.0\n")  # consider=0
    pred_path = tmp_path / "ignored_pred.txt"
    write_mot(pred_path, [(1, 1, 100, 100, 50, 100), (1, 2, 300, 200, 50, 100)])
    metrics = evaluate_clip(gt_path, pred_path)
    check("consider=0 rows ignored: 0 misses", metrics["num_misses"] == 0)
    check("consider=0 rows ignored: MOTA = 1.0",
          abs(metrics["mota"] - 1.0) < 1e-9, f"got {metrics['mota']}")

    # Multi-clip aggregation.
    pairs = []
    for clip_idx in range(2):
        gt_p = tmp_path / f"agg_{clip_idx}_gt.txt"
        pred_p = tmp_path / f"agg_{clip_idx}_pred.txt"
        ids = (1, 2) if clip_idx == 0 else (3, 4)
        rows = [
            (1, ids[0], 100, 100, 50, 100),
            (1, ids[1], 300, 200, 50, 100),
            (2, ids[0], 105, 100, 50, 100),
            (2, ids[1], 305, 200, 50, 100),
        ]
        write_gt(gt_p, rows)
        write_mot(pred_p, rows)
        pairs.append((f"clip_{clip_idx}", gt_p, pred_p))

    per_clip_results, combined = evaluate_clips(pairs)
    check("aggregation: 2 clip results returned", len(per_clip_results) == 2)
    check("aggregation: each clip has perfect MOTA",
          all(abs(c.metrics["mota"] - 1.0) < 1e-9 for c in per_clip_results))
    check("aggregation: OVERALL has 4 unique objects",
          combined["num_unique_objects"] == 4)

print()
print("All checks passed.")
