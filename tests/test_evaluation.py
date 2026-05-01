"""Tests for the evaluation module.

Some tests need py-motmetrics. These are marked to skip when the
library isn't installed, so the test suite still passes in environments
without it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# motmetrics 1.4.0 uses np.asfarray, which was removed in NumPy 2.0.
# Restore it as an alias before motmetrics is imported.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

from sports_tracker.evaluation import (
    DEFAULT_METRICS,
    METRIC_LABELS,
    _fmt_value,
    format_summary_table,
    ClipEvalResult,
)

motmetrics = pytest.importorskip("motmetrics")


# --- Tests that need motmetrics ---------------------------------------------


def write_mot(path: Path, rows: list[tuple]) -> None:
    """Write a MOT-format file from (frame, id, x, y, w, h) tuples."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            frame, tid, x, y, w, h = r
            f.write(f"{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")


def write_gt(path: Path, rows: list[tuple]) -> None:
    """Write a SportsMOT-format gt.txt with the consider/class/visibility columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            frame, tid, x, y, w, h = r
            # consider=1, class=1, visibility=1.0 (per SportsMOT format)
            f.write(f"{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1.0\n")


def test_perfect_predictions_score_perfect_mota(tmp_path: Path):
    """If predictions match GT exactly (same boxes, same IDs, same frames),
    MOTA should be 1.0 and there should be zero misses, zero FPs, zero
    identity switches.
    """
    from sports_tracker.evaluation import evaluate_clip

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

    metrics = evaluate_clip(gt_path, pred_path, iou_threshold=0.5)
    assert metrics["mota"] == pytest.approx(1.0)
    assert metrics["num_misses"] == 0
    assert metrics["num_false_positives"] == 0
    assert metrics["num_switches"] == 0
    assert metrics["num_unique_objects"] == 2


def test_missed_predictions_decrease_mota(tmp_path: Path):
    """If we miss half the predictions, MOTA should drop accordingly."""
    from sports_tracker.evaluation import evaluate_clip

    gt_rows = [
        (1, 1, 100, 100, 50, 100),
        (1, 2, 300, 200, 50, 100),
        (2, 1, 105, 100, 50, 100),
        (2, 2, 305, 200, 50, 100),
    ]
    # Predictions only cover track 1.
    pred_rows = [
        (1, 1, 100, 100, 50, 100),
        (2, 1, 105, 100, 50, 100),
    ]
    gt_path = tmp_path / "gt.txt"
    pred_path = tmp_path / "pred.txt"
    write_gt(gt_path, gt_rows)
    write_mot(pred_path, pred_rows)

    metrics = evaluate_clip(gt_path, pred_path, iou_threshold=0.5)
    # MOTA = 1 - (FN + FP + IDs) / GT = 1 - (2 + 0 + 0) / 4 = 0.5
    assert metrics["mota"] == pytest.approx(0.5)
    assert metrics["num_misses"] == 2
    assert metrics["num_false_positives"] == 0


def test_id_switch_is_counted(tmp_path: Path):
    """If predicted IDs swap mid-clip while the GT IDs stay constant,
    the evaluator should count an identity switch.
    """
    from sports_tracker.evaluation import evaluate_clip

    # GT: two stable players for 4 frames.
    gt_rows = []
    for frame in range(1, 5):
        gt_rows.append((frame, 1, 100, 100, 50, 100))
        gt_rows.append((frame, 2, 300, 200, 50, 100))

    # Predictions: same boxes, but the IDs swap on frame 3.
    pred_rows = [
        (1, 1, 100, 100, 50, 100),
        (1, 2, 300, 200, 50, 100),
        (2, 1, 100, 100, 50, 100),
        (2, 2, 300, 200, 50, 100),
        (3, 99, 100, 100, 50, 100),  # ID swap: was 1, now 99
        (3, 2, 300, 200, 50, 100),
        (4, 99, 100, 100, 50, 100),
        (4, 2, 300, 200, 50, 100),
    ]
    gt_path = tmp_path / "gt.txt"
    pred_path = tmp_path / "pred.txt"
    write_gt(gt_path, gt_rows)
    write_mot(pred_path, pred_rows)

    metrics = evaluate_clip(gt_path, pred_path, iou_threshold=0.5)
    assert metrics["num_switches"] >= 1, (
        f"Expected at least 1 identity switch, got {metrics['num_switches']}"
    )


def test_ignored_gt_rows_are_excluded(tmp_path: Path):
    """A gt.txt row with the consider-flag (column 7) set to 0 should
    not contribute to misses if we don't predict it. This is critical
    for SportsMOT, which uses the flag to mark non-player annotations.
    """
    from sports_tracker.evaluation import evaluate_clip

    gt_path = tmp_path / "gt.txt"
    pred_path = tmp_path / "pred.txt"

    # Two players, plus a row with consider=0 that should be ignored.
    with open(gt_path, "w") as f:
        f.write("1,1,100,100,50,100,1,1,1.0\n")
        f.write("1,2,300,200,50,100,1,1,1.0\n")
        # Ignored row -- a "phantom" entity we shouldn't be scored on.
        f.write("1,99,500,500,50,100,0,1,1.0\n")

    # Predict only the two real players.
    write_mot(
        pred_path,
        [(1, 1, 100, 100, 50, 100), (1, 2, 300, 200, 50, 100)],
    )

    metrics = evaluate_clip(gt_path, pred_path, iou_threshold=0.5)
    # If the ignored row WERE counted, we'd have 1 miss and MOTA < 1.0.
    assert metrics["num_misses"] == 0, (
        "consider=0 rows in gt.txt should not be counted as misses"
    )
    assert metrics["mota"] == pytest.approx(1.0)


def test_evaluate_clips_aggregates_correctly(tmp_path: Path):
    """The combined OVERALL row should have num_unique_objects equal to
    the sum across clips (when IDs don't collide across clips).
    """
    from sports_tracker.evaluation import evaluate_clips

    # Two clips, two distinct players each.
    pairs = []
    for clip_idx in range(2):
        gt_path = tmp_path / f"clip_{clip_idx}_gt.txt"
        pred_path = tmp_path / f"clip_{clip_idx}_pred.txt"
        # Use disjoint ID ranges: clip 0 has IDs 1,2; clip 1 has IDs 3,4
        ids = (1, 2) if clip_idx == 0 else (3, 4)
        rows = [
            (1, ids[0], 100, 100, 50, 100),
            (1, ids[1], 300, 200, 50, 100),
            (2, ids[0], 105, 100, 50, 100),
            (2, ids[1], 305, 200, 50, 100),
        ]
        write_gt(gt_path, rows)
        write_mot(pred_path, rows)
        pairs.append((f"clip_{clip_idx}", gt_path, pred_path))

    per_clip, combined = evaluate_clips(pairs)
    assert len(per_clip) == 2
    assert all(c.metrics["mota"] == pytest.approx(1.0) for c in per_clip)
    # py-motmetrics sums num_unique_objects across clips when the IDs are
    # disjoint, giving 4 total unique GT objects.
    assert combined["num_unique_objects"] == 4


# --- Tests that don't need motmetrics ---------------------------------------


def test_fmt_value_int_metrics():
    """Count metrics format as integers."""
    assert _fmt_value("num_misses", 5) == "5"
    assert _fmt_value("num_misses", 5.0) == "5"
    assert _fmt_value("num_unique_objects", 12) == "12"


def test_fmt_value_rate_metrics():
    """Rate metrics format as 3-decimal floats."""
    assert _fmt_value("mota", 0.5) == "0.500"
    assert _fmt_value("idf1", 0.123456) == "0.123"


def test_fmt_value_handles_none():
    assert _fmt_value("mota", None) == "-"


def test_format_summary_table_includes_all_metrics():
    per_clip = [
        ClipEvalResult(
            clip="clip_a",
            metrics={m: 0.5 if m in {"mota", "motp", "idf1", "idp", "idr"} else 1
                     for m in DEFAULT_METRICS},
        )
    ]
    combined = {m: 0.5 if m in {"mota", "motp", "idf1", "idp", "idr"} else 2
                for m in DEFAULT_METRICS}
    table = format_summary_table(per_clip, combined)
    # Sanity: every metric label and the OVERALL row should appear.
    for m in DEFAULT_METRICS:
        assert METRIC_LABELS[m] in table
    assert "OVERALL" in table
    assert "clip_a" in table
