"""Tracking evaluation against MOT-format ground truth.

Wraps the ``py-motmetrics`` library to compute the standard MOT
Challenge metrics: MOTA, IDF1, identity switches, MT/ML, FP, FN.

Note on file loading: motmetrics 1.4.0 ships ``mm.io.loadtxt``, which
internally calls ``pandas.read_csv`` with column expectations that are
incompatible with pandas 3.0's stricter CSV parser. Rather than
chase that breakage, we load files ourselves and hand motmetrics a
dataframe in the format it expects internally:

    columns: X, Y, Width, Height, Confidence, ClassId, Visibility
    index:   (FrameId, Id) MultiIndex

We then call ``compare_to_groundtruth`` with these dataframes, which
works fine because the file-loading is the only fragile part.

For ground truth files, we apply the SportsMOT/MOT16 convention that
column 7 (the "consider" flag) gates whether a row is included in
evaluation. Rows with consider == 0 are dropped before evaluation.

References:
    py-motmetrics: https://github.com/cheind/py-motmetrics
    Bernardin & Stiefelhagen, 2008 (MOTA/MOTP definitions)
    Ristani et al., 2016 (IDF1 definition)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional


# Standard set of metrics we care about for the report. Order matters:
# this is the column order used in the printed summary table.
DEFAULT_METRICS = [
    "mota",
    "motp",
    "idf1",
    "idp",
    "idr",
    "num_frames",
    "mostly_tracked",
    "partially_tracked",
    "mostly_lost",
    "num_false_positives",
    "num_misses",
    "num_switches",
    "num_fragmentations",
    "num_unique_objects",
]

# Pretty names used in the printed summary table.
METRIC_LABELS = {
    "mota": "MOTA",
    "motp": "MOTP",
    "idf1": "IDF1",
    "idp": "IDP",
    "idr": "IDR",
    "num_frames": "Frames",
    "mostly_tracked": "MT",
    "partially_tracked": "PT",
    "mostly_lost": "ML",
    "num_false_positives": "FP",
    "num_misses": "FN",
    "num_switches": "IDs",
    "num_fragmentations": "Frag",
    "num_unique_objects": "GT",
}


@dataclass
class ClipEvalResult:
    """Per-clip evaluation result."""

    clip: str
    metrics: dict = field(default_factory=dict)


def _load_motmetrics():
    """Lazy-import motmetrics so non-eval code paths don't pay for it.

    Also installs a NumPy 2.0 compatibility shim. py-motmetrics 1.4.0
    uses ``np.asfarray``, which was removed in NumPy 2.0. We restore it
    as an alias for ``np.asarray(..., dtype=float)`` -- the documented
    replacement -- before importing motmetrics. This is harmless for
    older NumPy and necessary for newer NumPy.
    """
    import numpy as np

    if not hasattr(np, "asfarray"):
        np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

    import motmetrics as mm

    return mm


def _load_mot_dataframe(path: Path, is_gt: bool) -> "pandas.DataFrame":
    """Load a MOT-format file into the dataframe motmetrics expects.

    We bypass ``mm.io.loadtxt`` because its pandas integration is
    broken on pandas 3.0. The internal format motmetrics works with
    is straightforward (see issue #12 in the py-motmetrics repo):

        columns: X, Y, Width, Height, Confidence, ClassId, Visibility
        index:   (FrameId, Id)

    Args:
        path: file path to a 9- or 10-column MOT-format CSV.
        is_gt: when True, drop rows whose 7th column ("consider") is 0.
            This is the SportsMOT/MOT16 convention for ignored
            annotations. For prediction files we keep every row.
    """
    import pandas as pd

    # Parse the CSV ourselves with very permissive whitespace handling.
    # SportsMOT writes ", " between fields, and our writer doesn't, so
    # skipinitialspace handles both cases.
    raw = pd.read_csv(
        path,
        header=None,
        skipinitialspace=True,
        comment="#",
        # MOT format has up to 10 columns; we read at most 9 because
        # neither GT nor predictions use the world-z field meaningfully.
        usecols=list(range(9)),
        names=[
            "FrameId",
            "Id",
            "X",
            "Y",
            "Width",
            "Height",
            "Confidence",
            "ClassId",
            "Visibility",
        ],
        engine="python",  # avoid C engine quirks with mixed delimiters
    )

    if is_gt:
        # Drop rows the dataset author marked "ignore".
        raw = raw[raw["Confidence"] >= 1].copy()

    # motmetrics expects (FrameId, Id) as a MultiIndex.
    df = raw.set_index(["FrameId", "Id"]).sort_index()

    # The 'Confidence' column means different things in GT vs predictions:
    # in GT it's the consider-flag (already filtered above), in predictions
    # it's a detection score. motmetrics doesn't actually use the value
    # for anything except identifying the row, so this is fine.
    return df[["X", "Y", "Width", "Height", "Confidence", "ClassId", "Visibility"]]


def evaluate_clip(
    gt_path: Path,
    pred_path: Path,
    iou_threshold: float = 0.5,
) -> dict:
    """Compute MOT metrics for one clip.

    Args:
        gt_path: ground-truth file (e.g. ``gt/gt.txt`` from SportsMOT).
        pred_path: predictions file from our tracker.
        iou_threshold: minimum IOU for a prediction to be considered a
            match against a ground-truth box. The MOT Challenge standard
            is 0.5; we follow that.

    Returns:
        A dict mapping metric name (e.g. ``mota``) to its value.
    """
    mm = _load_motmetrics()

    gt = _load_mot_dataframe(gt_path, is_gt=True)
    pred = _load_mot_dataframe(pred_path, is_gt=False)

    # The accumulator walks frame by frame, matching predictions to GT
    # via the Hungarian algorithm on a 1 - IOU cost matrix, recording
    # all the events (matches, misses, switches) needed to compute the
    # final summary metrics.
    acc = mm.utils.compare_to_groundtruth(gt, pred, "iou", distth=1.0 - iou_threshold)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=DEFAULT_METRICS,
        name="clip",
    )
    # summary is a pandas DataFrame with one row. Convert to dict.
    return {k: summary[k].iloc[0] for k in DEFAULT_METRICS}


def evaluate_clips(
    pairs: Iterable[tuple[str, Path, Path]],
    iou_threshold: float = 0.5,
) -> tuple[List[ClipEvalResult], dict]:
    """Evaluate multiple clips and produce both per-clip and combined metrics.

    The combined metrics are NOT a simple average. They're computed by
    accumulating events across all clips and computing the metrics on
    the joint accumulator -- i.e. one "big virtual sequence" made by
    concatenating all the clips. This matches the MOT Challenge
    aggregation convention.

    Args:
        pairs: iterable of (clip_name, gt_path, pred_path).
        iou_threshold: same as ``evaluate_clip``.

    Returns:
        (per_clip_results, combined_metrics)
    """
    mm = _load_motmetrics()

    accs = []
    names = []
    per_clip: List[ClipEvalResult] = []
    mh = mm.metrics.create()

    for clip_name, gt_path, pred_path in pairs:
        gt = _load_mot_dataframe(gt_path, is_gt=True)
        pred = _load_mot_dataframe(pred_path, is_gt=False)
        acc = mm.utils.compare_to_groundtruth(
            gt, pred, "iou", distth=1.0 - iou_threshold
        )
        accs.append(acc)
        names.append(clip_name)

        clip_summary = mh.compute(acc, metrics=DEFAULT_METRICS, name=clip_name)
        per_clip.append(
            ClipEvalResult(
                clip=clip_name,
                metrics={k: clip_summary[k].iloc[0] for k in DEFAULT_METRICS},
            )
        )

    if not accs:
        return [], {}

    combined_summary = mh.compute_many(
        accs,
        metrics=DEFAULT_METRICS,
        names=names,
        generate_overall=True,
    )
    # The "OVERALL" row is the aggregated result we want for the report.
    overall_row = combined_summary.loc["OVERALL"]
    combined = {k: overall_row[k] for k in DEFAULT_METRICS}

    return per_clip, combined


def format_summary_table(
    per_clip: List[ClipEvalResult],
    combined: Optional[dict] = None,
) -> str:
    """Format evaluation results as a human-readable text table."""
    metrics = DEFAULT_METRICS

    # Column widths: 24 for clip name, then varies by metric.
    name_w = max(24, max((len(c.clip) for c in per_clip), default=24))
    col_widths = {}
    for m in metrics:
        label = METRIC_LABELS.get(m, m)
        # Width is max(label, longest formatted value).
        w = len(label)
        for c in per_clip:
            v = c.metrics[m]
            w = max(w, len(_fmt_value(m, v)))
        if combined is not None and m in combined:
            w = max(w, len(_fmt_value(m, combined[m])))
        col_widths[m] = max(w, 6)

    # Header.
    lines = []
    header = f"{'clip':<{name_w}}  " + "  ".join(
        f"{METRIC_LABELS.get(m, m):>{col_widths[m]}}" for m in metrics
    )
    lines.append(header)
    lines.append("-" * len(header))

    for c in per_clip:
        row = f"{c.clip:<{name_w}}  " + "  ".join(
            f"{_fmt_value(m, c.metrics[m]):>{col_widths[m]}}" for m in metrics
        )
        lines.append(row)

    if combined is not None:
        lines.append("-" * len(header))
        row = f"{'OVERALL':<{name_w}}  " + "  ".join(
            f"{_fmt_value(m, combined[m]):>{col_widths[m]}}" for m in metrics
        )
        lines.append(row)

    return "\n".join(lines)


def _fmt_value(metric: str, value) -> str:
    """Format a metric value for display.

    Percent metrics get formatted with two decimals; counts as integers.
    """
    if value is None:
        return "-"
    # Detect whether to format as int (counts) or float (rates).
    int_metrics = {
        "num_frames",
        "mostly_tracked",
        "partially_tracked",
        "mostly_lost",
        "num_false_positives",
        "num_misses",
        "num_switches",
        "num_fragmentations",
        "num_unique_objects",
    }
    try:
        if metric in int_metrics:
            return f"{int(value)}"
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)
