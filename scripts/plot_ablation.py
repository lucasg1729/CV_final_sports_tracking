"""Plot ablation results from per-tag summary.csv files.

Reads results/<tag>/summary.csv for each ablation run, extracts the
OVERALL row's metrics, and produces line plots of MOTA / IDF1 vs the
swept parameter.

Usage:
    python scripts/plot_ablation.py \
        --param det_conf --tags conf-0.45 conf-0.50 conf-0.55 conf-0.60 \
        --out results/figures/mota_vs_conf.png

    python scripts/plot_ablation.py \
        --param max_age --tags maxage-1 maxage-5 maxage-15 maxage-30 maxage-60 \
        --out results/figures/mota_vs_maxage.png

The --param flag is just used as the x-axis label; the actual numeric
values are extracted from each tag name (e.g. "conf-0.55" -> 0.55,
"maxage-30" -> 30).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sports_tracker.config import REPO_ROOT


def extract_value(tag: str) -> float:
    """Pull the numeric value out of a tag name.

    'conf-0.55' -> 0.55, 'maxage-30' -> 30.0, 'iou-0.3' -> 0.3.
    Falls back to the raw string if no number is found, which would
    fail later -- worth knowing immediately rather than producing a
    silently misleading plot.
    """
    match = re.search(r"[-_](\d+\.?\d*)$", tag)
    if not match:
        raise ValueError(
            f"Could not extract numeric value from tag {tag!r}. "
            f"Tags should end in -<number>, e.g. 'conf-0.55' or 'maxage-30'."
        )
    return float(match.group(1))


def load_overall_row(csv_path: Path) -> dict[str, float]:
    """Read a summary.csv and return the OVERALL row as a dict."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("clip") == "OVERALL":
                # Convert numeric fields. The CSV stores them as strings.
                out = {}
                for k, v in row.items():
                    if k == "clip":
                        continue
                    try:
                        out[k] = float(v)
                    except (TypeError, ValueError):
                        out[k] = v  # leave non-numeric as string
                return out
    raise ValueError(f"No OVERALL row found in {csv_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tags",
        nargs="+",
        required=True,
        help="Space-separated list of tag directories under results/. "
        "Each must contain a summary.csv with an OVERALL row.",
    )
    ap.add_argument(
        "--param",
        required=True,
        help="Name of the parameter being swept (used as x-axis label).",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Path to write the figure to. Parent directory will be created.",
    )
    ap.add_argument(
        "--results-dir",
        default=None,
        help="Where the tag directories live. Default: results/.",
    )
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=["MOTA", "IDF1"],
        help="Which metrics to plot. Default: MOTA and IDF1.",
    )
    args = ap.parse_args()

    # Lazy import so this script doesn't blow up at import time on
    # systems without matplotlib.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is not installed. Install with:\n"
            "    pip install matplotlib"
        )
        return 1

    results_dir = (
        Path(args.results_dir) if args.results_dir else REPO_ROOT / "results"
    )

    # Load each tag's summary CSV.
    points = []  # list of (param_value, metric_dict)
    for tag in args.tags:
        csv_path = results_dir / tag / "summary.csv"
        if not csv_path.exists():
            print(f"[skip] {tag}: no summary at {csv_path}")
            continue
        try:
            value = extract_value(tag)
            metrics = load_overall_row(csv_path)
        except Exception as exc:
            print(f"[fail] {tag}: {exc}")
            continue
        points.append((value, metrics, tag))

    if not points:
        print("No data loaded; nothing to plot.")
        return 1

    # Sort by parameter value so the line plot is monotonic on x.
    points.sort(key=lambda p: p[0])

    xs = [p[0] for p in points]
    print(f"Plotting {len(points)} points for {args.param}: {xs}")

    fig, ax_mota = plt.subplots(figsize=(7, 4.5))

    # Two-axis plot: MOTA on left, IDF1 (and other rate metrics) on right.
    # This makes the relative shapes of both curves comparable at a glance,
    # since MOTA and IDF1 typically live in the same 0-1 range but can
    # peak at different x values.
    ax_idf1 = ax_mota.twinx() if len(args.metrics) > 1 else None

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, metric in enumerate(args.metrics):
        ys = [p[1].get(metric) for p in points]
        # Skip a metric entirely if any point is missing it.
        if any(y is None for y in ys):
            print(f"[skip] metric {metric!r}: not found in all summaries")
            continue
        target_ax = ax_mota if i == 0 or ax_idf1 is None else ax_idf1
        color = colors[i % len(colors)]
        target_ax.plot(xs, ys, marker="o", color=color, label=metric, linewidth=2)
        target_ax.set_ylabel(metric, color=color)
        target_ax.tick_params(axis="y", labelcolor=color)

        # Annotate the peak point so it's easy to read off the figure.
        # Stagger MOTA vs IDF1 vertically so labels don't overlap when
        # both metrics peak at the same x value.
        peak_idx = max(range(len(ys)), key=lambda j: ys[j])
        y_offset = 8 if i == 0 else -14
        target_ax.annotate(
            f"{metric}={ys[peak_idx]:.3f}",
            xy=(xs[peak_idx], ys[peak_idx]),
            xytext=(6, y_offset),
            textcoords="offset points",
            color=color,
            fontsize=9,
            fontweight="bold",
        )

    ax_mota.set_xlabel(args.param)
    ax_mota.set_title(f"Tracking metrics vs {args.param}")
    ax_mota.grid(True, alpha=0.3)

    # Combined legend (matplotlib's twin-axis legend is fiddly).
    lines, labels = ax_mota.get_legend_handles_labels()
    if ax_idf1 is not None:
        l2, lab2 = ax_idf1.get_legend_handles_labels()
        lines += l2
        labels += lab2
    ax_mota.legend(lines, labels, loc="best")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
