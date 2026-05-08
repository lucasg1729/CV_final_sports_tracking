"""Compare multiple tracker configurations side-by-side as a bar chart

Reads the OVERALL row from each summary.csv under results/<tag>/ and
plots one bar per (tag, metric) combination as a multi-panel figure
(one panel per metric).

Usage:
    python scripts/plot_comparison.py \
        --tags maxage-5 maxage-15 yolo-bytetrack yolo-botsort \
        --labels "Ours (max_age=5)" "Ours (max_age=15)" "ByteTrack" "BoT-SORT" \
        --metrics MOTA IDF1 IDs Frag \
        --out results/figures/comparison.png

The --labels flag is optional: if omitted, the tag names are used.
Higher-is-better metrics (MOTA, IDF1, MT) and lower-is-better metrics
(IDs, Frag, FP, FN) are detected from a built-in list and bar colors
are adjusted accordingly so the "best" bar is always highlighted.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sports_tracker.config import REPO_ROOT


# Metrics where higher = better. Anything not in this set is treated
# as lower = better so (FP, FN, IDs, Frag, ML)
HIGHER_IS_BETTER = {"MOTA", "MOTP", "IDF1", "IDP", "IDR", "MT"}


def load_overall_row(csv_path: Path) -> dict[str, float]:
    """Return the OVERALL row of a summary CSV as a metric dict"""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("clip") == "OVERALL":
                out = {}
                for k, v in row.items():
                    if k == "clip":
                        continue
                    try:
                        out[k] = float(v)
                    except (TypeError, ValueError):
                        out[k] = v
                return out
    raise ValueError(f"No OVERALL row in {csv_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tags",
        nargs="+",
        required=True,
        help="Tag directories under results/. Each must contain summary.csv.",
    )
    ap.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Display labels (one per tag). Defaults to the tag names.",
    )
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=["MOTA", "IDF1", "IDs", "Frag"],
        help="Metric columns to plot. Default: MOTA IDF1 IDs Frag.",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Path to write the figure to.",
    )
    ap.add_argument(
        "--results-dir",
        default=None,
        help="Where the tag directories live. Default: results/.",
    )
    ap.add_argument(
        "--title",
        default="Tracker comparison",
        help="Figure suptitle.",
    )
    args = ap.parse_args()

    if args.labels and len(args.labels) != len(args.tags):
        ap.error(
            f"--labels has {len(args.labels)} entries but --tags has {len(args.tags)}"
        )
    labels = args.labels or args.tags

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required: pip install matplotlib")
        return 1

    results_dir = (
        Path(args.results_dir) if args.results_dir else REPO_ROOT / "results"
    )

    rows = []  # list of (label, metrics_dict)
    for tag, label in zip(args.tags, labels):
        csv_path = results_dir / tag / "summary.csv"
        if not csv_path.exists():
            print(f"[skip] {tag}: no summary at {csv_path}")
            continue
        try:
            metrics = load_overall_row(csv_path)
        except Exception as exc:
            print(f"[fail] {tag}: {exc}")
            continue
        rows.append((label, metrics))

    if not rows:
        print("No data; nothing to plot.")
        return 1

    # square grid of subplots, one per metric
    n_metrics = len(args.metrics)
    n_cols = min(n_metrics, 2)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.5 * n_cols, 3.5 * n_rows), squeeze=False
    )
    fig.suptitle(args.title, fontsize=13, fontweight="bold")

    bar_labels = [r[0] for r in rows]
    x_positions = list(range(len(bar_labels)))

    # Use a single base color per panel, with the best bar highlighted 
    # in a darker shade
    base_color = "#7aa6d6"
    best_color = "#1f5fa3"

    for idx, metric in enumerate(args.metrics):
        ax = axes[idx // n_cols][idx % n_cols]
        values = [r[1].get(metric) for r in rows]
        if any(v is None for v in values):
            ax.text(
                0.5, 0.5, f"{metric}: missing",
                ha="center", va="center", transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue

        higher_better = metric in HIGHER_IS_BETTER
        best_idx = max(range(len(values)), key=lambda i: values[i] if higher_better else -values[i])

        colors = [best_color if i == best_idx else base_color for i in range(len(values))]
        bars = ax.bar(x_positions, values, color=colors, edgecolor="white", linewidth=1.0)

        # Annotate each bar with its value above it
        for bar, v in zip(bars, values):
            offset = (max(values) - min(values)) * 0.02 if max(values) != min(values) else 0.01
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{v:.3f}" if isinstance(v, float) and abs(v) < 100 else f"{int(v)}",
                ha="center", va="bottom", fontsize=9,
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(bar_labels, rotation=15, ha="right", fontsize=9)
        ax.set_title(
            f"{metric}  ({'higher = better' if higher_better else 'lower = better'})",
            fontsize=10,
        )
        ax.grid(True, axis="y", alpha=0.3)

        # Pad y-axis so annotations don't get cropped
        ymin = min(0, min(values) * 1.1) if min(values) < 0 else 0
        ymax = max(values) * 1.18 if max(values) > 0 else max(values) * 0.85
        ax.set_ylim(ymin, ymax)

    # Hide any unused subplot cells
    for j in range(n_metrics, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_axis_off()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
