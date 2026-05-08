"""MOT Challenge format I/O.

The MOT Challenge text format is the de facto standard for multi-object
tracking benchmarks. Each line is a single (frame, track) pair:

    frame, id, bb_x1, bb_y1, bb_w, bb_h, conf, x, y, z

Where:
    frame   1-indexed frame number
    id      integer track id (-1 in detection files, but >0 in tracking output)
    bb_*    bounding box in TOP-LEFT + WIDTH/HEIGHT format. Note: this is
            DIFFERENT from our internal [x1, y1, x2, y2] representation;
            we convert at I/O boundaries.
    conf    detection confidence; 1.0 for ground truth
    x, y, z 3D world coordinates. Always -1, -1, -1 for 2D tracking.

SportsMOT ground truth files use this format with one extra convention:
gt.txt rows have additional fields after column 6 indicating whether
the row is "valid" (column 7 = 1), the class label, and visibility.
For our purposes we only read columns 1-6
"""

from __future__ import annotations

from pathlib import Path
from typing import List, NamedTuple

import numpy as np


class MotRow(NamedTuple):
    """One row of a MOT-format file"""

    frame: int
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


def write_mot_file(
    out_path: Path,
    rows: List[MotRow],
) -> None:
    """Write tracking results in MOT Challenge format

    Rows are sorted by (frame, track_id) before writing, matching the
    convention expected by the motmetrics evaluator

    Args:
        out_path: where to write. Parent dir is created if needed
        rows: tracking output rows. The internal [x1, y1, x2, y2] format
            is converted to MOT's top-left + width/height format
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_rows = sorted(rows, key=lambda r: (r.frame, r.track_id))

    with open(out_path, "w") as f:
        for r in sorted_rows:
            w = r.x2 - r.x1
            h = r.y2 - r.y1
            # MOT format: frame, id, x, y, w, h, conf, x_world, y_world, z_world
            # World coords are -1 for 2D tracking
            f.write(
                f"{r.frame},{r.track_id},"
                f"{r.x1:.2f},{r.y1:.2f},{w:.2f},{h:.2f},"
                f"{r.confidence:.4f},-1,-1,-1\n"
            )


def read_mot_file(path: Path) -> List[MotRow]:
    """Read a MOT-format file and return rows in [x1, y1, x2, y2] format

    Handles both prediction files and ground-truth files (which may have
    extra columns after column 6 in some datasets including SportsMOT).
    Only the first 6 numeric fields are interpreted
    """
    path = Path(path)
    rows: List[MotRow] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 6:
                raise ValueError(
                    f"Malformed MOT row in {path}: '{line}' (need >=6 fields)"
                )
            frame = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            # Confidence is column 7 (index 6); default to 1.0 if absent
            confidence = float(parts[6]) if len(parts) > 6 else 1.0
            rows.append(
                MotRow(
                    frame=frame,
                    track_id=track_id,
                    x1=x,
                    y1=y,
                    x2=x + w,
                    y2=y + h,
                    confidence=confidence,
                )
            )
    return rows


def rows_to_dict(rows: List[MotRow]) -> dict[int, list[MotRow]]:
    """Group MOT rows by frame number for quick per-frame lookup"""
    by_frame: dict[int, list[MotRow]] = {}
    for r in rows:
        by_frame.setdefault(r.frame, []).append(r)
    return by_frame
