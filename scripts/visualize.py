"""Render tracker output onto SportsMOT frames

Three modes:

1. --frame N         render a single frame as PNG (for figures)
2. --video           render every frame and write as MP4
3. --side-by-side    GT boxes left, predictions right, same frame

ID colors are hash-based and stable across frames. Side-by-side mode 
uses GT IDs and prediction IDs in the same color space, making 
identity errors more obvious
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sports_tracker.config import REPO_ROOT, load_config, resolve_path
from sports_tracker.mot_io import MotRow, read_mot_file, rows_to_dict


# Box visual style
BOX_THICKNESS = 2
LABEL_FONT_SCALE = 0.5
LABEL_THICKNESS = 1
LABEL_PAD = 4
TEXT_COLOR = (255, 255, 255)


def color_for_id(track_id: int) -> tuple[int, int, int]:
    """Map a track ID to a stable BGR color.

    Uses hash(id) modulo a hand-picked palette of distinguishable colors.
    The palette avoids near-grey and near-white so boxes stand out on
    typical sports footage backgrounds (greens, browns, lighter wood
    floors). All colors are in OpenCV's BGR ordering.
    """
    palette = [
        (0, 100, 255),      # orange
        (0, 200, 255),      # gold
        (50, 220, 50),      # green
        (255, 100, 0),      # blue
        (200, 50, 200),     # magenta
        (0, 255, 255),      # yellow
        (255, 0, 100),      # purple-blue
        (100, 200, 255),    # peach
        (50, 200, 200),     # mustard
        (255, 50, 50),      # red-blue
        (200, 100, 0),      # navy
        (0, 50, 255),       # red
    ]
    return palette[track_id % len(palette)]


def draw_box(image, bbox, color, label):
    """Draw a single bounding box with a labeled tab on top

    Args:
        image: BGR numpy array (modified in place)
        bbox: (x1, y1, x2, y2) in pixel coordinates
        color: BGR tuple
        label: short string to draw above the box
    """
    import cv2

    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, BOX_THICKNESS)

    # Compute label background size so text reads against a solid block
    # rather than against busy basketball court textures
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_THICKNESS
    )
    label_x1 = x1
    label_y2 = y1
    label_y1 = max(0, y1 - th - LABEL_PAD * 2)
    label_x2 = x1 + tw + LABEL_PAD * 2

    cv2.rectangle(image, (label_x1, label_y1), (label_x2, label_y2), color, -1)
    cv2.putText(
        image,
        label,
        (label_x1 + LABEL_PAD, label_y2 - LABEL_PAD - baseline + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        LABEL_FONT_SCALE,
        TEXT_COLOR,
        LABEL_THICKNESS,
        cv2.LINE_AA,
    )


def render_frame(
    image,
    rows: list[MotRow],
    title: str | None = None,
) -> "np.ndarray":
    """Draw all boxes from rows onto a copy of image"""
    import cv2

    canvas = image.copy()
    for r in rows:
        color = color_for_id(r.track_id)
        draw_box(canvas, (r.x1, r.y1, r.x2, r.y2), color, f"ID {r.track_id}")

    if title is not None:
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(
            canvas, title,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 255, 255), 2, cv2.LINE_AA,
        )

    return canvas


def render_side_by_side(
    image,
    gt_rows: list[MotRow],
    pred_rows: list[MotRow],
    gt_title: str = "Ground truth",
    pred_title: str = "Predictions",
) -> "np.ndarray":
    """Build a horizontal panel with GT on the left, predictions on the right"""
    import cv2

    left = render_frame(image, gt_rows, title=gt_title)
    right = render_frame(image, pred_rows, title=pred_title)
    return cv2.hconcat([left, right])


def load_frames_dir(clip_dir: Path) -> list[Path]:
    img_dir = clip_dir / "img1"
    paths = sorted(img_dir.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No JPG frames in {img_dir}")
    return paths


def cmd_single_frame(args, sportsmot_root, split):
    """Render one frame as a PNG"""
    import cv2

    clip_dir = sportsmot_root / split / args.clip
    frame_paths = load_frames_dir(clip_dir)
    if args.frame < 1 or args.frame > len(frame_paths):
        print(f"Frame {args.frame} out of range [1, {len(frame_paths)}].")
        return 1

    image = cv2.imread(str(frame_paths[args.frame - 1]))
    if image is None:
        print(f"Failed to read {frame_paths[args.frame - 1]}.")
        return 1

    pred_rows = read_mot_file(args.predictions) if args.predictions else []
    pred_by_frame = rows_to_dict(pred_rows)

    if args.side_by_side:
        gt_path = clip_dir / "gt" / "gt.txt"
        gt_rows = read_mot_file(gt_path) if gt_path.exists() else []
        gt_by_frame = rows_to_dict(gt_rows)
        canvas = render_side_by_side(
            image,
            gt_by_frame.get(args.frame, []),
            pred_by_frame.get(args.frame, []),
        )
    else:
        canvas = render_frame(
            image,
            pred_by_frame.get(args.frame, []),
            title=f"{args.clip}  frame {args.frame}",
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    print(f"Wrote {out_path}")
    return 0


def cmd_video(args, sportsmot_root, split):
    """Render every frame of a clip with overlaid boxes and write to MP4"""
    import cv2

    clip_dir = sportsmot_root / split / args.clip
    frame_paths = load_frames_dir(clip_dir)

    pred_rows = read_mot_file(args.predictions) if args.predictions else []
    pred_by_frame = rows_to_dict(pred_rows)

    gt_by_frame = {}
    if args.side_by_side:
        gt_path = clip_dir / "gt" / "gt.txt"
        if gt_path.exists():
            gt_by_frame = rows_to_dict(read_mot_file(gt_path))

    # Read the first frame to get dimensions and FPS
    first = cv2.imread(str(frame_paths[0]))
    h, w = first.shape[:2]
    out_w = w * 2 if args.side_by_side else w

    # FPS should pull from seqinfo.ini, else default
    import configparser
    fps = 25.0
    seqinfo = clip_dir / "seqinfo.ini"
    if seqinfo.exists():
        parser = configparser.ConfigParser()
        parser.optionxform = str
        parser.read(seqinfo)
        try:
            fps = float(parser["Sequence"]["frameRate"])
        except (KeyError, ValueError):
            pass

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # mp4v works on macOS/Linux without extra ffmpeg installs
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, h))
    if not writer.isOpened():
        print(f"Failed to open video writer for {out_path}.")
        return 1

    for i, frame_path in enumerate(frame_paths, start=1):
        image = cv2.imread(str(frame_path))
        if image is None:
            print(f"  warning: failed to read {frame_path}, skipping")
            continue

        if args.side_by_side:
            canvas = render_side_by_side(
                image,
                gt_by_frame.get(i, []),
                pred_by_frame.get(i, []),
            )
        else:
            canvas = render_frame(
                image, pred_by_frame.get(i, []),
                title=f"{args.clip}  frame {i}",
            )

        writer.write(canvas)

        if i % 100 == 0 or i == len(frame_paths):
            print(f"  rendered {i}/{len(frame_paths)} frames")

    writer.release()
    print(f"Wrote {out_path}  ({len(frame_paths)} frames @ {fps:.0f} fps)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, help="Clip name (folder under val/).")
    ap.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Path to a MOT-format predictions file. If omitted, only GT is shown "
        "(only meaningful with --side-by-side).",
    )
    ap.add_argument("--split", default=None, help="Override split from config.")

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--frame", type=int, help="Render a single 1-indexed frame as PNG.")
    mode.add_argument("--video", action="store_true", help="Render every frame as MP4.")

    ap.add_argument(
        "--side-by-side",
        action="store_true",
        help="Render GT and predictions side-by-side on each frame.",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output path. Use .png for --frame, .mp4 for --video.",
    )
    args = ap.parse_args()

    cfg = load_config()
    sportsmot_root = resolve_path(cfg["sportsmot_root"])
    split = args.split or cfg["split"]

    if args.frame is not None:
        return cmd_single_frame(args, sportsmot_root, split)
    return cmd_video(args, sportsmot_root, split)


if __name__ == "__main__":
    sys.exit(main())
