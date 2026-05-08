"""Verify that the SportsMOT dataset is laid out correctly.

Run this after downloading and extracting SportsMOT. It checks that:
  - The dataset root exists
  - The configured split (train/val/test) exists under it
  - Each clip directory has the expected MOT-format structure:
      img1/        with frame JPGs numbered from 000001.jpg
      gt/gt.txt    ground-truth annotations  (skipped for test split)
      seqinfo.ini  metadata (frame rate, resolution, length)
  - The frame count in seqinfo.ini matches the number of JPGs on disk

Usage:
    python scripts/verify_data.py
    python scripts/verify_data.py --split train
    python scripts/verify_data.py --quiet      # only print summary + errors
"""

from __future__ import annotations

import argparse
import configparser
import sys
from pathlib import Path

# Make the package importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sports_tracker.config import load_config, resolve_path


def parse_seqinfo(path: Path) -> dict[str, str]:
    """Parse a SportsMOT seqinfo.ini file.

    seqinfo.ini uses a [Sequence] section with key=value pairs:
      name, imDir, frameRate, seqLength, imWidth, imHeight, imExt
    """
    parser = configparser.ConfigParser()
    # By default ConfigParser lower-cases all keys, but SportsMOT uses
    # camelCase (seqLength, imDir, imWidth, ...). Preserves case
    parser.optionxform = str
    parser.read(path)
    if "Sequence" not in parser:
        raise ValueError(f"{path} has no [Sequence] section")
    return dict(parser["Sequence"])


def check_clip(clip_dir: Path, expect_gt: bool) -> tuple[bool, list[str], dict]:
    """Check a single clip directory.

    Returns (ok, errors, info). `info` contains parsed metadata when
    available; on hard failures it may be empty.
    """
    errors: list[str] = []
    info: dict = {"name": clip_dir.name}

    # seqinfo.ini is the entry point for everything else
    seqinfo_path = clip_dir / "seqinfo.ini"
    if not seqinfo_path.exists():
        errors.append("missing seqinfo.ini")
        return False, errors, info

    try:
        meta = parse_seqinfo(seqinfo_path)
    except Exception as exc:
        errors.append(f"could not parse seqinfo.ini: {exc}")
        return False, errors, info

    info.update(meta)

    # img1/ should exist and contain JPGs
    img_dir = clip_dir / meta.get("imDir", "img1")
    if not img_dir.is_dir():
        errors.append(f"missing image directory {img_dir.name}/")
    else:
        ext = meta.get("imExt", ".jpg")
        frames = sorted(img_dir.glob(f"*{ext}"))
        info["frames_on_disk"] = len(frames)

        # Cross-check against the declared seqLength
        try:
            declared = int(meta.get("seqLength", "0"))
            if declared != len(frames):
                errors.append(
                    f"seqLength={declared} but found {len(frames)} {ext} frames"
                )
        except ValueError:
            errors.append(f"seqLength is not an integer: {meta.get('seqLength')!r}")

        # First frame is conventionally 000001.{ext}
        if frames and frames[0].name != f"000001{ext}":
            errors.append(f"first frame is {frames[0].name}, expected 000001{ext}")

    # gt/gt.txt should only be present in train and val splits, not test
    gt_path = clip_dir / "gt" / "gt.txt"
    if expect_gt:
        if not gt_path.exists():
            errors.append("missing gt/gt.txt")
        else:
            try:
                with open(gt_path) as f:
                    n_rows = sum(1 for _ in f)
                info["gt_rows"] = n_rows
                if n_rows == 0:
                    errors.append("gt.txt is empty")
            except Exception as exc:
                errors.append(f"could not read gt.txt: {exc}")

    return len(errors) == 0, errors, info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--split",
        default=None,
        help="Override the split from config (train/val/test).",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Only print errors and final summary, not per-clip lines.",
    )
    args = ap.parse_args()

    cfg = load_config()
    sportsmot_root = resolve_path(cfg["sportsmot_root"])
    split = args.split or cfg["split"]
    expect_gt = split != "test"

    print(f"Repo root          : {sportsmot_root.parent.parent}")
    print(f"SportsMOT root     : {sportsmot_root}")
    print(f"Split              : {split}")
    print()

    if not sportsmot_root.exists():
        print(f"ERROR: dataset root does not exist: {sportsmot_root}")
        print("       Have you downloaded SportsMOT? See README for instructions.")
        return 1

    split_dir = sportsmot_root / split
    if not split_dir.exists():
        print(f"ERROR: split directory does not exist: {split_dir}")
        print(f"       Expected layout: {sportsmot_root}/{{train,val,test}}/")
        return 1

    clip_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    if not clip_dirs:
        print(f"ERROR: no clip directories found inside {split_dir}")
        return 1

    n_ok = 0
    n_bad = 0
    bad_clips: list[tuple[str, list[str]]] = []

    for clip_dir in clip_dirs:
        ok, errors, info = check_clip(clip_dir, expect_gt=expect_gt)
        if ok:
            n_ok += 1
            if not args.quiet:
                fr = info.get("frameRate", "?")
                w = info.get("imWidth", "?")
                h = info.get("imHeight", "?")
                length = info.get("frames_on_disk", "?")
                gt = info.get("gt_rows", "-")
                print(
                    f"  OK   {clip_dir.name:<30}  "
                    f"{length} frames @ {fr} fps, {w}x{h}, gt rows: {gt}"
                )
        else:
            n_bad += 1
            bad_clips.append((clip_dir.name, errors))
            print(f"  FAIL {clip_dir.name}")
            for err in errors:
                print(f"         - {err}")

    print()
    print(f"Summary: {n_ok} clips OK, {n_bad} clips with problems")

    if bad_clips:
        print()
        print("Problems detected. The dataset may be partially downloaded or")
        print("corrupted. Re-extract the affected clips and rerun this script.")
        return 1

    print(f"All clips in {split} split look good.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
