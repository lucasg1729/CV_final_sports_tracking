"""Run the MOT I/O checks without needing pytest."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sports_tracker.mot_io import MotRow, read_mot_file, rows_to_dict, write_mot_file


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        sys.exit(1)


def approx(a, b, tol=0.01):
    return abs(a - b) <= tol


with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)

    print("Write/read round-trip")
    print("-" * 60)

    rows = [
        MotRow(frame=1, track_id=1, x1=10.0, y1=20.0, x2=60.0, y2=120.0, confidence=1.0),
        MotRow(frame=1, track_id=2, x1=100.0, y1=200.0, x2=150.0, y2=300.0, confidence=1.0),
        MotRow(frame=2, track_id=1, x1=15.0, y1=25.0, x2=65.0, y2=125.0, confidence=1.0),
    ]
    path = tmp_path / "out.txt"
    write_mot_file(path, rows)
    check("file written", path.exists())

    loaded = read_mot_file(path)
    check("3 rows loaded", len(loaded) == 3)
    all_match = True
    for original, recovered in zip(rows, loaded):
        if not (
            recovered.frame == original.frame
            and recovered.track_id == original.track_id
            and approx(recovered.x1, original.x1)
            and approx(recovered.y1, original.y1)
            and approx(recovered.x2, original.x2)
            and approx(recovered.y2, original.y2)
        ):
            all_match = False
    check("round-trip preserves coordinates", all_match)

    print()
    print("Sorting and edge cases")
    print("-" * 60)

    # Unsorted input.
    rows = [
        MotRow(frame=2, track_id=1, x1=0, y1=0, x2=10, y2=10, confidence=1.0),
        MotRow(frame=1, track_id=2, x1=0, y1=0, x2=10, y2=10, confidence=1.0),
        MotRow(frame=1, track_id=1, x1=0, y1=0, x2=10, y2=10, confidence=1.0),
    ]
    path = tmp_path / "sort.txt"
    write_mot_file(path, rows)
    loaded = read_mot_file(path)
    check(
        "rows sorted by (frame, track_id)",
        [(r.frame, r.track_id) for r in loaded] == [(1, 1), (1, 2), (2, 1)],
        f"got {[(r.frame, r.track_id) for r in loaded]}",
    )

    # SportsMOT-style row with extra columns.
    path = tmp_path / "gt.txt"
    path.write_text("1,1,10.0,20.0,50.0,100.0,1,1,1.0\n")
    loaded = read_mot_file(path)
    check("extra columns: 1 row loaded", len(loaded) == 1)
    r = loaded[0]
    check(
        "extra columns: x,y,w,h converted to x1,y1,x2,y2",
        r.frame == 1
        and r.track_id == 1
        and approx(r.x1, 10.0)
        and approx(r.y1, 20.0)
        and approx(r.x2, 60.0)
        and approx(r.y2, 120.0),
    )

    # Blank lines and comments.
    path = tmp_path / "comments.txt"
    path.write_text("# comment\n1,1,10,20,50,100,1.0,-1,-1,-1\n\n2,1,15,25,50,100,1.0,-1,-1,-1\n")
    loaded = read_mot_file(path)
    check("blank lines and comments skipped", len(loaded) == 2)

    # Malformed row should raise.
    path = tmp_path / "bad.txt"
    path.write_text("1,2,3\n")
    raised = False
    try:
        read_mot_file(path)
    except ValueError:
        raised = True
    check("malformed row raises ValueError", raised)

    # Parent directory creation.
    nested = tmp_path / "a" / "b" / "c.txt"
    write_mot_file(
        nested, [MotRow(frame=1, track_id=1, x1=0, y1=0, x2=1, y2=1, confidence=1.0)]
    )
    check("write_mot_file creates parent directories", nested.exists())

    # Empty file.
    path = tmp_path / "empty.txt"
    write_mot_file(path, [])
    check("empty input: file exists", path.exists())
    check("empty input: file is empty", path.read_text() == "")
    check("empty input: read returns empty list", read_mot_file(path) == [])

    # Grouping helper.
    rows = [
        MotRow(frame=1, track_id=1, x1=0, y1=0, x2=1, y2=1, confidence=1.0),
        MotRow(frame=1, track_id=2, x1=0, y1=0, x2=1, y2=1, confidence=1.0),
        MotRow(frame=2, track_id=1, x1=0, y1=0, x2=1, y2=1, confidence=1.0),
    ]
    d = rows_to_dict(rows)
    check("rows_to_dict: keys", set(d.keys()) == {1, 2})
    check("rows_to_dict: frame 1 has 2 rows", len(d[1]) == 2)
    check("rows_to_dict: frame 2 has 1 row", len(d[2]) == 1)

print()
print("All checks passed.")
