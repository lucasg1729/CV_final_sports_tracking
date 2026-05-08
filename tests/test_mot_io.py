"""Tests for the MOT format I/O module."""

from __future__ import annotations

from pathlib import Path

import pytest

from sports_tracker.mot_io import MotRow, read_mot_file, rows_to_dict, write_mot_file


def test_write_then_read_roundtrip(tmp_path: Path):
    """Write some rows, read them back; data should be identical
    (modulo conversion between [x1,y1,x2,y2] and [x,y,w,h])
    """
    rows = [
        MotRow(frame=1, track_id=1, x1=10.0, y1=20.0, x2=60.0, y2=120.0, confidence=1.0),
        MotRow(frame=1, track_id=2, x1=100.0, y1=200.0, x2=150.0, y2=300.0, confidence=1.0),
        MotRow(frame=2, track_id=1, x1=15.0, y1=25.0, x2=65.0, y2=125.0, confidence=1.0),
    ]
    path = tmp_path / "out.txt"
    write_mot_file(path, rows)
    loaded = read_mot_file(path)
    assert len(loaded) == 3
    for original, recovered in zip(rows, loaded):
        assert recovered.frame == original.frame
        assert recovered.track_id == original.track_id
        assert recovered.x1 == pytest.approx(original.x1, abs=0.01)
        assert recovered.y1 == pytest.approx(original.y1, abs=0.01)
        assert recovered.x2 == pytest.approx(original.x2, abs=0.01)
        assert recovered.y2 == pytest.approx(original.y2, abs=0.01)


def test_write_sorts_rows(tmp_path: Path):
    """Output should be sorted by (frame, track_id) regardless of input order"""
    rows = [
        MotRow(frame=2, track_id=1, x1=0, y1=0, x2=10, y2=10, confidence=1.0),
        MotRow(frame=1, track_id=2, x1=0, y1=0, x2=10, y2=10, confidence=1.0),
        MotRow(frame=1, track_id=1, x1=0, y1=0, x2=10, y2=10, confidence=1.0),
    ]
    path = tmp_path / "out.txt"
    write_mot_file(path, rows)
    loaded = read_mot_file(path)
    assert [(r.frame, r.track_id) for r in loaded] == [(1, 1), (1, 2), (2, 1)]


def test_read_handles_extra_columns(tmp_path: Path):
    """SportsMOT gt.txt has extra columns beyond the standard 7. Reader
    should pick up the first 6 numeric values plus confidence
    """
    path = tmp_path / "gt.txt"
    # SportsMOT-style row: frame, id, x, y, w, h, conf, class, vis
    path.write_text("1,1,10.0,20.0,50.0,100.0,1,1,1.0\n")
    loaded = read_mot_file(path)
    assert len(loaded) == 1
    r = loaded[0]
    assert r.frame == 1
    assert r.track_id == 1
    assert r.x1 == pytest.approx(10.0)
    assert r.y1 == pytest.approx(20.0)
    assert r.x2 == pytest.approx(60.0)
    assert r.y2 == pytest.approx(120.0)


def test_read_skips_blank_lines_and_comments(tmp_path: Path):
    path = tmp_path / "out.txt"
    path.write_text("# comment\n1,1,10,20,50,100,1.0,-1,-1,-1\n\n2,1,15,25,50,100,1.0,-1,-1,-1\n")
    loaded = read_mot_file(path)
    assert len(loaded) == 2


def test_read_rejects_malformed_row(tmp_path: Path):
    """A row with fewer than 6 fields should raise"""
    path = tmp_path / "out.txt"
    path.write_text("1,2,3\n")
    with pytest.raises(ValueError, match="Malformed"):
        read_mot_file(path)


def test_write_creates_parent_directories(tmp_path: Path):
    """write_mot_file should create the parent directory if needed,
    so callers don't have to mkdir manually before writing
    """
    nested = tmp_path / "a" / "b" / "c.txt"
    rows = [MotRow(frame=1, track_id=1, x1=0, y1=0, x2=1, y2=1, confidence=1.0)]
    write_mot_file(nested, rows)
    assert nested.exists()


def test_rows_to_dict_groups_correctly():
    rows = [
        MotRow(frame=1, track_id=1, x1=0, y1=0, x2=1, y2=1, confidence=1.0),
        MotRow(frame=1, track_id=2, x1=0, y1=0, x2=1, y2=1, confidence=1.0),
        MotRow(frame=2, track_id=1, x1=0, y1=0, x2=1, y2=1, confidence=1.0),
    ]
    d = rows_to_dict(rows)
    assert set(d.keys()) == {1, 2}
    assert len(d[1]) == 2
    assert len(d[2]) == 1


def test_write_empty(tmp_path: Path):
    """Writing an empty list should produce an empty file, not crash"""
    path = tmp_path / "empty.txt"
    write_mot_file(path, [])
    assert path.exists()
    assert path.read_text() == ""
    assert read_mot_file(path) == []
