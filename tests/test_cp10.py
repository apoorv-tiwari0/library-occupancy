"""
test_cp10.py — Validates ROI Reviewer logic (CP-10).

Tests all state transitions programmatically without opening a window.
"""

import sys, shutil, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from roi.reviewer import ROIReviewer
from utils.helpers import read_json


def make_frame():
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def make_draft():
    return {
        "cad_lab": {
            "seat_LAB1": [[50,50],[200,50],[200,200],[50,200]],
            "seat_LAB2": [[250,50],[400,50],[400,200],[250,200]],
            "seat_LAB3": [[450,50],[600,50],[600,200],[450,200]],
        },
        "section_B": {
            "seat_B1": [[10,10],[100,10],[100,100],[10,100]],
        }
    }


def make_reviewer(tmpdir):
    return ROIReviewer(
        section_id  = "cad_lab",
        frame       = make_frame(),
        draft_data  = make_draft(),
        output_path = Path(tmpdir) / "roi_polygons.json",
    )


def test_loads_correct_section(tmpdir):
    r = make_reviewer(tmpdir)
    assert len(r._seats) == 3, f"Expected 3 seats, got {len(r._seats)}"
    assert "seat_LAB1" in r._seats
    assert "section_B" not in r._seats   # other section must not bleed in
    print("  ✅ Loads correct section only")


def test_delete(tmpdir):
    r = make_reviewer(tmpdir)
    r._selected = "seat_LAB1"
    r._delete_selected()
    assert "seat_LAB1" not in r._seats
    assert len(r._seats) == 2
    print("  ✅ Delete works")


def test_delete_nothing_selected(tmpdir):
    r = make_reviewer(tmpdir)
    r._selected = None
    r._delete_selected()          # should not crash
    assert len(r._seats) == 3    # nothing removed
    print("  ✅ Delete with nothing selected is safe")


def test_add_polygon(tmpdir):
    r = make_reviewer(tmpdir)
    r._start_add()
    assert r._mode == "draw"
    for x, y in [(10,10),(100,10),(100,100),(10,100)]:
        r._mouse_callback(1, x, y, None, None)
    r._confirm_polygon()
    assert r._mode == "view"
    assert len(r._seats) == 4
    print(f"  ✅ Add polygon: now {len(r._seats)} seats")


def test_redraw(tmpdir):
    r = make_reviewer(tmpdir)
    r._selected = "seat_LAB2"
    r._start_redraw()
    assert r._mode == "draw"
    assert "seat_LAB2" not in r._seats   # removed while redrawing
    for x, y in [(300,300),(500,300),(500,500),(300,500)]:
        r._mouse_callback(1, x, y, None, None)
    r._confirm_polygon()
    assert "seat_LAB2" in r._seats
    assert r._seats["seat_LAB2"] == [[300,300],[500,300],[500,500],[300,500]]
    print("  ✅ Redraw replaces polygon correctly")


def test_cancel_draw(tmpdir):
    r = make_reviewer(tmpdir)
    r._start_add()
    r._mouse_callback(1, 10, 10, None, None)
    r._cancel_draw()
    assert r._mode == "view"
    assert r._current_polygon == []
    assert len(r._seats) == 3   # nothing added
    print("  ✅ Cancel draw leaves seats unchanged")


def test_minimum_vertices(tmpdir):
    r = make_reviewer(tmpdir)
    r._start_add()
    r._mouse_callback(1, 10, 10, None, None)
    r._mouse_callback(1, 50, 10, None, None)
    r._confirm_polygon()          # only 2 points — should be rejected
    assert r._mode == "draw"      # stays in draw mode
    assert len(r._seats) == 3     # no new seat added
    print("  ✅ < 3 vertices correctly rejected")


def test_other_sections_preserved(tmpdir):
    r = make_reviewer(tmpdir)
    r._selected = "seat_LAB1"
    r._delete_selected()
    r._save()
    saved = read_json(r.output_path)
    assert "section_B" in saved, "Other section was erased on save"
    assert len(saved["section_B"]) == 1
    print("  ✅ Other sections preserved after save")


def test_select_by_click(tmpdir):
    r = make_reviewer(tmpdir)
    # Click inside seat_LAB1 polygon [[50,50],[200,50],[200,200],[50,200]]
    r._try_select(125, 125)
    assert r._selected == "seat_LAB1", f"Expected seat_LAB1, got {r._selected}"
    # Click outside all polygons
    r._try_select(700, 700)
    assert r._selected is None
    print("  ✅ Click-to-select works")


if __name__ == "__main__":
    print("=" * 55)
    print("CP-10 — ROI Reviewer — Validation Tests")
    print("=" * 55)

    tmpdir = tempfile.mkdtemp(prefix="cp10_")
    try:
        print("\n[1] Section loading:")
        test_loads_correct_section(tmpdir)

        print("\n[2] Delete selected:")
        test_delete(tmpdir)

        print("\n[3] Delete with nothing selected:")
        test_delete_nothing_selected(tmpdir)

        print("\n[4] Add new polygon:")
        test_add_polygon(tmpdir)

        print("\n[5] Redraw polygon:")
        test_redraw(tmpdir)

        print("\n[6] Cancel draw:")
        test_cancel_draw(tmpdir)

        print("\n[7] Minimum vertex rejection:")
        test_minimum_vertices(tmpdir)

        print("\n[8] Other sections preserved:")
        test_other_sections_preserved(tmpdir)

        print("\n[9] Click-to-select:")
        test_select_by_click(tmpdir)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("\n" + "=" * 55)
    print("🎉 CP-10 PASSED — ROI reviewer ready")
    print("=" * 55)
    print("\nTo review your cad_lab draft:")
    print("  python roi/reviewer.py --section cad_lab --draft data/roi/roi_auto_draft.json --image data/sample_frames/cad_lab.jpg")