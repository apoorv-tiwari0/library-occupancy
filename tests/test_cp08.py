"""
test_cp08.py — Validates the ROI annotator tool (CP-08).

Since the annotator is interactive (requires mouse clicks), we test it
programmatically by simulating mouse events and key presses directly
on the ROIAnnotator object — no actual window interaction needed.

Tests:
  1. ROIAnnotator initialises and loads existing JSON correctly
  2. Polygon creation (simulated clicks + ENTER)
  3. Polygon undo (Z key)
  4. JSON output has correct nested structure
  5. Appending to existing ROI file works (doesn't overwrite other sections)
  6. Minimum vertex validation (< 3 points rejected)

Run from project root:
    python test_cp08.py
"""

import sys
import shutil
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from roi.annotator import ROIAnnotator
from utils.helpers import read_json, write_json


def make_dummy_frame(h=480, w=640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def simulate_annotator(
    section_id: str,
    roi_path: Path,
    polygons: list[list[tuple]],
) -> dict:
    """
    Create an ROIAnnotator, inject a dummy frame, simulate polygon clicks,
    and return the resulting ROI data — without opening any window.
    """
    ann = ROIAnnotator(
        section_id=section_id,
        source=0,           # won't be used — we inject frame directly
        roi_path=roi_path,
    )
    ann._base_frame = make_dummy_frame()

    for pts in polygons:
        # Simulate left clicks
        for x, y in pts:
            ann._mouse_callback(1, x, y, None, None)  # EVENT_LBUTTONDOWN = 1
        # Simulate ENTER to close polygon
        ann._close_polygon()

    # Save
    ann._save()
    return ann._roi_data


def test_basic_polygon_creation(tmpdir: Path) -> None:
    roi_path = tmpdir / "roi.json"
    polygons = [
        [(100, 100), (200, 100), (200, 200), (100, 200)],   # seat 1
        [(300, 100), (400, 100), (400, 200), (300, 200)],   # seat 2
    ]
    roi_data = simulate_annotator("section_A", roi_path, polygons)

    assert "section_A" in roi_data, "section_A missing from output"
    assert len(roi_data["section_A"]) == 2, \
        f"Expected 2 seats, got {len(roi_data['section_A'])}"

    for seat_id, pts in roi_data["section_A"].items():
        assert len(pts) == 4, f"{seat_id} should have 4 vertices"
        for pt in pts:
            assert len(pt) == 2, f"Each vertex should be [x, y], got {pt}"

    print(f"  ✅ Polygon creation: {list(roi_data['section_A'].keys())}")


def test_json_structure(tmpdir: Path) -> None:
    roi_path = tmpdir / "roi_struct.json"
    simulate_annotator("section_B", roi_path, [
        [(50, 50), (150, 50), (150, 150), (50, 150)],
    ])
    loaded = read_json(roi_path)
    assert isinstance(loaded, dict), "Top level must be a dict"
    assert "section_B" in loaded
    assert isinstance(loaded["section_B"], dict), "Section value must be a dict"
    for seat_id, pts in loaded["section_B"].items():
        assert isinstance(seat_id, str), "Seat ID must be a string"
        assert isinstance(pts, list),    "Points must be a list"
        assert all(isinstance(p, list) and len(p) == 2 for p in pts), \
            "Each point must be [x, y]"
    print(f"  ✅ JSON structure valid: {loaded}")


def test_append_does_not_overwrite(tmpdir: Path) -> None:
    """Annotating section_B should not erase section_A data."""
    roi_path = tmpdir / "roi_append.json"

    # First annotation — section_A
    simulate_annotator("section_A", roi_path, [
        [(10, 10), (100, 10), (100, 100), (10, 100)],
    ])

    # Second annotation — section_B (different annotator instance)
    simulate_annotator("section_B", roi_path, [
        [(200, 200), (300, 200), (300, 300), (200, 300)],
    ])

    final = read_json(roi_path)
    assert "section_A" in final, "section_A was erased!"
    assert "section_B" in final, "section_B missing"
    assert len(final["section_A"]) == 1
    assert len(final["section_B"]) == 1
    print(f"  ✅ Append works — both sections preserved: {list(final.keys())}")


def test_minimum_vertex_validation(tmpdir: Path) -> None:
    """Polygons with fewer than 3 vertices must be rejected."""
    roi_path = tmpdir / "roi_min.json"
    ann = ROIAnnotator(
        section_id="section_C",
        source=0,
        roi_path=roi_path,
    )
    ann._base_frame = make_dummy_frame()

    # Simulate only 2 clicks (not enough for a polygon)
    ann._mouse_callback(1, 100, 100, None, None)
    ann._mouse_callback(1, 200, 100, None, None)
    ann._close_polygon()   # should be rejected

    assert "section_C" not in ann._roi_data or \
           len(ann._roi_data.get("section_C", {})) == 0, \
        "Polygon with < 3 vertices should be rejected"
    print(f"  ✅ Minimum vertex validation works (< 3 points rejected)")


def test_undo(tmpdir: Path) -> None:
    """Undo should remove the last completed polygon."""
    roi_path = tmpdir / "roi_undo.json"
    ann = ROIAnnotator(
        section_id="section_D",
        source=0,
        roi_path=roi_path,
    )
    ann._base_frame = make_dummy_frame()

    # Add two polygons
    for x_off in [0, 200]:
        for x, y in [(50+x_off,50),(150+x_off,50),(150+x_off,150),(50+x_off,150)]:
            ann._mouse_callback(1, x, y, None, None)
        ann._close_polygon()

    assert len(ann._roi_data.get("section_D", {})) == 2

    # Undo last
    ann._undo_last()
    assert len(ann._roi_data.get("section_D", {})) == 1, \
        "After undo, should have 1 polygon"
    print(f"  ✅ Undo works — 2 polygons → undo → 1 polygon")


def test_multi_section_single_file(tmpdir: Path) -> None:
    """All 11 sections should coexist in one ROI file."""
    roi_path = tmpdir / "roi_full.json"
    sections = [f"section_{chr(65+i)}" for i in range(11)]  # A through K

    for sec in sections:
        simulate_annotator(sec, roi_path, [
            [(10, 10), (100, 10), (100, 100), (10, 100)],
            [(200, 10), (300, 10), (300, 100), (200, 100)],
        ])

    final = read_json(roi_path)
    assert len(final) == 11, f"Expected 11 sections, got {len(final)}"
    for sec in sections:
        assert sec in final, f"{sec} missing from ROI file"
        assert len(final[sec]) == 2, \
            f"{sec} should have 2 seats, got {len(final[sec])}"
    print(f"  ✅ All 11 sections coexist in one file: {list(final.keys())}")
    print(f"     Total seats: {sum(len(v) for v in final.values())}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("CP-08 — ROI Annotator Tool — Validation Test")
    print("=" * 60)

    tmpdir = Path(tempfile.mkdtemp(prefix="cp08_"))

    try:
        print("\n[1] Basic polygon creation:")
        test_basic_polygon_creation(tmpdir)

        print("\n[2] JSON structure validation:")
        test_json_structure(tmpdir)

        print("\n[3] Append without overwrite:")
        test_append_does_not_overwrite(tmpdir)

        print("\n[4] Minimum vertex validation:")
        test_minimum_vertex_validation(tmpdir)

        print("\n[5] Undo functionality:")
        test_undo(tmpdir)

        print("\n[6] All 11 sections in one file:")
        test_multi_section_single_file(tmpdir)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("\n" + "=" * 60)
    print("🎉  CP-08 PASSED — ROI annotator ready")
    print("=" * 60)
    print("\nTo use the annotator on real footage:")
    print("  python roi/annotator.py --section section_A --image data/sample_frames/section_A.jpg")
    print("  python roi/annotator.py --section section_A --video data/test_videos/section_A.mp4")