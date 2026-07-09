"""
test_zcp01.py — Validates zone annotation tool (ZCP-01)

Tests:
  1. ZoneAnnotator initialises correctly
  2. Zone confirmation works (simulated clicks + ENTER)
  3. JSON structure is correct {section: {near/mid/far: [[x,y],...]}}
  4. Auto-advance near→mid→far after each ENTER
  5. Undo clears a zone
  6. Append to existing file (other sections preserved)
  7. Minimum vertex validation (< 3 rejected)
  8. All 11 sections can coexist in one file

Run from project root:
    python test_zcp01.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from roi.zone_annotator import ZoneAnnotator, ZONE_NAMES


def make_annotator(section_id: str, zone_path: Path) -> ZoneAnnotator:
    ann = ZoneAnnotator(
        section_id = section_id,
        source     = 0,
        zone_path  = zone_path,
    )
    ann._base_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    return ann


def sim_zone(ann: ZoneAnnotator, zone: str, pts: list[tuple]) -> None:
    """Simulate clicking points and pressing ENTER to confirm a zone."""
    ann._active_zone  = zone
    ann._current_poly = []
    for x, y in pts:
        ann._mouse_cb(1, x, y, None, None)
    ann._confirm_zone()


def near_pts(): return [(50,50),(200,50),(200,200),(50,200)]
def mid_pts():  return [(50,200),(200,200),(200,350),(50,350)]
def far_pts():  return [(50,350),(200,350),(200,450),(50,450)]


def test_init(tmpdir: Path) -> None:
    ann = make_annotator("cad_lab", tmpdir / "z.json")
    assert ann.section_id   == "cad_lab"
    assert ann._active_zone == "near"
    assert all(len(ann._zones[z]) == 0 for z in ZONE_NAMES)
    print("  ✅ Init — all zones empty, active=near")


def test_confirm_zone(tmpdir: Path) -> None:
    ann = make_annotator("cad_lab", tmpdir / "z.json")
    sim_zone(ann, "near", near_pts())
    assert len(ann._zones["near"]) == 4
    assert ann._active_zone == "mid"   # auto-advanced
    print("  ✅ Zone confirmed — 4 vertices, auto-advanced to mid")


def test_all_three_zones(tmpdir: Path) -> None:
    ann = make_annotator("cad_lab", tmpdir / "z.json")
    sim_zone(ann, "near", near_pts())
    sim_zone(ann, "mid",  mid_pts())
    sim_zone(ann, "far",  far_pts())
    ann._save()

    data = json.loads((tmpdir / "z.json").read_text())
    assert "cad_lab" in data
    for z in ZONE_NAMES:
        assert z in data["cad_lab"], f"Zone '{z}' missing"
        assert len(data["cad_lab"][z]) == 4, f"Zone '{z}' should have 4 pts"
    print("  ✅ All 3 zones confirmed and saved correctly")


def test_json_structure(tmpdir: Path) -> None:
    ann = make_annotator("g_huss", tmpdir / "z.json")
    sim_zone(ann, "near", near_pts())
    sim_zone(ann, "mid",  mid_pts())
    sim_zone(ann, "far",  far_pts())
    ann._save()

    data = json.loads((tmpdir / "z.json").read_text())
    assert isinstance(data, dict)
    assert "g_huss" in data
    for z in ZONE_NAMES:
        pts = data["g_huss"][z]
        assert isinstance(pts, list)
        assert all(isinstance(p, list) and len(p) == 2 for p in pts)
    print("  ✅ JSON structure valid: {section: {near/mid/far: [[x,y],...]}}")


def test_auto_advance(tmpdir: Path) -> None:
    ann = make_annotator("cad_lab", tmpdir / "z.json")
    assert ann._active_zone == "near"
    sim_zone(ann, "near", near_pts())
    assert ann._active_zone == "mid"
    sim_zone(ann, "mid", mid_pts())
    assert ann._active_zone == "far"
    print("  ✅ Auto-advance: near → mid → far after each ENTER")


# def test_undo(tmpdir: Path) -> None:
#     ann = make_annotator("cad_lab", tmpdir / "z.json")
#     sim_zone(ann, "near", near_pts())
#     assert len(ann._zones["near"]) == 4
#     ann._active_zone = "near"   # set active zone to the one we want to undo
#     ann._undo_zone()
#     assert len(ann._zones["near"]) == 0
#     print("  ✅ Undo clears confirmed zone")
def test_undo(tmpdir: Path) -> None:
    ann = ZoneAnnotator(
        section_id = "cad_lab",
        source     = 0,
        zone_path  = tmpdir / "z_undo.json",  # fresh file
    )
    ann._base_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    sim_zone(ann, "near", near_pts())
    assert len(ann._zones["near"]) == 4
    ann._active_zone = "near"
    ann._undo_zone()
    assert len(ann._zones["near"]) == 0
    print("  ✅ Undo clears confirmed zone")


# def test_min_vertices(tmpdir: Path) -> None:
#     ann = make_annotator("cad_lab", tmpdir / "z.json")
#     ann._active_zone  = "near"
#     ann._current_poly = [[10,10],[20,10]]   # only 2 points
#     ann._confirm_zone()
#     assert len(ann._zones["near"]) == 0
#     print("  ✅ < 3 vertices rejected")

def test_min_vertices(tmpdir: Path) -> None:
    # Use a fresh path — no existing data that could pre-populate zones
    ann = ZoneAnnotator(
        section_id = "cad_lab",
        source     = 0,
        zone_path  = tmpdir / "z_minvert.json",  # fresh file
    )
    ann._base_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ann._active_zone  = "near"
    ann._current_poly = [[10,10],[20,10]]   # only 2 points
    ann._confirm_zone()
    assert len(ann._zones["near"]) == 0
    print("  ✅ < 3 vertices rejected")

def test_append_preserves_other_sections(tmpdir: Path) -> None:
    zp = tmpdir / "z.json"
    ann_a = make_annotator("cad_lab", zp)
    sim_zone(ann_a, "near", near_pts())
    sim_zone(ann_a, "mid",  mid_pts())
    sim_zone(ann_a, "far",  far_pts())
    ann_a._save()

    ann_b = make_annotator("g_huss", zp)
    sim_zone(ann_b, "near", near_pts())
    sim_zone(ann_b, "mid",  mid_pts())
    sim_zone(ann_b, "far",  far_pts())
    ann_b._save()

    data = json.loads(zp.read_text())
    assert "cad_lab" in data, "cad_lab was erased!"
    assert "g_huss"  in data
    print("  ✅ Append preserves other sections")


def test_all_11_sections(tmpdir: Path) -> None:
    sections = [
        "cad_lab","focused_reading_area","g_hall_2","g_huss",
        "hindi_section","ip_camera_19","ipc","main_computer_room",
        "reference_2","reference_area","weeding_out_area"
    ]
    zp = tmpdir / "z.json"
    for sec in sections:
        ann = make_annotator(sec, zp)
        sim_zone(ann, "near", near_pts())
        sim_zone(ann, "mid",  mid_pts())
        sim_zone(ann, "far",  far_pts())
        ann._save()

    data = json.loads(zp.read_text())
    assert len(data) == 11
    for sec in sections:
        assert sec in data
        for z in ZONE_NAMES:
            assert len(data[sec][z]) == 4
    print(f"  ✅ All 11 sections in one file — {len(data)} sections, 3 zones each")


if __name__ == "__main__":
    print("=" * 55)
    print("ZCP-01 — Zone Annotation Tool")
    print("=" * 55)

    tmpdir = Path(tempfile.mkdtemp(prefix="zcp01_"))
    try:
        print("\n[1] Init:")
        test_init(tmpdir)

        print("\n[2] Confirm zone:")
        test_confirm_zone(tmpdir)

        print("\n[3] All 3 zones confirmed and saved:")
        test_all_three_zones(tmpdir)

        print("\n[4] JSON structure:")
        test_json_structure(tmpdir)

        print("\n[5] Auto-advance near→mid→far:")
        test_auto_advance(tmpdir)

        print("\n[6] Undo:")
        test_undo(tmpdir)

        print("\n[7] Minimum vertex validation:")
        test_min_vertices(tmpdir)

        print("\n[8] Append preserves other sections:")
        test_append_preserves_other_sections(tmpdir)

        print("\n[9] All 11 sections in one file:")
        test_all_11_sections(tmpdir)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("\n" + "=" * 55)
    print("🎉 ZCP-01 PASSED — zone annotation tool ready")
    print("=" * 55)
    print("\nTo annotate a real section:")
    print("  python roi/zone_annotator.py --section cad_lab --image data/sample_frames/cad_lab.jpg")