"""
test_zcp02.py — Validates zone config validator (ZCP-02)

Tests:
  1. Validator loads and runs on a valid config
  2. All 11 sections detected in real zone_config.json
  3. All 33 zones (11 × 3) valid
  4. Missing section detected correctly
  5. Missing zone detected correctly
  6. Too few vertices detected correctly
  7. Degenerate polygon (zero area) detected correctly
  8. Invalid vertex format detected correctly
  9. Visualize zones on real sample frames (saves preview images)
  10. Validate real zone_config.json end-to-end

Run from project root:
    python test_zcp02.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from roi.zone_validator import ZoneValidator, ZONE_NAMES

ZONE_PATH    = Path(r"C:\IITD_Internship\library-occupancy\data\roi\zone_config.json")
SAMPLE_DIR   = Path("data/sample_frames")
PREVIEW_DIR  = Path("data/roi_previews/zones")

SECTIONS = [
    "cad_lab","focused_reading_area","g_hall_2","g_huss",
    "hindi_section","ip_camera_19","ipc","main_computer_room",
    "reference_2","reference_area","weeding_out_area"
]


def make_valid_config(sections: list[str]) -> dict:
    """Create a valid zone config dict for testing."""
    return {
        sec: {
            "near": [[10,10],[200,10],[200,150],[10,150]],
            "mid":  [[10,150],[200,150],[200,300],[10,300]],
            "far":  [[10,300],[200,300],[200,450],[10,450]],
        }
        for sec in sections
    }


def test_valid_config(tmpdir: Path) -> None:
    zp   = tmpdir / "valid.json"
    data = make_valid_config(SECTIONS)
    zp.write_text(json.dumps(data))

    v      = ZoneValidator(zp)
    report = v.validate()
    assert report.passed(), f"Valid config failed:\n{report.summary()}"
    assert len(report.sections_found)   == 11
    assert len(report.sections_missing) == 0
    assert report.valid_zones           == 33
    assert len(report.errors())         == 0
    print(f"  ✅ Valid config: 11 sections, 33 zones, 0 errors")


def test_missing_section(tmpdir: Path) -> None:
    zp   = tmpdir / "miss_sec.json"
    data = make_valid_config(SECTIONS[:-2])   # drop last 2 sections
    zp.write_text(json.dumps(data))

    v      = ZoneValidator(zp)
    report = v.validate()
    assert not report.passed()
    assert len(report.sections_missing) == 2
    print(f"  ✅ Missing sections detected: {report.sections_missing}")


def test_missing_zone(tmpdir: Path) -> None:
    zp   = tmpdir / "miss_zone.json"
    data = make_valid_config(SECTIONS)
    del data["cad_lab"]["far"]   # remove far zone from cad_lab
    zp.write_text(json.dumps(data))

    v      = ZoneValidator(zp)
    report = v.validate()
    errors = [e for e in report.errors() if e.section_id == "cad_lab"]
    assert len(errors) > 0
    assert any("far" in e.message or e.zone == "far" for e in errors)
    print(f"  ✅ Missing zone detected: {errors[0].message}")


def test_too_few_vertices(tmpdir: Path) -> None:
    zp   = tmpdir / "few_verts.json"
    data = make_valid_config(SECTIONS)
    data["g_huss"]["near"] = [[10,10],[50,10]]   # only 2 vertices
    zp.write_text(json.dumps(data))

    v      = ZoneValidator(zp)
    report = v.validate()
    errors = [e for e in report.errors()
              if e.section_id == "g_huss" and e.zone == "near"]
    assert len(errors) > 0
    print(f"  ✅ Too few vertices detected: {errors[0].message}")


def test_degenerate_polygon(tmpdir: Path) -> None:
    zp   = tmpdir / "degen.json"
    data = make_valid_config(SECTIONS)
    # Collinear points — zero area polygon
    data["g_huss"]["mid"] = [[10,10],[20,10],[30,10],[40,10]]
    zp.write_text(json.dumps(data))

    v      = ZoneValidator(zp)
    report = v.validate()
    errors = [e for e in report.errors()
              if e.section_id == "g_huss" and e.zone == "mid"]
    assert len(errors) > 0
    print(f"  ✅ Degenerate polygon detected: {errors[0].message}")


def test_invalid_vertex(tmpdir: Path) -> None:
    zp   = tmpdir / "bad_vert.json"
    data = make_valid_config(SECTIONS)
    data["cad_lab"]["near"] = [[10,10],[200,10],["bad","vertex"],[10,150]]
    zp.write_text(json.dumps(data))

    v      = ZoneValidator(zp)
    report = v.validate()
    errors = [e for e in report.errors() if e.section_id == "cad_lab"]
    assert len(errors) > 0
    print(f"  ✅ Invalid vertex detected: {errors[0].message}")


def test_missing_file(tmpdir: Path) -> None:
    v      = ZoneValidator(tmpdir / "nonexistent.json")
    report = v.validate()
    assert not report.passed()
    assert len(report.sections_missing) == 11
    print(f"  ✅ Missing file handled gracefully")


def test_visualize_zones() -> None:
    """Save zone overlay previews for all sections that have sample frames."""
    if not ZONE_PATH.exists():
        print(f"  ⚠️  Skipped — {ZONE_PATH} not found")
        return

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    v = ZoneValidator(ZONE_PATH)
    saved = 0

    for sec in SECTIONS:
        img_path = SAMPLE_DIR / f"{sec}.jpg"
        if not img_path.exists():
            continue
        frame   = cv2.imread(str(img_path))
        out     = PREVIEW_DIR / f"{sec}_zones.jpg"
        v.visualize(sec, frame, out_path=out)
        saved += 1

    print(f"  ✅ Zone previews saved: {saved}/11 → {PREVIEW_DIR}/")


def test_real_zone_config() -> None:
    """Validate the actual zone_config.json you annotated."""
    if not ZONE_PATH.exists():
        print(f"  ⚠️  Skipped — {ZONE_PATH} not found")
        return

    sample_frames = {
        sec: str(SAMPLE_DIR / f"{sec}.jpg")
        for sec in SECTIONS
        if (SAMPLE_DIR / f"{sec}.jpg").exists()
    }

    v      = ZoneValidator(ZONE_PATH, sample_frames=sample_frames)
    report = v.validate()

    print(f"\n  Real zone_config.json validation:")
    print(f"  " + "\n  ".join(report.summary().splitlines()))

    if report.passed():
        print(f"\n  ✅ Real zone config PASSED")
    else:
        print(f"\n  ❌ Real zone config has issues — fix before proceeding to ZCP-03")

    # Fail test if there are errors (warnings are OK)
    assert len(report.errors()) == 0, \
        f"Fix these errors before ZCP-03:\n" + \
        "\n".join(f"  [{e.section_id}][{e.zone}] {e.message}" for e in report.errors())


if __name__ == "__main__":
    print("=" * 55)
    print("ZCP-02 — Zone Config Validator")
    print("=" * 55)

    tmpdir = Path(tempfile.mkdtemp(prefix="zcp02_"))

    try:
        print("\n[1] Valid config passes:")
        test_valid_config(tmpdir)

        print("\n[2] Missing section detected:")
        test_missing_section(tmpdir)

        print("\n[3] Missing zone detected:")
        test_missing_zone(tmpdir)

        print("\n[4] Too few vertices detected:")
        test_too_few_vertices(tmpdir)

        print("\n[5] Degenerate polygon detected:")
        test_degenerate_polygon(tmpdir)

        print("\n[6] Invalid vertex detected:")
        test_invalid_vertex(tmpdir)

        print("\n[7] Missing file handled:")
        test_missing_file(tmpdir)

        print("\n[8] Zone visualizations saved:")
        test_visualize_zones()

        print("\n[9] Real zone_config.json end-to-end:")
        test_real_zone_config()

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("\n" + "=" * 55)
    print("🎉 ZCP-02 PASSED — zone config validator ready")
    print("=" * 55)