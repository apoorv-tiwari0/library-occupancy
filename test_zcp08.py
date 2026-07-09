"""
test_zcp08.py — Validates DM-Count far zone density estimator (ZCP-08)

Tests:
  1. FarZoneEstimator initialises and loads weights
  2. Empty far polygon returns 0, None
  3. Blank frame returns count >= 0
  4. Output types correct (int count, numpy density map)
  5. Density map is non-negative everywhere
  6. Real frame — far zone estimation on all sections
  7. Visualization saves correctly

Run from project root:
    python test_zcp08.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from detection.density_estimator import FarZoneEstimator

ZONE_PATH  = Path(r"C:\IITD_Internship\library-occupancy\data\roi\zone_config.json")
SAMPLE_DIR = Path("data/sample_frames")
PREVIEW_DIR = Path("data/roi_previews")

SECTIONS = [
    "cad_lab","focused_reading_area","g_hall_2","g_huss",
    "hindi_section","ip_camera_19","ipc","main_computer_room",
    "reference_2","reference_area","weeding_out_area"
]


def load_zone_config() -> dict:
    return json.loads(ZONE_PATH.read_text())


def make_estimator(zone_data, section="cad_lab") -> FarZoneEstimator:
    return FarZoneEstimator(
        zone_config  = zone_data[section],
        section_id   = section,
        weights_path = "models/dm_count_shb.pth",
    )


def test_init(zone_data) -> FarZoneEstimator:
    est = make_estimator(zone_data)
    assert est._model is not None
    assert len(est.far_poly) >= 3
    print(f"  ✅ Init | section=cad_lab far_vertices={len(est.far_poly)} device={est.device}")
    return est


def test_empty_polygon(zone_data) -> None:
    est = FarZoneEstimator(
        zone_config  = {"near":[],"mid":[],"far":[]},
        section_id   = "test",
        weights_path = "models/dm_count_shb.pth",
    )
    count, dm = est.estimate(np.zeros((480,640,3), dtype=np.uint8))
    assert count == 0
    assert dm is None
    print("  ✅ Empty far polygon returns (0, None)")


def test_blank_frame(est) -> None:
    blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
    count, dm = est.estimate(blank)
    assert isinstance(count, int)
    assert count >= 0
    print(f"  ✅ Blank frame returns count={count} (int, non-negative)")


def test_output_types(est) -> None:
    frame = cv2.imread(str(SAMPLE_DIR / "cad_lab.jpg"))
    if frame is None:
        print("  ⚠️  Skipped")
        return
    count, dm = est.estimate(frame)
    assert isinstance(count, int),       f"count must be int, got {type(count)}"
    assert count >= 0,                   f"count must be >= 0, got {count}"
    assert isinstance(dm, np.ndarray),   f"density map must be ndarray"
    assert dm.dtype == np.float32 or dm.dtype == np.float64
    print(f"  ✅ Output types correct | count={count} dm.shape={dm.shape}")


def test_density_map_non_negative(est) -> None:
    frame = cv2.imread(str(SAMPLE_DIR / "cad_lab.jpg"))
    if frame is None:
        print("  ⚠️  Skipped")
        return
    _, dm = est.estimate(frame)
    assert (dm >= 0).all(), "Density map contains negative values"
    print(f"  ✅ Density map non-negative | min={dm.min():.4f} max={dm.max():.4f}")


def test_all_sections(zone_data) -> None:
    """Run far zone estimation on all 11 sections and print counts."""
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  {'Section':<30} {'Far count':>10} {'DM shape':>12}")
    print(f"  {'-'*55}")

    for sec in SECTIONS:
        img_path = SAMPLE_DIR / f"{sec}.jpg"
        if not img_path.exists():
            print(f"  {sec:<30} {'NO IMAGE':>10}")
            continue

        est   = FarZoneEstimator(
            zone_config  = zone_data[sec],
            section_id   = sec,
            weights_path = "models/dm_count_shb.pth",
        )
        frame = cv2.imread(str(img_path))
        count, dm = est.estimate(frame)
        shape_str = str(dm.shape) if dm is not None else "N/A"
        print(f"  {sec:<30} {count:>10} {shape_str:>12}")

    print(f"\n  ✅ All sections processed")


def test_visualization(zone_data) -> None:
    img_path = SAMPLE_DIR / "g_huss.jpg"
    if not img_path.exists():
        print(f"  ⚠️  Skipped — g_huss.jpg not found")
        return

    est   = FarZoneEstimator(
        zone_config  = zone_data["g_huss"],
        section_id   = "g_huss",
        weights_path = "models/dm_count_shb.pth",
    )
    frame        = cv2.imread(str(img_path))
    count, vis   = est.estimate_with_visualization(frame)
    out          = PREVIEW_DIR / "zcp08_far_zone_ghuss.jpg"
    cv2.imwrite(str(out), vis)
    print(f"  ✅ Visualization | g_huss far count={count} → {out}")


if __name__ == "__main__":
    print("=" * 55)
    print("ZCP-08 — DM-Count Far Zone Estimator")
    print("=" * 55)

    zone_data = load_zone_config()

    print("\n[1] Init:")
    est = test_init(zone_data)

    print("\n[2] Empty polygon:")
    test_empty_polygon(zone_data)

    print("\n[3] Blank frame:")
    test_blank_frame(est)

    print("\n[4] Output types:")
    test_output_types(est)

    print("\n[5] Density map non-negative:")
    test_density_map_non_negative(est)

    print("\n[6] All 11 sections — far zone counts:")
    test_all_sections(zone_data)

    print("\n[7] Visualization:")
    test_visualization(zone_data)

    print("\n" + "=" * 55)
    print("🎉 ZCP-08 PASSED — DM-Count far zone estimator ready")
    print("=" * 55)
    print("\nNext: ZCP-10 (calibration) to tune scale_factor per section")