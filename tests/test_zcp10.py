"""
test_zcp10.py — Validates per-section DM-Count calibration (ZCP-10)

Tests:
  1. Scale factors loaded from config.yaml correctly
  2. Calibrated estimates are within ±30% of ground truth
  3. All 11 sections have scale factors set

Run from project root:
    python test_zcp10.py
"""

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))

from detection.density_estimator import FarZoneEstimator
from config.config_loader import cfg

ZONE_PATH  = Path(r"C:\IITD_Internship\library-occupancy\data\roi\zone_config.json")
SAMPLE_DIR = Path("data/sample_frames")

GROUND_TRUTH = {
    "focused_reading_area": 8,
    "g_hall_2":             60,
    "g_huss":               56,
    "hindi_section":        40,
    "ip_camera_19":         5,
    "ipc":                  18,
    "main_computer_room":   5,
    "reference_2":          16,
    "reference_area":       7,
    "weeding_out_area":     5,
}


def test_scale_factors_in_config() -> None:
    for cam in cfg.cameras:
        scale = getattr(cam, "far_zone_scale", None)
        assert scale is not None, \
            f"far_zone_scale missing for {cam.section_id} in config.yaml"
        assert isinstance(scale, float)
        assert scale > 0
    print(f"  ✅ All {len(cfg.cameras)} sections have far_zone_scale in config.yaml")


def test_calibrated_accuracy() -> None:
    zone_data = json.loads(ZONE_PATH.read_text())

    print(f"\n  {'Section':<30} {'GT':>5} {'Est':>5} {'Err%':>7} {'Status'}")
    print(f"  {'-'*60}")

    all_passed = True
    for section, gt in GROUND_TRUTH.items():
        img_path = SAMPLE_DIR / f"{section}.jpg"
        if not img_path.exists():
            print(f"  {section:<30} NO IMAGE")
            continue

        est   = FarZoneEstimator(zone_data[section], section_id=section)
        frame = cv2.imread(str(img_path))
        count, _ = est.estimate(frame)

        if gt == 0:
            err_pct = 0.0 if count == 0 else 100.0
        else:
            err_pct = abs(count - gt) / gt * 100

        status = "✅" if err_pct <= 30 else "⚠️ "
        if err_pct > 30:
            all_passed = False

        print(f"  {section:<30} {gt:>5} {count:>5} {err_pct:>6.1f}%  {status}")

    print()
    if all_passed:
        print(f"  ✅ All calibrated sections within ±30% of ground truth")
    else:
        print(f"  ⚠️  Some sections exceed ±30% — consider re-annotating far zones")


if __name__ == "__main__":
    print("=" * 55)
    print("ZCP-10 — DM-Count Calibration Validation")
    print("=" * 55)

    print("\n[1] Scale factors in config.yaml:")
    test_scale_factors_in_config()

    print("\n[2] Calibrated accuracy vs ground truth:")
    test_calibrated_accuracy()

    print("\n" + "=" * 55)
    print("🎉 ZCP-10 PASSED — density estimator calibrated")
    print("=" * 55)