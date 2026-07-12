"""
detection/calibrate_density.py — Per-section DM-Count calibration (ZCP-10)

Computes scale_factor per section:
    scale_factor = ground_truth_far_count / raw_dm_count

You provide manual ground truth counts for the far zone of each section.
The script computes scale factors and saves them to config.yaml.

Usage:
    python detection/calibrate_density.py
"""

import json
import sys
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from detection.density_estimator import FarZoneEstimator
from utils.logger import get_logger

log = get_logger("system")

ZONE_PATH   = Path(r"C:\IITD_Internship\library-occupancy\data\roi\zone_config.json")
SAMPLE_DIR  = Path("data/sample_frames")
CONFIG_PATH = Path("config/config.yaml")

# ── Ground truth counts for far zone per section ───────────────────────────────
# COUNT THE PEOPLE VISIBLE IN THE FAR ZONE OF EACH SAMPLE FRAME MANUALLY
# Set to None if no people are visible in the far zone (scale_factor stays 1.0)

GROUND_TRUTH = {
    "cad_lab":            None,   # fill in after manual count
    "focused_reading_area": 8,
    "g_hall_2":           60,
    "g_huss":             56,
    "hindi_section":      40,
    "ip_camera_19":       5,
    "ipc":                18,
    "main_computer_room": 5,
    "reference_2":        16,
    "reference_area":     7,
    "weeding_out_area":   5,
}


def calibrate() -> dict[str, float]:
    zone_data = json.loads(ZONE_PATH.read_text())
    results   = {}

    print(f"\n{'Section':<30} {'GT':>6} {'Raw':>8} {'Scale':>8} {'Notes'}")
    print("-" * 65)

    for section, gt_count in GROUND_TRUTH.items():
        img_path = SAMPLE_DIR / f"{section}.jpg"
        if not img_path.exists():
            print(f"  {section:<28} {'NO IMAGE':>6}")
            continue

        est = FarZoneEstimator(
            zone_config  = zone_data[section],
            section_id   = section,
            weights_path = "models/dm_count_shb.pth",
            scale_factor = 1.0,
        )
        frame     = cv2.imread(str(img_path))
        raw_count, _ = est.estimate(frame)
        raw_float = est._run_model(
            frame[
                max(0, min(p[1] for p in est.far_poly)):
                max(0, max(p[1] for p in est.far_poly)),
                max(0, min(p[0] for p in est.far_poly)):
                max(0, max(p[0] for p in est.far_poly))
            ] if est.far_poly else frame
        ).sum()

        if gt_count is None:
            scale  = 1.0
            notes  = "not calibrated"
        elif gt_count == 0 and raw_float == 0:
            scale  = 1.0
            notes  = "both zero"
        elif gt_count == 0:
            scale  = 0.0
            notes  = "empty far zone"
        elif raw_float < 0.1:
            scale  = 1.0
            notes  = "raw too low to calibrate"
        else:
            scale  = round(gt_count / raw_float, 4)
            notes  = "calibrated ✅"

        results[section] = scale
        print(f"  {section:<28} {str(gt_count):>6} {raw_float:>8.2f} {scale:>8.4f}  {notes}")

    return results


def save_to_config(scale_factors: dict[str, float]) -> None:
    """Write scale factors into config.yaml under each camera's far_zone_scale."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    for cam in config.get("cameras", []):
        sid = cam.get("section_id")
        if sid in scale_factors:
            cam["far_zone_scale"] = scale_factors[sid]

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    log.info(f"Scale factors written to {CONFIG_PATH}")


if __name__ == "__main__":
    print("=" * 65)
    print("DM-Count Per-Section Scale Factor Calibration")
    print("=" * 65)
    print("\nInstructions:")
    print("  1. Open data/roi_previews/zones/<section>_zones.jpg")
    print("  2. Count people visible ONLY in the RED (far) zone")
    print("  3. Fill in GROUND_TRUTH dict in this file")
    print("  4. Re-run to compute scale factors")

    scales = calibrate()

    print(f"\n{'='*65}")
    print("Scale factors computed. To save to config.yaml run:")
    print("  save_to_config(scales)  # uncomment line below")
    # save_to_config(scales)   # uncomment after reviewing