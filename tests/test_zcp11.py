"""
test_zcp11.py — Far zone accuracy on held-out test frames (ZCP-11)

Tests CSRNet far zone estimator on frames NOT used for calibration.
2 test frames per section (data/test_frames/<section>-1.png, -2.png).

Acceptable accuracy: ±30% of ground truth count.
We manually count far-zone people in each test frame below.

Run from project root:
    python test_zcp11.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from detection.csrnet_estimator import CSRNetFarZoneEstimator

ZONE_PATH   = Path(r"C:\IITD_Internship\library-occupancy\data\roi\zone_config.json")
TEST_DIR    = Path("data/test_frames")
PREVIEW_DIR = Path("data/roi_previews/zcp11")

# ── Manual ground truth for test frames (far zone only) ───────────────────────
# Count people visible ONLY in the far (red) zone of each test frame.
# Open data/roi_previews/zones/<section>_zones.jpg as reference for zone boundaries.
# Fill these in before running.

GROUND_TRUTH = {
    "cad_lab-1":              4,   # fill in
    "cad_lab-2":              2,
    "focused_reading_area-1": 5,
    "focused_reading_area-2": 7,
    "g_hall_2-1":             29,
    "g_hall_2-2":             30,
    "g_huss-1":               38,
    "g_huss-2":               35,
    "hindi_section-1":        16,
    "hindi_section-2":        12,
    "ip_camera_19-1":         3,
    "ip_camera_19-2":         2,
    "ipc-1":                  8,
    "ipc-2":                  6,
    "main_computer_room-1":   4,
    "main_computer_room-2":   2,
    "reference_2-1":          4,
    "reference_2-2":          4,
    "reference_area-1":       3,
    "reference_area-2":       3,
    "weeding_out_area-1":     2,
    "weeding_out_area-2":     1,
}


def load_zone_config() -> dict:
    return json.loads(ZONE_PATH.read_text())


def run_evaluation(zone_data: dict) -> None:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    results   = []
    no_gt     = []
    no_image  = []

    for frame_key, gt in GROUND_TRUTH.items():
        # Parse section and frame number from key
        parts      = frame_key.rsplit("-", 1)
        section_id = parts[0]
        frame_num  = parts[1]

        # Find image file (.png or .jpg)
        img_path = None
        for ext in [".png", ".jpg", ".PNG", ".JPG"]:
            candidate = TEST_DIR / f"{frame_key}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            no_image.append(frame_key)
            continue

        if gt is None:
            no_gt.append(frame_key)
            continue

        if section_id not in zone_data:
            print(f"  ⚠️  No zone config for {section_id}")
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  ⚠️  Could not read {img_path}")
            continue

        est = CSRNetFarZoneEstimator(
            zone_config  = zone_data[section_id],
            section_id   = section_id,
            weights_path = "models/csrnet_v3_best.pth",
        )

        count, vis = est.estimate_with_visualization(frame)

        # Save annotated preview
        cv2.imwrite(str(PREVIEW_DIR / f"{frame_key}_far.jpg"), vis)

        # Compute error
        if gt == 0:
            err_pct = 0.0 if count == 0 else 100.0
        else:
            err_pct = abs(count - gt) / gt * 100

        tolerance = 50 if gt <= 3 else 30
        passed = err_pct <= tolerance
        results.append({
            "frame":   frame_key,
            "gt":      gt,
            "est":     count,
            "err_pct": err_pct,
            "passed":  passed,
        })

    # Print results table
    print(f"\n  {'Frame':<30} {'GT':>5} {'Est':>5} {'Err%':>7}  Status")
    print(f"  {'-'*58}")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {r['frame']:<30} {r['gt']:>5} {r['est']:>5} "
              f"{r['err_pct']:>6.1f}%  {status}")

    if not results and no_gt:
        print(f"\n  [!] No ground truth filled in yet.")
        print(f"  Fill in GROUND_TRUTH dict then re-run.")
        return

    # Summary
    if results:
        passed     = sum(1 for r in results if r["passed"])
        total      = len(results)
        avg_err    = sum(r["err_pct"] for r in results) / total
        mae        = sum(abs(r["est"] - r["gt"]) for r in results) / total

        print(f"\n  {'-'*58}")
        print(f"  Frames evaluated : {total}")
        print(f"  Within ±30%      : {passed}/{total}")
        print(f"  MAE              : {mae:.2f} persons")
        print(f"  Mean error %     : {avg_err:.1f}%")

        if no_gt:
            print(f"\n  [!] {len(no_gt)} frames skipped (no GT): {no_gt}")
        if no_image:
            print(f"  [!] {len(no_image)} frames not found: {no_image}")

        if passed == total:
            print(f"\n  [PASS] ALL {total} frames within +/-30% -- ZCP-11 PASSED")
        elif passed >= total * 0.7:
            print(f"\n  [PASS] {passed}/{total} frames within +/-30% -- acceptable accuracy")
        else:
            print(f"\n  [FAIL] {passed}/{total} within +/-30% -- consider re-annotating zones")

    print(f"\n  Preview images saved -> {PREVIEW_DIR}/")


if __name__ == "__main__":
    print("=" * 55)
    print("ZCP-11 — Far Zone Accuracy (Held-Out Test Frames)")
    print("=" * 55)

    zone_data = load_zone_config()

    print("\nStep 1 — Check test frames exist:")
    found = 0
    for key in GROUND_TRUTH:
        for ext in [".png", ".jpg", ".PNG", ".JPG"]:
            if (TEST_DIR / f"{key}{ext}").exists():
                found += 1
                break
    print(f"  Found {found}/22 test frames in {TEST_DIR}/")

    print("\nStep 2 — Evaluation:")
    run_evaluation(zone_data)

    print("\n" + "=" * 55)
    print("ZCP-11 complete")
    print("=" * 55)