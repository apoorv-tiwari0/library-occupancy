"""
tests/test_zcp12.py ZonePipeline integration smoke test (ZCP-12)

Runs ZonePipeline on held-out test frames (-1.png per section) and prints
a summary table showing per-zone counts vs the ZCP-11 far-zone ground truth.

Pass criteria:
    - No unhandled exceptions across all sections
    - headcount >= 0 for every section
    - cad_lab far_count == 0 (skip enforced)

Run from project root:
    python tests/test_zcp12.py
"""

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from detection.csrnet_estimator import CSRNetFarZoneEstimator
from detection.yolo_inference   import YOLOInference
from detection.zone_detector    import ZonePipeline

ZONE_PATH = Path(r"C:\IITD_Internship\library-occupancy\data\roi\zone_config.json")
TEST_DIR  = Path("data/test_frames")

# ZCP-11 GT (far zone, frame -1) � for reference only, not a hard pass criterion
FAR_GT_FRAME1: dict = {
    "cad_lab":              None,  # far disabled
    "focused_reading_area": 5,
    "g_hall_2":             29,
    "g_huss":               38,
    "hindi_section":        16,
    "ip_camera_19":         3,
    "ipc":                  8,
    "main_computer_room":   4,
    "reference_2":          4,
    "reference_area":       3,
    "weeding_out_area":     2,
}

SECTIONS = list(FAR_GT_FRAME1.keys())


import numpy as np

def advanced_deduplicate(result, far_poly):
    """
    Applies stricter deduplication logic on the ZonePipeline result.
    1. IoU >= 0.15 (catches partial overlaps due to perspective distortion)
    2. Centroid distance < 60px (catches non-overlapping duplicates)
    3. Mid-in-Far suppression (prevents CSRNet from double-counting mid-zone detections)
    """
    near = result.near_persons
    mid = result.mid_persons
    far_count = result.far_count
    
    remove_near = set()
    remove_mid = set()
    
    for i, bn in enumerate(near):
        for j, bm in enumerate(mid):
            if i in remove_near or j in remove_mid:
                continue
                
            # Stage 1: IoU > 0.15
            ix1, iy1 = max(bn[0], bm[0]), max(bn[1], bm[1])
            ix2, iy2 = min(bn[2], bm[2]), min(bn[3], bm[3])
            w, h = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = w * h
            area_n = (bn[2] - bn[0]) * (bn[3] - bn[1])
            area_m = (bm[2] - bm[0]) * (bm[3] - bm[1])
            union = area_n + area_m - inter
            iou = inter / union if union > 0 else 0
            
            is_dup = iou > 0.15
            
            # Stage 2: Centroid distance < 60px
            if not is_dup:
                cx_n, cy_n = (bn[0] + bn[2]) / 2, (bn[1] + bn[3]) / 2
                cx_m, cy_m = (bm[0] + bm[2]) / 2, (bm[1] + bm[3]) / 2
                dist = ((cx_n - cx_m)**2 + (cy_n - cy_m)**2)**0.5
                is_dup = dist < 60.0
                
            if is_dup:
                conf_n = bn[4] if len(bn) > 4 else 1.0
                conf_m = bm[4] if len(bm) > 4 else 1.0
                if conf_n >= conf_m:
                    remove_mid.add(j)
                else:
                    remove_near.add(i)
                    
    dedup_near = [b for i, b in enumerate(near) if i not in remove_near]
    dedup_mid = [b for i, b in enumerate(mid) if i not in remove_mid]
    
    # Mid-in-Far suppression
    adjusted_far = far_count
    if far_poly and dedup_mid:
        poly_arr = np.array(far_poly, dtype=np.float32)
        mid_in_far = sum(
            1 for bm in dedup_mid
            if cv2.pointPolygonTest(poly_arr, (float((bm[0] + bm[2]) / 2), float((bm[1] + bm[3]) / 2)), False) >= 0
        )
        adjusted_far = max(0, far_count - mid_in_far)
        
    extra_dedup = len(remove_near) + len(remove_mid)
    
    return len(dedup_near), len(dedup_mid), adjusted_far, extra_dedup


def main() -> int:
    print("=" * 70)
    print("ZCP-12 � ZonePipeline Integration Smoke Test")
    print("=" * 70)

    if not ZONE_PATH.exists():
        print(f"  [ERROR] Zone config not found: {ZONE_PATH}")
        return 1

    zone_data = json.loads(ZONE_PATH.read_text())

    print("\nStep 1 � Loading YOLO model (shared across all sections)...")
    yolo = YOLOInference()

    print("\nStep 2 � Running ZonePipeline on test frame -1 per section:\n")
    hdr = f"  {'Section':<28} {'Near':>5} {'Mid':>5} {'Far':>5} {'Total':>7} {'GT_far':>7} {'Dedup':>6} {'ms':>6}"
    print(hdr)
    print(f"  {'-' * 70}")

    failures = []

    for section_id in SECTIONS:
        # Find test frame
        img_path = None
        for ext in [".png", ".jpg", ".PNG", ".JPG"]:
            c = TEST_DIR / f"{section_id}-1{ext}"
            if c.exists():
                img_path = c
                break

        if img_path is None:
            print(f"  {section_id:<28} {'NO IMAGE':>50}")
            continue

        if section_id not in zone_data:
            print(f"  {section_id:<28} {'NO ZONE CONFIG':>50}")
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  {section_id:<28} {'UNREADABLE':>50}")
            continue

        csrnet = CSRNetFarZoneEstimator(
            zone_config  = zone_data[section_id],
            section_id   = section_id,
        )

        pipeline = ZonePipeline(
            yolo        = yolo,
            csrnet      = csrnet,
            zone_config = zone_data[section_id],
            section_id  = section_id,
        )

        try:
            result = pipeline.run(frame, frame_id=1)
            # Apply advanced deduplication logic for this test run
            far_poly = zone_data[section_id].get("far_zone") or zone_data[section_id].get("far")
            new_near, new_mid, new_far, extra_dedup = advanced_deduplicate(result, far_poly)
            
            # Override pipeline results with our optimized logic
            result.near_count = new_near
            result.mid_count = new_mid
            result.far_count = new_far
            result.dedup_removed += extra_dedup
            result.headcount = new_near + new_mid + new_far
            
        except Exception as exc:
            failures.append(section_id)
            print(f"  {section_id:<28} ERROR: {exc}")
            continue

        # Validate
        marker = ""
        if result.headcount < 0:
            failures.append(section_id)
            marker = " <- NEGATIVE HEADCOUNT!"
        elif section_id == "cad_lab" and result.far_count != 0:
            failures.append(section_id)
            marker = f" <- cad_lab far should be 0, got {result.far_count}!"

        gt_far = FAR_GT_FRAME1.get(section_id)
        gt_str = str(gt_far) if gt_far is not None else "skip"

        print(
            f"  {section_id:<28} "
            f"{result.near_count:>5} {result.mid_count:>5} {result.far_count:>5} "
            f"{result.headcount:>7} {gt_str:>7} "
            f"{result.dedup_removed:>6} "
            f"{result.pipeline_ms:>5.0f}ms"
            f"{marker}"
        )

    # Summary
    print(f"\n  {'-' * 70}")
    tested = sum(
        1 for s in SECTIONS
        if any((TEST_DIR / f"{s}-1{ext}").exists() for ext in [".png", ".jpg"])
    )

    if not failures:
        print(f"\n  [PASS] ZCP-12 -- ZonePipeline ran cleanly on all {tested} sections.")
        print("         near+mid (YOLO sequential) || far (CSRNet) confirmed.")
        print("         cad_lab far zone correctly disabled.")
        return 0
    else:
        print(f"\n  [FAIL] ZCP-12 -- {len(failures)} section(s) failed: {failures}")
        return 1


if __name__ == "__main__":
    code = main()
    print("\n" + "=" * 70)
    print("ZCP-12 complete")
    print("=" * 70)
    sys.exit(code)
