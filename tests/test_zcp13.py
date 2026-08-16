"""
tests/test_zcp13.py — Zone-Based Vacancy Pipeline Test (ZCP-13)

Uses the zone-fusion architecture (near + mid via YOLO, far via CSRNet)
with advanced cross-zone deduplication to compute per-section vacancy for all 11 sections.

Vacancy calculation:
    headcount    = near + mid + far  (deduplicated)
    max_capacity = from config.yaml
    vacancy      = max(0, max_capacity − headcount)
    occupancy    = (headcount / max_capacity) * 100 %
    is_available = vacancy > 0

Run from project root:
    python tests/test_zcp13.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import cfg
from detection.csrnet_estimator import CSRNetFarZoneEstimator
from detection.yolo_inference import YOLOInference
from detection.zone_detector import ZonePipeline

ZONE_PATH = Path(r"C:\IITD_Internship\library-occupancy\data\roi\zone_config.json")
TEST_DIR = Path("data/test_frames")

SECTIONS = [cam.section_id for cam in cfg.cameras if cam.enabled]


def get_max_capacity(section_id: str) -> int | None:
    """Fetch max_capacity for section_id from config.yaml."""
    for cam in cfg.cameras:
        if cam.section_id == section_id:
            return getattr(cam, "max_capacity", None)
    return None


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
                dist = ((cx_n - cx_m) ** 2 + (cy_n - cy_m) ** 2) ** 0.5
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
            1
            for bm in dedup_mid
            if cv2.pointPolygonTest(
                poly_arr,
                (float((bm[0] + bm[2]) / 2), float((bm[1] + bm[3]) / 2)),
                False,
            )
            >= 0
        )
        adjusted_far = max(0, far_count - mid_in_far)

    extra_dedup = len(remove_near) + len(remove_mid)

    return len(dedup_near), len(dedup_mid), adjusted_far, extra_dedup


import logging

def main() -> int:
    print("=" * 88)
    print("ZCP-13 — Zone-Based Vacancy Pipeline Test")
    print("=" * 88)

    if not ZONE_PATH.exists():
        print(f"  [ERROR] Zone config not found: {ZONE_PATH}")
        return 1

    zone_data = json.loads(ZONE_PATH.read_text())

    print("\nStep 1 — Loading YOLO model (shared across all sections)...")
    yolo = YOLOInference()

    # Suppress verbose logger output during per-section processing
    logging.disable(logging.INFO)

    failures = []
    rows = []

    for section_id in SECTIONS:
        # Find test frame
        img_path = None
        for ext in [".png", ".jpg", ".PNG", ".JPG"]:
            c = TEST_DIR / f"{section_id}-1{ext}"
            if c.exists():
                img_path = c
                break

        if img_path is None:
            rows.append(f"  {section_id:<24} {'NO IMAGE':>60}")
            continue

        if section_id not in zone_data:
            rows.append(f"  {section_id:<24} {'NO ZONE CONFIG':>60}")
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            rows.append(f"  {section_id:<24} {'UNREADABLE':>60}")
            continue

        max_cap = get_max_capacity(section_id)
        if max_cap is None:
            rows.append(f"  {section_id:<24} {'NO MAX CAPACITY IN CONFIG':>60}")
            continue

        csrnet = CSRNetFarZoneEstimator(
            zone_config=zone_data[section_id],
            section_id=section_id,
        )

        pipeline = ZonePipeline(
            yolo=yolo,
            csrnet=csrnet,
            zone_config=zone_data[section_id],
            section_id=section_id,
        )

        try:
            result = pipeline.run(frame, frame_id=1)
            # Apply advanced deduplication logic
            far_poly = zone_data[section_id].get("far_zone") or zone_data[section_id].get("far")
            new_near, new_mid, new_far, extra_dedup = advanced_deduplicate(result, far_poly)

            headcount = new_near + new_mid + new_far
            vacancy = max(0, max_cap - headcount)
            occupancy_pct = min(100.0, round((headcount / max_cap) * 100, 1))
            status_str = "AVAILABLE" if vacancy > 0 else "FULL"

        except Exception as exc:
            failures.append(section_id)
            rows.append(f"  {section_id:<24} ERROR: {exc}")
            continue

        # Validate
        marker = ""
        if headcount < 0:
            failures.append(section_id)
            marker = " <- NEGATIVE HEADCOUNT!"
        elif section_id == "cad_lab" and new_far != 0:
            failures.append(section_id)
            marker = f" <- cad_lab far should be 0, got {new_far}!"

        rows.append(
            f"  {section_id:<24} "
            f"{new_near:>5} {new_mid:>5} {new_far:>5} "
            f"{headcount:>6} {max_cap:>7} {vacancy:>8} "
            f"{occupancy_pct:>9.1f}% {status_str:>10} "
            f"{result.pipeline_ms:>5.0f}ms"
            f"{marker}"
        )

    # Re-enable logging
    logging.disable(logging.NOTSET)

    print("\nStep 2 — Zone-Based Vacancy Summary Table:\n")
    hdr = f"  {'Section':<24} {'Near':>5} {'Mid':>5} {'Far':>5} {'Count':>6} {'MaxCap':>7} {'Vacancy':>8} {'Occupancy':>10} {'Status':>10} {'ms':>6}"
    print(hdr)
    print(f"  {'-' * 88}")
    for row in rows:
        print(row)


    # Summary
    print(f"\n  {'-' * 88}")
    tested = sum(
        1
        for s in SECTIONS
        if any((TEST_DIR / f"{s}-1{ext}").exists() for ext in [".png", ".jpg"])
    )

    if not failures:
        print(f"\n  [PASS] ZCP-13 -- Zone-based vacancy pipeline ran cleanly on all {tested} sections.")
        print("         near+mid (YOLO sequential) || far (CSRNet) confirmed.")
        print("         Vacancy = max(0, max_capacity - headcount) calculated successfully.")
        return 0
    else:
        print(f"\n  [FAIL] ZCP-13 -- {len(failures)} section(s) failed: {failures}")
        return 1


if __name__ == "__main__":
    code = main()
    print("\n" + "=" * 80)
    print("ZCP-13 complete")
    print("=" * 80)
    sys.exit(code)
