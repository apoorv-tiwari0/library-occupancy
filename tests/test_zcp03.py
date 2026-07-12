"""
test_zcp03.py — Validates near zone detector (ZCP-03)

Tests:
  1. NearZoneDetector initialises correctly
  2. get_zone_crop returns correct bounding box
  3. map_boxes_to_frame shifts coordinates correctly
  4. filter_by_polygon keeps only in-polygon detections
  5. Empty near polygon returns empty results
  6. Empty crop (degenerate zone) handled gracefully
  7. Real frame — cad_lab.jpg near zone detection
  8. Coordinate mapping is correct (boxes in full-frame space)

Run from project root:
    python test_zcp03.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from detection.zone_detector import (
    NearZoneDetector,
    get_zone_crop,
    map_boxes_to_frame,
    filter_by_polygon,
)
from detection.yolo_inference import YOLOInference

ZONE_PATH  = Path(r"C:\IITD_Internship\library-occupancy\data\roi\zone_config.json")
SAMPLE_DIR = Path("data/sample_frames")
SECTION    = "weeding_out_area"


def load_zone_config() -> dict:
    return json.loads(ZONE_PATH.read_text())


def test_get_zone_crop() -> None:
    frame   = np.zeros((480, 640, 3), dtype=np.uint8)
    polygon = [[100,100],[300,100],[300,300],[100,300]]
    crop, x1, y1, x2, y2 = get_zone_crop(frame, polygon)

    assert x1 == 100 and y1 == 100
    assert x2 == 301 and y2 == 301
    assert crop.shape == (201, 201, 3)
    print(f"  ✅ get_zone_crop: crop={crop.shape} offset=({x1},{y1})")


def test_get_zone_crop_clamped() -> None:
    """Polygon that extends beyond frame bounds should be clamped."""
    frame   = np.zeros((480, 640, 3), dtype=np.uint8)
    polygon = [[-50,-50],[700,-50],[700,500],[-50,500]]
    crop, x1, y1, x2, y2 = get_zone_crop(frame, polygon)

    assert x1 >= 0 and y1 >= 0
    assert x2 <= 640 and y2 <= 480
    print(f"  ✅ get_zone_crop: out-of-bounds polygon clamped correctly")


def test_map_boxes_to_frame() -> None:
    # Box in crop-local coords: x1=10, y1=20, x2=50, y2=80
    boxes = [[10.0, 20.0, 50.0, 80.0, 0.9]]
    mapped = map_boxes_to_frame(boxes, x1_off=100, y1_off=150)
    assert mapped[0][0] == 110.0   # 10 + 100
    assert mapped[0][1] == 170.0   # 20 + 150
    assert mapped[0][2] == 150.0   # 50 + 100
    assert mapped[0][3] == 230.0   # 80 + 150
    assert mapped[0][4] == 0.9     # conf unchanged
    print(f"  ✅ map_boxes_to_frame: offset applied correctly")


def test_map_boxes_with_class() -> None:
    boxes  = [[10.0, 20.0, 50.0, 80.0, 0.9, 63.0]]
    mapped = map_boxes_to_frame(boxes, x1_off=50, y1_off=50, has_class=True)
    assert mapped[0][5] == 63.0   # class_id preserved
    print(f"  ✅ map_boxes_to_frame: class_id preserved")


def test_filter_by_polygon_inside() -> None:
    polygon = [[0,0],[200,0],[200,200],[0,200]]
    # Centroid at (100,100) — inside polygon
    boxes = [[80.0, 80.0, 120.0, 120.0, 0.9]]
    kept  = filter_by_polygon(boxes, polygon)
    assert len(kept) == 1
    print(f"  ✅ filter_by_polygon: inside detection kept")


def test_filter_by_polygon_outside() -> None:
    polygon = [[0,0],[200,0],[200,200],[0,200]]
    # Centroid at (300,300) — outside polygon
    boxes = [[280.0, 280.0, 320.0, 320.0, 0.9]]
    kept  = filter_by_polygon(boxes, polygon)
    assert len(kept) == 0
    print(f"  ✅ filter_by_polygon: outside detection removed")


def test_empty_polygon(yolo) -> None:
    zone_config = {"near": [], "mid": [[0,0],[100,0],[100,100],[0,100]], "far": []}
    detector    = NearZoneDetector(yolo, zone_config, section_id=SECTION)
    frame       = np.zeros((480, 640, 3), dtype=np.uint8)
    persons, objects = detector.detect(frame)
    assert persons == [] and objects == []
    print(f"  ✅ Empty near polygon returns empty results")


def test_real_frame(yolo, zone_data) -> None:
    img_path = SAMPLE_DIR / f"{SECTION}.jpg"
    if not img_path.exists():
        print(f"  ⚠️  Skipped — {img_path} not found")
        return

    frame    = cv2.imread(str(img_path))
    h, w     = frame.shape[:2]
    detector = NearZoneDetector(yolo, zone_data[SECTION], section_id=SECTION)
    persons, objects = detector.detect(frame)

    print(f"  ✅ Real frame — {SECTION} near zone:")
    print(f"     Persons detected: {len(persons)}")
    print(f"     Objects detected: {len(objects)}")

    # All boxes must be within full-frame bounds
    for box in persons:
        x1, y1, x2, y2, conf = box
        assert 0 <= x1 < w and 0 <= x2 <= w, \
            f"X coords out of frame: x1={x1} x2={x2} w={w}"
        assert 0 <= y1 < h and 0 <= y2 <= h, \
            f"Y coords out of frame: y1={y1} y2={y2} h={h}"
    print(f"     ✅ All {len(persons)} boxes within full-frame bounds ({w}×{h})")

    # Save visualization
    vis = frame.copy()
    near_poly = np.array(zone_data[SECTION]["near"], dtype=np.int32)
    cv2.polylines(vis, [near_poly], isClosed=True, color=(0,255,0), thickness=2)
    overlay = vis.copy()
    cv2.fillPoly(overlay, [near_poly], (0,255,0))
    cv2.addWeighted(overlay, 0.15, vis, 0.85, 0, vis)

    for i, box in enumerate(persons):
        x1, y1, x2, y2, conf = box
        cv2.rectangle(vis, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
        cv2.putText(vis, f"#{i+1} {conf:.2f}",
                    (int(x1), max(0,int(y1)-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.rectangle(vis, (0,0), (320,28), (0,0,0), -1)
    cv2.putText(vis, f"NEAR zone | persons={len(persons)}",
                (6,18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    out = Path("data/roi_previews/zcp03_near_zone.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), vis)
    print(f"     → zcp03_near_zone.jpg saved")


if __name__ == "__main__":
    print("=" * 55)
    print("ZCP-03 — Near Zone Detector")
    print("=" * 55)

    print("\n[1] get_zone_crop basic:")
    test_get_zone_crop()

    print("\n[2] get_zone_crop out-of-bounds clamping:")
    test_get_zone_crop_clamped()

    print("\n[3] map_boxes_to_frame:")
    test_map_boxes_to_frame()

    print("\n[4] map_boxes_to_frame with class:")
    test_map_boxes_with_class()

    print("\n[5] filter_by_polygon — inside:")
    test_filter_by_polygon_inside()

    print("\n[6] filter_by_polygon — outside:")
    test_filter_by_polygon_outside()

    print("\nLoading YOLO model...")
    yolo      = YOLOInference()
    zone_data = load_zone_config()

    print("\n[7] Empty polygon handled gracefully:")
    test_empty_polygon(yolo)

    print("\n[8] Real frame — near zone detection:")
    test_real_frame(yolo, zone_data)

    print("\n" + "=" * 55)
    print("🎉 ZCP-03 PASSED — near zone detector ready")
    print("=" * 55)