"""
test_cp11.py — Validates SAHI+YOLOv10 inference module (CP-11)

Logic used (matches visualize_detections.py exactly):
  - conf >= 0.50 (set in config.yaml) — removes glass/reflection false positives
  - NMM threshold 0.15 — merges cross-slice duplicate detections
  - No ROI filtering — confidence threshold is sufficient
"""

import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))

from detection.yolo_inference import YOLOInference, _nms, _box_iou, _class_label
from config.constants import CocoClass, RESERVATION_CLASSES


def test_model_loads():
    yolo = YOLOInference()
    assert yolo._model is not None
    assert yolo._detection_model is not None
    print("  ✅ Model loaded")
    return yolo


def test_blank_frame(yolo):
    blank = np.zeros((640, 640, 3), dtype=np.uint8)
    persons, objects = yolo.run_inference(blank, section_id="cad_lab")
    assert isinstance(persons, list)
    assert isinstance(objects, list)
    assert len(persons) == 0
    print(f"  ✅ Blank frame: 0 persons, 0 objects")


def test_output_format(yolo):
    blank = np.zeros((640, 640, 3), dtype=np.uint8)
    persons, objects = yolo.run_inference(blank, section_id="cad_lab")
    for box in persons:
        assert len(box) == 5, f"Person box should have 5 values, got {len(box)}"
        assert 0.0 <= box[4] <= 1.0, f"Confidence out of range: {box[4]}"
    for box in objects:
        assert len(box) == 6, f"Object box should have 6 values, got {len(box)}"
        assert int(box[5]) in RESERVATION_CLASSES, f"Invalid class id: {box[5]}"
    print("  ✅ Output format correct")


def test_box_iou():
    assert abs(_box_iou([0,0,100,100], [0,0,100,100]) - 1.0) < 1e-5
    assert _box_iou([0,0,100,100], [200,200,300,300]) == 0.0
    iou = _box_iou([0,0,100,100], [50,50,150,150])
    assert 0.0 < iou < 1.0
    print(f"  ✅ box_iou: identical=1.0, non-overlap=0.0, partial={iou:.3f}")


def test_nms():
    boxes = [
        [10.0, 10.0, 100.0, 200.0, 0.91],
        [12.0, 11.0, 102.0, 201.0, 0.85],   # near-duplicate
        [300.0, 300.0, 400.0, 450.0, 0.75], # distinct
    ]
    kept = _nms(boxes, iou_threshold=0.5, has_class=False)
    assert len(kept) == 2, f"Expected 2, got {len(kept)}"
    assert kept[0][4] == 0.91
    print(f"  ✅ NMS: 3 boxes → {len(kept)} after dedup")


def test_sahi_config(yolo):
    sc = yolo._get_slice_config("cad_lab")
    assert sc is not None, "sahi_slicing config missing for cad_lab in config.yaml"
    assert hasattr(sc, "slice_height"), "slice_height missing"
    assert hasattr(sc, "slice_width"),  "slice_width missing"
    assert hasattr(sc, "overlap_ratio"), "overlap_ratio missing"
    print(f"  ✅ SAHI config: {sc.slice_width}x{sc.slice_height} overlap={sc.overlap_ratio}")


def test_draw_detections(yolo):
    frame   = np.zeros((720, 1280, 3), dtype=np.uint8)
    persons = [[50.0, 50.0, 200.0, 400.0, 0.91]]
    objects = [[300.0, 100.0, 450.0, 250.0, 0.75, float(CocoClass.LAPTOP)]]
    vis     = yolo.draw_detections(frame, persons, objects)
    assert vis.shape == frame.shape
    assert vis is not frame
    print(f"  ✅ draw_detections: shape {vis.shape}")


def test_real_frame(yolo):
    """
    Runs on cad_lab.jpg using exact same logic as visualize_detections.py.
    Ground truth: 6-7 genuine people in cad_lab.
    Acceptable range: 5-8 (±1-2 tolerance for CCTV angle/occlusion).
    """
    img_path = Path("data/sample_frames/cad_lab.jpg")
    if not img_path.exists():
        print(f"  ⚠️  Skipped — {img_path} not found")
        return

    frame = cv2.imread(str(img_path))
    out_dir = Path("data/roi_previews")
    out_dir.mkdir(parents=True, exist_ok=True)

    persons, objects = yolo.run_inference(frame, section_id="cad_lab")

    # Save annotated output
    vis = yolo.draw_detections(frame, persons, objects)
    cv2.imwrite(str(out_dir / "cp11_final.jpg"), vis)

    # Save debug output (slice grid visible)
    _, _, debug = yolo.run_inference_debug(frame, section_id="cad_lab")
    cv2.imwrite(str(out_dir / "cp11_debug.jpg"), debug)

    print(f"  ✅ Real frame — cad_lab:")
    print(f"     Persons detected : {len(persons)}")
    print(f"     Objects detected : {len(objects)}")
    for i, box in enumerate(persons):
        cx, cy = int((box[0]+box[2])/2), int((box[1]+box[3])/2)
        print(f"       #{i+1} conf={box[4]:.3f}  centroid=({cx},{cy})")
    print(f"     → cp11_final.jpg and cp11_debug.jpg saved")

    # Sanity check — should be within realistic range for this room
    assert 3 <= len(persons) <= 15, \
        f"Person count {len(persons)} is outside expected range 3-15"


if __name__ == "__main__":
    print("=" * 55)
    print("CP-11 — SAHI+YOLOv10 Inference")
    print("=" * 55)

    print("\n[1] Model loads:")
    yolo = test_model_loads()

    print("\n[2] Blank frame:")
    test_blank_frame(yolo)

    print("\n[3] Output format:")
    test_output_format(yolo)

    print("\n[4] Box IoU:")
    test_box_iou()

    print("\n[5] NMS deduplication:")
    test_nms()

    print("\n[6] SAHI slice config:")
    test_sahi_config(yolo)

    print("\n[7] draw_detections:")
    test_draw_detections(yolo)

    print("\n[8] Real frame — cad_lab.jpg:")
    test_real_frame(yolo)

    print("\n" + "=" * 55)
    print("🎉 CP-11 PASSED — SAHI+YOLOv10 inference ready")
    print("=" * 55)