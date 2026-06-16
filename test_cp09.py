"""
test_cp09.py — Validates CP-09 auto-calibration pipeline.

Tests all pure-Python logic (polygon conversion, IoU, duplicate removal,
visualisation) without loading the heavy DINO/SAM2 models.
The full model pipeline is tested via a live integration run at the end.
"""

import sys
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from roi.auto_calibrator import (
    _mask_to_polygon,
    _box_iou,
    _remove_duplicate_boxes,
    visualise_results,
)
from utils.helpers import read_json


def test_mask_to_polygon():
    # Solid white square in centre of a blank mask
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:300, 100:300] = 255

    poly = _mask_to_polygon(mask)
    assert poly is not None,      "Should detect a valid polygon"
    assert len(poly) >= 3,        "Polygon needs at least 3 points"
    assert all(len(p) == 2 for p in poly), "Each point must be [x, y]"
    print(f"  ✅ mask_to_polygon: {len(poly)} vertices")


def test_mask_to_polygon_noise():
    # Tiny blob — should be rejected (below min_area)
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[10:15, 10:15] = 255    # 5x5 = 25 px < min_area 500

    poly = _mask_to_polygon(mask)
    assert poly is None, "Tiny mask should return None"
    print("  ✅ mask_to_polygon: tiny mask correctly rejected")


def test_box_iou():
    # Identical boxes → IoU = 1.0
    assert abs(_box_iou([0,0,100,100], [0,0,100,100]) - 1.0) < 1e-5

    # Non-overlapping → IoU = 0.0
    assert _box_iou([0,0,100,100], [200,200,300,300]) == 0.0

    # Partial overlap
    iou = _box_iou([0,0,100,100], [50,50,150,150])
    assert 0.0 < iou < 1.0, f"Expected partial IoU, got {iou}"
    print(f"  ✅ box_iou: identical=1.0, non-overlap=0.0, partial={iou:.3f}")


def test_remove_duplicate_boxes():
    boxes = [
        [0,   0,  100, 100],   # original
        [5,   5,  105, 105],   # near-duplicate (high IoU)
        [300, 300, 400, 400],  # distinct box
    ]
    kept = _remove_duplicate_boxes(boxes, iou_threshold=0.5)
    assert len(kept) == 2, f"Expected 2 boxes after dedup, got {len(kept)}"
    print(f"  ✅ remove_duplicate_boxes: {len(boxes)} → {len(kept)}")


def test_visualise_results(tmpdir):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    roi_data = {
        "section_A": {
            "seat_A1": [[50,50],[200,50],[200,200],[50,200]],
            "seat_A2": [[250,50],[400,50],[400,200],[250,200]],
        }
    }
    save_path = Path(tmpdir) / "vis_test.jpg"
    vis = visualise_results(frame, roi_data, "section_A", save_path=save_path)

    assert vis is not None
    assert vis.shape == frame.shape
    assert save_path.exists(), "Visualisation image not saved"
    print(f"  ✅ visualise_results: image saved ({vis.shape})")


def test_full_pipeline(image_path: str, section_id: str = "section_A"):
    """
    Integration test — runs the real DINO+SAM2 pipeline on your image.
    Only called when --integration flag is passed.
    """
    from roi.auto_calibrator import run_auto_calibration
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "draft_roi.json"
        roi_data = run_auto_calibration(
            section_id      = section_id,
            image_path      = image_path,
            roi_output_path = output,
        )

        assert isinstance(roi_data, dict), "Output must be a dict"
        assert section_id in roi_data,     f"{section_id} missing from output"
        seats = roi_data[section_id]
        assert len(seats) > 0, "No seats detected — check image or lower thresholds"

        for seat_id, pts in seats.items():
            assert isinstance(pts, list)
            assert all(len(p) == 2 for p in pts)

        print(f"  ✅ Full pipeline: {len(seats)} seats detected")
        for sid, pts in seats.items():
            print(f"     {sid}: {len(pts)} vertices")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--integration", action="store_true",
                        help="Run full DINO+SAM2 pipeline (requires models)")
    parser.add_argument("--image",   default=None, help="Image path for integration test")
    parser.add_argument("--section", default="section_A")
    args = parser.parse_args()

    print("=" * 60)
    print("CP-09 — Auto-calibration — Unit Tests")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp(prefix="cp09_")
    try:
        print("\n[1] mask_to_polygon (valid mask):")
        test_mask_to_polygon()

        print("\n[2] mask_to_polygon (noise rejection):")
        test_mask_to_polygon_noise()

        print("\n[3] box_iou:")
        test_box_iou()

        print("\n[4] remove_duplicate_boxes:")
        test_remove_duplicate_boxes()

        print("\n[5] visualise_results:")
        test_visualise_results(tmpdir)

        print("\n" + "=" * 60)
        print("🎉 CP-09 unit tests PASSED")
        print("=" * 60)

        if args.integration:
            if not args.image:
                print("\n❌ Pass --image <path> to run integration test.")
            else:
                print(f"\n[6] Full DINO+SAM2 pipeline on: {args.image}")
                test_full_pipeline(args.image, args.section)
                print("\n🎉 CP-09 integration test PASSED")
        else:
            print("\nTo run the full model pipeline:")
            print("  python test_cp09.py --integration --image data/sample_frames/section_A.jpg")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)