"""
test_cp14.py — Validates per-section inference pipeline (CP-14)

Tests:
  1. SectionPipeline initialises correctly
  2. run() returns a SectionResult with correct types
  3. SectionResult.to_dict() is fully serialisable
  4. is_available() and occupancy_pct() work correctly
  5. Blank frame returns 0 headcount, all seats FREE
  6. Real frame end-to-end — cad_lab.jpg
  7. MultiSectionPipeline builds from config + ROI
  8. run_with_visualization returns annotated frame

Run from project root:
    python test_cp14.py
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from detection.pipeline import SectionPipeline, MultiSectionPipeline, SectionResult
from detection.yolo_inference import YOLOInference
from ingestion.preprocessor import Preprocessor

ROI_PATH = Path(r"C:\IITD_Internship\library-occupancy\data\roi\roi_polygons.json")
SECTION  = "cad_lab"


def load_roi() -> dict:
    if not ROI_PATH.exists():
        raise FileNotFoundError(f"ROI file not found: {ROI_PATH}")
    return json.loads(ROI_PATH.read_text())


def make_blank(h=1080, w=1920) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_pipeline_init(yolo, roi_data) -> SectionPipeline:
    pipeline = SectionPipeline(
        section_id    = SECTION,
        seat_polygons = roi_data[SECTION],
        max_capacity  = 24,
        yolo          = yolo,
    )
    assert pipeline.section_id    == SECTION
    assert pipeline.max_capacity  == 24
    assert len(pipeline.seat_polygons) > 0
    print(f"  ✅ SectionPipeline init | seats={len(pipeline.seat_polygons)}")
    return pipeline


def test_result_types(pipeline) -> None:
    result = pipeline.run(make_blank())
    assert isinstance(result, SectionResult)
    assert isinstance(result.section_id,   str)
    assert isinstance(result.timestamp,    str)
    assert isinstance(result.headcount,    int)
    assert isinstance(result.vacancy,      int)
    assert isinstance(result.seat_states,  dict)
    assert isinstance(result.counts,       dict)
    assert isinstance(result.person_boxes, list)
    assert isinstance(result.object_boxes, list)
    assert isinstance(result.inference_ms, float)
    assert isinstance(result.pipeline_ms,  float)
    print(f"  ✅ SectionResult types all correct")


def test_to_dict(pipeline) -> None:
    result = pipeline.run(make_blank())
    d = result.to_dict()
    # Must be JSON serialisable
    json_str = json.dumps(d)
    assert len(json_str) > 0
    assert "section_id"   in d
    assert "headcount"    in d
    assert "vacancy"      in d
    assert "seat_states"  in d
    assert "is_available" in d
    print(f"  ✅ to_dict() is JSON serialisable ({len(json_str)} chars)")


def test_is_available(pipeline) -> None:
    result = pipeline.run(make_blank())
    # Blank frame → 0 people → vacancy = max_capacity → available
    assert result.is_available() == True
    assert result.occupancy_pct() == 0.0
    print(f"  ✅ is_available=True, occupancy_pct=0.0 on blank frame")


def test_blank_frame(pipeline) -> None:
    result = pipeline.run(make_blank())
    assert result.headcount == 0
    assert result.vacancy   == pipeline.max_capacity
    assert all(v == "free" for v in result.seat_states.values()), \
        f"Expected all FREE on blank frame, got: {set(result.seat_states.values())}"
    print(
        f"  ✅ Blank frame | headcount=0 vacancy={result.vacancy} "
        f"all {len(result.seat_states)} seats FREE"
    )


def test_real_frame(pipeline) -> None:
    img_path = Path(f"data/sample_frames/{SECTION}.jpg")
    if not img_path.exists():
        print(f"  ⚠️  Skipped — {img_path} not found")
        return

    frame  = cv2.imread(str(img_path))
    result = pipeline.run(frame)

    out_dir = Path("data/roi_previews")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save visualization
    from detection.classifier import SeatClassifier
    clf = pipeline.classifier
    vis = clf.draw_states(
        frame, pipeline.seat_polygons,
        result.seat_states, result.headcount, result.max_capacity
    )
    cv2.imwrite(str(out_dir / "cp14_pipeline.jpg"), vis)

    print(f"  ✅ Real frame — {SECTION}:")
    print(f"     Headcount    : {result.headcount}")
    print(f"     Vacancy      : {result.vacancy}/{result.max_capacity}")
    print(f"     Occupancy    : {result.occupancy_pct()}%")
    print(f"     Available    : {result.is_available()}")
    print(f"     Occupied seats: {result.counts['occupied']}")
    print(f"     Reserved seats: {result.counts['reserved']}")
    print(f"     Free seats    : {result.counts['free']}")
    print(f"     Inference     : {result.inference_ms:.0f}ms")
    print(f"     Pipeline total: {result.pipeline_ms:.0f}ms")
    print(f"     → cp14_pipeline.jpg saved")

    assert result.headcount    >= 0
    assert result.pipeline_ms  >  0
    assert result.inference_ms >  0
    assert len(result.seat_states) == len(pipeline.seat_polygons)


def test_run_with_visualization(pipeline) -> None:
    img_path = Path(f"data/sample_frames/{SECTION}.jpg")
    if not img_path.exists():
        print(f"  ⚠️  Skipped — {img_path} not found")
        return

    frame        = cv2.imread(str(img_path))
    result, vis  = pipeline.run_with_visualization(frame)

    assert vis.shape == frame.shape, "Visualization shape must match input frame"
    assert isinstance(result, SectionResult)

    out = Path("data/roi_previews/cp14_viz.jpg")
    cv2.imwrite(str(out), vis)
    print(f"  ✅ run_with_visualization | shape={vis.shape} → cp14_viz.jpg saved")


def test_multi_section_pipeline(roi_data) -> None:
    multi = MultiSectionPipeline(roi_data)
    sections = multi.sections()
    assert len(sections) > 0
    print(f"  ✅ MultiSectionPipeline | sections={sections}")

    # Run on blank frames for all sections
    blank_frames = {sid: make_blank() for sid in sections}
    t0      = time.perf_counter()
    results = multi.run_all(blank_frames)
    elapsed = (time.perf_counter() - t0) * 1000

    assert len(results) == len(sections)
    for sid, result in results.items():
        assert result.section_id == sid
        assert result.headcount  == 0

    print(
        f"  ✅ run_all | {len(sections)} sections | "
        f"total={elapsed:.0f}ms avg={elapsed/len(sections):.0f}ms/section"
    )


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("CP-14 — Per-Section Inference Pipeline")
    print("=" * 55)

    roi_data = load_roi()

    print("\nLoading shared YOLO model (once)...")
    yolo = YOLOInference()

    print("\n[1] Pipeline init:")
    pipeline = test_pipeline_init(yolo, roi_data)

    print("\n[2] Result types:")
    test_result_types(pipeline)

    print("\n[3] to_dict() serialisable:")
    test_to_dict(pipeline)

    print("\n[4] is_available / occupancy_pct:")
    test_is_available(pipeline)

    print("\n[5] Blank frame → all FREE:")
    test_blank_frame(pipeline)

    print("\n[6] Real frame end-to-end:")
    test_real_frame(pipeline)

    print("\n[7] run_with_visualization:")
    test_run_with_visualization(pipeline)

    print("\n[8] MultiSectionPipeline:")
    test_multi_section_pipeline(roi_data)

    print("\n" + "=" * 55)
    print("🎉 CP-14 PASSED — per-section pipeline ready")
    print("=" * 55)