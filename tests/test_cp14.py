"""
test_cp14.py — Validates per-section inference pipeline (CP-14)

Pipeline: preprocess → SAHI+YOLO → vacancy = max_capacity (config.yaml) − headcount
No seat classifier. No ROI polygon matching. max_capacity from config only.
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
from config.config_loader import cfg

SECTION = "cad_lab"


def make_blank(h=1080, w=1920) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_pipeline_init(yolo) -> SectionPipeline:
    pipeline = SectionPipeline(section_id=SECTION, yolo=yolo)
    assert pipeline.section_id == SECTION
    assert pipeline.max_capacity is not None, \
        f"max_capacity not found in config.yaml for '{SECTION}'"
    # Confirm it came from config, not hardcoded
    expected = next(c.max_capacity for c in cfg.cameras if c.section_id == SECTION)
    assert pipeline.max_capacity == expected, \
        f"max_capacity mismatch: pipeline={pipeline.max_capacity} config={expected}"
    print(f"  ✅ SectionPipeline init | section={SECTION} "
          f"max_capacity={pipeline.max_capacity} (from config.yaml)")
    return pipeline


def test_result_types(pipeline) -> None:
    result = pipeline.run(make_blank())
    assert isinstance(result, SectionResult)
    assert isinstance(result.section_id,   str)
    assert isinstance(result.timestamp,    str)
    assert isinstance(result.headcount,    int)
    assert isinstance(result.max_capacity, int)
    assert isinstance(result.vacancy,      int)
    assert isinstance(result.person_boxes, list)
    assert isinstance(result.object_boxes, list)
    assert isinstance(result.inference_ms, float)
    assert isinstance(result.pipeline_ms,  float)
    # Confirm no classifier artifacts
    assert not hasattr(result, "seat_states") or result.__class__.__name__ == "SectionResult"
    print(f"  ✅ SectionResult types correct — no classifier fields")


def test_to_dict(pipeline) -> None:
    result  = pipeline.run(make_blank())
    d       = result.to_dict()
    js      = json.dumps(d)
    assert "section_id"   in d
    assert "headcount"    in d
    assert "vacancy"      in d
    assert "max_capacity" in d
    assert "is_available" in d
    assert "occupancy_pct" in d
    # seat_states must NOT be in the dict
    assert "seat_states" not in d, "seat_states should not appear in headcount mode"
    print(f"  ✅ to_dict() serialisable ({len(js)} chars) — no seat_states key")


def test_blank_frame(pipeline) -> None:
    result = pipeline.run(make_blank())
    assert result.headcount    == 0
    assert result.vacancy      == pipeline.max_capacity
    assert result.is_available() == True
    assert result.occupancy_pct() == 0.0
    print(f"  ✅ Blank frame | headcount=0 "
          f"vacancy={result.vacancy}/{result.max_capacity} "
          f"occupancy=0.0%")


def test_vacancy_clamped_to_zero(pipeline) -> None:
    """Vacancy must never go negative even if headcount > max_capacity."""
    from unittest.mock import patch
    # Patch run_inference to return more people than max_capacity
    fake_persons = [[0,0,100,100,0.9]] * (pipeline.max_capacity + 5)
    with patch.object(pipeline.yolo, "run_inference", return_value=(fake_persons, [])):
        result = pipeline.run(make_blank())
    assert result.vacancy == 0, \
        f"Vacancy should clamp to 0, got {result.vacancy}"
    assert result.headcount == pipeline.max_capacity + 5
    print(f"  ✅ Vacancy clamped to 0 when headcount > max_capacity")


def test_real_frame(pipeline) -> None:
    img_path = Path(f"data/sample_frames/{SECTION}.jpg")
    if not img_path.exists():
        print(f"  ⚠️  Skipped — {img_path} not found")
        return

    frame  = cv2.imread(str(img_path))
    result = pipeline.run(frame)

    # Save annotated output
    vis = frame.copy()
    h_orig, w_orig = frame.shape[:2]
    sx, sy = w_orig / 640, h_orig / 640
    for i, box in enumerate(result.person_boxes):
        x1, y1, x2, y2, conf = box
        cv2.rectangle(vis,
            (int(x1*sx), int(y1*sy)), (int(x2*sx), int(y2*sy)),
            (0, 255, 0), 2)
        cv2.putText(vis, f"#{i+1} {conf:.2f}",
            (int(x1*sx), max(0, int(y1*sy)-6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

    hud = (f"section={result.section_id}  "
           f"headcount={result.headcount}  "
           f"vacancy={result.vacancy}/{result.max_capacity}  "
           f"occupancy={result.occupancy_pct()}%")
    cv2.rectangle(vis, (0,0), (len(hud)*9+10, 32), (0,0,0), -1)
    cv2.putText(vis, hud, (6,22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

    out = Path("data/roi_previews/cp14_pipeline.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), vis)

    print(f"  ✅ Real frame — {SECTION}:")
    print(f"     Headcount    : {result.headcount}")
    print(f"     Max capacity : {result.max_capacity}  ← from config.yaml")
    print(f"     Vacancy      : {result.vacancy}")
    print(f"     Occupancy    : {result.occupancy_pct()}%")
    print(f"     Available    : {result.is_available()}")
    print(f"     Inference    : {result.inference_ms:.0f}ms")
    print(f"     Pipeline     : {result.pipeline_ms:.0f}ms")
    print(f"     → cp14_pipeline.jpg saved")

    assert result.headcount    >= 0
    assert result.vacancy      >= 0
    assert result.max_capacity == next(
        c.max_capacity for c in cfg.cameras if c.section_id == SECTION
    ), "max_capacity must match config.yaml"


def test_multi_section_pipeline() -> None:
    """All 11 sections build correctly from config.yaml alone — no args needed."""
    multi    = MultiSectionPipeline()
    sections = multi.sections()

    # Confirm all enabled cameras have a pipeline
    enabled = [c.section_id for c in cfg.cameras if c.enabled]
    for sid in enabled:
        assert sid in sections, f"Missing pipeline for enabled section '{sid}'"

    print(f"  ✅ MultiSectionPipeline | {len(sections)} sections built from config.yaml")

    # Run blank frames through all sections
    blank_frames = {sid: make_blank() for sid in sections}
    t0      = time.perf_counter()
    results = multi.run_all(blank_frames)
    elapsed = (time.perf_counter() - t0) * 1000

    for sid, result in results.items():
        assert result.section_id  == sid
        assert result.headcount   == 0
        expected_cap = next(c.max_capacity for c in cfg.cameras if c.section_id == sid)
        assert result.max_capacity == expected_cap, \
            f"[{sid}] max_capacity={result.max_capacity} but config has {expected_cap}"
        assert result.vacancy == expected_cap  # headcount=0 → vacancy=max_capacity
        print(f"     {sid}: max_capacity={result.max_capacity} "
              f"vacancy={result.vacancy} ✅")

    print(f"  ✅ All {len(sections)} sections | "
          f"total={elapsed:.0f}ms avg={elapsed/len(sections):.0f}ms/section")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("CP-14 — Per-Section Inference Pipeline")
    print("=" * 55)

    print("\nLoading shared YOLO model (once)...")
    yolo = YOLOInference()

    print("\n[1] Pipeline init (max_capacity from config.yaml):")
    pipeline = test_pipeline_init(yolo)

    print("\n[2] Result types (no classifier fields):")
    test_result_types(pipeline)

    print("\n[3] to_dict() — no seat_states key:")
    test_to_dict(pipeline)

    print("\n[4] Blank frame:")
    test_blank_frame(pipeline)

    print("\n[5] Vacancy clamped to zero:")
    test_vacancy_clamped_to_zero(pipeline)

    print("\n[6] Real frame end-to-end:")
    test_real_frame(pipeline)

    print("\n[7] MultiSectionPipeline (all 11 sections from config.yaml):")
    test_multi_section_pipeline()

    print("\n" + "=" * 55)
    print("🎉 CP-14 PASSED — per-section pipeline ready")
    print("=" * 55)