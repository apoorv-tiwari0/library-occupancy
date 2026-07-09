"""
test_cp13.py — Validates temporal smoother (CP-13)
"""

import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from tracking.smoother import TemporalSmoother, MultiSectionSmoother, _majority_vote
from detection.pipeline import SectionResult, MultiSectionPipeline


def make_result(headcount=0, section_id="cad_lab", max_cap=18) -> SectionResult:
    vacancy = max(0, max_cap - headcount) if max_cap else None
    return SectionResult(
        section_id   = section_id,
        timestamp    = "2026-06-18T00:00:00",
        headcount    = headcount,
        max_capacity = max_cap,
        vacancy      = vacancy,
        person_boxes = [],
        object_boxes = [],
        inference_ms = 0.0,
        pipeline_ms  = 0.0,
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_init() -> TemporalSmoother:
    s = TemporalSmoother(section_id="cad_lab")
    assert s.window_size > 0
    assert 0 < s.ema_alpha <= 1
    assert s.buffer_fill() == 0.0
    print(f"  ✅ Init | window={s.window_size} ema_alpha={s.ema_alpha}")
    return s


def test_warmup(s: TemporalSmoother) -> None:
    """Before buffer half-fills, raw headcount passes through unchanged."""
    s.reset()
    out = s.update(make_result(headcount=5))
    assert out.headcount == 5
    print("  ✅ Warmup — raw headcount passed through")


def test_blip_suppressed(s: TemporalSmoother) -> None:
    """One-frame spike (person briefly detected) should be smoothed out."""
    s.reset()
    for _ in range(s.window_size):
        s.update(make_result(headcount=4))
    # One spike frame
    out = s.update(make_result(headcount=12))
    assert out.headcount == 4, \
        f"Blip should be suppressed, got headcount={out.headcount}"
    print(f"  ✅ Blip suppressed — headcount stayed at 4 despite spike to 12")


def test_genuine_change(s: TemporalSmoother) -> None:
    """Consistent new headcount for N frames should update the output."""
    s.reset()
    for _ in range(s.window_size):
        s.update(make_result(headcount=2))
    for _ in range(s.window_size):
        s.update(make_result(headcount=8))
    out = s.update(make_result(headcount=8))
    assert out.headcount == 8, \
        f"Genuine change not picked up, got headcount={out.headcount}"
    print(f"  ✅ Genuine change — headcount updated to 8 after {s.window_size} frames")


def test_vacancy_recalculated(s: TemporalSmoother) -> None:
    """Vacancy must be recalculated from smoothed headcount, not raw."""
    s.reset()
    for _ in range(s.window_size):
        s.update(make_result(headcount=6, max_cap=18))
    out = s.update(make_result(headcount=6, max_cap=18))
    assert out.vacancy == 12, \
        f"Expected vacancy=12 (18-6), got {out.vacancy}"
    print(f"  ✅ Vacancy recalculated from smoothed headcount: 18-6=12")


def test_vacancy_not_negative(s: TemporalSmoother) -> None:
    s.reset()
    out = s.update(make_result(headcount=50, max_cap=18))
    assert out.vacancy >= 0
    print(f"  ✅ Vacancy non-negative even with headcount > max_capacity")


def test_buffer_fill(s: TemporalSmoother) -> None:
    s.reset()
    assert s.buffer_fill() == 0.0
    for _ in range(s.window_size):
        s.update(make_result())
    assert s.buffer_fill() == 1.0
    print(f"  ✅ buffer_fill: 0.0 → 1.0 after {s.window_size} frames")


def test_reset(s: TemporalSmoother) -> None:
    for _ in range(5):
        s.update(make_result(headcount=7))
    s.reset()
    assert s.buffer_fill()   == 0.0
    assert s._ema            is None
    assert s._frame_count    == 0
    assert len(s._buffer)    == 0
    print("  ✅ reset() clears all state")


def test_majority_vote() -> None:
    buf = deque([3, 3, 4, 3, 5, 3, 4])
    assert _majority_vote(buf) == 3
    print("  ✅ _majority_vote: [3,3,4,3,5,3,4] → 3")


def test_multi_section_smoother() -> None:
    ms = MultiSectionSmoother()
    for _ in range(10):
        ms.update("sec_A", make_result(headcount=5, section_id="sec_A", max_cap=20))
        ms.update("sec_B", make_result(headcount=0, section_id="sec_B", max_cap=15))

    out_a = ms.update("sec_A", make_result(headcount=5, section_id="sec_A", max_cap=20))
    out_b = ms.update("sec_B", make_result(headcount=0, section_id="sec_B", max_cap=15))

    assert out_a.headcount == 5
    assert out_b.headcount == 0
    fills = ms.buffer_fills()
    assert "sec_A" in fills and "sec_B" in fills
    print(f"  ✅ MultiSectionSmoother — sec_A=5 sec_B=0, isolated correctly")


def test_real_pipeline_integration() -> None:
    """Smoother on top of real pipeline on cad_lab.jpg — 15 frames."""
    img_path = Path("data/sample_frames/cad_lab.jpg")
    if not img_path.exists():
        print(f"  ⚠️  Skipped — {img_path} not found")
        return

    frame    = cv2.imread(str(img_path))
    pipeline = MultiSectionPipeline()
    smoother = TemporalSmoother(section_id="cad_lab")

    raw_result = smoothed_result = None
    for _ in range(15):
        raw_result      = pipeline.run("cad_lab", frame)
        smoothed_result = smoother.update(raw_result)

    print(f"  ✅ Real pipeline + smoother (15 frames):")
    print(f"     Raw      headcount={raw_result.headcount} vacancy={raw_result.vacancy}")
    print(f"     Smoothed headcount={smoothed_result.headcount} vacancy={smoothed_result.vacancy}")
    print(f"     Buffer fill: {smoother.buffer_fill():.0%}")

    assert smoothed_result.headcount >= 0
    assert smoothed_result.vacancy   >= 0
    assert smoother.buffer_fill()    == 1.0


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("CP-13 — Temporal Smoother")
    print("=" * 55)

    print("\n[1] Init:")
    s = test_init()

    print("\n[2] Warmup passthrough:")
    test_warmup(s)

    print("\n[3] Blip suppression:")
    test_blip_suppressed(s)

    print("\n[4] Genuine state change:")
    test_genuine_change(s)

    print("\n[5] Vacancy recalculated from smoothed headcount:")
    test_vacancy_recalculated(s)

    print("\n[6] Vacancy non-negative:")
    test_vacancy_not_negative(s)

    print("\n[7] Buffer fill tracking:")
    test_buffer_fill(s)

    print("\n[8] Reset:")
    test_reset(s)

    print("\n[9] Majority vote helper:")
    test_majority_vote()

    print("\n[10] MultiSectionSmoother isolation:")
    test_multi_section_smoother()

    print("\n[11] Real pipeline integration (15 frames):")
    test_real_pipeline_integration()

    print("\n" + "=" * 55)
    print("🎉 CP-13 PASSED — temporal smoother ready")
    print("=" * 55)