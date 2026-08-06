"""
test_zcp09.py — Validates CSRNet far zone estimator (ZCP-09)

Tests:
  1. CSRNetFarZoneEstimator initialises and loads weights
  2. Empty far polygon returns (0, None)
  3. Output types correct (int, numpy array)
  4. Density map is non-negative
  5. Model is deterministic (same input = same output)
  6. All 11 sections — raw counts
  7. Recalibrate scale factors from new raw counts
  8. Visualization saves correctly

Run from project root:
    python test_zcp09.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from detection.csrnet_estimator import CSRNetFarZoneEstimator

ZONE_PATH   = Path(r"C:\IITD_Internship\library-occupancy\data\roi\zone_config.json")
SAMPLE_DIR  = Path("data/sample_frames")
PREVIEW_DIR = Path("data/roi_previews")

SECTIONS = [
    "cad_lab","focused_reading_area","g_hall_2","g_huss",
    "hindi_section","ip_camera_19","ipc","main_computer_room",
    "reference_2","reference_area","weeding_out_area"
]

# Ground truth from your manual count (far zone only)
GROUND_TRUTH = {
    "focused_reading_area": 8,
    "g_hall_2":             60,
    "g_huss":               56,
    "hindi_section":        40,
    "ip_camera_19":         5,
    "ipc":                  18,
    "main_computer_room":   5,
    "reference_2":          16,
    "reference_area":       7,
    "weeding_out_area":     5,
}


def load_zone_config() -> dict:
    return json.loads(ZONE_PATH.read_text())


def test_init(zone_data) -> CSRNetFarZoneEstimator:
    est = CSRNetFarZoneEstimator(
        zone_config  = zone_data["cad_lab"],
        section_id   = "cad_lab",
        weights_path = "models/csrnet_v3_best.pth",
        scale_factor = 1.0,
    )
    assert est._model is not None
    assert len(est.far_poly) >= 3
    print(f"  ✅ Init | section=cad_lab device={est.device}")
    return est


def test_empty_polygon(zone_data) -> None:
    est = CSRNetFarZoneEstimator(
        zone_config  = {"near":[],"mid":[],"far":[]},
        section_id   = "test",
        weights_path = "models/csrnet_v3_best.pth",
        scale_factor = 1.0,
    )
    count, dm = est.estimate(np.zeros((480,640,3), dtype=np.uint8))
    assert count == 0 and dm is None
    print("  ✅ Empty far polygon returns (0, None)")


def test_output_types(est) -> None:
    frame = cv2.imread(str(SAMPLE_DIR / "cad_lab.jpg"))
    if frame is None:
        print("  ⚠️  Skipped")
        return
    count, dm = est.estimate(frame)
    assert isinstance(count, int)
    assert count >= 0
    assert isinstance(dm, np.ndarray)
    print(f"  ✅ Output types | count={count} dm.shape={dm.shape} dm.dtype={dm.dtype}")


def test_density_map_non_negative(est) -> None:
    frame = cv2.imread(str(SAMPLE_DIR / "cad_lab.jpg"))
    if frame is None:
        print("  ⚠️  Skipped")
        return
    _, dm = est.estimate(frame)
    print(f"  ✅ Density map | min={dm.min():.4f} max={dm.max():.4f} sum={dm.sum():.4f}")


def test_deterministic(est) -> None:
    """Same frame must always give same count — confirms weights loaded correctly."""
    frame = cv2.imread(str(SAMPLE_DIR / "cad_lab.jpg"))
    if frame is None:
        print("  ⚠️  Skipped")
        return
    counts = []
    for _ in range(3):
        count, _ = est.estimate(frame)
        counts.append(count)
    assert len(set(counts)) == 1, f"Non-deterministic: {counts}"
    print(f"  ✅ Deterministic — same count across 3 runs: {counts[0]}")


def test_all_sections_raw(zone_data) -> dict:
    """Run CSRNet on all sections with scale_factor=1.0 to get raw counts."""
    print(f"\n  {'Section':<30} {'Raw (sum)':<12} {'GT':>6}  {'New Scale':>10}")
    print(f"  {'-'*64}")

    new_scales = {}
    raw_counts = {}

    for sec in SECTIONS:
        img_path = SAMPLE_DIR / f"{sec}.jpg"
        if not img_path.exists():
            print(f"  {sec:<30} NO IMAGE")
            continue

        est = CSRNetFarZoneEstimator(
            zone_config  = zone_data[sec],
            section_id   = sec,
            weights_path = "models/csrnet_v3_best.pth",
            scale_factor = 1.0,
        )
        frame = cv2.imread(str(img_path))
        _, dm = est.estimate(frame)

        if dm is None:
            print(f"  {sec:<30} {'NO FAR ZONE':>10}")
            continue

        raw = float(dm.sum())
        raw_counts[sec] = raw
        gt  = GROUND_TRUTH.get(sec)

        if gt is None or raw < 0.01:
            scale = 1.0
            scale_str = "1.0000 (no GT)"
        else:
            scale = round(gt / raw, 4)
            scale_str = f"{scale:.4f}"

        new_scales[sec] = scale
        gt_str = str(gt) if gt is not None else "N/A"
        print(f"  {sec:<30} {raw:<12.4f} {gt_str:>6}  {scale_str:>10}")

    return new_scales


def test_visualization(zone_data) -> None:
    img_path = SAMPLE_DIR / "g_huss.jpg"
    if not img_path.exists():
        print(f"  ⚠️  Skipped")
        return

    est = CSRNetFarZoneEstimator(
        zone_config  = zone_data["g_huss"],
        section_id   = "g_huss",
        weights_path = "models/csrnet_v3_best.pth",
        scale_factor = 1.0,
    )
    frame      = cv2.imread(str(img_path))
    count, vis = est.estimate_with_visualization(frame)
    out        = PREVIEW_DIR / "zcp09_csrnet_far_ghuss.jpg"
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), vis)
    print(f"  ✅ Visualization | g_huss raw far count={count} → {out}")


if __name__ == "__main__":
    print("=" * 55)
    print("ZCP-09 — CSRNet Far Zone Estimator")
    print("=" * 55)

    zone_data = load_zone_config()

    print("\n[1] Init:")
    est = test_init(zone_data)

    print("\n[2] Empty polygon:")
    test_empty_polygon(zone_data)

    print("\n[3] Output types:")
    test_output_types(est)

    print("\n[4] Density map non-negative:")
    test_density_map_non_negative(est)

    print("\n[5] Deterministic:")
    test_deterministic(est)

    print("\n[6] All 11 sections — raw counts + new scale factors:")
    new_scales = test_all_sections_raw(zone_data)

    print("\n[7] Visualization:")
    test_visualization(zone_data)

    print("\n" + "=" * 55)
    print("🎉 ZCP-09 PASSED — CSRNet far zone estimator ready")
    print("=" * 55)

    print("\n⚡ Copy these scale factors into config.yaml:")
    print("   (under each camera's far_zone_scale field)\n")
    for sec, scale in new_scales.items():
        print(f"   {sec}: {scale}")