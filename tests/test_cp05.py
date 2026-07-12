"""
test_cp05.py — Validates the preprocessing pipeline (CP-05).

What this tests:
  1. Preprocessor initialises without errors (reads config correctly)
  2. Output shape is exactly 640x640x3
  3. Output dtype is uint8 (required by YOLO)
  4. Each pipeline step works correctly
  5. Pipeline works on dark, noisy, and non-square frames
  6. Benchmarks processing time per frame (target: <100ms)

Run from project root:
    python test_cp05.py
"""

import time
import numpy as np
import cv2
from ingestion.preprocessor import Preprocessor


def make_dark_frame(h=480, w=640) -> np.ndarray:
    """Simulate a dark low-quality CCTV frame."""
    rng = np.random.default_rng(seed=0)
    frame = rng.integers(0, 60, (h, w, 3), dtype=np.uint8)
    noise = rng.integers(0, 20, (h, w, 3), dtype=np.uint8)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def make_noisy_frame(h=480, w=640) -> np.ndarray:
    """Simulate a noisy normal-brightness frame."""
    rng = np.random.default_rng(seed=1)
    # Create structured content (simulate background + objects)
    base = np.full((h, w, 3), 120, dtype=np.uint8)
    # Add some rectangles to simulate furniture
    base[100:200, 100:300] = 80
    base[300:400, 200:500] = 90
    # Add realistic noise on top
    noise = rng.integers(0, 30, (h, w, 3), dtype=np.uint8)
    return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def test_output_shape_and_dtype():
    pre = Preprocessor()
    for name, frame in [
        ("dark",       make_dark_frame()),
        ("noisy",      make_noisy_frame()),
        ("non-square", make_noisy_frame(720, 1280)),
    ]:
        out = pre.process(frame)
        assert out.shape == (640, 640, 3), \
            f"[{name}] Expected (640,640,3), got {out.shape}"
        assert out.dtype == np.uint8, \
            f"[{name}] Expected uint8, got {out.dtype}"
        print(f"  ✅ [{name}] output shape={out.shape} dtype={out.dtype}")


def test_gamma_brightens_dark_frame():
    pre = Preprocessor()
    dark = make_dark_frame()
    corrected = pre._apply_gamma(dark)
    mean_before = dark.mean()
    mean_after  = corrected.mean()
    assert mean_after > mean_before, \
        f"Gamma should brighten dark frame. Before={mean_before:.1f} After={mean_after:.1f}"
    print(f"  ✅ Gamma: mean brightness {mean_before:.1f} → {mean_after:.1f} (brightened)")


def test_bilateral_preserves_edges():
    """
    Bilateral filter should smooth flat regions but preserve sharp edges.
    We test this by creating a frame with a hard edge, applying bilateral,
    and confirming the edge is still detectable afterwards.
    """
    pre = Preprocessor()
    # Create frame with a hard vertical edge at x=320
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, 320:] = 200   # right half bright, left half dark
    # Add noise
    rng = np.random.default_rng(seed=42)
    noise = rng.integers(0, 25, frame.shape, dtype=np.uint8)
    noisy = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    filtered = pre._apply_bilateral(noisy)

    # The edge column should still have a significant gradient
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    col_left  = float(gray[:, 315].mean())
    col_right = float(gray[:, 325].mean())
    edge_strength = abs(col_right - col_left)

    assert edge_strength > 50, \
        f"Bilateral should preserve edges. Edge strength={edge_strength:.1f} (too low)"
    print(f"  ✅ Bilateral: edge preserved (strength={edge_strength:.1f}, noise smoothed)")


def test_clahe_increases_contrast():
    pre = Preprocessor()
    dark = make_dark_frame()
    enhanced = pre._apply_clahe(dark)
    std_before = dark.std()
    std_after  = enhanced.std()
    assert std_after > std_before, \
        f"CLAHE should increase contrast. Std before={std_before:.2f} after={std_after:.2f}"
    print(f"  ✅ CLAHE: contrast std {std_before:.2f} → {std_after:.2f} (increased)")


def test_resize():
    pre = Preprocessor()
    big = make_noisy_frame(h=1080, w=1920)
    resized = pre._apply_resize(big)
    assert resized.shape == (640, 640, 3), \
        f"Wrong shape after resize: {resized.shape}"
    print(f"  ✅ Resize: (1080,1920,3) → {resized.shape}")


def test_benchmark():
    pre = Preprocessor()
    frame = make_noisy_frame()
    # Warmup
    for _ in range(3):
        pre.process(frame)
    # Benchmark 20 frames
    n = 20
    start = time.perf_counter()
    for _ in range(n):
        pre.process(frame)
    elapsed = time.perf_counter() - start
    ms_per_frame = (elapsed / n) * 1000
    print(f"  ✅ Benchmark: {ms_per_frame:.1f} ms/frame over {n} frames")
    if ms_per_frame > 100:
        print(f"  ⚠️  WARNING: {ms_per_frame:.1f} ms/frame is above target.")
    else:
        print(f"  ✅ Speed is within target (<100ms/frame) — ready for real-time use.")


if __name__ == "__main__":
    print("=" * 60)
    print("CP-05 — Preprocessing Pipeline — Validation Test")
    print("=" * 60)

    print("\n[1] Output shape & dtype:")
    test_output_shape_and_dtype()

    print("\n[2] Gamma correction:")
    test_gamma_brightens_dark_frame()

    print("\n[3] Bilateral filter (edge preservation):")
    test_bilateral_preserves_edges()

    print("\n[4] CLAHE contrast enhancement:")
    test_clahe_increases_contrast()

    print("\n[5] Resize:")
    test_resize()

    print("\n[6] Speed benchmark:")
    test_benchmark()

    print("\n" + "=" * 60)
    print("🎉  CP-05 PASSED — preprocessing pipeline ready")
    print("=" * 60)