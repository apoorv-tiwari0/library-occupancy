"""
test_cp06.py — Validates multi-camera parallel ingestion (CP-06).

Run from project root:
    python test_cp06.py
"""

import os
import sys
import time
import tempfile
import shutil
import queue
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config.config_loader import cfg
from ingestion.stream_manager import StreamManager
from ingestion.frame_extractor import FramePacket


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_synthetic_video(path: str, n_frames: int = 120,
                          width: int = 640, height: int = 480,
                          section_label: str = "S") -> None:
    """Write a short AVI with section label stamped on each frame.
    Using 120 frames (4 seconds @ 30fps) to give preprocessor workers
    enough time to process before extraction finishes.
    """
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(path, fourcc, 30.0, (width, height))
    rng = np.random.default_rng(seed=42)
    for i in range(n_frames):
        frame = rng.integers(40, 200, (height, width, 3), dtype=np.uint8)
        cv2.putText(frame, f"{section_label} F{i+1}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


def patch_config_with_videos(video_paths: dict) -> None:
    """Monkey-patch cfg.cameras stream_urls to point to synthetic video files."""
    for cam in cfg.cameras:
        if cam.section_id in video_paths:
            cam.stream_url = video_paths[cam.section_id]


def drain_queue_with_timeout(
    manager: StreamManager,
    section_id: str = None,
    target_per_section: int = 5,
    timeout: float = 15.0,
) -> dict:
    """
    Collect frames until we have target_per_section from every section,
    or timeout elapses. Waits for preprocessor workers to finish flushing
    even after extractors have stopped.
    """
    collected = {cam.section_id: [] for cam in cfg.cameras if cam.enabled}
    deadline  = time.time() + timeout
    empty_streak = 0

    while time.time() < deadline:
        pkt = manager.get_frame(section_id=section_id, timeout=0.3)

        if pkt is None:
            empty_streak += 1
            # Give up only after 10 consecutive empty polls AND
            # extractors are all done — means queues are truly drained
            if empty_streak >= 10 and not manager.active_sections():
                break
            continue

        empty_streak = 0
        collected[pkt.section_id].append(pkt)

        if all(len(v) >= target_per_section for v in collected.values()):
            break

    return collected


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_stream_manager_builds() -> None:
    manager = StreamManager(preprocess=False)
    n_enabled = sum(1 for c in cfg.cameras if c.enabled)
    assert len(manager._extractors) == n_enabled, \
        f"Expected {n_enabled} extractors, got {len(manager._extractors)}"
    print(f"  ✅ StreamManager built | {n_enabled} camera(s) configured")


def test_parallel_ingestion(video_dir: str) -> dict:
    """
    Core test: verify frames arrive from ALL sections concurrently.
    """
    video_paths = {}
    for cam in cfg.cameras:
        if not cam.enabled:
            continue
        vpath = os.path.join(video_dir, f"{cam.section_id}.avi")
        make_synthetic_video(vpath, n_frames=120,
                              section_label=cam.section_id[-1].upper())
        video_paths[cam.section_id] = vpath

    patch_config_with_videos(video_paths)

    manager = StreamManager(preprocess=True)
    manager.start()

    start_time = time.time()
    collected  = drain_queue_with_timeout(manager, target_per_section=5)
    elapsed    = time.time() - start_time

    manager.stop()

    print(f"  ✅ Collected frames in {elapsed:.1f}s:")
    for section_id, pkts in collected.items():
        print(f"     {section_id}: {len(pkts)} frame(s)")
        assert len(pkts) > 0, \
            f"No frames received from {section_id} — check extractor or preprocessor worker"

    print(f"  ✅ All sections delivered frames (parallel ingestion confirmed)")
    return collected


def test_packet_fields(collected: dict) -> None:
    """Verify every packet has correct metadata and preprocessed shape."""
    for section_id, pkts in collected.items():
        for pkt in pkts[:3]:
            assert pkt.section_id == section_id, \
                f"section_id mismatch: {pkt.section_id} != {section_id}"
            assert pkt.camera_id,    "camera_id must not be empty"
            assert pkt.frame_no > 0, "frame_no must be > 0"
            assert pkt.timestamp,    "timestamp must not be empty"
            assert pkt.frame.shape == (640, 640, 3), \
                f"Expected preprocessed shape (640,640,3), got {pkt.frame.shape}"
            assert pkt.frame.dtype == np.uint8
    print(f"  ✅ All packets have correct fields and preprocessed shape (640,640,3)")


def test_per_section_queue_isolation(video_dir: str) -> None:
    """Verify per-section queues only receive their own section's frames."""
    video_paths = {}
    for cam in cfg.cameras:
        if not cam.enabled:
            continue
        vpath = os.path.join(video_dir, f"{cam.section_id}_iso.avi")
        make_synthetic_video(vpath, n_frames=120,
                              section_label=cam.section_id[-1].upper())
        video_paths[cam.section_id] = vpath

    patch_config_with_videos(video_paths)
    manager = StreamManager(preprocess=False)
    manager.start()

    for section_id in [cam.section_id for cam in cfg.cameras if cam.enabled]:
        collected = drain_queue_with_timeout(
            manager, section_id=section_id, target_per_section=3
        )
        pkts = collected.get(section_id, [])
        for pkt in pkts:
            assert pkt.section_id == section_id, \
                f"Queue contamination! Got {pkt.section_id} in {section_id} queue"
        print(f"  ✅ [{section_id}] isolation OK — {len(pkts)} frame(s), no contamination")

    manager.stop()


def test_queue_sizes_api(video_dir: str) -> None:
    """Verify queue_sizes() returns a dict with correct section keys."""
    for cam in cfg.cameras:
        if cam.enabled:
            vpath = os.path.join(video_dir, f"{cam.section_id}_qs.avi")
            make_synthetic_video(vpath, n_frames=60)
            cam.stream_url = vpath

    manager = StreamManager(preprocess=False)
    manager.start()
    time.sleep(0.5)
    sizes = manager.queue_sizes()
    manager.stop()

    assert isinstance(sizes, dict)
    for cam in cfg.cameras:
        if cam.enabled:
            assert cam.section_id in sizes, \
                f"{cam.section_id} missing from queue_sizes()"
    print(f"  ✅ queue_sizes() API works: {sizes}")


def test_clean_shutdown(video_dir: str) -> None:
    """Verify stop() completes without hanging."""
    for cam in cfg.cameras:
        if cam.enabled:
            vpath = os.path.join(video_dir, f"{cam.section_id}_shut.avi")
            make_synthetic_video(vpath, n_frames=30)
            cam.stream_url = vpath

    manager = StreamManager(preprocess=False)
    manager.start()
    time.sleep(0.5)

    t0 = time.time()
    manager.stop()
    elapsed = time.time() - t0

    assert elapsed < 10, f"stop() took too long: {elapsed:.1f}s"
    print(f"  ✅ Clean shutdown in {elapsed:.2f}s")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("CP-06 — Multi-Camera Parallel Ingestion — Validation Test")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp(prefix="cp06_")

    try:
        print("\n[1] StreamManager builds correctly:")
        test_stream_manager_builds()

        print("\n[2] Parallel ingestion from all sections:")
        collected = test_parallel_ingestion(tmpdir)

        print("\n[3] Packet fields and preprocessed shape:")
        test_packet_fields(collected)

        print("\n[4] Per-section queue isolation:")
        test_per_section_queue_isolation(tmpdir)

        print("\n[5] Queue sizes API:")
        test_queue_sizes_api(tmpdir)

        print("\n[6] Clean shutdown:")
        test_clean_shutdown(tmpdir)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("\n" + "=" * 60)
    print("🎉  CP-06 PASSED — multi-camera parallel ingestion ready")
    print("=" * 60)