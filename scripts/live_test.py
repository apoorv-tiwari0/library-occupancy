"""
scripts/live_test.py — Real-time multi-section occupancy test

Ingests all 11 section video files simultaneously via StreamManager,
runs the inference pipeline on each frame, and displays a live
vacancy table in the terminal. No dashboard — pure terminal output.

This validates the full pipeline end-to-end before backend/frontend work.

Usage:
    python scripts/live_test.py

Requirements:
    - All 11 video files in data/test_videos/ named by section_id
      e.g. data/test_videos/cad_lab.mp4
    - config.yaml with all 11 cameras configured
    - data/roi/roi_polygons.json (not used for inference but kept for reference)

Output:
    Live-updating terminal table showing per-section headcount and vacancy.
    Press Ctrl+C to stop.
"""

import sys
import os
import time
import threading
import queue
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import cfg
from detection.pipeline import MultiSectionPipeline, SectionResult
from utils.logger import get_logger

log = get_logger("system")

# ── Config ─────────────────────────────────────────────────────────────────────

VIDEO_DIR   = Path("data/test_videos")   # folder with section_id.mp4 files
SAMPLE_RATE = 5                           # process every Nth frame
DISPLAY_INTERVAL = 2.0                    # redraw terminal table every N seconds

# ── Shared state ───────────────────────────────────────────────────────────────

# Latest result per section — written by worker threads, read by display thread
latest_results: dict[str, SectionResult] = {}
results_lock = threading.Lock()

stop_event = threading.Event()


# ── Worker: one per section ────────────────────────────────────────────────────

def section_worker(
    section_id:   str,
    video_path:   Path,
    pipeline,     # SectionPipeline instance
) -> None:
    """
    Reads frames from a video file, runs pipeline, updates latest_results.
    Loops the video when it ends (simulating a continuous live feed).
    """
    log.info(f"[{section_id}] Worker started | source={video_path.name}")

    while not stop_event.is_set():
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            log.error(f"[{section_id}] Cannot open {video_path}")
            time.sleep(2)
            continue

        frame_no = 0
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                # Video ended — loop back
                log.info(f"[{section_id}] Video ended, looping...")
                break

            frame_no += 1
            if frame_no % SAMPLE_RATE != 0:
                continue

            try:
                result = pipeline.run(section_id, frame)
                with results_lock:
                    latest_results[section_id] = result
            except Exception as e:
                log.error(f"[{section_id}] Pipeline error: {e}")

        cap.release()

    log.info(f"[{section_id}] Worker stopped.")


# ── Display: terminal table ────────────────────────────────────────────────────

def status_bar(headcount: int, max_cap: int, width: int = 20) -> str:
    """ASCII progress bar: filled = occupied, empty = vacant."""
    if not max_cap:
        return "N/A"
    filled = min(max_cap, headcount)
    pct    = filled / max_cap
    n_fill = round(pct * width)
    bar    = "█" * n_fill + "░" * (width - n_fill)
    return f"[{bar}] {headcount}/{max_cap}"


def display_loop() -> None:
    """Redraws the terminal table every DISPLAY_INTERVAL seconds."""
    sections_ordered = [c.section_id for c in cfg.cameras if c.enabled]

    while not stop_event.is_set():
        time.sleep(DISPLAY_INTERVAL)

        # Clear terminal
        os.system("cls" if os.name == "nt" else "clear")

        print("=" * 72)
        print(f"  LIBRARY OCCUPANCY — LIVE ({time.strftime('%H:%M:%S')})")
        print("=" * 72)
        print(f"  {'SECTION':<25} {'STATUS':<12} {'BAR':<26} {'VACANCY'}")
        print("-" * 72)

        with results_lock:
            snapshot = dict(latest_results)

        total_vacancy = 0
        total_capacity = 0

        for sid in sections_ordered:
            if sid not in snapshot:
                print(f"  {sid:<25} {'LOADING...':<12}")
                continue

            r    = snapshot[sid]
            bar  = status_bar(r.headcount, r.max_capacity)
            pct  = r.occupancy_pct() or 0.0
            vac  = r.vacancy if r.vacancy is not None else "N/A"

            if pct >= 90:
                status = "FULL"
            elif pct >= 60:
                status = "BUSY"
            elif pct >= 20:
                status = "MODERATE"
            else:
                status = "AVAILABLE"

            print(f"  {sid:<25} {status:<12} {bar:<26} {vac}")

            if r.max_capacity:
                total_capacity += r.max_capacity
            if r.vacancy is not None:
                total_vacancy += r.vacancy

        print("-" * 72)
        print(f"  {'TOTAL LIBRARY':<25} {'':<12} {'':<26} "
              f"{total_vacancy}/{total_capacity} vacant")
        print("=" * 72)
        print("\n  Press Ctrl+C to stop.\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 55)
    print("Live Multi-Section Occupancy Test")
    print("=" * 55)

    # 1. Find video files
    print("\nLooking for video files in:", VIDEO_DIR)
    video_map: dict[str, Path] = {}
    for cam in cfg.cameras:
        if not cam.enabled:
            continue
        sid = cam.section_id
        # Accept .mp4, .avi, .mov
        for ext in [".mp4", ".avi", ".mov", ".MP4", ".AVI"]:
            vpath = VIDEO_DIR / f"{sid}{ext}"
            if vpath.exists():
                video_map[sid] = vpath
                print(f"  ✅ {sid:<30} → {vpath.name}")
                break
        else:
            print(f"  ⚠️  {sid:<30} → NO VIDEO FILE FOUND (skipping)")

    if not video_map:
        print("\n❌ No video files found. Add videos to:", VIDEO_DIR)
        print("   Files should be named: cad_lab.mp4, g_huss.mp4, etc.")
        return

    print(f"\nFound {len(video_map)}/11 section videos.")

    # 2. Build pipeline (loads YOLO once)
    print("\nLoading pipeline (YOLO model)...")
    pipeline = MultiSectionPipeline()

    # 3. Start one worker thread per section that has a video
    print(f"\nStarting {len(video_map)} worker threads...")
    workers = []
    for sid, vpath in video_map.items():
        if sid not in pipeline.sections():
            log.warning(f"No pipeline for '{sid}' — skipping")
            continue
        t = threading.Thread(
            target = section_worker,
            args   = (sid, vpath, pipeline),
            name   = f"worker-{sid}",
            daemon = True,
        )
        t.start()
        workers.append(t)

    # 4. Start display thread
    display_t = threading.Thread(
        target=display_loop, name="display", daemon=True
    )
    display_t.start()

    print(f"\n✅ Running — {len(workers)} sections active. Press Ctrl+C to stop.\n")

    # 5. Wait for Ctrl+C
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\nStopping...")
        stop_event.set()

    # 6. Wait for workers
    for t in workers:
        t.join(timeout=3)

    print("Stopped cleanly.")


if __name__ == "__main__":
    main()