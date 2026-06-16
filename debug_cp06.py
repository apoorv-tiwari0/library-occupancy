"""
debug_cp06.py — Pinpoint where frames are being lost in CP-06 pipeline.
Run: python debug_cp06.py
"""
import os, sys, time, queue, tempfile, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from config.config_loader import cfg
from ingestion.frame_extractor import FrameExtractor, FramePacket

def make_video(path, n_frames=30):
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(path, fourcc, 30.0, (640, 480))
    rng = np.random.default_rng(seed=0)
    for i in range(n_frames):
        frame = rng.integers(40, 200, (480, 640, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

tmpdir = tempfile.mkdtemp(prefix="dbg_")
vpath = os.path.join(tmpdir, "test.avi")
make_video(vpath, n_frames=30)
print(f"Synthetic video: {vpath}")

# Step 1 — test raw extractor queue directly
print("\n--- Step 1: Raw extractor queue ---")
raw_q = queue.Queue(maxsize=50)
cam = cfg.cameras[0]
cam.stream_url = vpath

ext = FrameExtractor(
    camera_id=cam.camera_id,
    section_id=cam.section_id,
    stream_url=vpath,
    output_queue=raw_q,
    sample_rate=1,   # sample EVERY frame so we get max packets
)
ext.start()
time.sleep(3)
ext.stop()

raw_count = raw_q.qsize()
print(f"Raw queue size after extraction: {raw_count}")

if raw_count == 0:
    print("❌ PROBLEM: Frames not reaching raw queue at all!")
    print("   Check FrameExtractor.run() — the extractor thread may be failing silently.")
else:
    print(f"✅ {raw_count} frames in raw queue")

# Step 2 — test preprocessor directly on one frame
print("\n--- Step 2: Preprocessor on one frame ---")
if raw_count > 0:
    pkt = raw_q.get_nowait()
    print(f"   Raw frame shape: {pkt.frame.shape}, dtype: {pkt.frame.dtype}")
    try:
        from ingestion.preprocessor import Preprocessor
        pre = Preprocessor()
        out = pre.process(pkt.frame)
        print(f"   Preprocessed shape: {out.shape}, dtype: {out.dtype}")
        print("   ✅ Preprocessor works on raw frame")
    except Exception as e:
        print(f"   ❌ Preprocessor failed: {e}")

# Step 3 — test StreamManager with sample_rate=1 and direct queue inspection
print("\n--- Step 3: StreamManager internal queue inspection ---")
from ingestion.stream_manager import StreamManager
cam.stream_url = vpath
# Also patch second camera if present
if len(cfg.cameras) > 1:
    cfg.cameras[1].stream_url = vpath

manager = StreamManager(preprocess=True)
manager.start()
time.sleep(5)  # give plenty of time

# Check internal queue sizes BEFORE stopping
print(f"   Raw queue sizes:     { {k: v.qsize() for k,v in manager._raw_queues.items()} }")
print(f"   Section queue sizes: { {k: v.qsize() for k,v in manager._section_queues.items()} }")
print(f"   Merged queue size:   {manager._merged_queue.qsize()}")

manager.stop()

shutil.rmtree(tmpdir, ignore_errors=True)
print("\nDone.")