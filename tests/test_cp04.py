"""
test_cp04.py — Runs the CP-04 self-test for the frame extractor.

Place this file at the project root (same level as config/) and run:
    python test_cp04.py
"""

import sys
from pathlib import Path

# Make sure project root is on the path so relative imports resolve
sys.path.insert(0, str(Path(__file__).parent))

from ingestion.frame_extractor import _run_cp04_test

if __name__ == "__main__":
    _run_cp04_test()