"""CP-03 verification test — Logging & Utilities."""

import shutil
import numpy as np
from pathlib import Path

from utils.logger import get_logger
from utils.helpers import (
    resize_frame,
    get_timestamp,
    read_json,
    write_json,
    validate_section_id,
    is_video_file,
)


def test_logger():
    log = get_logger("system")
    log.debug("debug message — only visible if log_level is DEBUG")
    log.info("CP-03 logger test — INFO")
    log.warning("CP-03 logger test — WARNING")
    log.error("CP-03 logger test — ERROR")

    # Same name must return the same cached instance
    log2 = get_logger("system")
    assert log is log2, "Logger cache broken — returned different instance"

    # Section-specific log
    sec_log = get_logger("section_A")
    sec_log.info("Section A logger working")

    # Log files must have been created
    assert Path("logs/system.log").exists(),    "logs/system.log not created"
    assert Path("logs/section_A.log").exists(), "logs/section_A.log not created"

    print(" Logger — log files created, caching works")


def test_resize_frame():
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)  # 640x480 BGR frame

    resized = resize_frame(dummy, 640, 640)
    assert resized.shape == (640, 640, 3), f"Unexpected shape: {resized.shape}"

    # Already correct size — should return unchanged
    same = resize_frame(dummy, 640, 480)
    assert same is dummy, "resize_frame should return original when size matches"

    print(" resize_frame — correct output shape")


def test_timestamp():
    ts = get_timestamp()
    assert "T" in ts and "+" in ts, f"Unexpected timestamp format: {ts}"
    print(f" get_timestamp — {ts}")


def test_json_io():
    test_path = Path("data/test_temp/test_output.json")
    payload = {"section_A": {"seat_A1": [[0, 0], [100, 0], [100, 100]]}}

    write_json(payload, test_path)
    assert test_path.exists(), "write_json did not create file"

    loaded = read_json(test_path)
    assert loaded == payload, "read_json returned different data"

    # Cleanup
    shutil.rmtree(test_path.parent)
    print(" write_json / read_json — round-trip verified")


def test_validate_section_id():
    assert validate_section_id("section_A") is True
    assert validate_section_id("section_B") is True
    assert validate_section_id("section_FAKE") is False
    print(" validate_section_id — valid/invalid cases pass")


def test_is_video_file():
    assert is_video_file("footage.mp4")  is True
    assert is_video_file("clip.avi")     is True
    assert is_video_file("image.jpg")    is False
    assert is_video_file("notes.txt")    is False
    print(" is_video_file — extension checks pass")


if __name__ == "__main__":
    test_logger()
    test_resize_frame()
    test_timestamp()
    test_json_io()
    test_validate_section_id()
    test_is_video_file()
    print("\n CP-03 PASSED — logging & utilities ready")