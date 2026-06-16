"""
helpers.py — Shared utility functions used across all modules.

Functions:
    resize_frame          — Resize an OpenCV frame to target dimensions
    get_timestamp         — ISO-8601 timestamp string for DB/logging
    read_json             — Load a JSON file safely
    write_json            — Write a dict to JSON with pretty-printing
    validate_section_id   — Check a section_id exists in config
    is_video_file         — True if a path points to a supported video file
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from config.config_loader import cfg
from config.constants import VIDEO_EXTENSIONS


# ── Frame utilities ────────────────────────────────────────────────────────────

def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Resize an OpenCV BGR frame to (width, height).

    Args:
        frame:  Input frame as a NumPy array (H x W x C).
        width:  Target width in pixels.
        height: Target height in pixels.

    Returns:
        Resized frame. Returns original frame unchanged if already correct size.
    """
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


# ── Timestamp ──────────────────────────────────────────────────────────────────

def get_timestamp() -> str:
    """
    Return the current UTC time as an ISO-8601 string.

    Example output: '2025-08-14T10:32:05.123456+00:00'

    Always uses UTC so timestamps are consistent across timezones.
    """
    return datetime.now(tz=timezone.utc).isoformat()


# ── JSON I/O ───────────────────────────────────────────────────────────────────

def read_json(path: str | Path) -> dict:
    """
    Load and return a JSON file as a Python dict.

    Args:
        path: Path to the .json file.

    Returns:
        Parsed dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: dict, path: str | Path, indent: int = 2) -> None:
    """
    Serialise a dict to a JSON file, creating parent directories if needed.

    Args:
        data:   Dictionary to serialise.
        path:   Destination file path.
        indent: Pretty-print indentation level (default 2).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_section_id(section_id: str) -> bool:
    """
    Return True if section_id matches a camera entry in config.yaml.

    Args:
        section_id: String to validate, e.g. "section_A".

    Returns:
        True if found, False otherwise.
    """
    valid_ids = {cam.section_id for cam in cfg.cameras}
    return section_id in valid_ids


def is_video_file(path: str | Path) -> bool:
    """
    Return True if the path has a recognised video file extension.

    Args:
        path: File path to check.

    Returns:
        True if the extension is in VIDEO_EXTENSIONS.
    """
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS