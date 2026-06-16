"""
constants.py — Immutable project-wide literals.

These never change at runtime. If it can vary by environment or user config,
it belongs in config.yaml instead. Only truly fixed values live here.
"""

from enum import Enum


# ── Seat occupancy states ──────────────────────────────────────────────────────

class SeatStatus(str, Enum):
    FREE     = "free"
    OCCUPIED = "occupied"
    RESERVED = "reserved"


# ── YOLO COCO class IDs used in this project ──────────────────────────────────
# Full COCO list: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml

class CocoClass(int, Enum):
    PERSON      = 0
    BACKPACK    = 24
    HANDBAG     = 26
    SUITCASE    = 28
    BOTTLE      = 39
    LAPTOP      = 63
    BOOK        = 73

# Convenience set — all object classes that indicate seat reservation
RESERVATION_CLASSES: frozenset[int] = frozenset({
    CocoClass.BACKPACK,
    CocoClass.HANDBAG,
    CocoClass.SUITCASE,
    CocoClass.BOTTLE,
    CocoClass.LAPTOP,
    CocoClass.BOOK,
})


# ── Redis key schema ───────────────────────────────────────────────────────────
# Key format: "occupancy:{section_id}:{seat_id}"
# Example:    "occupancy:section_A:seat_A1"

REDIS_KEY_PREFIX = "occupancy"


# ── Database table name ────────────────────────────────────────────────────────

DB_OCCUPANCY_TABLE = "seat_occupancy"


# ── Supported video file extensions (for local file testing) ──────────────────

VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".avi", ".mkv", ".mov"})