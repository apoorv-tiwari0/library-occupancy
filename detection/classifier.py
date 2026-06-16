"""
detection/classifier.py — Seat occupancy classifier (CP-12)

Takes person_boxes and object_boxes from YOLOInference and the seat ROI
polygons for a section, then classifies each seat as:

    FREE       — no person, no reservation object detected
    OCCUPIED   — a person's centroid or bounding box overlaps the seat ROI
    RESERVED   — no person but a reservation object (bag/laptop/book) overlaps

Classification logic:
    1. For each seat polygon, check all person detections:
       - If any person centroid falls inside the polygon → OCCUPIED
       - If person box overlaps polygon bbox with IoU > threshold → OCCUPIED
    2. If not OCCUPIED, check reservation objects the same way → RESERVED
    3. Otherwise → FREE

Headcount:
    Total persons detected in the section (from YOLO) is also returned
    as a raw headcount for sections where max_capacity is configured.
    Vacancy = max_capacity - headcount.

Usage:
    from detection.classifier import SeatClassifier, SeatState
    clf = SeatClassifier()
    states, headcount = clf.classify(
        seat_polygons = roi_data["cad_lab"],
        person_boxes  = persons,
        object_boxes  = objects,
    )
    # states = {"seat_LAB1": "occupied", "seat_LAB2": "free", ...}
    # headcount = 7
"""

import sys
from pathlib import Path
from enum import Enum

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import cfg
from utils.logger import get_logger

log = get_logger("system")


class SeatState(str, Enum):
    FREE      = "free"
    OCCUPIED  = "occupied"
    RESERVED  = "reserved"


class SeatClassifier:
    """
    Classifies each seat ROI as FREE, OCCUPIED, or RESERVED.

    Args:
        iou_threshold: Minimum IoU between a detection bbox and seat bbox
                       to count as an overlap. Default from config.yaml.
        use_centroid:  If True (default), also check if person centroid
                       falls inside the seat polygon — catches cases where
                       IoU is low but person is clearly sitting at the seat.
    """

    def __init__(
        self,
        iou_threshold: float | None = None,
        use_centroid:  bool = True,
    ) -> None:
        self.iou_threshold = iou_threshold or cfg.classifier.iou_threshold
        self.use_centroid  = use_centroid

        log.info(
            f"SeatClassifier ready | "
            f"iou_threshold={self.iou_threshold} "
            f"use_centroid={self.use_centroid}"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def classify(
        self,
        seat_polygons: dict[str, list[list[int]]],
        person_boxes:  list[list[float]],
        object_boxes:  list[list[float]],
    ) -> tuple[dict[str, str], int]:
        """
        Classify every seat in the section.

        Args:
            seat_polygons: {seat_id: [[x,y], ...]} from roi_polygons.json
            person_boxes:  [[x1,y1,x2,y2,conf], ...] from YOLOInference
            object_boxes:  [[x1,y1,x2,y2,conf,cls], ...] from YOLOInference

        Returns:
            states:    {seat_id: "free"|"occupied"|"reserved"}
            headcount: total number of persons detected in this section
        """
        headcount = len(person_boxes)
        states    = {}

        for seat_id, pts in seat_polygons.items():
            poly     = np.array(pts, dtype=np.float32)
            seat_box = _poly_bbox(poly)

            # Signal 1 — person overlap
            occupied = self._any_overlap(
                poly, seat_box, person_boxes, has_class=False
            )

            if occupied:
                states[seat_id] = SeatState.OCCUPIED
                continue

            # Signal 2 — reservation object overlap
            reserved = self._any_overlap(
                poly, seat_box, object_boxes, has_class=True
            )

            states[seat_id] = SeatState.RESERVED if reserved else SeatState.FREE

        counts = _count_states(states)
        log.info(
            f"Classified {len(states)} seats | "
            f"occupied={counts[SeatState.OCCUPIED]} "
            f"reserved={counts[SeatState.RESERVED]} "
            f"free={counts[SeatState.FREE]} "
            f"headcount={headcount}"
        )

        return states, headcount

    def classify_with_capacity(
        self,
        seat_polygons: dict[str, list[list[int]]],
        person_boxes:  list[list[float]],
        object_boxes:  list[list[float]],
        max_capacity:  int | None = None,
    ) -> dict:
        """
        classify() plus vacancy calculation using max_capacity.

        If max_capacity is provided, vacancy = max_capacity - headcount.
        This is the headcount-based approach for large open sections
        where individual seat ROIs may not cover every chair.

        Returns a result dict with all occupancy data for this section.
        """
        states, headcount = self.classify(
            seat_polygons, person_boxes, object_boxes
        )
        counts = _count_states(states)

        vacancy = None
        if max_capacity is not None and max_capacity > 0:
            vacancy = max(0, max_capacity - headcount)

        return {
            "seat_states":  states,
            "headcount":    headcount,
            "max_capacity": max_capacity,
            "vacancy":      vacancy,
            "counts": {
                "occupied": counts[SeatState.OCCUPIED],
                "reserved": counts[SeatState.RESERVED],
                "free":     counts[SeatState.FREE],
                "total":    len(states),
            }
        }

    # ── Overlap detection ──────────────────────────────────────────────────────

    def _any_overlap(
        self,
        poly:      np.ndarray,
        seat_box:  list[float],
        boxes:     list[list[float]],
        has_class: bool,
    ) -> bool:
        """
        Return True if ANY detection in boxes overlaps with the seat polygon.

        Two-stage check (fast bbox first, then precise polygon):
          1. IoU between detection bbox and seat bounding box > threshold
          2. OR detection centroid falls inside the seat polygon
        """
        for box in boxes:
            det_box = box[:4]

            # Stage 1 — fast bbox IoU check
            if _box_iou(det_box, seat_box) >= self.iou_threshold:
                return True

            # Stage 2 — centroid inside polygon
            if self.use_centroid:
                cx = float((box[0] + box[2]) / 2)
                cy = float((box[1] + box[3]) / 2)
                if cv2.pointPolygonTest(poly, (cx, cy), measureDist=False) >= 0:
                    return True

        return False

    # ── Visualisation ──────────────────────────────────────────────────────────

    def draw_states(
        self,
        frame:         np.ndarray,
        seat_polygons: dict[str, list[list[int]]],
        states:        dict[str, str],
        headcount:     int,
        max_capacity:  int | None = None,
    ) -> np.ndarray:
        """
        Draw seat polygons color-coded by state on a copy of the frame.

        Green  = FREE
        Red    = OCCUPIED
        Yellow = RESERVED
        """
        COLOR = {
            SeatState.FREE:     (0,   255,  0),    # green
            SeatState.OCCUPIED: (0,   0,    255),   # red
            SeatState.RESERVED: (0,   215,  255),   # yellow
        }
        ALPHA = 0.35   # polygon fill transparency

        vis = frame.copy()
        overlay = frame.copy()

        for seat_id, pts in seat_polygons.items():
            state  = states.get(seat_id, SeatState.FREE)
            color  = COLOR[state]
            poly   = np.array(pts, dtype=np.int32)

            # Filled polygon on overlay
            cv2.fillPoly(overlay, [poly], color)
            # Outline on vis
            cv2.polylines(vis, [poly], isClosed=True, color=color, thickness=2)

            # Seat ID label at centroid
            cx = int(sum(p[0] for p in pts) / len(pts))
            cy = int(sum(p[1] for p in pts) / len(pts))
            cv2.putText(vis, seat_id, (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Blend overlay
        cv2.addWeighted(overlay, ALPHA, vis, 1 - ALPHA, 0, vis)

        # HUD
        counts  = _count_states(states)
        vacancy = max(0, max_capacity - headcount) if max_capacity else "N/A"
        hud = (
            f"headcount={headcount}  "
            f"occupied={counts[SeatState.OCCUPIED]}  "
            f"reserved={counts[SeatState.RESERVED]}  "
            f"free={counts[SeatState.FREE]}  "
            f"vacancy={vacancy}"
        )
        cv2.rectangle(vis, (0, 0), (len(hud) * 9 + 10, 28), (0, 0, 0), -1)
        cv2.putText(vis, hud, (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        return vis


# ── Helpers ────────────────────────────────────────────────────────────────────

def _poly_bbox(poly: np.ndarray) -> list[float]:
    """Return [x1, y1, x2, y2] bounding box of a polygon."""
    x1 = float(poly[:, 0].min())
    y1 = float(poly[:, 1].min())
    x2 = float(poly[:, 0].max())
    y2 = float(poly[:, 1].max())
    return [x1, y1, x2, y2]


def _box_iou(a: list[float], b: list[float]) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _count_states(states: dict[str, str]) -> dict:
    """Count seats per state."""
    counts = {
        SeatState.OCCUPIED: 0,
        SeatState.RESERVED: 0,
        SeatState.FREE:     0,
    }
    for s in states.values():
        counts[SeatState(s)] += 1
    return counts