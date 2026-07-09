"""
detection/zone_detector.py — Zone-based detection (ZCP-03/04/05)

Splits a frame into near/mid/far zones and runs the appropriate
detector on each zone crop:

    NEAR — SAHI+YOLOv10 (existing YOLOInference)
    MID  — SAHI+YOLOv10 + ByteTrack (ZCP-05)
    FAR  — DM-Count density estimator (ZCP-08)

This module implements the NEAR zone detector (ZCP-03).
MID and FAR are stubs until their respective checkpoints.

Coordinate mapping:
    All detections are returned in FULL-FRAME coordinates regardless
    of which zone they were detected in. The zone crop offset is added
    back after detection.

Usage:
    from detection.zone_detector import NearZoneDetector
    detector = NearZoneDetector(yolo, zone_config["cad_lab"])
    persons, objects = detector.detect(frame)
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from detection.yolo_inference import YOLOInference
from utils.logger import get_logger

log = get_logger("system")


def get_zone_crop(
    frame:    np.ndarray,
    polygon:  list[list[int]],
) -> tuple[np.ndarray, int, int, int, int]:
    """
    Crop the frame to the bounding box of a zone polygon.

    Returns:
        crop:  Cropped BGR frame
        x1,y1: Top-left of crop in full-frame coords
        x2,y2: Bottom-right of crop in full-frame coords
    """
    h, w   = frame.shape[:2]
    pts    = np.array(polygon, dtype=np.int32)
    rx, ry, rw, rh = cv2.boundingRect(pts)

    # Clamp to frame bounds
    x1 = max(0, rx)
    y1 = max(0, ry)
    x2 = min(w, rx + rw)
    y2 = min(h, ry + rh)

    crop = frame[y1:y2, x1:x2]
    return crop, x1, y1, x2, y2


def map_boxes_to_frame(
    boxes:  list[list[float]],
    x1_off: int,
    y1_off: int,
    has_class: bool = False,
) -> list[list[float]]:
    """
    Map bounding boxes from crop-local coords to full-frame coords.

    Args:
        boxes:     List of [x1,y1,x2,y2,conf] or [x1,y1,x2,y2,conf,cls]
        x1_off:    X offset of crop in full frame
        y1_off:    Y offset of crop in full frame
        has_class: True if boxes have a class_id at index 5

    Returns:
        Boxes with coordinates shifted to full-frame space.
    """
    mapped = []
    for box in boxes:
        new_box = [
            box[0] + x1_off,
            box[1] + y1_off,
            box[2] + x1_off,
            box[3] + y1_off,
            box[4],
        ]
        if has_class:
            new_box.append(box[5])
        mapped.append(new_box)
    return mapped


def filter_by_polygon(
    boxes:   list[list[float]],
    polygon: list[list[int]],
) -> list[list[float]]:
    """
    Keep only detections whose centroid falls inside the zone polygon.
    Removes detections in the bounding-box crop that are outside the
    actual polygon (the crop is rectangular but the polygon may not be).
    """
    if not polygon:
        return boxes
    poly = np.array(polygon, dtype=np.float32)
    kept = []
    for box in boxes:
        cx = float((box[0] + box[2]) / 2)
        cy = float((box[1] + box[3]) / 2)
        if cv2.pointPolygonTest(poly, (cx, cy), measureDist=False) >= 0:
            kept.append(box)
    return kept


class NearZoneDetector:
    """
    Detects people in the NEAR zone using SAHI+YOLOv10.

    The near zone contains the closest seats to the camera where
    people are largest in frame — SAHI+YOLO works best here.

    Args:
        yolo:        Shared YOLOInference instance.
        zone_config: Zone polygons for this section
                     {"near": [[x,y],...], "mid": ..., "far": ...}
        section_id:  Section identifier for SAHI slice config lookup.
    """

    def __init__(
        self,
        yolo:        YOLOInference,
        zone_config: dict[str, list[list[int]]],
        section_id:  str,
    ) -> None:
        self.yolo        = yolo
        self.zone_config = zone_config
        self.section_id  = section_id
        self.near_poly   = zone_config.get("near", [])

        log.info(
            f"NearZoneDetector | section={section_id} "
            f"near_polygon_vertices={len(self.near_poly)}"
        )

    def detect(
        self,
        frame: np.ndarray,
    ) -> tuple[list[list[float]], list[list[float]]]:
        """
        Detect persons and objects in the near zone.

        Args:
            frame: Full BGR frame (any resolution).

        Returns:
            person_boxes: [[x1,y1,x2,y2,conf], ...] in full-frame coords
            object_boxes: [[x1,y1,x2,y2,conf,cls], ...] in full-frame coords
        """
        if not self.near_poly:
            log.warning(f"[{self.section_id}] Near zone polygon not configured")
            return [], []

        # Crop to near zone bounding box
        crop, x1, y1, x2, y2 = get_zone_crop(frame, self.near_poly)

        if crop.size == 0:
            log.warning(f"[{self.section_id}] Near zone crop is empty")
            return [], []

        # Run SAHI+YOLO on the crop
        persons_crop, objects_crop = self.yolo.run_inference(
            crop, section_id=self.section_id
        )

        # Map back to full-frame coordinates
        persons = map_boxes_to_frame(persons_crop, x1, y1, has_class=False)
        objects = map_boxes_to_frame(objects_crop, x1, y1, has_class=True)

        # Filter: keep only detections whose centroid is inside the polygon
        # (crop is rectangular but polygon may not be)
        persons = filter_by_polygon(persons, self.near_poly)
        objects = filter_by_polygon(objects, self.near_poly)

        log.info(
            f"[{self.section_id}] Near zone | "
            f"persons={len(persons)} objects={len(objects)}"
        )

        return persons, objects

class MidZoneDetector:
    """
    Detects and tracks people in the MID zone using SAHI+YOLO + ByteTrack.

    The mid zone contains medium-distance seats where people are medium-sized.
    ByteTrack assigns persistent track IDs across frames, which:
      - Prevents double-counting people who briefly leave/re-enter
      - Provides dwell time data per person
      - Smooths detection flickering better than majority vote alone

    Args:
        yolo:        Shared YOLOInference instance.
        zone_config: Zone polygons {"near":..., "mid":..., "far":...}
        section_id:  Section identifier.
    """

    def __init__(
        self,
        yolo:        YOLOInference,
        zone_config: dict[str, list[list[int]]],
        section_id:  str,
        fps:         float = 10.0,
    ) -> None:
        from tracking.byte_track import ByteTracker

        self.yolo        = yolo
        self.zone_config = zone_config
        self.section_id  = section_id
        self.mid_poly    = zone_config.get("mid", [])
        self._fps        = fps
        self._tracker    = ByteTracker(
            track_thresh = 0.45,
            track_buffer = 30,
            match_thresh = 0.8,
            frame_rate   = fps,
        )
        self._active_tracks: dict[int, list[float]] = {}

        log.info(
            f"MidZoneDetector | section={section_id} "
            f"mid_polygon_vertices={len(self.mid_poly)}"
        )

    def reset(self) -> None:
        from tracking.byte_track import ByteTracker
        self._tracker = ByteTracker(
            track_thresh = 0.45,
            track_buffer = 30,
            match_thresh = 0.8,
            frame_rate   = self._fps,
        )
        self._active_tracks = {}
        log.info(f"[{self.section_id}] MidZoneDetector reset.")

    def detect(
        self,
        frame:    np.ndarray,
        frame_id: int = 0,
    ) -> tuple[list[list[float]], list[int]]:
        if not self.mid_poly:
            log.warning(f"[{self.section_id}] Mid zone polygon not configured")
            return [], []

        crop, x1, y1, x2, y2 = get_zone_crop(frame, self.mid_poly)
        if crop.size == 0:
            return [], []

        persons_crop, _ = self.yolo.run_inference(
            crop, section_id=self.section_id
        )
        persons_full = map_boxes_to_frame(persons_crop, x1, y1)
        persons_full = filter_by_polygon(persons_full, self.mid_poly)

        if not persons_full:
            self._active_tracks = {}
            return [], []

        # boxmot expects [x1,y1,x2,y2,conf,cls] as float32 numpy array
        dets = np.array([
            [b[0], b[1], b[2], b[3], b[4], 0.0]
            for b in persons_full
        ], dtype=np.float32)

        try:
            # boxmot ByteTrack update returns Nx8: [x1,y1,x2,y2,id,conf,cls,idx]
            tracks = self._tracker.update(dets, frame)
        except Exception as e:
            log.warning(f"[{self.section_id}] ByteTrack error: {e} — raw detections")
            return persons_full, list(range(len(persons_full)))

        person_boxes = []
        track_ids    = []

        for track in tracks:
            try:
                bx1, by1, bx2, by2 = float(track[0]), float(track[1]), float(track[2]), float(track[3])
                tid  = int(track[4])
                conf = float(track[5]) if len(track) > 5 else 0.8
                person_boxes.append([bx1, by1, bx2, by2, conf])
                track_ids.append(tid)
            except (IndexError, ValueError):
                continue

        self._active_tracks = {tid: box for tid, box in zip(track_ids, person_boxes)}

        log.info(
            f"[{self.section_id}] Mid zone | "
            f"detections={len(persons_full)} "
            f"tracked={len(person_boxes)} "
            f"active_ids={track_ids}"
        )
        return person_boxes, track_ids

    

    def active_track_count(self) -> int:
        """Number of currently active tracks."""
        return len(self._active_tracks)

class FarZoneDetector:
    """
    Estimates people count in the FAR zone using density estimation.

    People are small/distant in the far zone — individual detection
    fails reliably. Density estimation (CSRNet/DM-Count) treats the
    zone as a crowd and estimates total count from learned density maps.

    Args:
        density_estimator: DensityEstimator instance.
        zone_config:       Zone polygons for this section.
        section_id:        Section identifier.
    """

    def __init__(
        self,
        density_estimator,
        zone_config: dict[str, list[list[int]]],
        section_id:  str,
    ) -> None:
        self.estimator   = density_estimator
        self.zone_config = zone_config
        self.section_id  = section_id
        self.far_poly    = zone_config.get("far", [])

        log.info(
            f"FarZoneDetector | section={section_id} "
            f"far_polygon_vertices={len(self.far_poly)}"
        )

    def detect(self, frame: np.ndarray) -> float:
        """
        Estimate people count in the far zone.

        Args:
            frame: Full BGR frame.

        Returns:
            Estimated people count (float). Use round() for integer.
        """
        if not self.far_poly:
            log.warning(f"[{self.section_id}] Far zone polygon not configured")
            return 0.0

        crop, x1, y1, x2, y2 = get_zone_crop(frame, self.far_poly)
        if crop.size == 0:
            return 0.0

        count = self.estimator.estimate(crop)

        log.info(
            f"[{self.section_id}] Far zone | "
            f"estimated={count:.2f} → {round(count)}"
        )

        return count