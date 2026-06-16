"""
detection/pipeline.py — Per-section inference pipeline (CP-14)

Wires together the full detection stack for one library section:

    FramePacket (from StreamManager)
        ↓
    Preprocessor        — gamma → bilateral → CLAHE → resize
        ↓
    YOLOInference       — SAHI person detection + object detection
        ↓
    SeatClassifier      — FREE / OCCUPIED / RESERVED per seat + headcount
        ↓
    SectionResult       — single dataclass with everything downstream needs

One SectionPipeline instance is created per section and reused across
frames. The preprocessor and YOLO model are shared across sections via
dependency injection to avoid loading the model N times.

Usage:
    from detection.pipeline import SectionPipeline, SectionResult

    yolo = YOLOInference()
    pipeline = SectionPipeline(
        section_id   = "cad_lab",
        seat_polygons = roi_data["cad_lab"],
        max_capacity  = 24,
        yolo         = yolo,
    )
    result = pipeline.run(frame)
    print(result.headcount, result.vacancy, result.seat_states)
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import cfg
from detection.classifier import SeatClassifier, SeatState
from detection.yolo_inference import YOLOInference
from ingestion.preprocessor import Preprocessor
from utils.helpers import get_timestamp
from utils.logger import get_logger

log = get_logger("system")


@dataclass
class SectionResult:
    """
    All occupancy data produced by one pipeline pass on one frame.

    Attributes:
        section_id:   Which section this result is for.
        timestamp:    ISO timestamp of the frame processed.
        headcount:    Raw person count from YOLO.
        max_capacity: Configured max capacity for this section.
        vacancy:      max_capacity - headcount (clamped to 0).
        seat_states:  Per-seat classification {seat_id: "free"|"occupied"|"reserved"}.
        counts:       Aggregated counts {"occupied": N, "reserved": N, "free": N}.
        person_boxes: Raw YOLO person detections [[x1,y1,x2,y2,conf], ...].
        object_boxes: Raw YOLO object detections [[x1,y1,x2,y2,conf,cls], ...].
        inference_ms: Time taken for YOLO inference in milliseconds.
        pipeline_ms:  Total pipeline time in milliseconds.
    """
    section_id:   str
    timestamp:    str
    headcount:    int
    max_capacity: int | None
    vacancy:      int | None
    seat_states:  dict[str, str]
    counts:       dict[str, int]
    person_boxes: list[list[float]]
    object_boxes: list[list[float]]
    inference_ms: float
    pipeline_ms:  float

    def is_available(self) -> bool:
        """True if vacancy > 0 or any seat is FREE."""
        if self.vacancy is not None:
            return self.vacancy > 0
        return self.counts.get("free", 0) > 0

    def occupancy_pct(self) -> float | None:
        """Occupancy as a percentage of max_capacity. None if not configured."""
        if not self.max_capacity:
            return None
        return round(min(100.0, self.headcount / self.max_capacity * 100), 1)

    def to_dict(self) -> dict:
        """Serialisable dict for Redis / API response."""
        return {
            "section_id":    self.section_id,
            "timestamp":     self.timestamp,
            "headcount":     self.headcount,
            "max_capacity":  self.max_capacity,
            "vacancy":       self.vacancy,
            "occupancy_pct": self.occupancy_pct(),
            "is_available":  self.is_available(),
            "seat_states":   self.seat_states,
            "counts":        self.counts,
            "inference_ms":  round(self.inference_ms, 1),
            "pipeline_ms":   round(self.pipeline_ms, 1),
        }


class SectionPipeline:
    """
    Full inference pipeline for one library section.

    Instantiate once per section. Call run(frame) on every processed frame.

    Args:
        section_id:    Section identifier (must match config.yaml and ROI file).
        seat_polygons: {seat_id: [[x,y],...]} from roi_polygons.json.
        max_capacity:  Maximum people this section can hold. None = not configured.
        yolo:          Shared YOLOInference instance (model loaded once).
        preprocessor:  Optional shared Preprocessor. Created if not provided.
        iou_threshold: Passed to SeatClassifier. Defaults to config value.
    """

    def __init__(
        self,
        section_id:    str,
        seat_polygons: dict[str, list[list[int]]],
        max_capacity:  int | None,
        yolo:          YOLOInference,
        preprocessor:  Preprocessor | None = None,
        iou_threshold: float | None = None,
    ) -> None:
        self.section_id    = section_id
        self.seat_polygons = seat_polygons
        self.max_capacity  = max_capacity
        self.yolo          = yolo
        self.preprocessor  = preprocessor or Preprocessor()
        self.classifier    = SeatClassifier(iou_threshold=iou_threshold)

        log.info(
            f"SectionPipeline ready | section={section_id} "
            f"seats={len(seat_polygons)} max_capacity={max_capacity}"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, frame: np.ndarray) -> SectionResult:
        """
        Run the full pipeline on a single raw BGR frame.

        Steps:
            1. Preprocess (gamma → bilateral → CLAHE → resize)
            2. YOLO inference (persons + objects)
            3. Seat classification (FREE/OCCUPIED/RESERVED + headcount)
            4. Pack into SectionResult

        Args:
            frame: Raw BGR numpy array from the camera.

        Returns:
            SectionResult with all occupancy data.
        """
        t_start = time.perf_counter()
        ts      = get_timestamp()

        # Step 1 — preprocess
        processed = self.preprocessor.process(frame)

        # Step 2 — YOLO inference
        t_infer = time.perf_counter()
        persons, objects = self.yolo.run_inference(
            processed, section_id=self.section_id
        )
        inference_ms = (time.perf_counter() - t_infer) * 1000

        # Step 3 — classify seats
        result = self.classifier.classify_with_capacity(
            seat_polygons = self.seat_polygons,
            person_boxes  = persons,
            object_boxes  = objects,
            max_capacity  = self.max_capacity,
        )

        pipeline_ms = (time.perf_counter() - t_start) * 1000

        section_result = SectionResult(
            section_id   = self.section_id,
            timestamp    = ts,
            headcount    = result["headcount"],
            max_capacity = result["max_capacity"],
            vacancy      = result["vacancy"],
            seat_states  = {k: v.value if hasattr(v, "value") else v
                           for k, v in result["seat_states"].items()},
            counts       = result["counts"],
            person_boxes = persons,
            object_boxes = objects,
            inference_ms = inference_ms,
            pipeline_ms  = pipeline_ms,
        )

        log.info(
            f"[{self.section_id}] Pipeline | "
            f"headcount={section_result.headcount} "
            f"vacancy={section_result.vacancy} "
            f"occupied={section_result.counts['occupied']} "
            f"free={section_result.counts['free']} "
            f"pipeline={pipeline_ms:.0f}ms "
            f"inference={inference_ms:.0f}ms"
        )

        return section_result

    def run_with_visualization(
        self,
        frame: np.ndarray,
    ) -> tuple["SectionResult", np.ndarray]:
        """
        Same as run() but also returns an annotated frame showing
        seat states (color-coded polygons) + person bounding boxes.
        Useful for debugging and dashboard preview.
        """
        result = self.run(frame)

        # Draw seat states on the ORIGINAL (not preprocessed) frame
        vis = self.classifier.draw_states(
            frame        = frame,
            seat_polygons = self.seat_polygons,
            states       = result.seat_states,
            headcount    = result.headcount,
            max_capacity = result.max_capacity,
        )

        # Also draw person boxes
        import cv2
        for box in result.person_boxes:
            x1, y1, x2, y2, conf = box
            # Scale boxes back from 640x640 to original frame size
            h_orig, w_orig = frame.shape[:2]
            sx = w_orig / 640
            sy = h_orig / 640
            cv2.rectangle(vis,
                (int(x1*sx), int(y1*sy)),
                (int(x2*sx), int(y2*sy)),
                (0, 255, 0), 2)

        return result, vis


class MultiSectionPipeline:
    """
    Manages one SectionPipeline per library section.

    Shares a single YOLOInference and Preprocessor instance across all
    sections — the model is loaded once regardless of how many sections
    there are.

    Usage:
        pipeline = MultiSectionPipeline(roi_data)
        results  = pipeline.run_all(frames_dict)
        # frames_dict = {"cad_lab": frame1, "g_huss": frame2, ...}
    """

    def __init__(
        self,
        roi_data: dict[str, dict],
    ) -> None:
        # Shared model — loaded once
        self._yolo         = YOLOInference()
        self._preprocessor = Preprocessor()
        self._pipelines: dict[str, SectionPipeline] = {}

        for cam in cfg.cameras:
            if not cam.enabled:
                continue
            sid = cam.section_id
            if sid not in roi_data:
                log.warning(f"No ROI data for section '{sid}' — skipping")
                continue

            self._pipelines[sid] = SectionPipeline(
                section_id    = sid,
                seat_polygons = roi_data[sid],
                max_capacity  = getattr(cam, "max_capacity", None),
                yolo          = self._yolo,
                preprocessor  = self._preprocessor,
            )

        log.info(
            f"MultiSectionPipeline ready | "
            f"sections={list(self._pipelines.keys())}"
        )

    def run(
        self,
        section_id: str,
        frame:      np.ndarray,
    ) -> "SectionResult":
        """Run pipeline for a single section."""
        if section_id not in self._pipelines:
            raise KeyError(f"No pipeline configured for section '{section_id}'")
        return self._pipelines[section_id].run(frame)

    def run_all(
        self,
        frames: dict[str, np.ndarray],
    ) -> dict[str, "SectionResult"]:
        """
        Run pipeline for multiple sections.
        In production this is called from parallel worker threads —
        one frame per section arrives from StreamManager.
        """
        results = {}
        for section_id, frame in frames.items():
            if section_id in self._pipelines:
                results[section_id] = self._pipelines[section_id].run(frame)
            else:
                log.warning(f"run_all: no pipeline for '{section_id}' — skipped")
        return results

    def sections(self) -> list[str]:
        """Return list of configured section IDs."""
        return list(self._pipelines.keys())