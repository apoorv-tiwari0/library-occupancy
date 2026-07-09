"""
detection/pipeline.py — Per-section inference pipeline (CP-14)

Pipeline per section:
    Raw frame
        ↓
    Preprocessor        — gamma → bilateral → CLAHE → resize to 640x640
        ↓
    YOLOInference       — SAHI+YOLOv10 person detection
        ↓
    Vacancy calculation — vacancy = max_capacity (config.yaml) − headcount
        ↓
    SectionResult       — headcount, vacancy, occupancy_pct, is_available

No seat-by-seat classification. No ROI polygon matching.
max_capacity comes exclusively from config.yaml per section.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import cfg
from detection.yolo_inference import YOLOInference
from ingestion.preprocessor import Preprocessor
from utils.helpers import get_timestamp
from utils.logger import get_logger

log = get_logger("system")


def _get_max_capacity(section_id: str) -> int | None:
    """Read max_capacity for a section from config.yaml. Returns None if not set."""
    for cam in cfg.cameras:
        if cam.section_id == section_id:
            return getattr(cam, "max_capacity", None)
    return None


@dataclass
class SectionResult:
    """
    Occupancy data for one section from one frame.

    Attributes:
        section_id:   Section identifier.
        timestamp:    ISO timestamp of the processed frame.
        headcount:    Number of people detected by YOLO.
        max_capacity: From config.yaml — maximum people allowed in section.
        vacancy:      max_capacity − headcount (clamped to 0).
        person_boxes: Raw YOLO detections [[x1,y1,x2,y2,conf], ...].
        object_boxes: Reservation object detections [[x1,y1,x2,y2,conf,cls], ...].
        inference_ms: YOLO inference time in milliseconds.
        pipeline_ms:  Total pipeline time in milliseconds.
    """
    section_id:   str
    timestamp:    str
    headcount:    int
    max_capacity: int | None
    vacancy:      int | None
    person_boxes: list[list[float]]
    object_boxes: list[list[float]]
    inference_ms: float
    pipeline_ms:  float

    def is_available(self) -> bool:
        """True if there is at least one vacancy."""
        if self.vacancy is None:
            return True
        return self.vacancy > 0

    def occupancy_pct(self) -> float | None:
        """Percentage of max_capacity occupied. None if max_capacity not set."""
        if not self.max_capacity:
            return None
        return round(min(100.0, self.headcount / self.max_capacity * 100), 1)

    def to_dict(self) -> dict:
        """JSON-serialisable dict for Redis / API / dashboard."""
        return {
            "section_id":    self.section_id,
            "timestamp":     self.timestamp,
            "headcount":     self.headcount,
            "max_capacity":  self.max_capacity,
            "vacancy":       self.vacancy,
            "occupancy_pct": self.occupancy_pct(),
            "is_available":  self.is_available(),
            "inference_ms":  round(self.inference_ms, 1),
            "pipeline_ms":   round(self.pipeline_ms, 1),
        }


class SectionPipeline:
    """
    Full inference pipeline for one library section.

    Reads max_capacity from config.yaml automatically.
    Instantiate once per section, call run(frame) on every new frame.

    Args:
        section_id:   Section identifier — must match config.yaml.
        yolo:         Shared YOLOInference instance (loaded once, shared).
        preprocessor: Optional shared Preprocessor. Created if not provided.
    """

    def __init__(
        self,
        section_id:   str,
        yolo:         YOLOInference,
        preprocessor: Preprocessor | None = None,
    ) -> None:
        self.section_id   = section_id
        self.yolo         = yolo
        self.preprocessor = preprocessor or Preprocessor()
        self.max_capacity = _get_max_capacity(section_id)

        log.info(
            f"SectionPipeline ready | section={section_id} "
            f"max_capacity={self.max_capacity} "
            f"(headcount mode — vacancy = max_capacity − headcount)"
        )

    def run(self, frame: np.ndarray) -> SectionResult:
        """
        Run the full pipeline on one raw BGR frame.

        Steps:
            1. Preprocess  — gamma → bilateral → CLAHE → resize 640×640
            2. YOLO        — SAHI person detection
            3. Vacancy     — max_capacity (config.yaml) − headcount

        Args:
            frame: Raw BGR numpy array from the camera.

        Returns:
            SectionResult with headcount and vacancy.
        """
        t_start = time.perf_counter()
        ts      = get_timestamp()

        # Step 1 — preprocess
        processed = self.preprocessor.process(frame)

        # Step 2 — YOLO person detection
        t_infer = time.perf_counter()
        persons, objects = self.yolo.run_inference(
            processed, section_id=self.section_id
        )
        inference_ms = (time.perf_counter() - t_infer) * 1000

        # Step 3 — headcount-based vacancy
        headcount = len(persons)
        vacancy   = (
            max(0, self.max_capacity - headcount)
            if self.max_capacity is not None else None
        )

        pipeline_ms = (time.perf_counter() - t_start) * 1000

        result = SectionResult(
            section_id   = self.section_id,
            timestamp    = ts,
            headcount    = headcount,
            max_capacity = self.max_capacity,
            vacancy      = vacancy,
            person_boxes = persons,
            object_boxes = objects,
            inference_ms = inference_ms,
            pipeline_ms  = pipeline_ms,
        )

        log.info(
            f"[{self.section_id}] headcount={headcount} "
            f"max_capacity={self.max_capacity} "
            f"vacancy={vacancy} "
            f"inference={inference_ms:.0f}ms "
            f"pipeline={pipeline_ms:.0f}ms"
        )

        return result


class MultiSectionPipeline:
    """
    Manages one SectionPipeline per library section.

    Loads YOLO model once and shares it across all sections.
    All max_capacity values come from config.yaml automatically.

    Usage:
        pipeline = MultiSectionPipeline()
        result   = pipeline.run("cad_lab", frame)
        results  = pipeline.run_all({"cad_lab": frame1, "g_huss": frame2})
    """

    def __init__(self) -> None:
        self._yolo         = YOLOInference()
        self._preprocessor = Preprocessor()
        self._pipelines: dict[str, SectionPipeline] = {}

        for cam in cfg.cameras:
            if not cam.enabled:
                continue
            self._pipelines[cam.section_id] = SectionPipeline(
                section_id   = cam.section_id,
                yolo         = self._yolo,
                preprocessor = self._preprocessor,
            )

        log.info(
            f"MultiSectionPipeline ready | "
            f"sections={list(self._pipelines.keys())}"
        )

    def run(self, section_id: str, frame: np.ndarray) -> SectionResult:
        """Run pipeline for one section."""
        if section_id not in self._pipelines:
            raise KeyError(f"No pipeline for section '{section_id}'")
        return self._pipelines[section_id].run(frame)

    def run_all(
        self, frames: dict[str, np.ndarray]
    ) -> dict[str, SectionResult]:
        """Run pipeline for multiple sections."""
        results = {}
        for section_id, frame in frames.items():
            if section_id in self._pipelines:
                results[section_id] = self._pipelines[section_id].run(frame)
            else:
                log.warning(f"run_all: no pipeline for '{section_id}' — skipped")
        return results

    def sections(self) -> list[str]:
        """Return configured section IDs."""
        return list(self._pipelines.keys())