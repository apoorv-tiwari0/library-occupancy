"""
detection/yolo_inference.py — Person detection with SAHI sliced inference (CP-11)

Pipeline:
  - SAHI get_sliced_prediction for persons (handles crowded scenes)
  - Standard YOLO full-frame pass for reservation objects (bags, laptops, books)

Key tuning values (set via config.yaml):
  model.confidence_threshold : 0.50  — filters background/reflection false positives
  sahi_slicing.postprocess_match_threshold : 0.15  — merges cross-slice duplicate detections

Usage:
    from detection.yolo_inference import YOLOInference
    yolo = YOLOInference()
    persons, objects = yolo.run_inference(frame, section_id="cad_lab")
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import cfg
from config.constants import CocoClass, RESERVATION_CLASSES
from utils.logger import get_logger

log = get_logger("system")

# NMM match threshold — merges cross-slice duplicate detections
# 0.15 = merge boxes that share ≥15% IoU (handles partial-body tile splits)
NMM_MATCH_THRESHOLD = 0.15


class YOLOInference:
    """
    Person + object detector using SAHI sliced inference.

    SAHI slices the frame into overlapping windows, runs YOLO on each,
    then merges with NMM. conf≥0.50 cleanly separates real people from
    glass-wall reflections and background detections.
    """

    def __init__(
        self,
        weights_path: str | None = None,
        device:       str | None = None,
        conf:         float | None = None,
    ) -> None:
        from sahi import AutoDetectionModel
        from ultralytics import YOLO

        self.weights_path = weights_path or cfg.model.weights_path
        self.device       = device       or cfg.model.device
        self.conf         = conf         or cfg.model.confidence_threshold

        log.info(
            f"Loading SAHI+YOLOv10 | weights={self.weights_path} "
            f"device={self.device} conf={self.conf}"
        )

        self._detection_model = AutoDetectionModel.from_pretrained(
            model_type           = "ultralytics",
            model_path           = self.weights_path,
            confidence_threshold = self.conf,
            device               = self.device,
        )

        self._model = YOLO(self.weights_path)
        self._model.to(self.device)

        log.info("SAHI+YOLOv10 loaded.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def run_inference(
        self,
        frame:      np.ndarray,
        section_id: str | None = None,
    ) -> tuple[list[list[float]], list[list[float]]]:
        """
        Run SAHI sliced inference for persons + full-frame pass for objects.

        Args:
            frame:      BGR numpy array (any resolution).
            section_id: Used to look up sahi_slicing config from config.yaml.

        Returns:
            person_boxes: [[x1, y1, x2, y2, conf], ...]
            object_boxes: [[x1, y1, x2, y2, conf, class_id], ...]
        """
        slice_cfg = self._get_slice_config(section_id)
        persons   = self._run_sahi(frame, slice_cfg)
        objects   = self._run_objects_fullframe(frame)

        log.info(
            f"Inference | section={section_id} "
            f"persons={len(persons)} objects={len(objects)}"
        )
        return persons, objects

    # ── SAHI person detection ──────────────────────────────────────────────────

    def _run_sahi(
        self,
        frame:     np.ndarray,
        slice_cfg: object | None,
    ) -> list[list[float]]:
        """
        Run SAHI get_sliced_prediction for person class only.

        SAHI slices the image, runs YOLO on each slice, merges with NMM.
        NMM threshold 0.15 correctly merges partial-body detections at
        tile boundaries without over-merging separate people.
        """
        from sahi.predict import get_sliced_prediction

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        if slice_cfg:
            slice_h = slice_cfg.slice_height
            slice_w = slice_cfg.slice_width
            overlap  = slice_cfg.overlap_ratio
        else:
            slice_h = min(512, h)
            slice_w = min(512, w)
            overlap  = 0.20

        result = get_sliced_prediction(
            image                       = frame_rgb,
            detection_model             = self._detection_model,
            slice_height                = slice_h,
            slice_width                 = slice_w,
            overlap_height_ratio        = overlap,
            overlap_width_ratio         = overlap,
            perform_standard_pred       = True,
            postprocess_type            = "NMM",
            postprocess_match_metric    = "IOU",
            postprocess_match_threshold = NMM_MATCH_THRESHOLD,
            verbose                     = 0,
        )

        person_boxes = []
        for pred in result.object_prediction_list:
            if pred.category.id != CocoClass.PERSON:
                continue
            bbox = pred.bbox
            person_boxes.append([
                bbox.minx, bbox.miny,
                bbox.maxx, bbox.maxy,
                pred.score.value,
            ])

        return person_boxes

    # ── Object detection (reservation items) ──────────────────────────────────

    def _run_objects_fullframe(
        self,
        frame: np.ndarray,
    ) -> list[list[float]]:
        """
        Run standard YOLO on full frame for reservation objects only.
        Bags, laptops, books are large enough to detect without slicing.
        """
        results = self._model(
            frame,
            conf    = self.conf,
            device  = self.device,
            verbose = False,
            imgsz   = 640,
        )

        object_boxes = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id not in RESERVATION_CLASSES:
                    continue
                conf   = float(box.conf[0].item())
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                object_boxes.append([x1, y1, x2, y2, conf, float(cls_id)])

        return object_boxes

    # ── Visualisation ──────────────────────────────────────────────────────────

    def draw_detections(
        self,
        frame:        np.ndarray,
        person_boxes: list[list[float]],
        object_boxes: list[list[float]],
        **kwargs,
    ) -> np.ndarray:
        """Draw person/object boxes on a copy of the frame."""
        vis = frame.copy()

        for i, box in enumerate(person_boxes):
            x1, y1, x2, y2, conf = box
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 0), 2)
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)
            cv2.circle(vis, (cx, cy), 6, (0, 255, 0), -1)
            cv2.putText(vis, f"#{i+1} {conf:.2f}",
                        (int(x1), max(0, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        for box in object_boxes:
            x1, y1, x2, y2, conf, cls_id = box
            lbl = _class_label(int(cls_id))
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 165, 255), 2)
            cv2.putText(vis, f"{lbl} {conf:.2f}",
                        (int(x1), max(0, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

        cv2.rectangle(vis, (0, 0), (340, 32), (0, 0, 0), -1)
        cv2.putText(vis,
                    f"persons={len(person_boxes)}  objects={len(object_boxes)}  conf>{self.conf}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return vis

    def run_inference_debug(
        self,
        frame:      np.ndarray,
        section_id: str | None = None,
    ) -> tuple[list, list, np.ndarray]:
        """
        Same as run_inference but also returns an annotated debug frame.
        Shows SAHI slice grid (blue lines) + final detections (green boxes).
        """
        from sahi.predict import get_sliced_prediction

        slice_cfg = self._get_slice_config(section_id)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        if slice_cfg:
            slice_h = slice_cfg.slice_height
            slice_w = slice_cfg.slice_width
            overlap  = slice_cfg.overlap_ratio
        else:
            slice_h = min(512, h)
            slice_w = min(512, w)
            overlap  = 0.20

        result = get_sliced_prediction(
            image                       = frame_rgb,
            detection_model             = self._detection_model,
            slice_height                = slice_h,
            slice_width                 = slice_w,
            overlap_height_ratio        = overlap,
            overlap_width_ratio         = overlap,
            perform_standard_pred       = True,
            postprocess_type            = "NMM",
            postprocess_match_metric    = "IOU",
            postprocess_match_threshold = NMM_MATCH_THRESHOLD,
            verbose                     = 0,
        )

        debug = frame.copy()

        # Draw SAHI slice grid
        cols = max(1, w // slice_w)
        rows = max(1, h // slice_h)
        for c in range(1, cols + 1):
            cv2.line(debug, (c * slice_w, 0), (c * slice_w, h), (60, 60, 200), 1)
        for r in range(1, rows + 1):
            cv2.line(debug, (0, r * slice_h), (w, r * slice_h), (60, 60, 200), 1)

        persons = []
        for pred in result.object_prediction_list:
            if pred.category.id != CocoClass.PERSON:
                continue
            bbox = pred.bbox
            persons.append([bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, pred.score.value])

        objects_out = self._run_objects_fullframe(frame)

        # Draw final detections
        for i, box in enumerate(persons):
            x1, y1, x2, y2, conf = box
            cv2.rectangle(debug, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)
            cv2.circle(debug, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(debug, f"#{i+1} {conf:.2f}",
                        (int(x1), max(0, int(y1)-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        cv2.rectangle(debug, (0, 0), (380, 28), (0, 0, 0), -1)
        cv2.putText(debug,
                    f"persons={len(persons)}  slices={cols}x{rows}  NMM={NMM_MATCH_THRESHOLD}",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return persons, objects_out, debug

    # ── Config helpers ─────────────────────────────────────────────────────────

    def _get_slice_config(self, section_id: str | None):
        """Return sahi_slicing config for a section from config.yaml."""
        for cam in cfg.cameras:
            if section_id is None or cam.section_id == section_id:
                return getattr(cam, "sahi_slicing", None)
        return None

    def _get_tile_config(self, section_id: str | None):
        """Legacy tile config lookup — kept for backward compatibility."""
        for cam in cfg.cameras:
            if section_id is None or cam.section_id == section_id:
                return getattr(cam, "tiled_inference", None)
        return None


# ── Module-level helpers ───────────────────────────────────────────────────────

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


def _nms(
    boxes:         list[list[float]],
    iou_threshold: float = 0.5,
    has_class:     bool  = False,
) -> list[list[float]]:
    """Greedy NMS — used for object deduplication."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    kept  = []
    for box in boxes:
        suppress = False
        for kept_box in kept:
            if has_class and int(box[5]) != int(kept_box[5]):
                continue
            if _box_iou(box[:4], kept_box[:4]) > iou_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(box)
    return kept


def _class_label(cls_id: int) -> str:
    try:
        return CocoClass(cls_id).name.lower()
    except ValueError:
        return f"class_{cls_id}"