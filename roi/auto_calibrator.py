"""
roi/auto_calibrator.py — Grounding DINO + SAM2 auto ROI calibration (CP-09)

Pipeline:
  1. Load an empty library frame
  2. Run Grounding DINO with text prompts to detect chairs/tables (zero-shot)
  3. Pass each bounding box to SAM2 to get a precise polygon mask
  4. Convert SAM2 masks → polygon contours
  5. Output a draft ROI JSON in the same nested format as the manual annotator

The output JSON is a DRAFT — always run through CP-10 (ROI review UI)
before using in production.

Usage:
    python roi/auto_calibrator.py --section section_A --image data/sample_frames/section_A.jpg
    python roi/auto_calibrator.py --section section_A --image data/sample_frames/section_A.jpg --merge
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import cfg
from utils.helpers import read_json, write_json
from utils.logger import get_logger

log = get_logger("auto_calibrator")

# ── Detection prompts — tune these if DINO misses seats ───────────────────────
DETECTION_PROMPTS = "chair . table . seat . desk . stool"

# ── DINO thresholds ────────────────────────────────────────────────────────────
DINO_BOX_THRESHOLD  = 0.30   # lower = more detections (more false positives)
DINO_TEXT_THRESHOLD = 0.25


# ── Model loaders (lazy — loaded once, cached) ─────────────────────────────────

_dino_model  = None
_sam2_predictor = None


def _load_dino():
    global _dino_model
    if _dino_model is not None:
        return _dino_model

    log.info("Loading Grounding DINO model...")
    from groundingdino.util.inference import load_model
    _dino_model = load_model(
        model_config_path = "models/GroundingDINO_SwinT_OGC.py",
        model_checkpoint_path = "models/groundingdino_swint_ogc.pth",
        device = cfg.model.device,
    )
    log.info("Grounding DINO loaded.")
    return _dino_model


def _load_sam2():
    global _sam2_predictor
    if _sam2_predictor is not None:
        return _sam2_predictor

    log.info("Loading SAM2 model...")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam2 = build_sam2(
        config_file = "configs/sam2.1/sam2.1_hiera_l.yaml",
        ckpt_path = "models/sam2_hiera_l.pt",
        device = cfg.model.device,
    )
    _sam2_predictor = SAM2ImagePredictor(sam2)
    log.info("SAM2 loaded.")
    return _sam2_predictor


# ── Core pipeline ──────────────────────────────────────────────────────────────

def detect_seats_dino(frame_rgb: np.ndarray) -> list[list[float]]:
    """
    Run Grounding DINO on an RGB frame.

    Returns:
        List of bounding boxes as [x1, y1, x2, y2] in pixel coords.
    """
    from groundingdino.util.inference import predict
    import torchvision.transforms as T

    model = _load_dino()
    h, w  = frame_rgb.shape[:2]

    # DINO expects a normalised tensor
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((800, 800)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(frame_rgb)

    boxes_norm, logits, phrases = predict(
        model       = model,
        image       = image_tensor,
        caption     = DETECTION_PROMPTS,
        box_threshold  = DINO_BOX_THRESHOLD,
        text_threshold = DINO_TEXT_THRESHOLD,
        device      = cfg.model.device,
    )

    # Convert normalised [cx, cy, w, h] → pixel [x1, y1, x2, y2]
    boxes_px = []
    for box, phrase, logit in zip(boxes_norm, phrases, logits):
        cx, cy, bw, bh = box.tolist()
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        # Clamp to frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        boxes_px.append([x1, y1, x2, y2])
        log.info(f"  DINO detected: '{phrase}' conf={logit:.2f} box={[x1,y1,x2,y2]}")

    log.info(f"Grounding DINO: {len(boxes_px)} seat candidates detected.")
    return boxes_px


def refine_boxes_with_sam2(
    frame_rgb: np.ndarray,
    boxes: list[list[float]],
) -> list[list[list[int]]]:
    """
    Pass each bounding box to SAM2 to get a precise polygon mask.

    Returns:
        List of polygons, one per box. Each polygon is [[x,y], [x,y], ...].
        Returns a rectangular fallback polygon if SAM2 mask is poor quality.
    """
    predictor = _load_sam2()
    predictor.set_image(frame_rgb)

    polygons = []
    for i, box in enumerate(boxes):
        box_np = np.array(box, dtype=np.float32)

        with torch.inference_mode():
            masks, scores, _ = predictor.predict(
                box            = box_np,
                multimask_output = True,
            )

        # Pick the highest-confidence mask
        best_idx  = int(np.argmax(scores))
        best_mask = masks[best_idx].astype(np.uint8) * 255

        polygon = _mask_to_polygon(best_mask)

        if polygon is None:
            # Fallback: use the bounding box itself as a 4-point polygon
            x1, y1, x2, y2 = [int(v) for v in box]
            polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            log.warning(f"  Box {i}: SAM2 mask too small, using bbox fallback.")
        else:
            log.info(f"  Box {i}: SAM2 polygon with {len(polygon)} vertices.")

        polygons.append(polygon)

    return polygons


def _mask_to_polygon(
    mask: np.ndarray,
    min_area: int = 500,
    epsilon_ratio: float = 0.02,
) -> list[list[int]] | None:
    """
    Convert a binary mask to a simplified polygon contour.

    Args:
        mask:          Binary mask (uint8, 0 or 255).
        min_area:      Minimum contour area to accept (filters noise).
        epsilon_ratio: Douglas-Peucker simplification strength.

    Returns:
        List of [x, y] points, or None if no valid contour found.
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    # Take the largest contour
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None

    # Simplify with Douglas-Peucker
    epsilon  = epsilon_ratio * cv2.arcLength(largest, closed=True)
    approx   = cv2.approxPolyDP(largest, epsilon, closed=True)
    polygon  = approx.reshape(-1, 2).tolist()

    return polygon if len(polygon) >= 3 else None


# ── Duplicate removal ──────────────────────────────────────────────────────────

def _remove_duplicate_boxes(
    boxes: list[list[float]],
    iou_threshold: float = 0.5,
) -> list[list[float]]:
    """
    Simple greedy NMS to remove highly overlapping DINO detections.
    Keeps the first occurrence when IoU > threshold.
    """
    if not boxes:
        return boxes

    kept = []
    for box in boxes:
        duplicate = False
        for kept_box in kept:
            if _box_iou(box, kept_box) > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)

    removed = len(boxes) - len(kept)
    if removed:
        log.info(f"Removed {removed} duplicate DINO detections (IoU > {iou_threshold}).")
    return kept


def _box_iou(a: list[float], b: list[float]) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)


# ── Visualisation ──────────────────────────────────────────────────────────────

def visualise_results(
    frame_bgr: np.ndarray,
    roi_data:  dict,
    section_id: str,
    save_path: str | Path | None = None,
) -> np.ndarray:
    """
    Draw all detected polygons on the frame and optionally save to disk.

    Returns the annotated frame (BGR).
    """
    vis = frame_bgr.copy()
    seats = roi_data.get(section_id, {})

    for seat_id, pts in seats.items():
        poly = np.array(pts, dtype=np.int32)
        cv2.polylines(vis, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
        cx = int(sum(p[0] for p in pts) / len(pts))
        cy = int(sum(p[1] for p in pts) / len(pts))
        cv2.putText(vis, seat_id, (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), vis)
        log.info(f"Visualisation saved → {save_path}")

    return vis


# ── Main orchestrator ──────────────────────────────────────────────────────────

def run_auto_calibration(
    section_id: str,
    image_path: str | Path,
    roi_output_path: str | Path,
    merge_with_existing: bool = False,
) -> dict:
    """
    Full auto-calibration pipeline for one section.

    Args:
        section_id:          e.g. "section_A"
        image_path:          Path to an empty library frame (JPG/PNG).
        roi_output_path:     Where to write the draft ROI JSON.
        merge_with_existing: If True, append to existing JSON rather than overwrite.

    Returns:
        Full ROI dictionary.
    """
    image_path      = Path(image_path)
    roi_output_path = Path(roi_output_path)

    # Load frame
    frame_bgr = cv2.imread(str(image_path))
    if frame_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    log.info(f"Frame loaded: {image_path} ({frame_bgr.shape[1]}x{frame_bgr.shape[0]})")

    # Step 1 — Grounding DINO detection
    boxes = detect_seats_dino(frame_rgb)
    if not boxes:
        log.warning("No seats detected by Grounding DINO. Try lowering DINO_BOX_THRESHOLD.")
        return {}

    # Step 2 — Remove duplicates
    boxes = _remove_duplicate_boxes(boxes)

    # Step 3 — SAM2 polygon refinement
    polygons = refine_boxes_with_sam2(frame_rgb, boxes)

    # Step 4 — Build ROI dict
    seat_prefix = section_id.split("_")[-1].upper()
    roi_section = {}
    for i, polygon in enumerate(polygons, start=1):
        seat_id = f"seat_{seat_prefix}{i}"
        roi_section[seat_id] = polygon

    # Step 5 — Merge or replace
    if merge_with_existing and roi_output_path.exists():
        existing = read_json(roi_output_path)
        existing[section_id] = roi_section
        roi_data = existing
        log.info(f"Merged with existing ROI file. Total sections: {len(roi_data)}")
    else:
        roi_data = {section_id: roi_section}

    # Step 6 — Save draft JSON
    write_json(roi_data, roi_output_path)
    log.info(
        f"Draft ROI saved → {roi_output_path} "
        f"({len(roi_section)} seats detected in {section_id})"
    )

    # Step 7 — Save visualisation
    vis_path = Path("data/roi_previews") / f"{section_id}_auto_calibration.jpg"
    visualise_results(frame_bgr, roi_data, section_id, save_path=vis_path)

    return roi_data


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Auto-calibrate seat ROIs using Grounding DINO + SAM2."
    )
    parser.add_argument("--section", required=True, help="Section ID (e.g. section_A)")
    parser.add_argument("--image",   required=True, help="Path to an empty library frame")
    parser.add_argument(
        "--output",
        default="data/roi/roi_auto_draft.json",
        help="Output path for draft ROI JSON",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge with existing ROI JSON instead of overwriting",
    )
    args = parser.parse_args()

    roi_data = run_auto_calibration(
        section_id          = args.section,
        image_path          = args.image,
        roi_output_path     = args.output,
        merge_with_existing = args.merge,
    )

    print(f"\nAuto-calibration complete.")
    print(f"  Section : {args.section}")
    print(f"  Seats   : {len(roi_data.get(args.section, {}))}")
    print(f"  Draft ROI saved to: {args.output}")
    print(f"\n⚠️  This is a DRAFT — run CP-10 review UI before using in production.")


if __name__ == "__main__":
    main()