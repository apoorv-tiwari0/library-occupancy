"""
roi/annotator.py — Interactive polygon ROI annotation tool (CP-08)

Opens a camera frame in an OpenCV window. The operator clicks to draw
polygon zones around each seat/table. When a polygon is complete, it is
saved with a unique seat ID. All polygons are stored in a nested JSON:

    {
      "section_A": {
        "seat_A1": [[x1,y1], [x2,y2], [x3,y3], ...],
        "seat_A2": [[x1,y1], ...]
      },
      "section_B": { ... }
    }

Controls:
    Left click      — add a polygon vertex
    Right click     — undo last vertex
    ENTER           — close & save the current polygon
    ESC             — discard the current polygon (start over)
    S               — save all polygons to JSON and continue
    Q               — save all polygons to JSON and quit

Usage:
    python roi/annotator.py --section section_A --image data/sample_frames/section_A.jpg
    python roi/annotator.py --section section_A --video data/test_videos/section_A.mp4
    python roi/annotator.py --section section_A --camera 0
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import cfg
from utils.helpers import read_json, write_json
from utils.logger import get_logger

log = get_logger("system")

# ── Colours (BGR) ──────────────────────────────────────────────────────────────
COL_COMPLETE   = (0,   255,  0)    # green  — completed polygon
COL_ACTIVE     = (0,   200, 255)   # yellow — polygon being drawn
COL_VERTEX     = (255,  0,   0)    # blue   — vertex dot
COL_LABEL      = (255, 255, 255)   # white  — seat ID label
COL_CROSSHAIR  = (180, 180, 180)   # grey   — cursor crosshair


class ROIAnnotator:
    """
    Interactive polygon annotation tool for one camera section.

    Args:
        section_id:  The section being annotated (e.g. "section_A").
        source:      Path to an image file, video file, or int for webcam index.
        roi_path:    Path to the existing ROI JSON file (to load/append to).
        seat_prefix: Prefix for auto-generated seat IDs (default = section letter).
    """

    def __init__(
        self,
        section_id:  str,
        source:      str | int,
        roi_path:    str | Path,
        seat_prefix: str | None = None,
    ) -> None:
        self.section_id  = section_id
        self.source      = source
        self.roi_path    = Path(roi_path)
        self.seat_prefix = seat_prefix or section_id.split("_")[-1].upper()

        # Load existing ROI data (so we can append, not overwrite)
        self._roi_data: dict = self._load_existing()

        # State for the current drawing session
        self._current_polygon: list[list[int]] = []   # vertices being drawn
        self._mouse_pos: tuple[int, int] = (0, 0)
        self._base_frame: np.ndarray | None = None    # clean frame to redraw on

        # Seat counter — start after existing seats for this section
        existing = self._roi_data.get(section_id, {})
        self._seat_counter = len(existing) + 1

        log.info(
            f"ROIAnnotator | section={section_id} source={source} "
            f"roi_path={roi_path} existing_seats={len(existing)}"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Launch the annotation window. Blocks until the user presses Q.

        Returns:
            The full updated ROI dictionary (all sections).
        """
        frame = self._get_frame()
        if frame is None:
            log.error("Could not load source frame. Check --image/--video/--camera.")
            return self._roi_data

        self._base_frame = frame.copy()
        win = cfg.roi.annotation_window_name
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)
        cv2.setMouseCallback(win, self._mouse_callback)

        self._print_instructions()

        while True:
            display = self._render()
            cv2.imshow(win, display)
            key = cv2.waitKey(20) & 0xFF

            if key == 13:    # ENTER — close current polygon
                self._close_polygon()
            elif key == 27:  # ESC — discard current polygon
                self._current_polygon.clear()
                log.info("Current polygon discarded.")
            elif key == ord('s') or key == ord('S'):
                self._save()
                log.info("Polygons saved. Continue annotating or press Q to quit.")
            elif key == ord('q') or key == ord('Q'):
                self._save()
                break
            elif key == ord('z') or key == ord('Z'):
                # Undo last completed polygon for this section
                self._undo_last()

        cv2.destroyAllWindows()
        return self._roi_data

    # ── Frame acquisition ──────────────────────────────────────────────────────

    def _get_frame(self) -> np.ndarray | None:
        """Load a single frame from the source (image, video, or camera)."""
        src = self.source

        # Image file
        if isinstance(src, str) and Path(src).suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".tiff"
        }:
            frame = cv2.imread(src)
            if frame is None:
                log.error(f"Could not read image: {src}")
            return frame

        # Video file or RTSP or webcam index
        cap = cv2.VideoCapture(src if isinstance(src, int) else src)
        if not cap.isOpened():
            log.error(f"Could not open source: {src}")
            return None

        # Skip to middle of video for a more representative frame
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 10:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)

        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    # ── Mouse callback ─────────────────────────────────────────────────────────

    def _mouse_callback(self, event, x, y, flags, param) -> None:
        self._mouse_pos = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            self._current_polygon.append([x, y])

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self._current_polygon:
                self._current_polygon.pop()
                log.info(f"Vertex removed. Polygon now has {len(self._current_polygon)} points.")

    # ── Rendering ──────────────────────────────────────────────────────────────

    def _render(self) -> np.ndarray:
        """Build the display frame with all annotations overlaid."""
        display = self._base_frame.copy()
        h, w    = display.shape[:2]

        # Draw crosshair at mouse position
        mx, my = self._mouse_pos
        cv2.line(display, (mx, 0), (mx, h), COL_CROSSHAIR, 1)
        cv2.line(display, (0, my), (w, my), COL_CROSSHAIR, 1)

        # Draw all completed polygons for this section
        section_data = self._roi_data.get(self.section_id, {})
        for seat_id, pts in section_data.items():
            poly = np.array(pts, dtype=np.int32)
            cv2.polylines(display, [poly], isClosed=True,
                          color=COL_COMPLETE, thickness=2)
            cv2.fillPoly(
                display,
                [poly],
                color=(*COL_COMPLETE[::-1], 40),   # slight green tint
            )
            # Draw vertices
            for pt in pts:
                cv2.circle(display, tuple(pt), 4, COL_VERTEX, -1)
            # Label at centroid
            cx = int(sum(p[0] for p in pts) / len(pts))
            cy = int(sum(p[1] for p in pts) / len(pts))
            cv2.putText(display, seat_id, (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_LABEL, 2)

        # Draw polygon currently being drawn
        if self._current_polygon:
            pts = self._current_polygon
            for i, pt in enumerate(pts):
                cv2.circle(display, tuple(pt), 5, COL_ACTIVE, -1)
                if i > 0:
                    cv2.line(display, tuple(pts[i-1]), tuple(pt),
                             COL_ACTIVE, 2)
            # Preview line from last vertex to mouse
            cv2.line(display, tuple(pts[-1]), self._mouse_pos,
                     COL_ACTIVE, 1)
            # Preview closing line from mouse to first vertex
            if len(pts) >= 2:
                cv2.line(display, self._mouse_pos, tuple(pts[0]),
                         COL_ACTIVE, 1)

        # HUD — seat count
        n_seats = len(section_data)
        hud = (f"Section: {self.section_id}  |  "
               f"Seats annotated: {n_seats}  |  "
               f"ENTER=save polygon  ESC=discard  Z=undo  S=save  Q=quit")
        cv2.rectangle(display, (0, h - 30), (w, h), (0, 0, 0), -1)
        cv2.putText(display, hud, (10, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_LABEL, 1)

        return display

    # ── Polygon management ─────────────────────────────────────────────────────

    def _close_polygon(self) -> None:
        """Save the current polygon as a new seat ROI."""
        if len(self._current_polygon) < 3:
            log.warning(
                f"Need at least 3 points for a polygon "
                f"(have {len(self._current_polygon)}). Keep clicking."
            )
            return

        seat_id = f"seat_{self.seat_prefix}{self._seat_counter}"
        if self.section_id not in self._roi_data:
            self._roi_data[self.section_id] = {}

        self._roi_data[self.section_id][seat_id] = [
            list(pt) for pt in self._current_polygon
        ]
        self._current_polygon = []
        self._seat_counter += 1
        log.info(
            f"Polygon saved as {seat_id} "
            f"({len(self._roi_data[self.section_id])} total seats in {self.section_id})"
        )

    def _undo_last(self) -> None:
        """Remove the last completed polygon for this section."""
        section_data = self._roi_data.get(self.section_id, {})
        if not section_data:
            log.warning("No polygons to undo.")
            return
        last_key = list(section_data.keys())[-1]
        del self._roi_data[self.section_id][last_key]
        self._seat_counter -= 1
        log.info(f"Undid polygon: {last_key}")

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load_existing(self) -> dict:
        """Load existing ROI JSON if it exists, else return empty dict."""
        if self.roi_path.exists():
            try:
                data = read_json(self.roi_path)
                log.info(
                    f"Loaded existing ROI file: {self.roi_path} "
                    f"({len(data)} section(s))"
                )
                return data
            except Exception as e:
                log.warning(f"Could not load existing ROI file: {e}. Starting fresh.")
        return {}

    def _save(self) -> None:
        """Write the ROI data to JSON."""
        write_json(self._roi_data, self.roi_path)
        n_seats = sum(len(v) for v in self._roi_data.values())
        log.info(
            f"ROI saved → {self.roi_path} "
            f"({len(self._roi_data)} section(s), {n_seats} total seats)"
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _print_instructions() -> None:
        print("\n" + "=" * 55)
        print("  ROI ANNOTATOR — Controls")
        print("=" * 55)
        print("  Left click   → add vertex to current polygon")
        print("  Right click  → remove last vertex")
        print("  ENTER        → finish & save current polygon")
        print("  ESC          → discard current polygon")
        print("  Z            → undo last completed polygon")
        print("  S            → save progress to JSON")
        print("  Q            → save and quit")
        print("=" * 55 + "\n")


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive ROI polygon annotator for library seats."
    )
    parser.add_argument(
        "--section", required=True,
        help="Section ID to annotate (e.g. section_A)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  help="Path to a still image file")
    group.add_argument("--video",  help="Path to a video file")
    group.add_argument("--camera", type=int, help="Webcam/capture device index")
    parser.add_argument(
        "--roi-path",
        default=cfg.roi.roi_save_path,
        help=f"ROI JSON output path (default: {cfg.roi.roi_save_path})"
    )
    parser.add_argument(
        "--seat-prefix",
        default=None,
        help="Prefix for seat IDs (default: section letter, e.g. A for section_A)"
    )

    args = parser.parse_args()
    source = args.image or args.video or args.camera

    annotator = ROIAnnotator(
        section_id  = args.section,
        source      = source,
        roi_path    = args.roi_path,
        seat_prefix = args.seat_prefix,
    )
    roi_data = annotator.run()

    print(f"\nFinal ROI summary:")
    for section, seats in roi_data.items():
        print(f"  {section}: {len(seats)} seat(s) — {list(seats.keys())}")


if __name__ == "__main__":
    main()