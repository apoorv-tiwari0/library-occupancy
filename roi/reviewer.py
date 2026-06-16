"""
roi/reviewer.py — Human review UI for auto-generated ROI polygons (CP-10)

Loads the draft JSON from CP-09 and lets the operator:
  - Accept polygons as-is
  - Delete false positives (D key or right-click)
  - Redraw bad polygons from scratch (R key)
  - Add missing seats manually (A key)
  - Save the cleaned result to the final roi_polygons.json

Controls:
    Left click        — select a polygon / add vertex (in draw mode)
    Right click       — deselect / remove last vertex (in draw mode)
    D                 — delete selected polygon
    R                 — redraw selected polygon (enter draw mode)
    A                 — add new polygon (enter draw mode)
    ENTER             — confirm drawn polygon
    ESC               — cancel current draw / deselect
    S                 — save progress
    Q                 — save and quit

Usage:
    python roi/reviewer.py --section cad_lab
                           --draft data/roi/roi_auto_draft.json
                           --output data/roi/roi_polygons.json
                           --image data/sample_frames/cad_lab.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import cfg
from utils.helpers import read_json, write_json
from utils.logger import get_logger

log = get_logger("reviewer")

# ── Colours (BGR) ──────────────────────────────────────────────────────────────
COL_NORMAL    = (0,   255,   0)    # green  — accepted polygon
COL_SELECTED  = (0,   165, 255)    # orange — currently selected
COL_DRAWING   = (0,   200, 255)    # yellow — polygon being drawn
COL_VERTEX    = (255,   0,   0)    # blue   — vertex dot
COL_LABEL     = (255, 255, 255)    # white  — seat ID text
COL_CROSSHAIR = (180, 180, 180)    # grey   — mouse crosshair
COL_HUD_BG    = (30,   30,  30)    # dark   — HUD background


class ROIReviewer:
    """
    Interactive polygon review tool for one camera section.

    Args:
        section_id:   Section being reviewed (e.g. "cad_lab").
        frame:        BGR frame to display polygons on.
        draft_data:   Full ROI dict loaded from the draft JSON.
        output_path:  Where to write the cleaned final JSON.
    """

    def __init__(
        self,
        section_id:  str,
        frame:       np.ndarray,
        draft_data:  dict,
        output_path: str | Path,
    ) -> None:
        self.section_id  = section_id
        self.frame       = frame.copy()
        self.output_path = Path(output_path)

        # Deep-copy this section's polygons so we don't mutate the original
        self._seats: dict[str, list] = dict(
            draft_data.get(section_id, {})
        )
        # Keep other sections intact for saving
        self._other_sections: dict = {
            k: v for k, v in draft_data.items() if k != section_id
        }

        self._selected:        str | None        = None   # currently selected seat_id
        self._mode:            str               = "view" # "view" | "draw"
        self._current_polygon: list[list[int]]   = []
        self._mouse_pos:       tuple[int, int]   = (0, 0)
        self._seat_counter:    int               = len(self._seats) + 1
        self._seat_prefix:     str               = section_id.split("_")[-1].upper()
        self._status_msg:      str               = "Review polygons. Click to select."

        log.info(
            f"ROIReviewer | section={section_id} "
            f"polygons={len(self._seats)} output={output_path}"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """Launch the review window. Blocks until Q is pressed."""
        win = "ROI Reviewer"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)
        cv2.setMouseCallback(win, self._mouse_callback)
        self._print_instructions()

        while True:
            cv2.imshow(win, self._render())
            key = cv2.waitKey(20) & 0xFF

            if self._mode == "view":
                if key == ord('d') or key == ord('D'):
                    self._delete_selected()
                elif key == ord('r') or key == ord('R'):
                    self._start_redraw()
                elif key == ord('a') or key == ord('A'):
                    self._start_add()
                elif key == 27:   # ESC
                    self._selected = None
                    self._status_msg = "Deselected."
                elif key == ord('s') or key == ord('S'):
                    self._save()
                elif key == ord('q') or key == ord('Q'):
                    self._save()
                    break

            elif self._mode == "draw":
                if key == 13:     # ENTER
                    self._confirm_polygon()
                elif key == 27:   # ESC
                    self._cancel_draw()

        cv2.destroyAllWindows()
        return self._build_output()

    # ── Mouse callback ─────────────────────────────────────────────────────────

    def _mouse_callback(self, event, x, y, flags, param) -> None:
        self._mouse_pos = (x, y)

        if self._mode == "draw":
            if event == cv2.EVENT_LBUTTONDOWN:
                self._current_polygon.append([x, y])
            elif event == cv2.EVENT_RBUTTONDOWN:
                if self._current_polygon:
                    self._current_polygon.pop()

        elif self._mode == "view":
            if event == cv2.EVENT_LBUTTONDOWN:
                self._try_select(x, y)
            elif event == cv2.EVENT_RBUTTONDOWN:
                self._selected = None
                self._status_msg = "Deselected."

    # ── Selection ──────────────────────────────────────────────────────────────

    def _try_select(self, x: int, y: int) -> None:
        """Select whichever polygon contains the click point."""
        for seat_id, pts in self._seats.items():
            poly = np.array(pts, dtype=np.int32)
            if cv2.pointPolygonTest(poly, (x, y), False) >= 0:
                self._selected   = seat_id
                self._status_msg = f"Selected: {seat_id}  |  D=delete  R=redraw"
                log.info(f"Selected {seat_id}")
                return
        self._selected   = None
        self._status_msg = "No polygon at click. Try again or press A to add."

    # ── Draw mode ──────────────────────────────────────────────────────────────

    def _start_redraw(self) -> None:
        if not self._selected:
            self._status_msg = "Select a polygon first, then press R."
            return
        # Remove old polygon; will be replaced when ENTER is pressed
        del self._seats[self._selected]
        self._mode           = "draw"
        self._current_polygon = []
        self._status_msg     = f"Redrawing {self._selected}. Click vertices. ENTER=confirm ESC=cancel."
        log.info(f"Redrawing {self._selected}")

    def _start_add(self) -> None:
        self._selected        = None
        self._mode            = "draw"
        self._current_polygon = []
        self._status_msg      = "Drawing new polygon. Click vertices. ENTER=confirm ESC=cancel."
        log.info("Adding new polygon.")

    def _confirm_polygon(self) -> None:
        if len(self._current_polygon) < 3:
            self._status_msg = f"Need ≥ 3 points (have {len(self._current_polygon)}). Keep clicking."
            return

        # Use selected ID if redrawing, else generate a new one
        if self._selected and self._selected not in self._seats:
            seat_id = self._selected   # restoring a redrawn polygon
        else:
            seat_id = f"seat_{self._seat_prefix}{self._seat_counter}"
            self._seat_counter += 1

        self._seats[seat_id] = [list(p) for p in self._current_polygon]
        self._mode            = "view"
        self._current_polygon = []
        self._selected        = seat_id
        self._status_msg      = f"Saved {seat_id} ({len(self._seats[seat_id])} pts). Select next or press Q."
        log.info(f"Polygon confirmed: {seat_id}")

    def _cancel_draw(self) -> None:
        self._mode            = "view"
        self._current_polygon = []
        self._selected        = None
        self._status_msg      = "Draw cancelled."
        log.info("Draw cancelled.")

    # ── Delete ─────────────────────────────────────────────────────────────────

    def _delete_selected(self) -> None:
        if not self._selected:
            self._status_msg = "Nothing selected. Click a polygon first."
            return
        del self._seats[self._selected]
        log.info(f"Deleted {self._selected}")
        self._status_msg = f"Deleted {self._selected}."
        self._selected   = None

    # ── Rendering ──────────────────────────────────────────────────────────────

    def _render(self) -> np.ndarray:
        display = self.frame.copy()
        h, w    = display.shape[:2]

        # Crosshair
        mx, my = self._mouse_pos
        cv2.line(display, (mx, 0), (mx, h), COL_CROSSHAIR, 1)
        cv2.line(display, (0, my), (w, my), COL_CROSSHAIR, 1)

        # Accepted polygons
        for seat_id, pts in self._seats.items():
            poly  = np.array(pts, dtype=np.int32)
            color = COL_SELECTED if seat_id == self._selected else COL_NORMAL
            cv2.polylines(display, [poly], isClosed=True, color=color, thickness=2)

            # Semi-transparent fill
            overlay = display.copy()
            fill    = (0, 80, 200) if seat_id == self._selected else (0, 60, 0)
            cv2.fillPoly(overlay, [poly], fill)
            cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)

            # Vertices
            for pt in pts:
                cv2.circle(display, tuple(pt), 4, COL_VERTEX, -1)

            # Label at centroid
            cx = int(sum(p[0] for p in pts) / len(pts))
            cy = int(sum(p[1] for p in pts) / len(pts))
            cv2.putText(display, seat_id, (cx - 25, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_LABEL, 2)

        # Polygon currently being drawn
        if self._current_polygon:
            pts = self._current_polygon
            for i, pt in enumerate(pts):
                cv2.circle(display, tuple(pt), 5, COL_DRAWING, -1)
                if i > 0:
                    cv2.line(display, tuple(pts[i-1]), tuple(pt), COL_DRAWING, 2)
            cv2.line(display, tuple(pts[-1]), self._mouse_pos, COL_DRAWING, 1)
            if len(pts) >= 2:
                cv2.line(display, self._mouse_pos, tuple(pts[0]), COL_DRAWING, 1)

        # HUD bar at bottom
        hud_y = h - 55
        cv2.rectangle(display, (0, hud_y), (w, h), COL_HUD_BG, -1)
        mode_str = f"[{'DRAW' if self._mode == 'draw' else 'VIEW'}]"
        stats    = f"Section: {self.section_id}  |  Seats: {len(self._seats)}  |  {mode_str}"
        cv2.putText(display, stats, (10, hud_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)
        cv2.putText(display, self._status_msg, (10, hud_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_LABEL, 1)

        # Key hint bar
        hint = "Click=select  D=delete  R=redraw  A=add  S=save  Q=quit  ESC=deselect"
        cv2.putText(display, hint, (10, hud_y + 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)

        return display

    # ── Persistence ────────────────────────────────────────────────────────────

    def _build_output(self) -> dict:
        out = dict(self._other_sections)
        out[self.section_id] = self._seats
        return out

    def _save(self) -> None:
        data = self._build_output()
        write_json(data, self.output_path)
        total = sum(len(v) for v in data.values())
        log.info(
            f"Saved → {self.output_path} "
            f"({len(data)} section(s), {total} total seats)"
        )
        self._status_msg = f"Saved. {len(self._seats)} seats in {self.section_id}."

    @staticmethod
    def _print_instructions() -> None:
        print("\n" + "=" * 55)
        print("  ROI REVIEWER — Controls")
        print("=" * 55)
        print("  Left click   → select a polygon")
        print("  D            → delete selected polygon")
        print("  R            → redraw selected polygon")
        print("  A            → add a new polygon")
        print("  ENTER        → confirm drawn polygon")
        print("  ESC          → cancel draw / deselect")
        print("  S            → save progress")
        print("  Q            → save and quit")
        print("=" * 55 + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Human review UI for auto-generated ROI polygons."
    )
    parser.add_argument("--section", required=True,
                        help="Section ID to review (e.g. cad_lab)")
    parser.add_argument("--draft",
                        default="data/roi/roi_auto_draft.json",
                        help="Path to draft ROI JSON from CP-09")
    parser.add_argument("--output",
                        default=cfg.roi.roi_save_path,
                        help=f"Output path for cleaned ROI JSON (default: {cfg.roi.roi_save_path})")
    parser.add_argument("--image", default=None,
                        help="Background frame (default: data/sample_frames/<section>.jpg)")
    args = parser.parse_args()

    # Resolve background image
    image_path = args.image or f"data/sample_frames/{args.section}.jpg"
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Cannot read image: {image_path}")
        print("   Pass --image <path> to specify the background frame.")
        sys.exit(1)

    # Load draft
    draft_path = Path(args.draft)
    if not draft_path.exists():
        print(f"❌ Draft JSON not found: {draft_path}")
        print("   Run CP-09 auto_calibrator.py first.")
        sys.exit(1)

    draft_data = read_json(draft_path)
    if args.section not in draft_data:
        print(f"❌ Section '{args.section}' not found in draft JSON.")
        print(f"   Available sections: {list(draft_data.keys())}")
        sys.exit(1)

    reviewer = ROIReviewer(
        section_id  = args.section,
        frame       = frame,
        draft_data  = draft_data,
        output_path = args.output,
    )
    final_data = reviewer.run()

    print(f"\nReview complete.")
    print(f"  Section : {args.section}")
    print(f"  Seats   : {len(final_data.get(args.section, {}))}")
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()