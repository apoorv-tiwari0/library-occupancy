"""
roi/zone_annotator.py — Near/Mid/Far zone annotation tool (ZCP-01)

Draws exactly 3 zone polygons per section on a camera frame:
  - NEAR  (green)  — close to camera, people large in frame
  - MID   (yellow) — middle distance, people medium-sized
  - FAR   (red)    — far from camera, people small/distant

Output JSON structure:
{
  "cad_lab": {
    "near": [[x1,y1],[x2,y2],...],
    "mid":  [[x1,y1],[x2,y2],...],
    "far":  [[x1,y1],[x2,y2],...]
  },
  "g_huss": { ... }
}

Controls:
  1 / 2 / 3    — switch active zone (NEAR / MID / FAR)
  Left click   — add vertex to active zone
  Right click  — undo last vertex in active zone
  ENTER        — confirm current zone polygon
  ESC          — clear current zone (start over)
  S            — save progress to JSON
  Q            — save and quit

Usage:
  python roi/zone_annotator.py --section cad_lab --image data/sample_frames/cad_lab.jpg
  python roi/zone_annotator.py --section cad_lab --video data/test_videos/cad_lab.mp4
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import read_json, write_json
from utils.logger import get_logger

log = get_logger("system")

ZONE_NAMES  = ["near", "mid", "far"]
ZONE_COLORS = {
    "near": (0,   255,  0),    # green
    "mid":  (0,   215, 255),   # yellow
    "far":  (0,   0,   255),   # red
}
ZONE_KEYS = {"1": "near", "2": "mid", "3": "far"}
ALPHA = 0.25   # fill transparency


class ZoneAnnotator:
    """
    Interactive tool to draw near/mid/far zone polygons for one section.

    Args:
        section_id: Section being annotated.
        source:     Image path, video path, or webcam index.
        zone_path:  Path to zone_config.json (load/append).
    """

    def __init__(
        self,
        section_id: str,
        source:     str | int,
        zone_path:  str | Path,
    ) -> None:
        self.section_id = section_id
        self.source     = source
        self.zone_path  = Path(zone_path)

        # Load existing data
        self._data: dict = self._load_existing()

        # Per-zone polygon storage
        self._zones: dict[str, list[list[int]]] = {
            z: list(self._data.get(section_id, {}).get(z, []))
            for z in ZONE_NAMES
        }

        # Active zone being drawn
        self._active_zone   = "near"
        self._current_poly: list[list[int]] = []
        self._mouse_pos     = (0, 0)
        self._base_frame: np.ndarray | None = None

        log.info(
            f"ZoneAnnotator | section={section_id} "
            f"source={source} zone_path={zone_path}"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """Launch annotation window. Blocks until Q is pressed."""
        frame = self._get_frame()
        if frame is None:
            log.error("Could not load source frame.")
            return self._data

        self._base_frame = frame.copy()
        win = f"Zone Annotator — {self.section_id}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)
        cv2.setMouseCallback(win, self._mouse_cb)

        self._print_instructions()

        while True:
            cv2.imshow(win, self._render())
            key = cv2.waitKey(20) & 0xFF

            if key in [ord("1"), ord("2"), ord("3")]:
                self._active_zone   = ZONE_KEYS[chr(key)]
                self._current_poly  = []
                log.info(f"Active zone → {self._active_zone.upper()}")

            elif key == 13:   # ENTER — confirm zone
                self._confirm_zone()

            elif key == 27:   # ESC — clear current drawing
                self._current_poly = []
                log.info("Current polygon cleared.")

            elif key in [ord("s"), ord("S")]:
                self._save()

            elif key in [ord("q"), ord("Q")]:
                self._save()
                break

            elif key in [ord("z"), ord("Z")]:
                self._undo_zone()

        cv2.destroyAllWindows()
        return self._data

    # ── Frame acquisition ──────────────────────────────────────────────────────

    def _get_frame(self) -> np.ndarray | None:
        src = self.source
        if isinstance(src, str) and Path(src).suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp"
        }:
            frame = cv2.imread(src)
            if frame is None:
                log.error(f"Cannot read image: {src}")
            return frame

        cap = cv2.VideoCapture(src if isinstance(src, int) else src)
        if not cap.isOpened():
            log.error(f"Cannot open source: {src}")
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 10:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    # ── Mouse ──────────────────────────────────────────────────────────────────

    def _mouse_cb(self, event, x, y, flags, param) -> None:
        self._mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self._current_poly.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self._current_poly:
                self._current_poly.pop()

    # ── Rendering ──────────────────────────────────────────────────────────────

    def _render(self) -> np.ndarray:
        display = self._base_frame.copy()
        overlay = self._base_frame.copy()
        h, w    = display.shape[:2]
        mx, my  = self._mouse_pos

        # Crosshair
        cv2.line(display, (mx, 0), (mx, h), (180, 180, 180), 1)
        cv2.line(display, (0, my), (w, my), (180, 180, 180), 1)

        # Draw confirmed zones
        for zone_name in ZONE_NAMES:
            pts = self._zones[zone_name]
            if len(pts) < 3:
                continue
            color = ZONE_COLORS[zone_name]
            poly  = np.array(pts, dtype=np.int32)
            cv2.fillPoly(overlay, [poly], color)
            cv2.polylines(display, [poly], isClosed=True, color=color, thickness=2)
            for pt in pts:
                cv2.circle(display, tuple(pt), 4, color, -1)
            cx = int(sum(p[0] for p in pts) / len(pts))
            cy = int(sum(p[1] for p in pts) / len(pts))
            cv2.putText(display, zone_name.upper(), (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        cv2.addWeighted(overlay, ALPHA, display, 1 - ALPHA, 0, display)

        # Draw polygon currently being drawn
        color = ZONE_COLORS[self._active_zone]
        if self._current_poly:
            pts = self._current_poly
            for i, pt in enumerate(pts):
                cv2.circle(display, tuple(pt), 5, color, -1)
                if i > 0:
                    cv2.line(display, tuple(pts[i-1]), tuple(pt), color, 2)
            cv2.line(display, tuple(pts[-1]), (mx, my), color, 1)
            if len(pts) >= 2:
                cv2.line(display, (mx, my), tuple(pts[0]), color, 1)

        # HUD
        confirmed = sum(1 for z in ZONE_NAMES if len(self._zones[z]) >= 3)
        hud = (f"Section: {self.section_id}  |  "
               f"Active: {self._active_zone.upper()}  |  "
               f"Confirmed: {confirmed}/3  |  "
               f"1=NEAR 2=MID 3=FAR  ENTER=confirm ESC=clear S=save Q=quit Z=undo")
        cv2.rectangle(display, (0, h - 30), (w, h), (0, 0, 0), -1)
        cv2.putText(display, hud, (8, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return display

    # ── Zone management ────────────────────────────────────────────────────────

    def _confirm_zone(self) -> None:
        if len(self._current_poly) < 3:
            log.warning(f"Need ≥3 points. Have {len(self._current_poly)}.")
            return
        self._zones[self._active_zone] = [list(p) for p in self._current_poly]
        self._current_poly = []
        log.info(
            f"Zone {self._active_zone.upper()} confirmed "
            f"({len(self._zones[self._active_zone])} vertices)"
        )
        # Auto-advance to next zone
        idx = ZONE_NAMES.index(self._active_zone)
        if idx < len(ZONE_NAMES) - 1:
            self._active_zone = ZONE_NAMES[idx + 1]
            log.info(f"Auto-advanced to {self._active_zone.upper()}")

    def _undo_zone(self) -> None:
        self._zones[self._active_zone] = []
        log.info(f"Zone {self._active_zone.upper()} cleared.")

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load_existing(self) -> dict:
        if self.zone_path.exists():
            try:
                data = read_json(self.zone_path)
                log.info(f"Loaded existing zone config: {self.zone_path}")
                return data
            except Exception as e:
                log.warning(f"Could not load zone config: {e}")
        return {}

    def _save(self) -> None:
        self._data[self.section_id] = {
            z: self._zones[z] for z in ZONE_NAMES
        }
        write_json(self._data, self.zone_path)
        confirmed = sum(1 for z in ZONE_NAMES if len(self._zones[z]) >= 3)
        log.info(
            f"Saved zone config → {self.zone_path} "
            f"({confirmed}/3 zones confirmed for {self.section_id})"
        )

    @staticmethod
    def _print_instructions() -> None:
        print("\n" + "=" * 55)
        print("  ZONE ANNOTATOR — Controls")
        print("=" * 55)
        print("  1          → switch to NEAR zone (green)")
        print("  2          → switch to MID zone  (yellow)")
        print("  3          → switch to FAR zone  (red)")
        print("  Left click → add vertex")
        print("  Right click→ undo last vertex")
        print("  ENTER      → confirm current zone polygon")
        print("  ESC        → clear current drawing")
        print("  Z          → undo last confirmed zone")
        print("  S          → save progress")
        print("  Q          → save and quit")
        print("=" * 55)
        print("  Tip: zones auto-advance NEAR→MID→FAR after ENTER")
        print("=" * 55 + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw near/mid/far zone polygons for one library section."
    )
    parser.add_argument("--section", required=True,
                        help="Section ID (e.g. cad_lab)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  help="Path to still image")
    group.add_argument("--video",  help="Path to video file")
    group.add_argument("--camera", type=int, help="Webcam index")
    parser.add_argument("--zone-path",
                        default="data/roi/zone_config.json",
                        help="Output JSON path")
    args   = parser.parse_args()
    source = args.image or args.video or args.camera

    ann = ZoneAnnotator(
        section_id = args.section,
        source     = source,
        zone_path  = args.zone_path,
    )
    result = ann.run()

    print("\nZone summary:")
    for section, zones in result.items():
        print(f"  {section}:")
        for zone, pts in zones.items():
            status = f"{len(pts)} vertices" if len(pts) >= 3 else "⚠️  not confirmed"
            print(f"    {zone:<6} → {status}")


if __name__ == "__main__":
    main()