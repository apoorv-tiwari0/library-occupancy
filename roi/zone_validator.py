"""
roi/zone_validator.py — Zone config validator (ZCP-02)

Validates zone_config.json against:
  1. All expected sections present
  2. All 3 zones (near/mid/far) present per section
  3. Each zone has >= 3 vertices
  4. Each vertex is a valid [x, y] pair of numbers
  5. Zone polygons are non-degenerate (non-zero area)
  6. Zones are consistent with frame dimensions (no out-of-bounds coords)
     when a sample frame is provided

Usage:
    from roi.zone_validator import ZoneValidator
    validator = ZoneValidator("data/roi/zone_config.json")
    report = validator.validate()
    print(report.summary())
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import cfg
from utils.helpers import read_json
from utils.logger import get_logger

log = get_logger("system")

ZONE_NAMES = ["near", "mid", "far"]
MIN_VERTICES = 3
MIN_AREA = 100.0   # pixels^2 — below this a polygon is degenerate


@dataclass
class ZoneIssue:
    section_id: str
    zone:       str
    severity:   str   # "error" or "warning"
    message:    str


@dataclass
class ValidationReport:
    sections_found:   list[str]    = field(default_factory=list)
    sections_missing: list[str]    = field(default_factory=list)
    issues:           list[ZoneIssue] = field(default_factory=list)
    total_zones:      int          = 0
    valid_zones:      int          = 0

    def errors(self) -> list[ZoneIssue]:
        return [i for i in self.issues if i.severity == "error"]

    def warnings(self) -> list[ZoneIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def passed(self) -> bool:
        return len(self.errors()) == 0 and len(self.sections_missing) == 0

    def summary(self) -> str:
        lines = [
            f"Sections found   : {len(self.sections_found)}/11",
            f"Sections missing : {len(self.sections_missing)}",
            f"Zones valid      : {self.valid_zones}/{self.total_zones}",
            f"Errors           : {len(self.errors())}",
            f"Warnings         : {len(self.warnings())}",
            f"Result           : {'✅ PASSED' if self.passed() else '❌ FAILED'}",
        ]
        if self.sections_missing:
            lines.append(f"Missing sections : {self.sections_missing}")
        for issue in self.issues:
            tag = "❌" if issue.severity == "error" else "⚠️ "
            lines.append(f"  {tag} [{issue.section_id}][{issue.zone}] {issue.message}")
        return "\n".join(lines)


class ZoneValidator:
    """
    Validates zone_config.json for completeness and correctness.

    Args:
        zone_path:     Path to zone_config.json.
        sample_frames: Optional dict {section_id: frame_path} for
                       bounds checking against actual frame dimensions.
    """

    def __init__(
        self,
        zone_path:     str | Path,
        sample_frames: dict[str, str] | None = None,
    ) -> None:
        self.zone_path     = Path(zone_path)
        self.sample_frames = sample_frames or {}
        self._expected     = [c.section_id for c in cfg.cameras if c.enabled]

    def validate(self) -> ValidationReport:
        report = ValidationReport()

        # Load file
        if not self.zone_path.exists():
            log.error(f"Zone config not found: {self.zone_path}")
            report.sections_missing = list(self._expected)
            return report

        try:
            data = read_json(self.zone_path)
        except Exception as e:
            log.error(f"Cannot parse zone config: {e}")
            report.sections_missing = list(self._expected)
            return report

        # Check which sections are present
        report.sections_found   = [s for s in self._expected if s in data]
        report.sections_missing = [s for s in self._expected if s not in data]

        if report.sections_missing:
            log.warning(f"Missing sections: {report.sections_missing}")

        # Validate each section
        for section_id in report.sections_found:
            section_data = data[section_id]
            frame_shape  = self._get_frame_shape(section_id)

            for zone in ZONE_NAMES:
                report.total_zones += 1

                # Zone present?
                if zone not in section_data:
                    report.issues.append(ZoneIssue(
                        section_id, zone, "error",
                        f"Zone '{zone}' missing from section"
                    ))
                    continue

                pts = section_data[zone]

                # Min vertices
                if len(pts) < MIN_VERTICES:
                    report.issues.append(ZoneIssue(
                        section_id, zone, "error",
                        f"Only {len(pts)} vertices (need ≥{MIN_VERTICES})"
                    ))
                    continue

                # Valid [x,y] pairs
                valid_pts = True
                for i, pt in enumerate(pts):
                    if not (isinstance(pt, list) and len(pt) == 2):
                        report.issues.append(ZoneIssue(
                            section_id, zone, "error",
                            f"Vertex {i} is not a valid [x,y] pair: {pt}"
                        ))
                        valid_pts = False
                        break
                    if not all(isinstance(v, (int, float)) for v in pt):
                        report.issues.append(ZoneIssue(
                            section_id, zone, "error",
                            f"Vertex {i} contains non-numeric value: {pt}"
                        ))
                        valid_pts = False
                        break

                if not valid_pts:
                    continue

                poly = np.array(pts, dtype=np.float32)

                # Non-degenerate area
                area = float(cv2.contourArea(poly))
                if area < MIN_AREA:
                    report.issues.append(ZoneIssue(
                        section_id, zone, "error",
                        f"Zone area={area:.0f}px² is too small (min={MIN_AREA})"
                    ))
                    continue

                # Bounds check against frame dimensions
                if frame_shape is not None:
                    h, w = frame_shape[:2]
                    x_coords = [p[0] for p in pts]
                    y_coords = [p[1] for p in pts]
                    if min(x_coords) < 0 or max(x_coords) > w:
                        report.issues.append(ZoneIssue(
                            section_id, zone, "warning",
                            f"X coords [{min(x_coords):.0f}–{max(x_coords):.0f}] "
                            f"outside frame width {w}"
                        ))
                    if min(y_coords) < 0 or max(y_coords) > h:
                        report.issues.append(ZoneIssue(
                            section_id, zone, "warning",
                            f"Y coords [{min(y_coords):.0f}–{max(y_coords):.0f}] "
                            f"outside frame height {h}"
                        ))

                report.valid_zones += 1

        log.info(
            f"Validation complete | "
            f"sections={len(report.sections_found)}/11 "
            f"zones={report.valid_zones}/{report.total_zones} "
            f"errors={len(report.errors())} "
            f"warnings={len(report.warnings())}"
        )
        return report

    def _get_frame_shape(self, section_id: str):
        """Load frame shape for bounds checking. Returns None if not available."""
        if section_id not in self.sample_frames:
            return None
        path = self.sample_frames[section_id]
        frame = cv2.imread(str(path))
        return frame.shape if frame is not None else None

    def visualize(
        self,
        section_id: str,
        frame:      np.ndarray,
        out_path:   str | Path | None = None,
    ) -> np.ndarray:
        """
        Draw zone polygons on a frame for visual inspection.
        Green=near, Yellow=mid, Red=far.
        """
        COLORS = {"near": (0,255,0), "mid": (0,215,255), "far": (0,0,255)}
        ALPHA  = 0.25

        try:
            data = read_json(self.zone_path)
        except Exception:
            return frame

        if section_id not in data:
            return frame

        vis     = frame.copy()
        overlay = frame.copy()

        for zone, color in COLORS.items():
            pts = data[section_id].get(zone, [])
            if len(pts) < 3:
                continue
            poly = np.array(pts, dtype=np.int32)
            cv2.fillPoly(overlay, [poly], color)
            cv2.polylines(vis, [poly], isClosed=True, color=color, thickness=2)
            cx = int(sum(p[0] for p in pts) / len(pts))
            cy = int(sum(p[1] for p in pts) / len(pts))
            cv2.putText(vis, zone.upper(), (cx-20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.addWeighted(overlay, ALPHA, vis, 1 - ALPHA, 0, vis)

        if out_path:
            cv2.imwrite(str(out_path), vis)
            log.info(f"Zone visualization saved → {out_path}")

        return vis