"""
tracking/smoother.py — Temporal smoother for occupancy states (CP-13)

Problem: Raw per-frame headcounts are noisy. A person briefly standing up,
a detection dropping for one frame, or a momentary occlusion can flip the
headcount and cause the vacancy display to flicker.

Solution:
  - N-frame majority vote on headcount (rounds to most common value
    in the last N frames)
  - EMA (exponential moving average) as an alternative — smoother
    transitions but slightly more lag

Usage:
    from tracking.smoother import TemporalSmoother, MultiSectionSmoother

    smoother = TemporalSmoother(section_id="cad_lab")
    smoothed = smoother.update(raw_result)
"""

import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import cfg
from detection.pipeline import SectionResult
from utils.logger import get_logger

log = get_logger("system")


class TemporalSmoother:
    """
    Smooths per-section headcount using N-frame majority vote.

    Args:
        section_id:  Section this smoother is attached to.
        window_size: Rolling buffer size. Default from config.yaml.
        ema_alpha:   EMA factor (0-1). Higher = more responsive.
                     Default from config.yaml.
    """

    def __init__(
        self,
        section_id:  str,
        window_size: int   | None = None,
        ema_alpha:   float | None = None,
    ) -> None:
        self.section_id  = section_id
        self.window_size = window_size or cfg.smoothing.window_size
        self.ema_alpha   = ema_alpha   or cfg.smoothing.ema_alpha

        self._buffer: deque[int] = deque(maxlen=self.window_size)
        self._ema:    float | None = None
        self._frame_count: int = 0

        log.info(
            f"TemporalSmoother | section={section_id} "
            f"window={self.window_size} ema_alpha={self.ema_alpha}"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, result: SectionResult) -> SectionResult:
        """
        Feed a raw SectionResult through the smoother.
        Returns a new SectionResult with smoothed headcount and vacancy.
        The original result is not modified.
        """
        self._frame_count += 1
        self._buffer.append(result.headcount)

        # Update EMA
        if self._ema is None:
            self._ema = float(result.headcount)
        else:
            self._ema = self.ema_alpha * result.headcount + (1 - self.ema_alpha) * self._ema

        # Use majority vote once buffer has enough data, else raw
        if len(self._buffer) >= max(1, self.window_size // 2):
            smoothed_headcount = _majority_vote(self._buffer)
        else:
            smoothed_headcount = result.headcount  # warmup — pass through raw

        smoothed_vacancy = (
            max(0, result.max_capacity - smoothed_headcount)
            if result.max_capacity is not None else None
        )

        return SectionResult(
            section_id   = result.section_id,
            timestamp    = result.timestamp,
            headcount    = smoothed_headcount,
            max_capacity = result.max_capacity,
            vacancy      = smoothed_vacancy,
            person_boxes = result.person_boxes,
            object_boxes = result.object_boxes,
            inference_ms = result.inference_ms,
            pipeline_ms  = result.pipeline_ms,
        )

    def reset(self) -> None:
        """Clear buffer — call after camera reconnect."""
        self._buffer.clear()
        self._ema         = None
        self._frame_count = 0
        log.info(f"[{self.section_id}] Smoother reset.")

    def buffer_fill(self) -> float:
        """Buffer fullness as 0.0–1.0. Reaches 1.0 after window_size frames."""
        return min(1.0, len(self._buffer) / self.window_size)

    @property
    def ema_headcount(self) -> float | None:
        """Current EMA value (float). None before first update."""
        return self._ema


class MultiSectionSmoother:
    """
    One TemporalSmoother per section, created on first use.

    Usage:
        ms = MultiSectionSmoother()
        smoothed = ms.update("cad_lab", raw_result)
    """

    def __init__(
        self,
        window_size: int   | None = None,
        ema_alpha:   float | None = None,
    ) -> None:
        self._smoothers:  dict[str, TemporalSmoother] = {}
        self._window_size = window_size
        self._ema_alpha   = ema_alpha

    def update(self, section_id: str, result: SectionResult) -> SectionResult:
        if section_id not in self._smoothers:
            self._smoothers[section_id] = TemporalSmoother(
                section_id  = section_id,
                window_size = self._window_size,
                ema_alpha   = self._ema_alpha,
            )
        return self._smoothers[section_id].update(result)

    def reset(self, section_id: str) -> None:
        if section_id in self._smoothers:
            self._smoothers[section_id].reset()

    def buffer_fills(self) -> dict[str, float]:
        return {sid: s.buffer_fill() for sid, s in self._smoothers.items()}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _majority_vote(buf: deque) -> int:
    """Return the most common integer value in the buffer."""
    counts: dict[int, int] = {}
    for v in buf:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get)