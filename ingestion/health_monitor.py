"""
ingestion/health_monitor.py — Stream health monitor & auto-reconnect (CP-07)

Watches all active FrameExtractor threads. If a camera goes offline or stops
delivering frames, the monitor:
  1. Logs an alert with camera_id, section_id, and timestamp
  2. Waits for the configured reconnect timeout
  3. Spawns a fresh FrameExtractor to replace the dead one
  4. Notifies the StreamManager to rewire the new extractor's queue

This runs as a single background daemon thread inside StreamManager.

Usage (via StreamManager — not called directly):
    monitor = HealthMonitor(extractors, raw_queues, stop_event, cfg)
    monitor.start()
    monitor.stop()
"""

import queue
import threading
import time
from typing import Callable

from config.config_loader import cfg
from ingestion.frame_extractor import FrameExtractor, FramePacket
from utils.logger import get_logger

log = get_logger("system")


class CameraStatus:
    """Tracks the health state of a single camera."""
    ONLINE  = "online"
    OFFLINE = "offline"
    RECONNECTING = "reconnecting"


class HealthMonitor:
    """
    Watchdog that monitors extractor threads and auto-reconnects dead cameras.

    Args:
        extractors:      Dict of {section_id: FrameExtractor} — shared with StreamManager.
        raw_queues:      Dict of {section_id: queue.Queue} — raw frame queues.
        stop_event:      Threading event — set by StreamManager.stop() to shut down.
        on_reconnect:    Optional callback(section_id) fired after successful reconnect.
        check_interval:  Seconds between health checks (default from config).
        reconnect_timeout: Seconds to wait before attempting reconnect (default from config).
    """

    def __init__(
        self,
        extractors:        dict[str, FrameExtractor],
        raw_queues:        dict[str, queue.Queue],
        stop_event:        threading.Event,
        on_reconnect:      Callable[[str], None] | None = None,
        check_interval:    float | None = None,
        reconnect_timeout: float | None = None,
    ) -> None:
        self._extractors        = extractors
        self._raw_queues        = raw_queues
        self._stop_event        = stop_event
        self._on_reconnect      = on_reconnect
        self._check_interval    = check_interval    or cfg.ingestion.health_check_interval_sec
        self._reconnect_timeout = reconnect_timeout or cfg.ingestion.reconnect_timeout_sec

        # Track status and last-seen frame count per section
        self._status:      dict[str, str] = {
            sid: CameraStatus.ONLINE for sid in extractors
        }
        self._last_counts: dict[str, int] = {sid: 0 for sid in extractors}
        self._frame_counts: dict[str, int] = {sid: 0 for sid in extractors}

        self._thread: threading.Thread | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the watchdog thread."""
        self._thread = threading.Thread(
            target=self._run,
            name="health-monitor",
            daemon=True,
        )
        self._thread.start()
        log.info(
            f"HealthMonitor started | "
            f"check_interval={self._check_interval}s "
            f"reconnect_timeout={self._reconnect_timeout}s "
            f"watching={list(self._extractors.keys())}"
        )

    def stop(self) -> None:
        """Stop the watchdog thread (stop_event must already be set)."""
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self._check_interval + 2)
        log.info("HealthMonitor stopped.")

    def record_frame(self, section_id: str) -> None:
        """
        Called by the preprocessor worker each time a frame is successfully
        processed. Used to detect stalled-but-alive streams (thread running
        but no new frames arriving).
        """
        self._frame_counts[section_id] = self._frame_counts.get(section_id, 0) + 1

    def get_status(self) -> dict[str, str]:
        """Return a snapshot of current camera statuses."""
        return dict(self._status)

    # ── Internal watchdog loop ─────────────────────────────────────────────────

    def _run(self) -> None:
        """
        Main watchdog loop. Runs every check_interval seconds.

        Two failure modes detected:
          1. Thread dead — extractor thread has exited (stream disconnected)
          2. Frame stall — thread alive but no new frames in last interval
             (stream frozen, RTSP timeout, or decoder hung)
        """
        while not self._stop_event.is_set():
            time.sleep(self._check_interval)

            for section_id, extractor in list(self._extractors.items()):
                if self._stop_event.is_set():
                    break

                thread_alive  = extractor.is_running()
                current_count = self._frame_counts.get(section_id, 0)
                last_count    = self._last_counts.get(section_id, 0)
                frames_arrived = current_count > last_count

                self._last_counts[section_id] = current_count

                if thread_alive and frames_arrived:
                    # Healthy
                    if self._status[section_id] != CameraStatus.ONLINE:
                        log.info(
                            f"[{section_id}] Camera back ONLINE "
                            f"(frames flowing again)"
                        )
                    self._status[section_id] = CameraStatus.ONLINE

                elif thread_alive and not frames_arrived:
                    # Thread alive but no frames — possible stall
                    log.warning(
                        f"[{section_id}] ⚠️  Stream may be stalled — "
                        f"thread alive but no new frames in last "
                        f"{self._check_interval}s. Monitoring..."
                    )
                    # Don't reconnect immediately — wait one more cycle
                    # (brief pauses are normal during scene transitions)

                else:
                    # Thread dead — camera went offline
                    if self._status[section_id] != CameraStatus.OFFLINE:
                        log.error(
                            f"[{section_id}] ❌ Camera OFFLINE — "
                            f"extractor thread has exited. "
                            f"Attempting reconnect in {self._reconnect_timeout}s..."
                        )
                        self._status[section_id] = CameraStatus.OFFLINE

                    self._attempt_reconnect(section_id, extractor)

    def _attempt_reconnect(
        self,
        section_id: str,
        old_extractor: FrameExtractor,
    ) -> None:
        """
        Wait reconnect_timeout seconds then spawn a fresh FrameExtractor
        reusing the same raw queue and stream URL.
        """
        self._status[section_id] = CameraStatus.RECONNECTING
        log.info(
            f"[{section_id}] Waiting {self._reconnect_timeout}s before reconnect..."
        )

        # Sleep in small increments so we can react to stop_event quickly
        waited = 0.0
        while waited < self._reconnect_timeout:
            if self._stop_event.is_set():
                return
            time.sleep(0.5)
            waited += 0.5

        # Build a new extractor with the same config
        new_extractor = FrameExtractor(
            camera_id    = old_extractor.camera_id,
            section_id   = section_id,
            stream_url   = old_extractor.stream_url,
            output_queue = self._raw_queues[section_id],
            sample_rate  = old_extractor.sample_rate,
        )

        new_extractor.start()
        self._extractors[section_id] = new_extractor
        self._frame_counts[section_id] = 0
        self._last_counts[section_id]  = 0
        self._status[section_id] = CameraStatus.ONLINE

        log.info(
            f"[{section_id}] ✅ Reconnected — new extractor started "
            f"(camera={old_extractor.camera_id})"
        )

        if self._on_reconnect:
            try:
                self._on_reconnect(section_id)
            except Exception as e:
                log.error(f"[{section_id}] on_reconnect callback failed: {e}")