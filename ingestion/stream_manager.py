"""
ingestion/stream_manager.py — Multi-camera parallel ingestion manager (CP-06/07)
Now includes HealthMonitor integration for auto-reconnect (CP-07).
"""

import queue
import threading
import time
from typing import Optional

from config.config_loader import cfg
from ingestion.frame_extractor import FrameExtractor, FramePacket
from ingestion.preprocessor import Preprocessor
from ingestion.health_monitor import HealthMonitor
from utils.logger import get_logger

log = get_logger("system")


class StreamManager:
    """
    Manages parallel ingestion from all configured cameras.
    Includes automatic stream health monitoring and reconnection (CP-07).
    """

    def __init__(self, preprocess: bool = True) -> None:
        self._preprocess = preprocess

        self._raw_queues:     dict[str, queue.Queue] = {}
        self._section_queues: dict[str, queue.Queue] = {}
        self._merged_queue: queue.Queue = queue.Queue(
            maxsize=cfg.ingestion.frame_queue_maxsize * max(len(cfg.cameras), 1)
        )

        self._extractors:    dict[str, FrameExtractor]  = {}
        self._prep_threads:  dict[str, threading.Thread] = {}
        self._stop_event = threading.Event()
        self._health_monitor: HealthMonitor | None = None

        self._build_extractors()

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start all extractor threads, preprocessor workers, and health monitor."""
        self._stop_event.clear()

        for section_id, extractor in self._extractors.items():
            extractor.start()
            preprocessor = Preprocessor() if self._preprocess else None
            t = threading.Thread(
                target=self._preprocess_worker,
                args=(section_id, preprocessor),
                name=f"prep-{section_id}",
                daemon=True,
            )
            t.start()
            self._prep_threads[section_id] = t

        # Start health monitor
        self._health_monitor = HealthMonitor(
            extractors        = self._extractors,
            raw_queues        = self._raw_queues,
            stop_event        = self._stop_event,
            on_reconnect      = self._on_camera_reconnect,
        )
        self._health_monitor.start()

        log.info(
            f"StreamManager started | "
            f"cameras={len(self._extractors)} "
            f"sections={list(self._extractors.keys())} "
            f"preprocessing={'ON' if self._preprocess else 'OFF'}"
        )

    def stop(self) -> None:
        """Stop all threads cleanly."""
        self._stop_event.set()
        for extractor in self._extractors.values():
            extractor.stop()
        if self._health_monitor:
            self._health_monitor.stop()
        for t in self._prep_threads.values():
            t.join(timeout=5)
        log.info("StreamManager stopped — all camera threads shut down.")

    def get_frame(
        self,
        section_id: Optional[str] = None,
        timeout: float = 1.0,
    ) -> Optional[FramePacket]:
        target_queue = (
            self._section_queues[section_id]
            if section_id
            else self._merged_queue
        )
        try:
            return target_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def active_sections(self) -> list[str]:
        return [
            sid for sid, ext in self._extractors.items()
            if ext.is_running()
        ]

    def queue_sizes(self) -> dict[str, int]:
        return {
            sid: self._section_queues[sid].qsize()
            for sid in self._section_queues
        }

    def camera_statuses(self) -> dict[str, str]:
        """Return health status of all cameras (online/offline/reconnecting)."""
        if self._health_monitor:
            return self._health_monitor.get_status()
        return {}

    # ── Internal ───────────────────────────────────────────────────────────────

    def _build_extractors(self) -> None:
        for cam in cfg.cameras:
            if not cam.enabled:
                continue
            maxsize = cfg.ingestion.frame_queue_maxsize
            raw_q  = queue.Queue(maxsize=maxsize)
            proc_q = queue.Queue(maxsize=maxsize)
            self._raw_queues[cam.section_id]     = raw_q
            self._section_queues[cam.section_id] = proc_q
            self._extractors[cam.section_id] = FrameExtractor(
                camera_id    = cam.camera_id,
                section_id   = cam.section_id,
                stream_url   = cam.stream_url,
                output_queue = raw_q,
            )
        log.info(
            f"Built {len(self._extractors)} extractor(s): "
            f"{list(self._extractors.keys())}"
        )

    def _preprocess_worker(
        self,
        section_id:   str,
        preprocessor: Optional[Preprocessor],
    ) -> None:
        raw_q   = self._raw_queues[section_id]
        proc_q  = self._section_queues[section_id]
        sec_log = get_logger(section_id)
        sec_log.info(f"[{section_id}] Preprocessor worker started.")

        while not self._stop_event.is_set():
            try:
                packet: FramePacket = raw_q.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                if preprocessor is not None:
                    packet.frame = preprocessor.process(packet.frame)
            except Exception as e:
                sec_log.error(
                    f"[{section_id}] Preprocessing failed on "
                    f"frame {packet.frame_no}: {e}"
                )
                continue

            # Notify health monitor a frame was successfully processed
            if self._health_monitor:
                self._health_monitor.record_frame(section_id)

            try:
                proc_q.put_nowait(packet)
            except queue.Full:
                sec_log.warning(
                    f"[{section_id}] Section queue full — "
                    f"dropping frame {packet.frame_no}"
                )

            try:
                self._merged_queue.put_nowait(packet)
            except queue.Full:
                pass

        sec_log.info(f"[{section_id}] Preprocessor worker stopped.")

    def _on_camera_reconnect(self, section_id: str) -> None:
        """Called by HealthMonitor after a successful reconnect."""
        log.info(
            f"[{section_id}] StreamManager notified of reconnect — "
            f"pipeline resuming normally."
        )