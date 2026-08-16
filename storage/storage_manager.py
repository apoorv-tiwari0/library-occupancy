"""
storage/storage_manager.py — Unified Storage Coordinator for Redis & PostgreSQL/TimescaleDB.

Combines real-time Redis state management with persistent PostgreSQL historical logging
for simple single-call integration across the pipeline and FastAPI server.

Usage:
    from storage.storage_manager import StorageManager
    storage = StorageManager()
    storage.save_update(section_result)
    live_dashboard = storage.get_live_dashboard()
    history = storage.get_history("cad_lab", start_time="2026-08-16T00:00:00", end_time="2026-08-16T23:59:59")
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.db_logger import DatabaseLogger
from storage.redis_store import RedisLiveStore
from utils.logger import get_logger

log = get_logger("system")


class StorageManager:
    """
    Unified Manager interfacing with Redis (live state) and Database (historical logs).
    """

    def __init__(
        self,
        redis_store: RedisLiveStore | None = None,
        db_logger: DatabaseLogger | None = None,
    ) -> None:
        self.redis_store = redis_store or RedisLiveStore()
        self.db_logger = db_logger or DatabaseLogger()

        log.info(
            f"StorageManager initialized | "
            f"redis_connected={self.redis_store.is_connected()} "
            f"db_connected={self.db_logger.is_connected()}"
        )

    # ── Write Operations ──────────────────────────────────────────────────────

    def save_update(self, result: Any) -> dict[str, bool]:
        """
        Persist a single section occupancy update to both Redis (live) and DB (history).

        Args:
            result: SectionResult instance or dict payload.

        Returns:
            Dict {'redis': bool, 'db': bool}
        """
        redis_ok = self.redis_store.set_section_state(result)
        db_ok = self.db_logger.log_state(result)
        return {"redis": redis_ok, "db": db_ok}

    def save_all_updates(self, results: dict[str, Any]) -> dict[str, Any]:
        """
        Persist updates for multiple sections.

        Args:
            results: Dict mapping section_id -> SectionResult or dict.

        Returns:
            Dict with write count summaries.
        """
        redis_ok = self.redis_store.set_all_states(results)
        db_count = self.db_logger.log_all_states(results)
        return {
            "redis_success": redis_ok,
            "db_inserted_count": db_count,
            "total_sections": len(results),
        }

    # ── Read Operations for FastAPI & Analytics ───────────────────────────────

    def get_live_state(self, section_id: str) -> dict[str, Any] | None:
        """
        Get real-time state for a section (from Redis with fallback to DB).

        Args:
            section_id: Section identifier (e.g., 'cad_lab')

        Returns:
            Section state payload dict or None.
        """
        state = self.redis_store.get_section_state(section_id)
        if state:
            return state

        # Fallback to DB recent history if Redis key missing/offline
        history = self.db_logger.get_recent_history(section_id=section_id, limit=1)
        if history:
            return history[0]
        return None

    def get_live_dashboard(self) -> dict[str, dict[str, Any]]:
        """
        Get live state for all active library sections for instant FastAPI response.

        Returns:
            Dict mapping section_id -> state dict.
        """
        states = self.redis_store.get_all_states()
        if states:
            return states

        # Fallback: construct from DB recent records for each section
        history = self.db_logger.get_recent_history(limit=100)
        latest_per_section: dict[str, dict[str, Any]] = {}
        for rec in history:
            sid = rec["section_id"]
            if sid not in latest_per_section:
                latest_per_section[sid] = rec

        return latest_per_section

    def get_history(
        self,
        section_id: str | None = None,
        start_time: datetime | str | None = None,
        end_time: datetime | str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """
        Query historical time-range data for analytics graphs.
        """
        return self.db_logger.query_time_range(
            section_id=section_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )
