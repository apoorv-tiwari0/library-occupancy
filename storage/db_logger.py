# storage/db_logger.py
"""
storage/db_logger.py — PostgreSQL + TimescaleDB occupancy logger (CP-18)

Logs every SectionResult state change to the occupancy_log hypertable.
Uses a connection pool (SQLAlchemy) — engine created once, reused.
Only logs when headcount changes (avoids redundant identical rows).
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# pyrefly: ignore [missing-import]
from sqlalchemy import create_engine, text
# pyrefly: ignore [missing-import]
from sqlalchemy.pool import QueuePool

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import cfg
from detection.pipeline import SectionResult
from utils.logger import get_logger

log = get_logger("system")


class DBLogger:
    """
    Logs occupancy state changes to PostgreSQL + TimescaleDB.

    Uses SQLAlchemy connection pool for efficient reuse.
    Only writes a new row when headcount changes from previous value
    to avoid flooding the DB with identical consecutive states.

    Args:
        log_all: If True, log every frame regardless of change.
                 If False (default), only log on headcount change.
    """

    def __init__(self, log_all: bool = False) -> None:
        db            = cfg.database
        self._log_all = log_all

        # Build connection URL
        url = (
            f"postgresql+psycopg2://"
            f"{db.user}:{db.password}"
            f"@{db.host}:{db.port}"
            f"/{db.name}"
        )

        self._engine = create_engine(
            url,
            poolclass    = QueuePool,
            pool_size    = db.pool_size,
            max_overflow = db.max_overflow,
            pool_pre_ping = True,   # check connection health before use
        )

        # Track last headcount per section to detect changes
        self._last_headcount: dict[str, int] = {}

        # Test connection
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            log.info(
                f"DBLogger connected | "
                f"{db.host}:{db.port}/{db.name}"
            )
        except Exception as e:
            log.error(f"DBLogger connection failed: {e}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def log_result(self, result: SectionResult) -> bool:
        """
        Log a SectionResult to occupancy_log.

        Skips write if headcount hasn't changed since last log
        (unless log_all=True).

        Returns True if a row was written, False if skipped or failed.
        """
        sid          = result.section_id
        last         = self._last_headcount.get(sid)
        headcount_changed = (last is None or last != result.headcount)

        if not self._log_all and not headcount_changed:
            return False   # no change — skip write

        self._last_headcount[sid] = result.headcount

        try:
            with self._engine.begin() as conn:
                conn.execute(
                    text("""
                        INSERT INTO occupancy_log (
                            time, section_id, headcount, max_capacity,
                            vacancy, occupancy_pct, is_available,
                            inference_ms, pipeline_ms
                        ) VALUES (
                            CAST(:time AS TIMESTAMPTZ),
                            :section_id, :headcount, :max_capacity,
                            :vacancy, :occupancy_pct, :is_available,
                            :inference_ms, :pipeline_ms
                        )

                    """),
                    {
                        "time":         result.timestamp,
                        "section_id":   result.section_id,
                        "headcount":    result.headcount,
                        "max_capacity": result.max_capacity,
                        "vacancy":      result.vacancy,
                        "occupancy_pct": result.occupancy_pct(),
                        "is_available": result.is_available(),
                        "inference_ms": round(result.inference_ms, 1),
                        "pipeline_ms":  round(result.pipeline_ms, 1),
                    }
                )
            return True
        except Exception as e:
            log.error(f"DBLogger write failed [{sid}]: {e}")
            return False

    def get_latest(self, section_id: str) -> dict | None:
        """Get the most recent log entry for a section."""
        try:
            with self._engine.connect() as conn:
                row = conn.execute(
                    text("""
                        SELECT time, headcount, max_capacity, vacancy,
                               occupancy_pct, is_available
                        FROM   occupancy_log
                        WHERE  section_id = :sid
                        ORDER  BY time DESC
                        LIMIT  1
                    """),
                    {"sid": section_id}
                ).fetchone()
            if row is None:
                return None
            return dict(row._mapping)
        except Exception as e:
            log.warning(f"DBLogger get_latest failed [{section_id}]: {e}")
            return None

    def get_history(
        self,
        section_id: str,
        hours:      int = 24,
    ) -> list[dict]:
        """Get occupancy history for the last N hours."""
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    text("""
                        SELECT time, headcount, vacancy, occupancy_pct
                        FROM   occupancy_log
                        WHERE  section_id = :sid
                          AND  time > NOW() - INTERVAL ':hours hours'
                        ORDER  BY time ASC
                    """),
                    {"sid": section_id, "hours": hours}
                ).fetchall()
            return [dict(r._mapping) for r in rows]
        except Exception as e:
            log.warning(f"DBLogger get_history failed: {e}")
            return []

    def close(self) -> None:
        """Dispose connection pool cleanly."""
        self._engine.dispose()
        log.info("DBLogger connection pool closed.")

    # ── Compatibility & Helper API ─────────────────────────────────────────────

    def is_connected(self) -> bool:
        """Check if database connection pool is functional."""
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    @property
    def backend_name(self) -> str:
        return "PostgreSQL + TimescaleDB"

    def log_state(self, result) -> bool:
        """Alias for log_result."""
        return self.log_result(result)

    def log_all_states(self, results: dict | list) -> int:
        """Log state for multiple sections."""
        count = 0
        items = results.values() if isinstance(results, dict) else results
        for item in items:
            if self.log_result(item):
                count += 1
        return count

    def get_recent_history(
        self,
        section_id: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Fetch recent occupancy records."""
        try:
            with self._engine.connect() as conn:
                if section_id:
                    rows = conn.execute(
                        text("""
                            SELECT time, section_id, headcount, max_capacity, vacancy, occupancy_pct, is_available
                            FROM   occupancy_log
                            WHERE  section_id = :sid
                            ORDER  BY time DESC
                            LIMIT  :limit
                        """),
                        {"sid": section_id, "limit": limit}
                    ).fetchall()
                else:
                    rows = conn.execute(
                        text("""
                            SELECT time, section_id, headcount, max_capacity, vacancy, occupancy_pct, is_available
                            FROM   occupancy_log
                            ORDER  BY time DESC
                            LIMIT  :limit
                        """),
                        {"limit": limit}
                    ).fetchall()
            return [dict(r._mapping) for r in rows]
        except Exception as e:
            log.warning(f"DBLogger get_recent_history failed: {e}")
            return []

    def query_time_range(
        self,
        section_id: str | None = None,
        start_time: datetime | str | None = None,
        end_time: datetime | str | None = None,
        limit: int = 500,
    ) -> list[dict]:
        """Query occupancy history across section and timestamp criteria."""
        try:
            conditions = []
            params: dict = {"limit": limit}

            if section_id:
                conditions.append("section_id = :sid")
                params["sid"] = section_id

            if start_time:
                conditions.append("time >= CAST(:start_time AS TIMESTAMPTZ)")
                params["start_time"] = str(start_time)

            if end_time:
                conditions.append("time <= CAST(:end_time AS TIMESTAMPTZ)")
                params["end_time"] = str(end_time)


            where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
            query_str = f"""
                SELECT time, section_id, headcount, max_capacity, vacancy, occupancy_pct, is_available
                FROM   occupancy_log
                {where_clause}
                ORDER  BY time DESC
                LIMIT  :limit
            """
            with self._engine.connect() as conn:
                rows = conn.execute(text(query_str), params).fetchall()
            return [dict(r._mapping) for r in rows]
        except Exception as e:
            log.warning(f"DBLogger query_time_range failed: {e}")
            return []


DatabaseLogger = DBLogger