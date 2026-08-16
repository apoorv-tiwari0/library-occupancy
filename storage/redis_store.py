"""
storage/redis_store.py — Redis Live State Store for sub-5ms Dashboard Reads.

Key Structure:
    occupancy:live:<section_id>  — JSON payload with current section occupancy metrics (TTL 300s)
    occupancy:live:all           — Redis Hash mapping section_id -> JSON payload
    occupancy:live:last_update   — Global ISO timestamp of last section update

Usage:
    from storage.redis_store import RedisLiveStore
    store = RedisLiveStore()
    store.set_section_state(section_result)
    state = store.get_section_state("cad_lab")
"""

# storage/redis_store.py
"""
storage/redis_store.py — Redis live state store (CP-17)

Stores latest SectionResult per section as JSON.
Key schema: occupancy:{section_id}
TTL: 30 seconds (auto-expires if pipeline stops)
"""

import json
import sys
from pathlib import Path

# pyrefly: ignore [missing-import]
import redis

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import cfg
from detection.pipeline import SectionResult
from utils.logger import get_logger

log = get_logger("system")


class RedisStore:
    """
    Writes and reads occupancy state to/from Redis.
    One key per section, auto-expires after TTL seconds.
    """

    def __init__(self) -> None:
        r_cfg = cfg.redis
        self._client = redis.Redis(
            host            = r_cfg.host,
            port            = r_cfg.port,
            db              = r_cfg.db,
            password        = r_cfg.password or None,
            decode_responses = True,
            socket_timeout  = 2.0,
            socket_connect_timeout = 2.0,
        )
        self._prefix = r_cfg.key_prefix   # "occupancy"
        self._ttl    = getattr(r_cfg, "ttl_seconds", getattr(r_cfg, "state_ttl_sec", 30))  # 300 or 30


        # Test connection
        try:
            self._client.ping()
            log.info(f"RedisStore connected | {r_cfg.host}:{r_cfg.port}")
        except Exception as e:
            log.error(f"RedisStore connection failed: {e}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def store_result(self, result: SectionResult) -> bool:
        """
        Store a SectionResult as JSON in Redis.
        Key: occupancy:{section_id}
        Expires after TTL seconds.
        """
        key  = f"{self._prefix}:{result.section_id}"
        data = json.dumps(result.to_dict())
        try:
            self._client.setex(key, self._ttl, data)
            return True
        except Exception as e:
            log.warning(f"Redis write failed [{result.section_id}]: {e}")
            return False

    def get_result(self, section_id: str) -> dict | None:
        """Get latest occupancy state for one section."""
        key = f"{self._prefix}:{section_id}"
        try:
            data = self._client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            log.warning(f"Redis read failed [{section_id}]: {e}")
            return None

    def get_all_sections(self) -> dict[str, dict]:
        """
        Get latest occupancy state for all sections.
        Returns {section_id: result_dict} for all live keys.
        """
        pattern = f"{self._prefix}:*"
        try:
            keys    = self._client.keys(pattern)
            if not keys:
                return {}
            values  = self._client.mget(keys)
            results = {}
            for key, val in zip(keys, values):
                if val:
                    section_id = key.split(":", 1)[1]
                    results[section_id] = json.loads(val)
            return results
        except Exception as e:
            log.warning(f"Redis get_all failed: {e}")
            return {}

    def delete_section(self, section_id: str) -> None:
        """Remove a section's key (e.g. when camera goes offline)."""
        key = f"{self._prefix}:{section_id}"
        try:
            self._client.delete(key)
        except Exception as e:
            log.warning(f"Redis delete failed [{section_id}]: {e}")

    def ping(self) -> bool:
        """Check if Redis is reachable."""
        try:
            return self._client.ping()
        except Exception:
            return False

    # ── Compatibility & Helper API ─────────────────────────────────────────────

    def is_connected(self) -> bool:
        """Check if Redis client is connected and responding."""
        return self.ping()

    def set_section_state(self, result) -> bool:
        """Store a SectionResult or dict payload to Redis."""
        if isinstance(result, dict):
            sid = result.get("section_id")
            if not sid:
                return False
            key = f"{self._prefix}:{sid}"
            try:
                self._client.setex(key, self._ttl, json.dumps(result))
                return True
            except Exception as e:
                log.warning(f"Redis write failed [{sid}]: {e}")
                return False
        return self.store_result(result)

    def get_section_state(self, section_id: str) -> dict | None:
        """Alias for get_result."""
        return self.get_result(section_id)

    def get_all_states(self) -> dict[str, dict]:
        """Alias for get_all_sections."""
        return self.get_all_sections()

    def set_all_states(self, results: dict) -> bool:
        """Store multiple section results to Redis."""
        success = True
        for res in results.values():
            if not self.set_section_state(res):
                success = False
        return success

    def clear_all(self) -> None:
        """Clear all active section keys."""
        pattern = f"{self._prefix}:*"
        try:
            keys = self._client.keys(pattern)
            if keys:
                self._client.delete(*keys)
        except Exception as e:
            log.warning(f"Redis clear_all failed: {e}")


RedisLiveStore = RedisStore