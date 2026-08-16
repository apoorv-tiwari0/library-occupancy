"""
tests/test_storage.py — Unit and Integration Tests for Storage Subsystem (CP-15 / Storage)

Tests:
    1. RedisLiveStore: live state write, instant read, hash map, and TTL
    2. DatabaseLogger: schema creation, hypertable initialization, record insertion
    3. Time-Range Queries: filtering by section_id and timestamp ranges
    4. StorageManager: unified interface for pipeline SectionResult integration

Run:
    python tests/test_storage.py
"""

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))


from config.config_loader import cfg
from detection.pipeline import SectionResult
from storage.db_logger import DatabaseLogger
from storage.redis_store import RedisLiveStore
from storage.storage_manager import StorageManager


def make_dummy_result(section_id: str = "cad_lab", headcount: int = 6, max_cap: int = 18) -> SectionResult:
    vacancy = max(0, max_cap - headcount) if max_cap else None
    return SectionResult(
        section_id=section_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        headcount=headcount,
        max_capacity=max_cap,
        vacancy=vacancy,
        person_boxes=[[10, 10, 50, 100, 0.9]],
        object_boxes=[],
        inference_ms=120.5,
        pipeline_ms=150.2,
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_redis_store() -> None:
    print("\n[1] Testing RedisLiveStore...")
    redis_store = RedisLiveStore()

    if not redis_store.is_connected():
        print("  ⚠️ Redis server is offline (tested fallback/ping cleanly). Skipping live Redis key assertions.")
        return

    # Clear previous test keys
    redis_store.clear_all()

    res1 = make_dummy_result("cad_lab", headcount=6, max_cap=18)
    res2 = make_dummy_result("g_hall_2", headcount=30, max_cap=92)

    assert redis_store.set_section_state(res1) is True
    assert redis_store.set_section_state(res2) is True
    print("  ✅ set_section_state written for cad_lab and g_hall_2")

    # Read back single state
    state = redis_store.get_section_state("cad_lab")
    assert state is not None
    assert state["section_id"] == "cad_lab"
    assert state["headcount"] == 6
    assert state["vacancy"] == 12
    assert state["occupancy_pct"] == 33.3
    assert state["is_available"] is True
    print(f"  ✅ get_section_state read back correctly: headcount={state['headcount']} vacancy={state['vacancy']}")

    # Read back all states
    all_states = redis_store.get_all_states()
    assert "cad_lab" in all_states
    assert "g_hall_2" in all_states
    assert len(all_states) >= 2
    print(f"  ✅ get_all_states retrieved {len(all_states)} active sections for instant dashboard")


def test_database_logger() -> None:
    print("\n[2] Testing DatabaseLogger & Time-Range Queries...")
    db_logger = DatabaseLogger()
    assert db_logger.is_connected() is True
    print(f"  ✅ DatabaseLogger connected using backend: {db_logger.backend_name}")

    now = datetime.now(timezone.utc)
    t1 = (now - timedelta(minutes=30)).isoformat()
    t2 = (now - timedelta(minutes=15)).isoformat()
    t3 = now.isoformat()

    # Log 3 records across time
    r1 = make_dummy_result("focused_reading_area", headcount=10, max_cap=27)
    r1.timestamp = t1

    r2 = make_dummy_result("focused_reading_area", headcount=15, max_cap=27)
    r2.timestamp = t2

    r3 = make_dummy_result("g_huss", headcount=30, max_cap=30)
    r3.timestamp = t3

    assert db_logger.log_state(r1) is True
    assert db_logger.log_state(r2) is True
    assert db_logger.log_state(r3) is True
    print("  ✅ Logged 3 historical records to database")

    # Query all history
    recent = db_logger.get_recent_history(limit=10)
    assert len(recent) >= 3
    print(f"  ✅ get_recent_history returned {len(recent)} records")

    # Query specific section
    focused_history = db_logger.query_time_range(section_id="focused_reading_area")
    assert len(focused_history) >= 2
    for rec in focused_history:
        assert rec["section_id"] == "focused_reading_area"
    print(f"  ✅ Section filter query returned {len(focused_history)} records for 'focused_reading_area'")

    # Query specific time range
    start_filter = (now - timedelta(minutes=40)).isoformat()
    end_filter = (now - timedelta(minutes=20)).isoformat()
    filtered_time = db_logger.query_time_range(
        section_id="focused_reading_area",
        start_time=start_filter,
        end_time=end_filter,
    )
    assert len(filtered_time) >= 1
    print(f"  ✅ Time-range filter query ({start_filter[:16]} to {end_filter[:16]}) returned {len(filtered_time)} matching records")


def test_storage_manager() -> None:
    print("\n[3] Testing StorageManager Unified Coordinator...")
    manager = StorageManager()

    # Create dummy results for 3 sections
    res_map = {
        "ipc": make_dummy_result("ipc", headcount=9, max_cap=16),
        "reference_2": make_dummy_result("reference_2", headcount=11, max_cap=24),
        "weeding_out_area": make_dummy_result("weeding_out_area", headcount=1, max_cap=12),
    }

    # Save updates
    save_status = manager.save_all_updates(res_map)
    assert save_status["total_sections"] == 3
    assert save_status["db_inserted_count"] == 3
    print(f"  ✅ save_all_updates persisted {save_status['total_sections']} sections")

    # Fetch live dashboard
    dashboard = manager.get_live_dashboard()
    assert len(dashboard) >= 3
    print(f"  ✅ get_live_dashboard() returned {len(dashboard)} section states ready for FastAPI")

    # Fetch single section live state
    ipc_state = manager.get_live_state("ipc")
    assert ipc_state is not None
    assert ipc_state["section_id"] == "ipc"
    assert ipc_state["headcount"] == 9
    assert ipc_state["vacancy"] == 7
    print(f"  ✅ get_live_state('ipc') -> headcount={ipc_state['headcount']} vacancy={ipc_state['vacancy']}")

    # Fetch history via manager
    history = manager.get_history(section_id="ipc")
    assert len(history) >= 1
    assert history[0]["section_id"] == "ipc"
    print(f"  ✅ get_history('ipc') -> retrieved {len(history)} records")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("CP-15 / Storage — Redis & Database Subsystem Test")
    print("=" * 70)

    test_redis_store()
    test_database_logger()
    test_storage_manager()

    print("\n" + "=" * 70)
    print("🎉 CP-15 PASSED — Storage Subsystem & Time-Range Queries Ready")
    print("=" * 70)
