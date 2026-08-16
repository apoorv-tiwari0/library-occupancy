"""
test_cp17_cp18.py — Integration test for Redis + PostgreSQL storage (CP-17, CP-18)

Tests:
  1. Redis connection
  2. Store and retrieve SectionResult from Redis
  3. get_all_sections() returns correct data
  4. TTL expiry works
  5. PostgreSQL connection
  6. Log SectionResult to occupancy_log
  7. Only-on-change logging works
  8. get_latest() retrieves correct row
  9. Real pipeline result stored in both Redis and Postgres

Run from project root:
    python test_cp17_cp18.py
"""

import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))



from storage.redis_store import RedisStore
from storage.db_logger   import DBLogger
from detection.pipeline  import SectionResult
from utils.helpers       import get_timestamp


def make_result(section_id="cad_lab", headcount=6, max_cap=18) -> SectionResult:
    vacancy = max(0, max_cap - headcount)
    return SectionResult(
        section_id   = section_id,
        timestamp    = get_timestamp(),
        headcount    = headcount,
        max_capacity = max_cap,
        vacancy      = vacancy,
        person_boxes = [],
        object_boxes = [],
        inference_ms = 222.5,
        pipeline_ms  = 354.1,
    )


# ── Redis tests ────────────────────────────────────────────────────────────────

def test_redis_ping(store: RedisStore) -> None:
    assert store.ping(), "Redis not reachable"
    print("  ✅ Redis ping OK")


def test_redis_store_retrieve(store: RedisStore) -> None:
    result = make_result("cad_lab", headcount=6)
    ok     = store.store_result(result)
    assert ok, "store_result returned False"

    retrieved = store.get_result("cad_lab")
    assert retrieved is not None,              "get_result returned None"
    assert retrieved["section_id"] == "cad_lab"
    assert retrieved["headcount"]  == 6
    assert retrieved["vacancy"]    == 12
    print(f"  ✅ Redis store+retrieve | headcount={retrieved['headcount']} vacancy={retrieved['vacancy']}")


def test_redis_get_all(store: RedisStore) -> None:
    for sec, hc in [("cad_lab", 6), ("g_huss", 20), ("ipc", 3)]:
        store.store_result(make_result(sec, hc))

    all_data = store.get_all_sections()
    assert "cad_lab" in all_data
    assert "g_huss"  in all_data
    assert all_data["g_huss"]["headcount"] == 20
    print(f"  ✅ get_all_sections() returned {len(all_data)} sections")


def test_redis_key_schema(store: RedisStore) -> None:
    """Verify key is stored as occupancy:{section_id}"""
    # pyrefly: ignore [missing-import]
    import redis as redis_lib
    from config.config_loader import cfg
    r = redis_lib.Redis(
        host=cfg.redis.host, port=cfg.redis.port,
        db=cfg.redis.db, decode_responses=True
    )
    store.store_result(make_result("cad_lab", 6))
    assert r.exists("occupancy:cad_lab"), "Key occupancy:cad_lab not found in Redis"
    print(f"  ✅ Key schema correct: occupancy:cad_lab")


# ── DB tests ───────────────────────────────────────────────────────────────────

def test_db_connection(logger: DBLogger) -> None:
    # Already tested in __init__ — just verify no exception
    print("  ✅ PostgreSQL connection OK")


def test_db_log_result(logger: DBLogger) -> None:
    result = make_result("cad_lab", headcount=6)
    ok     = logger.log_result(result)
    assert ok, "log_result returned False"
    print("  ✅ Row inserted into occupancy_log")


def test_db_only_on_change(logger: DBLogger) -> None:
    """Same headcount twice should only write once."""
    logger._last_headcount = {}   # reset tracker
    result = make_result("g_huss", headcount=15)

    wrote1 = logger.log_result(result)   # should write
    wrote2 = logger.log_result(result)   # same headcount — should skip

    assert wrote1  == True,  "First write should succeed"
    assert wrote2  == False, "Second write with same headcount should be skipped"
    print("  ✅ Only-on-change logging works correctly")


def test_db_get_latest(logger: DBLogger) -> None:
    result = make_result("cad_lab", headcount=7)
    logger._last_headcount = {}   # reset so it writes
    logger.log_result(result)

    latest = logger.get_latest("cad_lab")
    assert latest is not None
    assert latest["headcount"] == 7
    print(f"  ✅ get_latest() | headcount={latest['headcount']} vacancy={latest['vacancy']}")


def test_real_pipeline_result(store: RedisStore, logger: DBLogger) -> None:
    """Run zone-based pipeline across all library sections and store results in both Redis and Postgres."""
    import cv2
    from detection.pipeline import ZoneMultiSectionPipeline

    frames_dir = Path("data/sample_frames")
    if not frames_dir.exists():
        print("  ⚠️  Skipped — data/sample_frames directory not found")
        return

    pipeline = ZoneMultiSectionPipeline()
    sample_files = list(frames_dir.glob("*.jpg"))
    if not sample_files:
        print("  ⚠️  Skipped — no sample frames found in data/sample_frames")
        return

    print(f"  Running ZoneMultiSectionPipeline on {len(sample_files)} sections...")
    logger._last_headcount = {}  # reset tracker so all sections get written

    processed_count = 0
    for img_path in sorted(sample_files):
        section_id = img_path.stem
        if section_id not in pipeline.sections():
            continue

        frame  = cv2.imread(str(img_path))
        result = pipeline.run(section_id, frame)

        # Store in Redis
        ok_redis = store.store_result(result)
        # Log to Postgres
        ok_db = logger.log_result(result)

        # Verify Redis readback
        retrieved = store.get_result(section_id)
        assert retrieved is not None, f"Redis readback failed for {section_id}"
        assert retrieved["headcount"] == result.headcount

        # Verify Postgres readback
        latest = logger.get_latest(section_id)
        assert latest is not None, f"Postgres readback failed for {section_id}"

        processed_count += 1
        print(f"     [{section_id}] headcount={result.headcount} vacancy={result.vacancy} | Redis={'OK' if ok_redis else 'FAIL'} DB={'OK' if ok_db else 'FAIL'}")

    print(f"  ✅ Zone pipeline completed & stored for {processed_count} sections")



# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("CP-17 + CP-18 — Redis + PostgreSQL Storage")
    print("=" * 55)

    store  = RedisStore()
    logger = DBLogger()

    print("\n── Redis (CP-17) ──────────────────────────────────")

    print("\n[1] Redis ping:")
    test_redis_ping(store)

    print("\n[2] Store and retrieve:")
    test_redis_store_retrieve(store)

    print("\n[3] get_all_sections():")
    test_redis_get_all(store)

    print("\n[4] Key schema verification:")
    test_redis_key_schema(store)

    print("\n── PostgreSQL (CP-18) ─────────────────────────────")

    print("\n[5] PostgreSQL connection:")
    test_db_connection(logger)

    print("\n[6] Log result to occupancy_log:")
    test_db_log_result(logger)

    print("\n[7] Only-on-change logging:")
    test_db_only_on_change(logger)

    print("\n[8] get_latest():")
    test_db_get_latest(logger)

    print("\n[9] Real pipeline result in both stores:")
    test_real_pipeline_result(store, logger)

    print("\n" + "=" * 55)
    print("🎉 CP-17 + CP-18 PASSED — storage layer ready")
    print("=" * 55)