-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Main occupancy log table
CREATE TABLE IF NOT EXISTS occupancy_log (
    time          TIMESTAMPTZ     NOT NULL,
    section_id    TEXT            NOT NULL,
    headcount     INTEGER         NOT NULL,
    max_capacity  INTEGER,
    vacancy       INTEGER,
    occupancy_pct NUMERIC(5,2),
    is_available  BOOLEAN,
    inference_ms  NUMERIC(8,1),
    pipeline_ms   NUMERIC(8,1)
);

-- Convert to TimescaleDB hypertable (partitioned by time)
SELECT create_hypertable(
    'occupancy_log', 'time',
    if_not_exists => TRUE
);

-- Index for fast per-section queries
CREATE INDEX IF NOT EXISTS idx_occupancy_section
    ON occupancy_log (section_id, time DESC);

-- Continuous aggregate: per-minute averages per section
CREATE MATERIALIZED VIEW IF NOT EXISTS occupancy_per_minute
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS bucket,
    section_id,
    AVG(headcount)     AS avg_headcount,
    AVG(vacancy)       AS avg_vacancy,
    MAX(headcount)     AS peak_headcount,
    MIN(vacancy)       AS min_vacancy
FROM occupancy_log
GROUP BY bucket, section_id
WITH NO DATA;

-- Retention policy: keep raw data for 90 days
SELECT add_retention_policy(
    'occupancy_log',
    INTERVAL '90 days',
    if_not_exists => TRUE
);

-- Summary view: latest state per section
CREATE OR REPLACE VIEW latest_occupancy AS
SELECT DISTINCT ON (section_id)
    section_id,
    time,
    headcount,
    max_capacity,
    vacancy,
    occupancy_pct,
    is_available
FROM occupancy_log
ORDER BY section_id, time DESC;