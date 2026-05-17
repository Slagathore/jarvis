-- 003_anomalies
-- ============================================================================
-- Table for AnomalyScorer (audit roadmap §25). Every fired anomaly is
-- persisted here so the dashboard can show a review queue and the nightly
-- auto_tune pass can measure the false-positive rate (invalidated / total)
-- and nudge the score threshold.
--
-- Behavioral PROFILES live on entity.metadata['pattern_profile'] (written by
-- PatternMiner) — they are not a table. This table is only the fired-anomaly
-- log + the user's "not actually unusual" feedback.
--
-- All statements idempotent so a retry after a partial failure is safe.

CREATE TABLE IF NOT EXISTS world_anomalies (
    id                  TEXT PRIMARY KEY,
    ts                  TEXT NOT NULL,      -- ts of the triggering event
    entity_id           TEXT,
    entity_name         TEXT,
    score               REAL NOT NULL DEFAULT 0,
    components          TEXT,               -- JSON: per-component scores
    event               TEXT,               -- JSON: the triggering entity_event
    invalidated         INTEGER NOT NULL DEFAULT 0,  -- user marked "not unusual"
    invalidated_reason  TEXT,
    invalidated_ts      TEXT
);
CREATE INDEX IF NOT EXISTS idx_world_anomalies_ts
    ON world_anomalies (ts DESC);
CREATE INDEX IF NOT EXISTS idx_world_anomalies_entity
    ON world_anomalies (entity_id, ts DESC);
