-- 002_belief_state
-- ============================================================================
-- Tables for the BeliefResolver (audit roadmap D4). The resolver tracks
-- competing *hypotheses* about each entity rather than a single projection,
-- with confidence split into identity / location / visibility / state so
-- "not detected for 30s" can decay visibility fast while location stays
-- high (the white-dog-on-a-white-blanket case).
--
-- These tables are written by the resolver in shadow mode; nothing consumes
-- them to drive behavior yet. WorldModel's world_entities remains the live
-- projection until D4b.
--
-- All statements idempotent so a retry after a partial failure is safe.

CREATE TABLE IF NOT EXISTS entity_beliefs (
    hypothesis_id          TEXT PRIMARY KEY,
    entity_key             TEXT NOT NULL,   -- 'person:3', 'cat:summer', ...
    entity_type            TEXT NOT NULL,   -- person | cat | dog | object
    state                  TEXT NOT NULL,   -- see belief.types.BeliefState
    room                   TEXT,
    camera                 TEXT,
    confidence_identity    REAL NOT NULL DEFAULT 0,
    confidence_location    REAL NOT NULL DEFAULT 0,
    confidence_visibility  REAL NOT NULL DEFAULT 0,
    confidence_state       REAL NOT NULL DEFAULT 0,
    is_primary             INTEGER NOT NULL DEFAULT 1,
    last_confirmed_ts      TEXT,
    last_evidence_ts       TEXT NOT NULL,
    evidence_breakdown     TEXT,            -- JSON
    updated_at             TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entity_beliefs_key
    ON entity_beliefs (entity_key);

CREATE TABLE IF NOT EXISTS belief_evidence (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             TEXT NOT NULL,
    entity_key     TEXT NOT NULL,
    source         TEXT NOT NULL,           -- vision.observation | manual | ...
    room           TEXT,
    camera         TEXT,
    evidence_type  TEXT NOT NULL,           -- sighting | absence | manual_tag
    score          REAL NOT NULL,           -- 0..1 strength of this evidence
    payload        TEXT                     -- JSON
);
CREATE INDEX IF NOT EXISTS idx_belief_evidence_key_ts
    ON belief_evidence (entity_key, ts DESC);
