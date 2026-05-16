-- 001_baseline
-- ============================================================================
-- Establishes the SchemaMigrator as the forward path for schema changes.
--
-- Core tables remain owned by DatabaseManager.SCHEMA_SQL (they already exist
-- in every live DB). This migration only adds retention-support indexes for
-- the Phase 0 nightly pruners — both reference SCHEMA_SQL tables that exist
-- by the time the migrator runs inside DatabaseManager.init().
--
-- All statements are idempotent (IF NOT EXISTS) so a retry after a partial
-- failure is safe.

-- Supports IdentityManager.prune_resolved_pending:
--   DELETE FROM identity_pending WHERE resolved IN (1,2) AND captured_at < ?
CREATE INDEX IF NOT EXISTS idx_identity_pending_resolved_captured
    ON identity_pending (resolved, captured_at);

-- Supports NotificationManager.prune_read:
--   DELETE FROM notifications WHERE read = 1 AND created_at < ?
-- (idx_notifications_unread already covers read+created_at; this is a
-- harmless explicit duplicate kept for clarity if that index is ever renamed.)
CREATE INDEX IF NOT EXISTS idx_notifications_read_created
    ON notifications (read, created_at);
