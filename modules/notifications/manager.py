"""
JARVIS — Ambient Home AI
========================
Mission: A persistent, dashboard-surfaced inbox for things that need Cole's
         attention. Identity drift conflicts, unknown-cluster captures, and
         system errors call notify(...) here instead of (only) printing to
         the log. The dashboard renders a bell icon with an unread count
         and a dropdown list; clicking a row navigates to the relevant
         card (pending review, person modal, etc.).

         Notifications persist across restarts so a drift event from
         yesterday is still there in the morning. mark_read() and delete()
         expose the dismissal API the dashboard uses.

Modules: modules/notifications.py
Classes: NotificationManager
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class NotificationManager:
    """Database-backed notification inbox. Pass a DatabaseManager + an optional
    async broadcast callback (orchestrator's _broadcast) for live UI updates."""

    def __init__(self, db: Any, broadcast: Optional[Any] = None) -> None:
        self._db = db
        self._broadcast = broadcast

    async def notify(
        self,
        kind: str,
        title: str,
        message: Optional[str] = None,
        target_type: Optional[str] = None,
        target_id: Optional[int] = None,
        action: Optional[str] = None,
        severity: str = "info",
    ) -> int:
        """Persist + broadcast. Returns the new notification id."""
        try:
            nid = await self._db.execute(
                "INSERT INTO notifications "
                "(kind, title, message, target_type, target_id, action, severity, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (kind, title, message, target_type, target_id, action, severity, _now_iso()),
            )
        except Exception as e:
            logger.warning(f"[Notifications] persist failed: {e}")
            return 0
        if self._broadcast is not None:
            try:
                await self._broadcast({
                    "type":        "notification.added",
                    "id":          nid,
                    "kind":        kind,
                    "title":       title,
                    "message":     message,
                    "target_type": target_type,
                    "target_id":   target_id,
                    "action":      action,
                    "severity":    severity,
                })
            except Exception as e:
                logger.debug(f"[Notifications] broadcast failed: {e}")
        return nid

    async def list_recent(self, limit: int = 50, only_unread: bool = False) -> list[dict]:
        sql = "SELECT * FROM notifications"
        params: tuple = ()
        if only_unread:
            sql += " WHERE read = 0"
        sql += " ORDER BY created_at DESC LIMIT ?"
        params = params + (int(limit),)
        try:
            rows = await self._db.fetchall(sql, params)
        except Exception:
            return []
        return [
            {
                "id":          int(r["id"]),
                "kind":        r["kind"],
                "title":       r["title"],
                "message":     r["message"],
                "target_type": r["target_type"],
                "target_id":   r["target_id"],
                "action":      r["action"],
                "severity":    r["severity"],
                "created_at":  r["created_at"],
                "read":        bool(r["read"]),
            }
            for r in rows
        ]

    async def unread_count(self) -> int:
        try:
            row = await self._db.fetchone(
                "SELECT COUNT(*) AS n FROM notifications WHERE read = 0"
            )
            return int(row["n"]) if row else 0
        except Exception:
            return 0

    async def mark_read(self, notification_id: int) -> bool:
        try:
            await self._db.execute(
                "UPDATE notifications SET read = 1 WHERE id = ?", (notification_id,)
            )
        except Exception as e:
            logger.debug(f"[Notifications] mark_read failed: {e}")
            return False
        if self._broadcast is not None:
            try:
                await self._broadcast({"type": "notification.read", "id": notification_id})
            except Exception:
                pass
        return True

    async def mark_all_read(self) -> bool:
        try:
            await self._db.execute("UPDATE notifications SET read = 1 WHERE read = 0")
        except Exception:
            return False
        if self._broadcast is not None:
            try:
                await self._broadcast({"type": "notification.read", "id": None, "all": True})
            except Exception:
                pass
        return True

    async def delete(self, notification_id: int) -> bool:
        try:
            await self._db.execute(
                "DELETE FROM notifications WHERE id = ?", (notification_id,)
            )
        except Exception:
            return False
        if self._broadcast is not None:
            try:
                await self._broadcast({"type": "notification.deleted", "id": notification_id})
            except Exception:
                pass
        return True

    async def dismiss_for_target(
        self, target_type: str, target_id: int,
    ) -> int:
        """Auto-dismiss every notification whose (target_type, target_id)
        points at this resource. Called by the resolve paths (pending
        review assign/reject, etc.) so a user who actions the underlying
        thing doesn't then have to click the bell to clear the matching
        alerts. Returns the number of rows deleted."""
        try:
            rows = await self._db.fetchall(
                "SELECT id FROM notifications "
                "WHERE target_type = ? AND target_id = ?",
                (target_type, int(target_id)),
            )
            if not rows:
                return 0
            ids = [int(r["id"]) for r in rows]
            placeholders = ",".join("?" for _ in ids)
            await self._db.execute(
                f"DELETE FROM notifications WHERE id IN ({placeholders})",
                tuple(ids),
            )
        except Exception as e:
            logger.debug(f"[Notifications] dismiss_for_target failed: {e}")
            return 0
        if self._broadcast is not None:
            try:
                await self._broadcast({
                    "type": "notification.deleted",
                    "id": None,
                    "ids": ids,
                    "target_type": target_type,
                    "target_id": int(target_id),
                })
            except Exception:
                pass
        return len(ids)

    async def dismiss_for_targets(
        self, target_type: str, target_ids: list,
    ) -> int:
        """Bulk variant of dismiss_for_target — one DELETE for an entire
        batch of resolved rows (avoid N round-trips when bulk-assigning
        50 pending reviews at once)."""
        if not target_ids:
            return 0
        try:
            ids_int = [int(t) for t in target_ids if t is not None]
            if not ids_int:
                return 0
            placeholders = ",".join("?" for _ in ids_int)
            rows = await self._db.fetchall(
                f"SELECT id FROM notifications "
                f"WHERE target_type = ? AND target_id IN ({placeholders})",
                (target_type, *ids_int),
            )
            if not rows:
                return 0
            del_ids = [int(r["id"]) for r in rows]
            del_placeholders = ",".join("?" for _ in del_ids)
            await self._db.execute(
                f"DELETE FROM notifications WHERE id IN ({del_placeholders})",
                tuple(del_ids),
            )
        except Exception as e:
            logger.debug(f"[Notifications] dismiss_for_targets failed: {e}")
            return 0
        if self._broadcast is not None:
            try:
                await self._broadcast({
                    "type": "notification.deleted",
                    "id": None,
                    "ids": del_ids,
                    "target_type": target_type,
                })
            except Exception:
                pass
        return len(del_ids)
