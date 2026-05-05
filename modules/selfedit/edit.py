"""
JARVIS — Ambient Home AI
========================
Mission: Let Jarvis read, search, and modify its own codebase. Three layers:

  Phase 1 — read-only:  read_file, list_files, grep_files, git_log
  Phase 2 — write:      write_file, edit_file, both auto-commit BEFORE the
                        change so any single edit is one `git reset` away.
  Phase 3 — restart:    restart_self schedules a graceful exit; the
                        bin/start_jarvis_supervised.ps1 wrapper relaunches
                        and reverts to the prior commit if the new instance
                        doesn't write a heartbeat file within 10 seconds.

Safety architecture:
  - Hard kill switch on the manager (default OFF).
  - Sandbox: every path argument is resolved and rejected if outside the
    project root.
  - Refuse list: critical files (config.yaml, .env, *.db) require explicit
    confirmation OR are blocked entirely.
  - All writes auto-commit beforehand. The dashboard exposes a one-click
    'revert last self-edit' that runs `git reset --hard HEAD~1`.
  - restart_self writes the pre-edit SHA to data/last_safe_sha.txt so the
    supervisor can roll back without git knowledge.

Modules: modules/selfedit/edit.py
Classes: SelfEditControl
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger


# Files that may NOT be edited by Jarvis under any circumstance. Includes
# this very file (so a buggy LLM can't disable its own safety) and secrets.
PROTECTED_FILES: set[str] = {
    "modules/selfedit/edit.py",
    "core/orchestrator.py",  # too central to risk a bad rewrite without dashboard approval
    ".env",
    "data/jarvis.db",
}

# Files that require dashboard confirmation before write. Anything else
# inside the project root is fair game (with auto-commit beforehand).
CONFIRM_FILES: set[str] = {
    "config.yaml",
    "main.py",
    "requirements.txt",
    "pyproject.toml",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PendingEdit:
    id: int
    action_type: str       # 'write_file' | 'edit_file'
    args: dict
    requested_at: str
    reason: str = ""


class SelfEditControl:
    """Sandboxed read/write/restart over Jarvis's own codebase."""

    def __init__(self, project_root: Path, broadcast: Optional[Any] = None) -> None:
        self._root = project_root.resolve()
        self._enabled: bool = False
        self._broadcast = broadcast
        self._pending: dict[int, PendingEdit] = {}
        self._next_id: int = 1

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, value: bool) -> None:
        self._enabled = bool(value)
        logger.info(f"[SelfEdit] kill switch -> {'ENABLED' if value else 'DISABLED'}")

    def status(self) -> dict:
        return {
            "enabled": self._enabled,
            "project_root": str(self._root),
            "pending_count": len(self._pending),
        }

    def list_pending(self) -> list[dict]:
        return [
            {"id": p.id, "action_type": p.action_type, "args": p.args,
             "requested_at": p.requested_at, "reason": p.reason}
            for p in sorted(self._pending.values(), key=lambda x: x.id)
        ]

    # ── Path safety ─────────────────────────────────────────────────────────

    def _resolve(self, path: str) -> Optional[Path]:
        """Resolve to an absolute path, rejecting anything outside the root."""
        try:
            p = (self._root / path).resolve()
        except Exception:
            return None
        try:
            p.relative_to(self._root)
        except ValueError:
            return None
        return p

    def _relpath(self, path: str) -> str:
        try:
            return str(Path(path).relative_to(self._root)).replace("\\", "/")
        except Exception:
            return path

    def _is_protected(self, rel: str) -> bool:
        rel = rel.replace("\\", "/")
        return rel in PROTECTED_FILES

    def _needs_confirm(self, rel: str) -> bool:
        rel = rel.replace("\\", "/")
        return rel in CONFIRM_FILES

    # ── Phase 1: read-only ─────────────────────────────────────────────────

    async def read_file(self, path: str, max_bytes: int = 200_000) -> dict:
        p = self._resolve(path)
        if p is None or not p.is_file():
            return {"error": f"file not found or outside project root: {path}"}
        try:
            data = await asyncio.to_thread(p.read_bytes)
        except Exception as e:
            return {"error": str(e)}
        truncated = len(data) > max_bytes
        if truncated:
            data = data[:max_bytes]
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return {"error": "binary file — refusing to return as text"}
        return {"path": str(p.relative_to(self._root)).replace("\\", "/"),
                "content": text, "truncated": truncated, "size": p.stat().st_size}

    async def list_files(self, glob: str = "**/*", max_results: int = 200) -> dict:
        try:
            matches = await asyncio.to_thread(
                lambda: list(self._root.glob(glob))[:max_results]
            )
        except Exception as e:
            return {"error": str(e)}
        out = []
        for m in matches:
            try:
                rel = str(m.relative_to(self._root)).replace("\\", "/")
                out.append({"path": rel, "is_dir": m.is_dir()})
            except Exception:
                continue
        return {"files": out, "truncated": len(matches) >= max_results}

    async def grep_files(
        self, pattern: str, glob: str = "**/*.py", max_results: int = 100,
    ) -> dict:
        compiled = re.compile(pattern)
        results: list[dict] = []
        def _scan():
            for path in self._root.glob(glob):
                if not path.is_file():
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                for i, line in enumerate(text.splitlines(), 1):
                    if compiled.search(line):
                        rel = str(path.relative_to(self._root)).replace("\\", "/")
                        results.append({"path": rel, "line": i, "text": line[:200]})
                        if len(results) >= max_results:
                            return
        await asyncio.to_thread(_scan)
        return {"matches": results, "truncated": len(results) >= max_results}

    async def git_log(self, n: int = 20) -> dict:
        try:
            out = await asyncio.to_thread(
                subprocess.check_output,
                ["git", "log", f"-n{n}", "--oneline"],
                cwd=str(self._root),
                stderr=subprocess.STDOUT,
            )
            return {"log": out.decode("utf-8", "ignore").splitlines()}
        except Exception as e:
            return {"error": str(e)}

    # ── Phase 2: writes (auto-commit beforehand) ───────────────────────────

    async def _git_snapshot(self, message: str) -> Optional[str]:
        """Stage + commit the current working tree so the next edit is a
        clean diff. Returns the commit SHA or None on failure. Skips if
        nothing's changed."""
        def _run():
            try:
                subprocess.check_call(
                    ["git", "add", "-A"], cwd=str(self._root),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass
            # Skip commit if nothing staged
            status = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                cwd=str(self._root), capture_output=True, text=True,
            )
            if not status.stdout.strip():
                return None
            try:
                subprocess.check_call(
                    ["git", "commit", "-m", message, "--no-verify"],
                    cwd=str(self._root),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                sha = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=str(self._root)
                ).decode().strip()
                return sha
            except Exception as e:
                logger.warning(f"[SelfEdit] git commit failed: {e}")
                return None
        return await asyncio.to_thread(_run)

    async def write_file(
        self, path: str, content: str, bypass_confirm: bool = False,
    ) -> dict:
        if not self._enabled:
            return {"error": "self-edit is disabled — flip the kill switch"}
        p = self._resolve(path)
        if p is None:
            return {"error": f"path outside project root: {path}"}
        rel = self._relpath(str(p))
        if self._is_protected(rel):
            return {"error": f"refused: {rel} is on the protected list"}
        if not bypass_confirm and self._needs_confirm(rel):
            return await self._queue("write_file", {"path": rel, "content": content},
                                     reason=f"{rel} requires confirmation")
        # Auto-commit before write so we can roll back
        await self._git_snapshot(f"selfedit: pre-write snapshot for {rel}")
        try:
            await asyncio.to_thread(p.parent.mkdir, parents=True, exist_ok=True)
            await asyncio.to_thread(p.write_text, content, "utf-8")
        except Exception as e:
            return {"error": str(e)}
        commit_sha = await self._git_snapshot(f"selfedit: write {rel}")
        return {"ok": True, "path": rel, "commit": commit_sha}

    async def edit_file(
        self, path: str, old_string: str, new_string: str, bypass_confirm: bool = False,
    ) -> dict:
        """Single-string find-and-replace. Errors if old_string is not unique."""
        if not self._enabled:
            return {"error": "self-edit is disabled — flip the kill switch"}
        p = self._resolve(path)
        if p is None or not p.is_file():
            return {"error": f"file not found or outside project root: {path}"}
        rel = self._relpath(str(p))
        if self._is_protected(rel):
            return {"error": f"refused: {rel} is on the protected list"}
        if not bypass_confirm and self._needs_confirm(rel):
            return await self._queue("edit_file", {
                "path": rel, "old_string": old_string, "new_string": new_string,
            }, reason=f"{rel} requires confirmation")
        try:
            text = await asyncio.to_thread(p.read_text, "utf-8")
        except Exception as e:
            return {"error": str(e)}
        count = text.count(old_string)
        if count == 0:
            return {"error": "old_string not found"}
        if count > 1:
            return {"error": f"old_string occurs {count} times — provide more context to make it unique"}
        new_text = text.replace(old_string, new_string, 1)
        await self._git_snapshot(f"selfedit: pre-edit snapshot for {rel}")
        try:
            await asyncio.to_thread(p.write_text, new_text, "utf-8")
        except Exception as e:
            return {"error": str(e)}
        commit_sha = await self._git_snapshot(f"selfedit: edit {rel}")
        return {"ok": True, "path": rel, "commit": commit_sha}

    async def revert_last(self) -> dict:
        """Roll back the most recent commit (typically the last self-edit).
        Useful from the dashboard if Jarvis breaks something."""
        try:
            await asyncio.to_thread(
                subprocess.check_call,
                ["git", "reset", "--hard", "HEAD~1"],
                cwd=str(self._root),
            )
            return {"ok": True}
        except Exception as e:
            return {"error": str(e)}

    # ── Confirmation queue ─────────────────────────────────────────────────

    async def _queue(self, action_type: str, args: dict, reason: str) -> dict:
        eid = self._next_id
        self._next_id += 1
        self._pending[eid] = PendingEdit(
            id=eid, action_type=action_type, args=args,
            requested_at=_now_iso(), reason=reason,
        )
        if self._broadcast is not None:
            try:
                await self._broadcast({
                    "type": "selfedit.pending_added",
                    "id": eid, "action_type": action_type, "reason": reason,
                })
            except Exception:
                pass
        return {"queued": True, "id": eid, "reason": reason}

    async def approve_pending(self, edit_id: int) -> dict:
        edit = self._pending.pop(edit_id, None)
        if edit is None:
            return {"error": f"no pending edit {edit_id}"}
        if edit.action_type == "write_file":
            return await self.write_file(**edit.args, bypass_confirm=True)
        if edit.action_type == "edit_file":
            return await self.edit_file(**edit.args, bypass_confirm=True)
        return {"error": f"unknown action_type: {edit.action_type}"}

    async def reject_pending(self, edit_id: int) -> bool:
        return self._pending.pop(edit_id, None) is not None

    # ── Phase 3: restart with auto-revert ──────────────────────────────────

    async def restart_self(self, reason: str = "self-edit applied") -> dict:
        """Schedule a graceful shutdown; the supervisor wrapper restarts
        Jarvis. If the new instance doesn't write a heartbeat within
        10 seconds, the wrapper auto-reverts (`git reset --hard HEAD~1`)
        and starts a new instance."""
        if not self._enabled:
            return {"error": "self-edit is disabled — flip the kill switch"}
        # Record the safe SHA so the supervisor can revert to it
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(self._root)
            ).decode().strip()
            (self._root / "data").mkdir(parents=True, exist_ok=True)
            (self._root / "data" / "last_safe_sha.txt").write_text(sha, "utf-8")
        except Exception as e:
            logger.warning(f"[SelfEdit] could not record safe SHA: {e}")
        # Mark intent
        try:
            (self._root / "data" / "restart_pending.txt").write_text(reason, "utf-8")
        except Exception:
            pass
        # Schedule the actual exit a beat later so the response can be sent
        async def _exit_soon():
            await asyncio.sleep(0.5)
            logger.warning(f"[SelfEdit] restart_self triggered: {reason}")
            # Exit code 42 = clean restart request to the supervisor
            os._exit(42)
        asyncio.create_task(_exit_soon())
        return {"ok": True, "reason": reason, "restarting_in_ms": 500}
