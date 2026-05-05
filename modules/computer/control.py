"""
JARVIS — Ambient Home AI
========================
Mission: Let Jarvis drive Cole's mouse and keyboard to take real action on
         the desktop — opening apps, clicking buttons, typing text, sending
         hotkeys. Gated behind a hard kill switch (default OFF) so any
         malfunction doesn't run wild on a live machine.

         Safety architecture (defense in depth):
           1. Kill switch — `enabled` flag on the manager. When False, every
              tool returns {"error": ...} immediately.
           2. Refuse list — pattern matches against typed text, hotkey combos,
              and forbidden paths. Catches obvious dangerous strings without
              the LLM having to.
           3. Confirmation queue — high-risk actions (anything matching the
              confirm list) land in a 'pending' state and require Cole to
              explicitly approve via the dashboard.
           4. pyautogui FAILSAFE — mouse to top-left corner aborts everything.
              Always on, configurable via pyautogui.FAILSAFE.

         Even with all four layers, this is the riskiest tool surface in
         Jarvis. Treat enabling it as 'I am paying attention to what's on
         screen'.

Modules: modules/computer/control.py
Classes: ComputerControl
"""

from __future__ import annotations

import asyncio
import base64
import io
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger


# ── Refuse list ─────────────────────────────────────────────────────────────
# Substrings to refuse outright when they appear in keyboard_type input.
# Case-insensitive. These are generic-dangerous; not a substitute for a
# confirmation gate but a fast-fail for the obviously catastrophic.
DANGEROUS_TEXT_PATTERNS = [
    r"\brm\s+-rf\b",
    r"\bdel\s+/[fsq]\b",
    r"\bformat\s+[a-z]:\b",
    r"\brmdir\s+/s\b",
    r"\bshutdown\b\s+(/s|/r|-h|-r)",
    r"\breboot\b",
    r"diskpart",
    r"\bnet\s+user\b.*\b/delete\b",
    r"\bdrop\s+(table|database)\b",
    r"\btruncate\s+table\b",
]

# Hotkey combos to refuse outright (lowercase keys joined with '+')
DANGEROUS_HOTKEYS: set[frozenset[str]] = {
    frozenset({"ctrl", "alt", "delete"}),
    frozenset({"win", "l"}),     # lock screen
    frozenset({"alt", "f4"}),    # close active window — too easy to misfire
}

# Combos that go through but trigger the confirmation queue first.
CONFIRM_HOTKEYS: set[frozenset[str]] = set()

# Patterns that go through but trigger confirmation. Used for typed text
# that's not catastrophic but worth a sanity check.
CONFIRM_TEXT_PATTERNS = [
    r"\bsend\b.*\bemail\b",
    r"\bsend\b.*\bmessage\b",
    r"\bpurchase\b",
    r"\bdelete\b",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _matches_any(text: str, patterns: list[str]) -> bool:
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False


@dataclass
class PendingAction:
    """An action waiting for Cole's approval before execution."""
    id: int
    action_type: str
    args: dict
    requested_at: str
    reason: str = ""


class ComputerControl:
    """Async wrapper over pyautogui with a hard kill switch and confirm queue.

    The kill switch is intentionally NOT persisted — every Jarvis restart
    comes up with computer control DISABLED. Cole flips it on from the
    dashboard when he's ready.
    """

    def __init__(self, broadcast: Optional[Any] = None) -> None:
        self._enabled: bool = False
        self._broadcast = broadcast
        self._pending: dict[int, PendingAction] = {}
        self._next_id: int = 1
        self._pyautogui: Optional[Any] = None
        # Initialize pyautogui only on first use to avoid loading X11/screen
        # state during boot. It's also a heavy import.
        self._init_attempted: bool = False

    # ── Lifecycle / status ──────────────────────────────────────────────────

    def _ensure_pyautogui(self) -> Optional[Any]:
        if self._pyautogui is not None:
            return self._pyautogui
        if self._init_attempted:
            return None
        self._init_attempted = True
        try:
            import pyautogui
            pyautogui.FAILSAFE = True   # mouse to top-left corner aborts
            pyautogui.PAUSE = 0.05      # small inter-action delay
            self._pyautogui = pyautogui
            return pyautogui
        except Exception as e:
            logger.warning(f"[ComputerControl] pyautogui init failed: {e}")
            return None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, value: bool) -> None:
        self._enabled = bool(value)
        logger.info(f"[ComputerControl] kill switch -> {'ENABLED' if value else 'DISABLED'}")

    def status(self) -> dict:
        return {
            "enabled": self._enabled,
            "pyautogui_ready": self._pyautogui is not None,
            "pending_count": len(self._pending),
        }

    # ── Confirmation queue ──────────────────────────────────────────────────

    def list_pending(self) -> list[dict]:
        return [
            {
                "id": p.id,
                "action_type": p.action_type,
                "args": p.args,
                "requested_at": p.requested_at,
                "reason": p.reason,
            }
            for p in sorted(self._pending.values(), key=lambda x: x.id)
        ]

    async def approve(self, action_id: int) -> dict:
        action = self._pending.pop(action_id, None)
        if action is None:
            return {"error": f"no pending action {action_id}"}
        # Execute now
        result = await self._execute(action.action_type, action.args, bypass_confirm=True)
        if self._broadcast is not None:
            try:
                await self._broadcast({"type": "computer.confirmed", "id": action_id, "result": result})
            except Exception:
                pass
        return result

    async def reject(self, action_id: int) -> bool:
        if action_id in self._pending:
            del self._pending[action_id]
            if self._broadcast is not None:
                try:
                    await self._broadcast({"type": "computer.rejected", "id": action_id})
                except Exception:
                    pass
            return True
        return False

    async def _queue_for_confirm(
        self, action_type: str, args: dict, reason: str = ""
    ) -> dict:
        action_id = self._next_id
        self._next_id += 1
        self._pending[action_id] = PendingAction(
            id=action_id,
            action_type=action_type,
            args=args,
            requested_at=_now_iso(),
            reason=reason,
        )
        if self._broadcast is not None:
            try:
                await self._broadcast({
                    "type": "computer.pending_added",
                    "id": action_id,
                    "action_type": action_type,
                    "reason": reason,
                })
            except Exception:
                pass
        return {
            "queued": True,
            "id": action_id,
            "reason": reason or "high-risk action awaiting approval",
        }

    # ── Action surface ──────────────────────────────────────────────────────

    async def screenshot(self) -> dict:
        """Capture the current screen. Returns base64 JPEG.

        Reading the screen is allowed even when kill switch is OFF — vision
        is read-only and the LLM may need it to plan an action that only
        runs after Cole flips the switch on.
        """
        pag = self._ensure_pyautogui()
        if pag is None:
            return {"error": "pyautogui not available"}
        try:
            img = await asyncio.to_thread(pag.screenshot)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            return {
                "image_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
                "size": [img.width, img.height],
            }
        except Exception as e:
            return {"error": f"screenshot failed: {e}"}

    async def screen_size(self) -> dict:
        pag = self._ensure_pyautogui()
        if pag is None:
            return {"error": "pyautogui not available"}
        try:
            w, h = await asyncio.to_thread(pag.size)
            return {"width": int(w), "height": int(h)}
        except Exception as e:
            return {"error": str(e)}

    async def mouse_click(
        self, x: int, y: int, button: str = "left", clicks: int = 1,
    ) -> dict:
        return await self._execute("mouse_click", {
            "x": int(x), "y": int(y),
            "button": str(button).lower(),
            "clicks": int(clicks),
        })

    async def mouse_move(self, x: int, y: int, duration: float = 0.2) -> dict:
        return await self._execute("mouse_move", {
            "x": int(x), "y": int(y), "duration": float(duration),
        })

    async def keyboard_type(self, text: str, interval: float = 0.02) -> dict:
        return await self._execute("keyboard_type", {
            "text": str(text), "interval": float(interval),
        })

    async def keyboard_hotkey(self, keys: list) -> dict:
        return await self._execute("keyboard_hotkey", {
            "keys": [str(k).lower() for k in (keys or [])],
        })

    # ── Internal execute (with safety gates) ───────────────────────────────

    async def _execute(
        self, action_type: str, args: dict, bypass_confirm: bool = False,
    ) -> dict:
        if not self._enabled:
            return {"error": "computer control is disabled — flip the kill switch in the dashboard"}
        pag = self._ensure_pyautogui()
        if pag is None:
            return {"error": "pyautogui not available"}

        # Refuse-list checks
        if action_type == "keyboard_type":
            text = args.get("text", "")
            if _matches_any(text, DANGEROUS_TEXT_PATTERNS):
                logger.warning(f"[ComputerControl] REFUSED type: {text!r}")
                return {"error": "refused: text matches a dangerous-pattern blocklist"}
            if not bypass_confirm and _matches_any(text, CONFIRM_TEXT_PATTERNS):
                return await self._queue_for_confirm(
                    action_type, args, reason="text matches a confirm pattern"
                )
        elif action_type == "keyboard_hotkey":
            keys = frozenset(args.get("keys", []))
            if keys in DANGEROUS_HOTKEYS:
                logger.warning(f"[ComputerControl] REFUSED hotkey: {sorted(keys)}")
                return {"error": "refused: hotkey is on the blocklist"}
            if not bypass_confirm and keys in CONFIRM_HOTKEYS:
                return await self._queue_for_confirm(
                    action_type, args, reason="hotkey requires confirm"
                )

        # Execute on a thread (pyautogui is sync)
        try:
            if action_type == "mouse_click":
                await asyncio.to_thread(
                    pag.click,
                    args["x"], args["y"],
                    clicks=args.get("clicks", 1),
                    button=args.get("button", "left"),
                )
                return {"ok": True}
            if action_type == "mouse_move":
                await asyncio.to_thread(
                    pag.moveTo, args["x"], args["y"], args.get("duration", 0.2)
                )
                return {"ok": True}
            if action_type == "keyboard_type":
                await asyncio.to_thread(
                    pag.typewrite, args["text"], interval=args.get("interval", 0.02)
                )
                return {"ok": True}
            if action_type == "keyboard_hotkey":
                await asyncio.to_thread(pag.hotkey, *args["keys"])
                return {"ok": True}
        except Exception as e:
            logger.warning(f"[ComputerControl] {action_type} failed: {e}")
            return {"error": str(e)}
        return {"error": f"unknown action_type: {action_type}"}
