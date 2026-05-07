"""
JARVIS — Ambient Home AI
========================
Mission: Per-room runtime tweaks that don't belong in config.yaml. Things
         like camera rotation/flip, image post-processing knobs, and the
         live speaker-volume override start as defaults from config.yaml
         but get hot-changed from the dashboard. Persisting them in a
         separate JSON file means the dashboard can mutate them without
         touching the user-edited YAML (which would clobber comments and
         layout).

         Design: a tiny key-value store keyed by room. Reads are in-memory
         and lock-free (cheap, called per frame). Writes go through an
         async lock and round-trip to disk synchronously inside a thread.

         Schema is intentionally open — managers consume the keys they
         care about and ignore the rest. New tweak keys (e.g. saturation,
         denoising strength) just start being read by the consumer; no
         schema migration needed.

Modules: core/room_settings.py
Classes: RoomSettings

#todo: Per-room enable flag for vision/audio so a single dashboard toggle
       can mute a room temporarily without editing config.yaml.
#todo: Migrate to SQLite once the dashboard accumulates enough other
       small KV state that JSON-on-disk gets racy.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Optional

from loguru import logger

# Recognized keys — declared here so consumers know what's expected. Unknown
# keys are silently kept (forward-compat with future dashboard fields).
_CAMERA_KEYS = {
    "rotation",     # int, one of 0/90/180/270 — clockwise degrees
    "flip_h",       # bool, mirror horizontally
    "flip_v",       # bool, mirror vertically
    "brightness",   # float, multiplier (1.0 = unchanged, range ~0.5-1.5)
    "contrast",     # float, multiplier (1.0 = unchanged, range ~0.5-1.5)
}
_SPEAKER_KEYS = {
    "volume",       # int, 0-100 — overrides the WyzeSshSpeakerSink default
    "muted",        # bool, if true SpeakerManager.play() returns False without playing
}


class RoomSettings:
    """In-memory + JSON-persisted per-room tweak store.

    Read path is sync and lock-free (each get() returns a copy). Write path
    is async with a lock — multiple dashboard clicks can't interleave a
    half-written file on disk.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._data: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._load()

    # ── Public read API ─────────────────────────────────────────────────

    def get(self, room: str) -> dict[str, Any]:
        """Return a snapshot of the room's settings dict (empty if none).
        Lock-free — callers can hit this on every frame without contention.
        """
        return dict(self._data.get(room, {}))

    def get_value(self, room: str, key: str, default: Any = None) -> Any:
        """Single-key fetch with a fallback. Hot path — keep simple."""
        return self._data.get(room, {}).get(key, default)

    def all(self) -> dict[str, dict[str, Any]]:
        """Snapshot of every room's settings — used by the dashboard's
        initial state hydration."""
        return {r: dict(v) for r, v in self._data.items()}

    # ── Public write API ────────────────────────────────────────────────

    async def update(self, room: str, **kwargs: Any) -> dict[str, Any]:
        """Merge kwargs into the room's settings dict, persist, return the
        new dict. Keys with value None are removed from the dict (the way
        to clear a previously-set tweak from the dashboard).
        """
        async with self._lock:
            current = self._data.setdefault(room, {})
            for k, v in kwargs.items():
                if v is None:
                    current.pop(k, None)
                else:
                    current[k] = v
            snapshot = dict(current)
            await asyncio.to_thread(self._save)
            logger.debug(f"[RoomSettings] '{room}' updated: {kwargs}")
            return snapshot

    async def clear_room(self, room: str) -> None:
        """Remove all tweaks for a room — reverts everything to config defaults."""
        async with self._lock:
            self._data.pop(room, None)
            await asyncio.to_thread(self._save)

    # ── Persistence ─────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load JSON from disk. Missing or unparseable file = empty dict —
        next write will create one. Don't crash startup over a bad
        settings file; the dashboard is recoverable.
        """
        if not self._path.exists():
            return
        try:
            text = self._path.read_text(encoding="utf-8")
            data = json.loads(text)
            if isinstance(data, dict):
                # Guard against the file containing non-dict per-room values
                self._data = {
                    r: dict(v) for r, v in data.items() if isinstance(v, dict)
                }
                logger.info(
                    f"[RoomSettings] Loaded {len(self._data)} room(s) from {self._path}"
                )
        except Exception as e:
            logger.warning(
                f"[RoomSettings] {self._path} unreadable ({e}); starting empty"
            )

    def _save(self) -> None:
        """Atomic-ish save — write to a sibling .tmp then rename."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8"
            )
            tmp.replace(self._path)
        except Exception as e:
            logger.warning(f"[RoomSettings] Save to {self._path} failed: {e}")

    # ── Validation helpers (used by the dashboard endpoint) ────────────

    @staticmethod
    def normalize(payload: dict[str, Any]) -> dict[str, Any]:
        """Coerce dashboard-supplied values into the expected types and
        ranges. Returns a clean dict suitable for update(). Unknown keys
        and unparseable values are dropped silently — only explicit
        `null` from the dashboard maps to the "clear this key" semantic
        (kept as None in the output so update() removes the key).

        The drop-vs-clear distinction matters: a typoed rotation like
        "banana" should be ignored, not interpreted as "reset to 0",
        because the latter changes user-visible state on a typo.
        """
        out: dict[str, Any] = {}

        # rotation: explicit None clears; valid int (0/90/180/270) sets;
        # everything else dropped.
        if "rotation" in payload:
            v = payload["rotation"]
            if v is None:
                out["rotation"] = None
            else:
                try:
                    iv = int(v)
                    if iv in (0, 90, 180, 270):
                        out["rotation"] = iv
                except (TypeError, ValueError):
                    pass

        # Boolean flags: explicit None clears; truthy/falsy coerced.
        for bkey in ("flip_h", "flip_v", "muted"):
            if bkey in payload:
                v = payload[bkey]
                if v is None:
                    out[bkey] = None
                else:
                    out[bkey] = bool(v)

        # Float multipliers: clamp valid; drop unparseable; None clears.
        for fkey in ("brightness", "contrast"):
            if fkey in payload:
                v = payload[fkey]
                if v is None:
                    out[fkey] = None
                else:
                    try:
                        fv = float(v)
                        out[fkey] = max(0.1, min(3.0, fv))
                    except (TypeError, ValueError):
                        pass

        # Volume: clamp valid to 0-100; drop unparseable; None clears.
        if "volume" in payload:
            v = payload["volume"]
            if v is None:
                out["volume"] = None
            else:
                try:
                    iv = int(v)
                    out["volume"] = max(0, min(100, iv))
                except (TypeError, ValueError):
                    pass

        return out
