"""
JARVIS — Ambient Home AI
========================
Mission: Duck (lower) per-application audio volumes on Windows while Jarvis is
         speaking, then restore them after. Stops the user's music/game/Discord
         from drowning out Jarvis's TTS without muting it entirely.

         Implementation uses pycaw (Python Core Audio Windows Library) to walk
         the active audio session list. We exclude Jarvis's own python process
         from the duck so TTS itself stays at full volume.

Modules: modules/voice/audio_focus.py
Classes: AudioFocus
Functions:
    AudioFocus.duck()    — Lower all non-excluded sessions to duck_factor
    AudioFocus.restore() — Restore the volumes recorded by the last duck()
    AudioFocus.duck_async() / restore_async()
"""

import asyncio
import os
from typing import Any, Optional

from loguru import logger

# Windows-only — graceful import so non-Windows hosts just no-op.
try:
    from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume
    import comtypes
    _PYCAW_AVAILABLE = True
except ImportError:
    AudioUtilities = None  # type: ignore[assignment]
    ISimpleAudioVolume = None  # type: ignore[assignment]
    comtypes = None  # type: ignore[assignment]
    _PYCAW_AVAILABLE = False


class AudioFocus:
    """
    Cross-app audio ducker. Call duck() before TTS playback, restore() after.

    Args:
        duck_factor:       Multiplier applied to other apps' volume (0.0–1.0).
                           0.2 = music gets quartered while Jarvis speaks.
        exclude_processes: Process names (case-insensitive, with .exe) to leave
                           alone. Defaults to Python so Jarvis itself keeps
                           full volume.
    """

    def __init__(
        self,
        duck_factor: float = 0.2,
        exclude_processes: Optional[set[str]] = None,
    ) -> None:
        self._duck_factor = float(max(0.0, min(1.0, duck_factor)))
        self._exclude: set[str] = (
            {p.lower() for p in (exclude_processes or set())}
            if exclude_processes
            else {"python.exe", "pythonw.exe"}
        )
        # session_id → (volume_iface, original_master_volume)
        self._restore_state: dict[int, tuple[Any, float]] = {}

    @property
    def available(self) -> bool:
        return _PYCAW_AVAILABLE

    def duck(self) -> None:
        """Lower all non-excluded sessions and remember their original volumes."""
        if not _PYCAW_AVAILABLE or AudioUtilities is None:
            return
        try:
            comtypes.CoInitialize()  # type: ignore[union-attr]
        except Exception:
            pass
        try:
            sessions = AudioUtilities.GetAllSessions()
        except Exception as e:
            logger.debug(f"[AudioFocus] GetAllSessions failed: {e}")
            return

        ducked = 0
        for s in sessions:
            try:
                proc = s.Process
                if proc is None:
                    continue
                if proc.name().lower() in self._exclude:
                    continue
                vol_iface = s._ctl.QueryInterface(ISimpleAudioVolume)
                cur = float(vol_iface.GetMasterVolume())
                if cur <= 0.001:
                    continue
                # session_id is process-unique within this snapshot
                self._restore_state[s.ProcessId] = (vol_iface, cur)
                vol_iface.SetMasterVolume(cur * self._duck_factor, None)
                ducked += 1
            except Exception as e:
                logger.debug(f"[AudioFocus] Skipping a session: {e}")

        if ducked:
            logger.debug(f"[AudioFocus] Ducked {ducked} session(s) to {int(self._duck_factor * 100)}%")

    def restore(self) -> None:
        """Restore volumes recorded by the most recent duck()."""
        if not self._restore_state:
            return
        for _pid, (vol_iface, original) in self._restore_state.items():
            try:
                vol_iface.SetMasterVolume(original, None)
            except Exception as e:
                logger.debug(f"[AudioFocus] Restore failed for one session: {e}")
        logger.debug(f"[AudioFocus] Restored {len(self._restore_state)} session(s)")
        self._restore_state.clear()

    async def duck_async(self) -> None:
        await asyncio.to_thread(self.duck)

    async def restore_async(self) -> None:
        await asyncio.to_thread(self.restore)
