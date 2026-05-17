"""
JARVIS — Ambient Home AI
========================
Mission: TTSRouter — a thin dispatcher in front of the two TTS backends
         (PiperTTS and KokoroTTS) so the voice menu can offer both and
         synthesis routes to whichever backend owns the selected voice.

         The voice layer holds ONE `self.tts` reference; making that a
         TTSRouter keeps every existing call site unchanged — the router
         mirrors the PiperTTS public surface (load / loaded / speak /
         speak_async / synthesize / synthesize_async / set_voice /
         available_voices / _active_voice / _sample_rate) and forwards
         each call to the active backend.

         VOICE MENU:
           available_voices() returns Piper voices PLUS Kokoro's voices,
           which KokoroTTS labels "KOK <id>". Selecting a "KOK "-prefixed
           label routes synthesis through Kokoro; anything else is Piper.

         LAZY KOKORO LOAD:
           Kokoro's model is a ~330 MB download on first ever use. To
           keep boot fast (and avoid a surprise download mid-startup),
           the router loads Piper at boot but defers KokoroTTS.load()
           until the first time a KOK voice is actually selected — see
           set_voice_async(). The KOK voices still appear in the menu
           before that (the id list is static).

Modules: modules/voice/tts_router.py
Classes: TTSRouter
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
from loguru import logger

from modules.voice.kokoro_tts import KokoroTTS
from modules.voice.tts import PiperTTS

_KOK_PREFIX = "KOK "


class TTSRouter:
    """Routes TTS calls to Piper or Kokoro by the active voice.

    Interface-compatible with PiperTTS so it can be dropped in as
    `self.tts` with no call-site changes. The active backend starts as
    Piper; set_voice()/set_voice_async() switch it.

    Config keys (from config["voice"]["tts"]):
        ...all PiperTTS keys (engine, voices, active_voice, ...)...
        kokoro.enabled:  master switch for the Kokoro backend (default True).
                         When False, no KOK voices are offered and the
                         router behaves exactly like a bare PiperTTS.
    """

    def __init__(self, config: dict) -> None:
        self._piper = PiperTTS(config)
        kcfg = (config.get("voice", {}).get("tts", {}).get("kokoro", {})) or {}
        self._kokoro_enabled = bool(kcfg.get("enabled", True))
        # KokoroTTS.__init__ is cheap — no model load happens here.
        self._kokoro: Any = KokoroTTS(config) if self._kokoro_enabled else None
        self._kokoro_loaded = False
        # The backend every forwarded call hits. Starts on Piper; the
        # office boots speaking with the configured Piper active_voice.
        self._active: Any = self._piper

    # ── Loading ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the Piper backend. Blocking — call once at startup.
        Kokoro is NOT loaded here (see module docstring — lazy load on
        first KOK-voice selection keeps boot off the ~330 MB download)."""
        self._piper.load()
        if self._kokoro_enabled:
            logger.info(
                "[TTSRouter] Piper ready; Kokoro available "
                "(loads on first KOK-voice selection)"
            )
        else:
            logger.info("[TTSRouter] Piper ready; Kokoro disabled in config")

    # ── PiperTTS-compatible surface (delegated to the active backend) ────────

    @property
    def loaded(self) -> bool:
        return bool(getattr(self._active, "loaded", False))

    @property
    def _sample_rate(self) -> int:
        """Sample rate of the ACTIVE backend's synthesized PCM. Piper is
        22.05 kHz, Kokoro 24 kHz — the node / Wyze speaker paths read
        this to resample correctly, so it must track the live backend."""
        return int(getattr(self._active, "_sample_rate", 22050))

    @property
    def _active_voice(self) -> str:
        """Menu label of the current voice — a raw name for Piper, a
        'KOK <id>' label for Kokoro."""
        name = str(getattr(self._active, "_active_voice", ""))
        if self._active is self._kokoro and name and not name.startswith(_KOK_PREFIX):
            return f"{_KOK_PREFIX}{name}"
        return name

    @property
    def _use_sapi(self) -> bool:
        """Exposed for parity with PiperTTS — only meaningful when Piper
        is the active backend."""
        return bool(getattr(self._piper, "_use_sapi", False))

    def available_voices(self) -> list[str]:
        """Piper voices + Kokoro 'KOK <id>' voices for the unified menu."""
        voices = list(self._piper.available_voices())
        if self._kokoro is not None:
            voices += self._kokoro.available_voices()
        return voices

    @staticmethod
    def _is_kokoro_voice(voice_name: str) -> bool:
        return (voice_name or "").strip().startswith(_KOK_PREFIX)

    def set_voice(self, voice_name: str) -> bool:
        """Synchronous voice switch. Piper voices switch instantly; a KOK
        voice only switches here if Kokoro is ALREADY loaded — otherwise
        use set_voice_async(), which performs the one-time lazy load.
        Returns True on success."""
        if self._is_kokoro_voice(voice_name):
            if self._kokoro is None:
                logger.warning(
                    f"[TTSRouter] Kokoro disabled — cannot select {voice_name!r}"
                )
                return False
            if not self._kokoro_loaded:
                logger.warning(
                    f"[TTSRouter] Kokoro not loaded yet — {voice_name!r} needs "
                    "set_voice_async() for the first selection"
                )
                return False
            if self._kokoro.set_voice(voice_name):
                self._active = self._kokoro
                logger.info(f"[TTSRouter] active backend → Kokoro ({voice_name})")
                return True
            return False
        if self._piper.set_voice(voice_name):
            self._active = self._piper
            logger.info(f"[TTSRouter] active backend → Piper ({voice_name})")
            return True
        return False

    async def set_voice_async(self, voice_name: str) -> bool:
        """Voice switch that can lazily load Kokoro. Piper switches in a
        thread (it runs a synthesis self-test); a first-time KOK voice
        triggers KokoroTTS.load() (the ~330 MB download) off the event
        loop. Returns True on success."""
        if not self._is_kokoro_voice(voice_name):
            ok = await asyncio.to_thread(self._piper.set_voice, voice_name)
            if ok:
                self._active = self._piper
                logger.info(f"[TTSRouter] active backend → Piper ({voice_name})")
            return ok

        if self._kokoro is None:
            logger.warning(
                f"[TTSRouter] Kokoro disabled — cannot select {voice_name!r}"
            )
            return False
        if not self._kokoro_loaded:
            logger.info(
                f"[TTSRouter] loading Kokoro backend for first use "
                f"({voice_name}) — first run downloads the model"
            )
            await asyncio.to_thread(self._kokoro.load)
            self._kokoro_loaded = True
        if not self._kokoro.loaded:
            logger.warning(
                f"[TTSRouter] Kokoro failed to load — staying on Piper"
            )
            return False
        ok = self._kokoro.set_voice(voice_name)
        if ok:
            self._active = self._kokoro
            logger.info(f"[TTSRouter] active backend → Kokoro ({voice_name})")
        return ok

    # ── Synthesis / playback (forwarded to the active backend) ───────────────

    def speak(self, text: str) -> None:
        self._active.speak(text)

    async def speak_async(self, text: str) -> None:
        await self._active.speak_async(text)

    def synthesize(self, text: str) -> np.ndarray:
        return self._active.synthesize(text)

    async def synthesize_async(self, text: str) -> np.ndarray:
        return await self._active.synthesize_async(text)
