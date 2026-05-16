"""
JARVIS — Ambient Home AI
========================
Mission: KokoroTTS — a second neural TTS backend alongside PiperTTS.
         Kokoro is an 82M-parameter open-weight TTS model (Apache-2.0)
         that produces notably more natural, expressive speech than
         Piper at 24 kHz, fully local.

         This class deliberately mirrors the PiperTTS public surface —
         load() / loaded / synthesize() / synthesize_async() / speak() /
         speak_async() / set_voice() / available_voices() / _sample_rate
         — so the voice layer can treat the two backends interchangeably.

         VOICE MENU: available_voices() returns each Kokoro voice with a
         "KOK " prefix so a unified menu can show Piper and Kokoro voices
         together and tell them apart at a glance. set_voice() accepts
         the prefixed or the raw id.

         Kokoro has no built-in "GLaDOS" voice — a specific timbre would
         need voice cloning, which Kokoro does not do. The GLaDOS *persona*
         is assigned one of Kokoro's stock voices via config; persona and
         voice are independent.

Modules: modules/voice/kokoro_tts.py
Classes: KokoroTTS

#todo: stream chunk-by-chunk (KPipeline already yields per-segment audio)
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
from loguru import logger

from modules.voice.audio_utils import play_audio_array, play_audio_array_async

KOKORO_SAMPLE_RATE = 24000
_KOK_PREFIX = "KOK "

# Kokoro v1.0 English voices. lang_code is derived from the id prefix:
# a* = American English, b* = British English.
_KOKORO_VOICES: tuple[str, ...] = (
    "af_heart", "af_bella", "af_nicole", "af_aoede", "af_kore", "af_sarah",
    "af_nova", "af_sky", "af_alloy", "af_jessica", "af_river",
    "am_adam", "am_michael", "am_fenrir", "am_puck", "am_echo", "am_eric",
    "am_liam", "am_onyx", "am_santa",
    "bf_emma", "bf_isabella", "bf_alice", "bf_lily",
    "bm_george", "bm_lewis", "bm_daniel", "bm_fable",
)


class KokoroTTS:
    """Kokoro neural TTS backend. Interface-compatible with PiperTTS.

    Config keys (from config["voice"]["tts"]["kokoro"], all optional):
        voice:    default Kokoro voice id (default "af_heart")
        speed:    speech-rate multiplier (default 1.0)
    """

    def __init__(self, config: dict) -> None:
        kcfg = (config.get("voice", {}).get("tts", {}).get("kokoro", {})) or {}
        self._default_voice = self._strip_prefix(kcfg.get("voice", "af_heart"))
        self._speed = float(kcfg.get("speed", 1.0))
        self._active_voice = self._default_voice
        self._sample_rate = KOKORO_SAMPLE_RATE
        self.loaded = False
        self._KPipeline: Any = None
        # One KPipeline per lang_code ('a'/'b') — created lazily, cached.
        self._pipelines: dict[str, Any] = {}

    # ── Loading ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Import Kokoro and warm the pipeline for the default voice.
        Blocking — call once at startup. Never raises: on failure
        loaded stays False and the voice layer keeps using Piper."""
        try:
            from kokoro import KPipeline
        except ImportError:
            logger.warning(
                "[KokoroTTS] 'kokoro' package not installed — "
                "backend unavailable (pip install kokoro)"
            )
            return
        self._KPipeline = KPipeline
        try:
            # Warming the default voice's pipeline downloads the model
            # (~330 MB) on first ever run, then it is cached.
            self._pipeline_for(self._default_voice)
            self.loaded = True
            logger.info(
                f"[KokoroTTS] ready ({len(_KOKORO_VOICES)} voices, "
                f"default '{self._default_voice}')"
            )
        except Exception as e:
            logger.warning(f"[KokoroTTS] load failed ({e}) — backend unavailable")
            self.loaded = False

    def _pipeline_for(self, voice: str) -> Any:
        """Lazily build + cache the KPipeline for the voice's language."""
        lang = "b" if voice.startswith("b") else "a"
        if lang not in self._pipelines:
            self._pipelines[lang] = self._KPipeline(lang_code=lang)
        return self._pipelines[lang]

    # ── Voice selection ──────────────────────────────────────────────────────

    @staticmethod
    def _strip_prefix(name: str) -> str:
        """Accept a menu label ('KOK af_heart') or a raw id ('af_heart')."""
        name = (name or "").strip()
        return name[len(_KOK_PREFIX):].strip() if name.startswith(_KOK_PREFIX) else name

    def available_voices(self) -> list[str]:
        """Kokoro voices as 'KOK <id>' menu labels."""
        return [f"{_KOK_PREFIX}{v}" for v in _KOKORO_VOICES]

    def set_voice(self, voice_name: str) -> bool:
        """Switch voice. Accepts a 'KOK '-prefixed label or a raw id.
        Returns True on success."""
        if not self.loaded:
            logger.warning("[KokoroTTS] cannot switch voice — backend not loaded")
            return False
        raw = self._strip_prefix(voice_name)
        if raw not in _KOKORO_VOICES:
            logger.warning(f"[KokoroTTS] unknown voice '{voice_name}'")
            return False
        try:
            self._pipeline_for(raw)  # ensure the lang pipeline exists
        except Exception as e:
            logger.warning(f"[KokoroTTS] could not switch to '{raw}': {e}")
            return False
        self._active_voice = raw
        logger.info(f"[KokoroTTS] voice -> {raw}")
        return True

    # ── Synthesis ────────────────────────────────────────────────────────────

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to a float32 mono numpy array at 24 kHz.
        Returns an empty array on failure rather than raising — the
        caller (or a Piper fallback) decides what to do with silence."""
        if not self.loaded or not text.strip():
            return np.zeros(0, dtype=np.float32)
        try:
            pipeline = self._pipeline_for(self._active_voice)
            chunks: list[np.ndarray] = []
            for _, _, audio in pipeline(
                text, voice=self._active_voice, speed=self._speed
            ):
                arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") \
                    else np.asarray(audio)
                chunks.append(arr.astype(np.float32).flatten())
            if not chunks:
                return np.zeros(0, dtype=np.float32)
            out = np.concatenate(chunks)
            logger.debug(
                f"[KokoroTTS] synthesized {len(out)/self._sample_rate:.2f}s "
                f"({self._active_voice})"
            )
            return out
        except Exception as e:
            logger.warning(f"[KokoroTTS] synthesis failed: {e}")
            return np.zeros(0, dtype=np.float32)

    async def synthesize_async(self, text: str) -> np.ndarray:
        """Non-blocking synthesize() — runs in a thread pool."""
        return await asyncio.to_thread(self.synthesize, text)

    def speak(self, text: str) -> None:
        """Synthesize and play, blocking until playback completes."""
        if not text.strip():
            return
        if not self.loaded:
            logger.warning(f"[KokoroTTS] not loaded — skipping: {text!r}")
            return
        audio = self.synthesize(text)
        if audio.size:
            play_audio_array(audio, self._sample_rate)

    async def speak_async(self, text: str) -> None:
        """Non-blocking synthesize + play."""
        if not text.strip() or not self.loaded:
            return
        audio = await self.synthesize_async(text)
        if audio.size:
            await play_audio_array_async(audio, self._sample_rate)
