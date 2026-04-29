"""
JARVIS — Ambient Home AI
========================
Mission: Text-to-speech synthesis using the Piper neural TTS engine.
         Piper produces high-quality, natural-sounding speech locally with no
         cloud dependency. The piper binary is called via subprocess; audio
         output is captured as raw PCM and played through sounddevice.

         Falls back to Windows SAPI (via win32com) if piper is not installed,
         so the system remains functional during development even without
         the voice models downloaded.

Modules: modules/voice/tts.py
Classes: PiperTTS
Functions:
    PiperTTS.__init__(config)      — Initialize with config
    PiperTTS.load()                — Validate piper binary and voice model exist
    PiperTTS.speak(text)           — Synthesize and play text (blocking)
    PiperTTS.speak_async(text)     — Async wrapper (runs in thread)
    PiperTTS.synthesize(text)      — Synthesize to numpy array (blocking)
    PiperTTS.synthesize_async(text)— Async wrapper
    PiperTTS._speak_sapi(text)     — Windows SAPI fallback

Variables:
    PiperTTS.loaded      — bool
    PiperTTS._piper_path — resolved binary path or None
    PiperTTS._model_path — .onnx voice model path or None
    PiperTTS._use_sapi   — bool, True when falling back to Windows SAPI

#todo: Add XTTS-v2 backend for voice cloning / custom voice
#todo: Pre-cache common phrases ("Looking into that...", "Done.") as WAV files
#todo: Add sentence-level streaming so first sentence plays while rest is synthesizing
#todo: Support multiple voices per persona (bedroom vs kitchen Jarvis voice)
#todo: Add SSML-like pause/emphasis markers for more natural speech
"""

import asyncio
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from core.exceptions import TTSError
from modules.voice.audio_utils import play_audio_array, play_audio_array_async


class PiperTTS:
    """
    Piper neural TTS with Windows SAPI fallback.

    Config keys used (from config["voice"]["tts"]):
        engine:        "piper" | "sapi"
        voice:         Piper voice model name (e.g., "en_US-ryan-high")
        piper_binary:  Path to piper executable or just "piper" for PATH lookup
        voices_dir:    Directory containing .onnx and .json voice files
        speed:         Speech rate multiplier (1.0 = normal)
        sample_rate:   Piper output sample rate (default 22050)

    Usage:
        tts = PiperTTS(config)
        tts.load()
        await tts.speak_async("Jarvis online and ready.")
    """

    def __init__(self, config: dict) -> None:
        self._cfg = config["voice"]["tts"]
        self.loaded: bool = False
        self._piper_path: Optional[str] = None
        self._model_path: Optional[str] = None
        self._use_sapi: bool = False
        self._sample_rate: int = self._cfg.get("sample_rate", 22050)

    def load(self) -> None:
        """
        Locate the piper binary and voice model file.
        If piper is unavailable, arms the SAPI fallback silently.

        Raises:
            TTSError: Only if both piper and SAPI are unavailable.
        """
        engine = self._cfg.get("engine", "piper")

        if engine == "sapi":
            self._arm_sapi()
            return

        # Try to find piper binary
        piper_bin = self._cfg.get("piper_binary", "piper")
        resolved = shutil.which(piper_bin)
        if resolved:
            self._piper_path = resolved
            logger.info(f"[TTS] Piper binary: {resolved}")
        else:
            logger.warning(f"[TTS] Piper binary '{piper_bin}' not found in PATH")
            logger.warning("[TTS] Download piper from: https://github.com/rhasspy/piper/releases")
            self._arm_sapi()
            return

        # Locate voice model
        voices_dir = Path(self._cfg.get("voices_dir", "data/voices"))
        voice_name = self._cfg.get("voice", "en_US-ryan-high")
        model_file = voices_dir / f"{voice_name}.onnx"

        if not model_file.exists():
            logger.warning(f"[TTS] Voice model not found: {model_file}")
            logger.warning(
                f"[TTS] Download from: https://huggingface.co/rhasspy/piper-voices/tree/main"
            )
            self._arm_sapi()
            return

        self._model_path = str(model_file)
        self.loaded = True
        logger.info(f"[TTS] Piper ready: {voice_name}")

    def _arm_sapi(self) -> None:
        """
        Attempt to enable Windows SAPI as TTS fallback.
        Sets self._use_sapi = True if pywin32 is available.
        """
        try:
            import win32com.client  # noqa: F401
            self._use_sapi = True
            self.loaded = True
            logger.warning("[TTS] Using Windows SAPI fallback — install piper for better quality")
        except ImportError:
            logger.error("[TTS] Neither piper nor pywin32 available — speech is disabled")
            self.loaded = False

    def speak(self, text: str) -> None:
        """
        Synthesize text and play it immediately. Blocks until playback completes.

        Args:
            text: The text to speak. Must be non-empty.

        Raises:
            TTSError: If synthesis fails and fallback is also unavailable.
        """
        if not text.strip():
            return

        if not self.loaded:
            logger.warning(f"[TTS] Not loaded — skipping: {text!r}")
            return

        if self._use_sapi:
            self._speak_sapi(text)
            return

        audio = self.synthesize(text)
        play_audio_array(audio, self._sample_rate)

    async def speak_async(self, text: str) -> None:
        """Non-blocking wrapper for speak(). Runs in a thread pool."""
        if not text.strip():
            return
        await asyncio.to_thread(self.speak, text)

    def synthesize(self, text: str) -> np.ndarray:
        """
        Run piper and capture raw PCM output as a float32 numpy array.
        Does not play the audio — returns it for the caller to handle.

        Piper writes raw 16-bit little-endian signed PCM to stdout when
        --output-raw is specified. We capture stdout and convert.

        Args:
            text: Text to synthesize.

        Returns:
            float32 numpy array, ready for sounddevice playback.

        Raises:
            TTSError: If the piper subprocess fails.
        """
        if not self._piper_path or not self._model_path:
            raise TTSError("Piper not configured — call load() first")

        cmd = [
            self._piper_path,
            "--model", self._model_path,
            "--output-raw",
            "--length-scale", str(1.0 / self._cfg.get("speed", 1.0)),
        ]

        try:
            result = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
        except subprocess.TimeoutExpired as e:
            raise TTSError(f"Piper timed out synthesizing: {text!r}") from e
        except FileNotFoundError as e:
            raise TTSError(f"Piper binary not found at {self._piper_path}") from e

        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            raise TTSError(f"Piper exited {result.returncode}: {stderr}")

        if not result.stdout:
            raise TTSError("Piper produced no audio output")

        # Convert raw 16-bit LE PCM → float32
        pcm = np.frombuffer(result.stdout, dtype=np.int16)
        audio = pcm.astype(np.float32) / 32768.0
        logger.debug(f"[TTS] Synthesized {len(audio) / self._sample_rate:.2f}s for: {text!r}")
        return audio

    async def synthesize_async(self, text: str) -> np.ndarray:
        """Non-blocking wrapper for synthesize(). Runs in a thread pool."""
        return await asyncio.to_thread(self.synthesize, text)

    def _speak_sapi(self, text: str) -> None:
        """
        Use Windows Speech API (SAPI) to speak text.
        Blocking — waits for speech to complete.
        Only called when piper is unavailable.
        """
        try:
            import win32com.client
            sapi = win32com.client.Dispatch("SAPI.SpVoice")
            sapi.Speak(text)
        except Exception as e:
            logger.error(f"[TTS] SAPI speak failed: {e}")
