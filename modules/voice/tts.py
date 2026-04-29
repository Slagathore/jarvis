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
        self._active_voice: str = ""
        self._use_sapi: bool = False
        self._sample_rate: int = self._cfg.get("sample_rate", 22050)

    def load(self) -> None:
        """
        Locate the piper binary and initial voice model.
        Supports absolute piper_binary path and a voices dict in config.
        Falls back to Windows SAPI if piper is unavailable.
        """
        engine = self._cfg.get("engine", "piper")

        if engine == "sapi":
            self._arm_sapi()
            return

        # Resolve piper binary — absolute path takes priority over PATH lookup
        piper_bin = self._cfg.get("piper_binary", "piper")
        if Path(piper_bin).is_file():
            self._piper_path = piper_bin
            logger.info(f"[TTS] Piper binary: {piper_bin}")
        else:
            resolved = shutil.which(piper_bin)
            if resolved:
                self._piper_path = resolved
                logger.info(f"[TTS] Piper binary: {resolved}")
            else:
                logger.warning(f"[TTS] Piper binary '{piper_bin}' not found")
                self._arm_sapi()
                return

        # Load the configured active voice
        voice_name = self._cfg.get("active_voice") or self._cfg.get("voice", "")
        if not self._load_voice(voice_name):
            self._arm_sapi()
            return
        if not self._probe_current_voice():
            logger.warning(
                f"[TTS] Voice '{voice_name}' failed Piper self-test — trying fallbacks"
            )
            if not self._load_fallback_voice(exclude=voice_name):
                self._arm_sapi()
                return

        self.loaded = True

    def _load_voice(self, voice_name: str) -> bool:
        """
        Resolve voice_name to a .onnx path using the voices dict, then fall back
        to voices_dir/{voice_name}.onnx for backward compatibility.
        Returns True if the model was found, False otherwise.
        """
        # Primary: check the voices dict (name → absolute .onnx path)
        voices = self._cfg.get("voices", {})
        if voice_name in voices:
            path = Path(voices[voice_name])
            if path.is_file():
                self._model_path = str(path)
                self._active_voice = voice_name
                logger.info(f"[TTS] Voice: {voice_name} ({path.name})")
                return True
            logger.warning(f"[TTS] Voice model not found: {path}")

        # Fallback: voices_dir/{voice_name}.onnx
        voices_dir = Path(self._cfg.get("voices_dir", "data/voices"))
        model_file = voices_dir / f"{voice_name}.onnx"
        if model_file.exists():
            self._model_path = str(model_file)
            self._active_voice = voice_name
            logger.info(f"[TTS] Voice: {voice_name}")
            return True

        logger.warning(f"[TTS] No model found for voice '{voice_name}'")
        return False

    def _load_fallback_voice(self, exclude: str) -> bool:
        """Try alternate configured voices until one passes synthesis self-test."""
        for voice_name in self.available_voices():
            if voice_name == exclude:
                continue
            if not self._load_voice(voice_name):
                continue
            if self._probe_current_voice():
                logger.warning(f"[TTS] Falling back to voice '{voice_name}'")
                return True
        logger.error("[TTS] No configured Piper voice passed self-test")
        return False

    def _probe_current_voice(self) -> bool:
        """
        Run a short synthesis to verify Piper can actually load the active voice.
        This catches broken/incompatible voice models early and avoids repeated
        runtime crashes on every speak() call.
        """
        if not self._piper_path or not self._model_path:
            return False

        probe_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                probe_path = tmp.name

            result = subprocess.run(
                [
                    self._piper_path,
                    "--model", self._model_path,
                    "--output_file", probe_path,
                    "--length-scale", str(1.0 / self._cfg.get("speed", 1.0)),
                ],
                input=b"Testing voice.",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=20,
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"[TTS] Voice '{self._active_voice}' self-test timed out")
            return False
        except Exception as e:
            logger.warning(f"[TTS] Voice '{self._active_voice}' self-test failed: {e}")
            return False
        finally:
            if probe_path:
                Path(probe_path).unlink(missing_ok=True)

        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            logger.warning(
                f"[TTS] Voice '{self._active_voice}' self-test crashed "
                f"(exit={result.returncode} / {self._format_exit_code(result.returncode)}): "
                f"{stderr or 'no stderr'}"
            )
            return False

        return True

    @staticmethod
    def _format_exit_code(returncode: int) -> str:
        """Return a stable hex rendering for Windows native crash codes."""
        unsigned = (returncode + (1 << 32)) % (1 << 32)
        return hex(unsigned)

    def set_voice(self, voice_name: str) -> bool:
        """
        Switch to a different voice at runtime. Returns True on success.
        The change takes effect on the next speak() call.
        """
        if not self._piper_path:
            logger.warning("[TTS] Cannot switch voice — piper not available")
            return False
        previous_voice = self._active_voice
        previous_model = self._model_path
        if self._load_voice(voice_name):
            if self._probe_current_voice():
                logger.info(f"[TTS] Switched to voice: {voice_name}")
                return True
            self._active_voice = previous_voice
            self._model_path = previous_model
            logger.warning(f"[TTS] Voice '{voice_name}' failed self-test — keeping '{previous_voice}'")
            return False
        logger.warning(f"[TTS] Voice '{voice_name}' not found — keeping '{self._active_voice}'")
        return False

    def available_voices(self) -> list[str]:
        """Return list of voice names defined in config."""
        return list(self._cfg.get("voices", {}).keys())

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
            raise TTSError(
                f"Piper exited {result.returncode} ({self._format_exit_code(result.returncode)}): "
                f"{stderr}"
            )

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
