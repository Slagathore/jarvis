"""
JARVIS — Ambient Home AI
========================
Mission: Speech-to-text transcription using faster-whisper (CTranslate2 backend).
         Loads the Whisper model once at startup and provides both sync and async
         transcription. GPU (CUDA float16) is used when available; falls back to
         CPU int8 automatically. Transcription runs in a thread pool to keep the
         event loop responsive.

Modules: modules/voice/stt.py
Classes: WhisperSTT
Functions:
    WhisperSTT.__init__(config)        — Store config, model not loaded yet
    WhisperSTT.load()                  — Load model (blocking, call from thread/init)
    WhisperSTT.transcribe(audio)       — Transcribe float32 numpy array (blocking)
    WhisperSTT.transcribe_async(audio) — Async wrapper (runs transcribe in thread)

Variables:
    WhisperSTT.loaded  — bool, True after load() succeeds
    WhisperSTT.model   — faster_whisper.WhisperModel instance

#todo: Add word-level timestamps for lip-sync or karaoke-style display
#todo: Add language detection mode (don't force language=en)
#todo: Stream transcription for real-time partial results
#todo: Add post-processing to strip transcription artifacts ("Uh," "um," etc.)
#todo: Cache last N transcripts for context — helps LLM understand incomplete sentences
"""

import asyncio
from typing import Optional

import numpy as np
from loguru import logger

from core.exceptions import STTError


class WhisperSTT:
    """
    Faster-Whisper speech-to-text wrapper.

    Config keys used (from config["voice"]["whisper"]):
        model_size:    "tiny" | "base" | "small" | "medium" | "large-v3"
        device:        "cuda" | "cpu"
        compute_type:  "float16" (GPU) | "int8" (CPU)
        language:      "en" (or None for auto-detect)
        beam_size:     int, 1 = fastest

    Usage:
        stt = WhisperSTT(config)
        stt.load()                           # Call once at startup
        text = await stt.transcribe_async(audio_array)
    """

    def __init__(self, config: dict) -> None:
        self._cfg = config["voice"]["whisper"]
        self.loaded: bool = False
        self.model = None  # faster_whisper.WhisperModel — loaded lazily

    def load(self) -> None:
        """
        Load the Whisper model. This is a blocking call (~2-10s depending on GPU).
        Call this from a thread or during startup before the event loop is busy.

        Falls back to CPU + int8 if CUDA is requested but unavailable.

        Raises:
            STTError: If the model cannot be loaded.
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise STTError("faster-whisper not installed. Run: pip install faster-whisper") from e

        model_size = self._cfg["model_size"]
        device = self._cfg["device"]
        compute_type = self._cfg["compute_type"]

        logger.info(f"[STT] Loading Whisper '{model_size}' on {device} ({compute_type})...")

        try:
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
            )
            self.loaded = True
            logger.info(f"[STT] Whisper '{model_size}' ready")

        except Exception as e:
            # If CUDA requested but unavailable, retry on CPU
            if device == "cuda":
                logger.warning(f"[STT] CUDA failed ({e}) — retrying on CPU/int8")
                try:
                    self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
                    self.loaded = True
                    logger.warning("[STT] Running Whisper on CPU — transcription will be slower")
                except Exception as cpu_err:
                    raise STTError(f"Whisper failed on both CUDA and CPU: {cpu_err}") from cpu_err
            else:
                raise STTError(f"Failed to load Whisper model: {e}") from e

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe a float32 numpy audio array to text.
        Audio must be 16kHz mono. Returns the stripped transcript string.
        Returns an empty string if nothing was heard.

        Args:
            audio: float32 numpy array at 16kHz mono.

        Returns:
            Transcribed text string, stripped of leading/trailing whitespace.

        Raises:
            STTError: If the model is not loaded or transcription fails.
        """
        if not self.loaded or self.model is None:
            raise STTError("Whisper model not loaded — call load() first")

        # Whisper expects float32, 16kHz. Ensure no clipping.
        audio = audio.astype(np.float32)
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / (np.abs(audio).max() + 1e-8)

        try:
            segments, _info = self.model.transcribe(
                audio,
                beam_size=self._cfg.get("beam_size", 1),
                language=self._cfg.get("language") or None,
                # Suppress hallucinations on silent or marginal audio. Higher
                # no_speech_threshold = drop more "I'm guessing" segments.
                no_speech_threshold=float(self._cfg.get("no_speech_threshold", 0.6)),
                log_prob_threshold=float(self._cfg.get("log_prob_threshold", -1.0)),
                condition_on_previous_text=False,
                vad_filter=bool(self._cfg.get("vad_filter", True)),
                vad_parameters=self._cfg.get("vad_parameters") or None,
            )
            text = " ".join(seg.text for seg in segments).strip()
            logger.debug(f"[STT] Transcript: {text!r}")
            return text

        except Exception as e:
            raise STTError(f"Transcription failed: {e}") from e

    async def transcribe_async(self, audio: np.ndarray) -> str:
        """
        Non-blocking wrapper for transcribe().
        Runs the CPU/GPU-bound transcription in a thread pool.
        The event loop remains responsive during transcription.
        """
        return await asyncio.to_thread(self.transcribe, audio)
