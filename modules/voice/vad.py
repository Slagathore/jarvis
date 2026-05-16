"""
JARVIS — Ambient Home AI
========================
Mission: VoiceActivityDetector — Stage 1 of the voice cascade. The
         always-on microphone produces audio continuously, but 80-95%
         of it is silence or non-speech. This stage answers one cheap
         question — "is there human-speech-like audio right now?" — so
         the expensive stages (wake-word, STT, LLM) only run on the
         fraction of audio that could possibly matter.

         BACKENDS (auto-selected, best first):
           silero  — Silero VAD via torch.hub. ~2 MB, neural, robust to
                     noise/music. Downloads once from GitHub, then cached.
           webrtc  — Google's WebRTC VAD via the `webrtcvad` package.
                     ~50 KB, pure-C, zero download, very fast. Less
                     discriminative than Silero but a solid fallback.

         The detector is stateful: a hangover window keeps `speech_active`
         True for a short tail after speech stops, so a natural mid-
         sentence pause does not prematurely close the segment.

Modules: modules/voice/vad.py
Classes: VoiceActivityDetector
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from loguru import logger

# Silero operates on fixed 512-sample windows at 16 kHz (32 ms).
_SILERO_WINDOW = 512
VAD_SAMPLE_RATE = 16000


class VoiceActivityDetector:
    """Stateful speech gate for 16 kHz mono audio.

    Config keys (from config["voice"]["vad"], all optional):
        backend:           "auto" | "silero" | "webrtc"   (default auto)
        threshold:         speech probability cutoff, silero only (0.5)
        webrtc_aggressiveness: 0-3, higher = more aggressive filter (2)
        hangover_ms:       keep speech_active this long after speech ends (320)
        min_speech_ms:     speech must persist this long to open (96)
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {})
        self._backend_pref = str(cfg.get("backend", "auto")).lower()
        self._threshold = float(cfg.get("threshold", 0.5))
        self._aggr = int(cfg.get("webrtc_aggressiveness", 2))
        self._hangover_s = float(cfg.get("hangover_ms", 320)) / 1000.0
        self._min_speech_s = float(cfg.get("min_speech_ms", 96)) / 1000.0

        self.backend: str = "none"
        self._silero: Any = None
        self._webrtc: Any = None
        self.loaded = False

        # Runtime state. Onset/hangover are tracked in *audio time*
        # (summed chunk durations), not wall-clock — so the gate behaves
        # identically for live mic input and for replayed/buffered audio,
        # and a burst of catch-up chunks can't skew the timing.
        self._speech_active = False
        self._speech_run_s = 0.0    # consecutive speech audio duration
        self._silence_run_s = 0.0   # consecutive non-speech audio duration
        self._residual = np.zeros(0, dtype=np.float32)

    # ── Loading ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load a VAD backend. Blocking — call once at startup. Never
        raises: on total failure the detector reports `loaded = False`
        and the cascade can treat every chunk as speech (fail-open)."""
        order = (["silero", "webrtc"] if self._backend_pref == "auto"
                 else [self._backend_pref])
        for backend in order:
            try:
                if backend == "silero" and self._load_silero():
                    self.backend = "silero"
                    break
                if backend == "webrtc" and self._load_webrtc():
                    self.backend = "webrtc"
                    break
            except Exception as e:
                logger.warning(f"[VAD] backend '{backend}' failed: {e}")
        self.loaded = self.backend != "none"
        if self.loaded:
            logger.info(f"[VAD] ready (backend={self.backend})")
        else:
            logger.warning("[VAD] no backend available — cascade will fail-open")

    def _load_silero(self) -> bool:
        import torch  # torch is a hard project dependency

        # Prefer the pip package (fully offline); fall back to torch.hub
        # (downloads once from GitHub, then cached locally).
        try:
            from silero_vad import load_silero_vad
            self._silero = load_silero_vad()
        except Exception:
            loaded = torch.hub.load(
                "snakers4/silero-vad", "silero_vad", trust_repo=True
            )
            # torch.hub returns either the model or a (model, utils) tuple.
            self._silero = loaded[0] if isinstance(loaded, tuple) else loaded
        self._silero.eval()
        self._torch = torch
        return True

    def _load_webrtc(self) -> bool:
        import webrtcvad
        self._webrtc = webrtcvad.Vad(self._aggr)
        return True

    # ── Detection ────────────────────────────────────────────────────────────

    def chunk_speech_prob(self, audio: np.ndarray) -> float:
        """Instantaneous speech likelihood for one audio chunk, 0.0-1.0.

        `audio` is float32 mono at 16 kHz in [-1, 1]. For webrtc (which
        is binary) this returns the fraction of 30 ms frames flagged as
        speech, which behaves like a probability for thresholding."""
        if not self.loaded or audio.size == 0:
            return 1.0  # fail-open: unknown → treat as possible speech
        audio = np.asarray(audio, dtype=np.float32)
        if self.backend == "silero":
            return self._silero_prob(audio)
        if self.backend == "webrtc":
            return self._webrtc_prob(audio)
        return 1.0

    def _silero_prob(self, audio: np.ndarray) -> float:
        # Silero needs exact 512-sample windows; buffer the remainder.
        buf = np.concatenate([self._residual, audio])
        n_windows = buf.size // _SILERO_WINDOW
        if n_windows == 0:
            self._residual = buf
            return 0.0
        self._residual = buf[n_windows * _SILERO_WINDOW:]
        probs = []
        with self._torch.no_grad():
            for i in range(n_windows):
                window = buf[i * _SILERO_WINDOW:(i + 1) * _SILERO_WINDOW]
                t = self._torch.from_numpy(window).float()
                probs.append(float(self._silero(t, VAD_SAMPLE_RATE).item()))
        return max(probs) if probs else 0.0

    def _webrtc_prob(self, audio: np.ndarray) -> float:
        # webrtcvad wants 16-bit PCM in 10/20/30 ms frames.
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        frame_len = int(VAD_SAMPLE_RATE * 0.03)  # 30 ms
        n = pcm16.size // frame_len
        if n == 0:
            return 0.0
        speech = 0
        for i in range(n):
            frame = pcm16[i * frame_len:(i + 1) * frame_len].tobytes()
            try:
                if self._webrtc.is_speech(frame, VAD_SAMPLE_RATE):
                    speech += 1
            except Exception:
                pass
        return speech / n

    def update(self, audio: np.ndarray) -> bool:
        """Feed the next chunk; return whether speech is currently active.

        Applies onset debouncing (speech must persist `min_speech_ms`
        before the gate opens) and a hangover tail (the gate stays open
        `hangover_ms` past the last speech so a pause does not cut a
        sentence in half). This is the boolean the cascade gates on."""
        audio = np.asarray(audio, dtype=np.float32)
        duration_s = audio.size / float(VAD_SAMPLE_RATE)
        is_speech = self.chunk_speech_prob(audio) >= self._threshold

        if is_speech:
            self._speech_run_s += duration_s
            self._silence_run_s = 0.0
            if (not self._speech_active
                    and self._speech_run_s >= self._min_speech_s):
                self._speech_active = True
        else:
            self._silence_run_s += duration_s
            self._speech_run_s = 0.0
            if (self._speech_active
                    and self._silence_run_s >= self._hangover_s):
                self._speech_active = False
        return self._speech_active

    @property
    def speech_active(self) -> bool:
        return self._speech_active

    def reset(self) -> None:
        """Clear runtime state — call between independent audio streams."""
        self._speech_active = False
        self._speech_run_s = 0.0
        self._silence_run_s = 0.0
        self._residual = np.zeros(0, dtype=np.float32)
