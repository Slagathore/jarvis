"""
JARVIS — Ambient Home AI
========================
Mission: ClapClassifier — open-vocabulary audio understanding. YAMNet
         (SoundEventClassifier) is a CLOSED-vocab classifier: it only
         knows AudioSet's 521 fixed classes. When a sound does not land
         confidently in any of them — YAMNet "does not understand it" —
         CLAP is the open-vocab fallback.

         CLAP (Contrastive Language-Audio Pretraining) is the audio
         analogue of CLIP: it embeds audio and free-text into a shared
         space, so it can score a clip against ARBITRARY natural-language
         descriptions with no retraining ("a microwave finishing",
         "a dog scratching at a door", "a smoke alarm chirping").

         MODEL: laion/clap-htsat-unfused via HuggingFace transformers.
         CLAP wants 48 kHz audio; clips are resampled from Jarvis's
         16 kHz at the edge.

Modules: modules/voice/clap_classifier.py
Classes: ClapClassifier

#todo: cache text embeddings for a fixed candidate list (they never change)
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import numpy as np
from loguru import logger

_CLAP_SAMPLE_RATE = 48000
_DEFAULT_MODEL = "laion/clap-htsat-unfused"


class ClapClassifier:
    """Zero-shot open-vocabulary audio classifier.

    Config keys (from config["voice"]["clap"], all optional):
        enabled:      bool — master switch (default True)
        model:        HF model id (default laion/clap-htsat-unfused)
        device:       "cuda" | "cpu" (default cuda if available)
        min_score:    softmax prob below which a CLAP match is ignored (0.40)
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", True))
        self._model_id = str(cfg.get("model", _DEFAULT_MODEL))
        self._device_pref = str(cfg.get("device", "cuda")).lower()
        self._min_score = float(cfg.get("min_score", 0.40))
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._device = "cpu"
        self.loaded = False

    # ── Loading ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the CLAP model + processor. Blocking — call once at
        startup. Never raises: on failure loaded=False and classify()
        returns nothing, so YAMNet alone still runs."""
        if not self.enabled:
            logger.info("[CLAP] disabled by config")
            return
        try:
            import torch
            from transformers import ClapModel, ClapProcessor

            self._torch = torch
            self._device = (
                "cuda" if (self._device_pref != "cpu" and torch.cuda.is_available())
                else "cpu"
            )
            model: Any = ClapModel.from_pretrained(self._model_id)
            self._model = model.to(self._device)
            self._model.eval()
            self._processor = ClapProcessor.from_pretrained(self._model_id)
            self.loaded = True
            logger.info(f"[CLAP] ready ({self._model_id}, {self._device})")
        except Exception as e:
            logger.warning(f"[CLAP] load failed ({e}) — open-vocab audio disabled")
            self.loaded = False

    # ── Classification ───────────────────────────────────────────────────────

    @staticmethod
    def _to_48k(audio: np.ndarray, src_rate: int) -> np.ndarray:
        """Resample a mono float32 clip to CLAP's 48 kHz."""
        audio = np.asarray(audio, dtype=np.float32).flatten()
        if src_rate == _CLAP_SAMPLE_RATE or audio.size == 0:
            return audio
        from scipy.signal import resample
        n = int(round(audio.size * _CLAP_SAMPLE_RATE / src_rate))
        return np.asarray(resample(audio, n), dtype=np.float32)

    def classify(
        self,
        audio: np.ndarray,
        candidate_labels: list[str],
        *,
        src_rate: int = 16000,
    ) -> list[tuple[str, float]]:
        """Score one clip against free-text `candidate_labels`.
        Returns [(label, probability)] sorted strongest-first. Labels are
        natural-language descriptions — CLAP needs no training on them."""
        if not self.loaded or audio.size == 0 or not candidate_labels:
            return []
        try:
            wav = self._to_48k(audio, src_rate)
            inputs = self._processor(
                text=candidate_labels, audios=wav,
                sampling_rate=_CLAP_SAMPLE_RATE,
                return_tensors="pt", padding=True,
            ).to(self._device)
            with self._torch.no_grad():
                out = self._model(**inputs)
            # logits_per_audio: (1, n_labels) — softmax → a distribution.
            probs = out.logits_per_audio.softmax(dim=-1)[0].cpu().numpy()
        except Exception as e:
            logger.debug(f"[CLAP] inference failed: {e}")
            return []
        scored = sorted(
            zip(candidate_labels, (float(p) for p in probs)),
            key=lambda kv: kv[1], reverse=True,
        )
        return scored

    def best_match(
        self,
        audio: np.ndarray,
        candidate_labels: list[str],
        *,
        src_rate: int = 16000,
    ) -> Optional[tuple[str, float]]:
        """Top candidate if it clears `min_score`, else None — the
        "CLAP recognised it / CLAP also doesn't know" decision."""
        scored = self.classify(audio, candidate_labels, src_rate=src_rate)
        if scored and scored[0][1] >= self._min_score:
            return scored[0]
        return None

    async def classify_async(
        self, audio: np.ndarray, candidate_labels: list[str],
        *, src_rate: int = 16000,
    ) -> list[tuple[str, float]]:
        """Non-blocking classify() — inference runs in a thread."""
        return await asyncio.to_thread(
            self.classify, audio, candidate_labels, src_rate=src_rate
        )
