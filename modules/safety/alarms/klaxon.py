"""
JARVIS — Safety
===============
Mission: Klaxon audio library for the alarm subsystem. Loads each
         alarm-type's klaxon sound file at boot, decodes and
         resamples to a single canonical PCM format, and serves the
         bytes to AlarmAudio's per-room fanout loop.

         Cole's preferred files live in `modules/safety/alarms/` next
         to this module (catescapealarm.mp3, dooralarm.mp3,
         firealarm.wav, clownalarm.wav). Filename → alarm_type
         mapping is fuzzy substring (case-insensitive); a missing
         file degrades to TTS-only for that alarm type rather than
         crashing the alarm.

         Decode path uses PyAV (already a dep — used by Wyze RTSP).
         Output is int16 mono at `target_rate` (default 16kHz to
         match Piper TTS), so SpeakerManager's per-room resamplers
         see exactly one canonical format.

         Per §29.5 the spec calls for distinct klaxons + 30s
         escalation. v4 ships with whatever Cole drops in this
         folder — we don't generate or vary them. Escalation lands
         when there are paired `_base` / `_escalated` files; until
         then, the same clip plays for both phases.

Modules: modules/safety/alarms/klaxon.py
Classes: KlaxonLibrary
Spec:    new 2.md §29.5.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

# Klaxon assets default to this directory (alongside the alarm modules).
# Override via KlaxonLibrary(klaxon_dir=...) in tests.
_DEFAULT_KLAXON_DIR = Path(__file__).resolve().parent

# Filename → alarm-type fuzzy match. Order matters: first hit wins, so
# put more specific tokens before generic ones. The same filename can
# appear with .mp3 or .wav — both are decoded.
_NAME_TOKENS: list[tuple[str, str]] = [
    ("catescape", "cat_escape"),
    ("cat_escape", "cat_escape"),
    ("door",      "door_open"),
    ("fire",      "fire"),
    # Decorative / non-standard. Available for explicit lookup but never
    # auto-selected by an alarm type.
    ("clown",     "clown"),
]


class KlaxonLibrary:
    """Loads + caches klaxon PCM by alarm type. Construct once at
    boot, share the instance with every alarm dispatcher."""

    def __init__(
        self,
        klaxon_dir: Optional[Path] = None,
        target_rate: int = 16000,
    ) -> None:
        self._dir = Path(klaxon_dir) if klaxon_dir else _DEFAULT_KLAXON_DIR
        self._target_rate = int(target_rate)
        # alarm_type → (pcm bytes int16 mono, rate). Empty when nothing
        # was found / loaded; per-alarm consumers fall back to TTS-only.
        self._cache: dict[str, tuple[bytes, int]] = {}
        # Source-file path per alarm type, for diagnostics + dashboard.
        self._sources: dict[str, Path] = {}

    def load_all(self) -> None:
        """Discover and decode every supported file in the klaxon dir.
        Non-fatal — a corrupt file logs a warning and is skipped."""
        if not self._dir.exists():
            logger.info(
                f"[Klaxon] dir {self._dir} missing — alarms will run "
                "TTS-only without klaxons"
            )
            return
        candidates: list[Path] = []
        for ext in ("*.mp3", "*.wav", "*.ogg", "*.flac"):
            candidates.extend(self._dir.glob(ext))
        if not candidates:
            logger.info(
                f"[Klaxon] no audio files in {self._dir} — TTS-only mode"
            )
            return
        for path in candidates:
            alarm_type = self._classify(path.name)
            if alarm_type is None:
                logger.debug(f"[Klaxon] skipping unmapped file '{path.name}'")
                continue
            # Don't clobber an already-loaded type — first match per type
            # wins. Lets you have e.g. firealarm.wav and not have a later
            # fire_backup.mp3 silently replace it.
            if alarm_type in self._cache:
                logger.debug(
                    f"[Klaxon] '{alarm_type}' already loaded, skipping "
                    f"'{path.name}'"
                )
                continue
            try:
                pcm = self._decode_to_int16_mono(path, self._target_rate)
            except Exception as e:
                logger.warning(
                    f"[Klaxon] decode failed for '{path.name}': {e}"
                )
                continue
            self._cache[alarm_type] = (pcm, self._target_rate)
            self._sources[alarm_type] = path
            logger.info(
                f"[Klaxon] '{alarm_type}' ← {path.name} "
                f"({len(pcm) // 2} samples @ {self._target_rate}Hz)"
            )

    def get(self, alarm_type: str) -> Optional[tuple[bytes, int]]:
        """PCM bytes + rate for `alarm_type`, or None if unmapped."""
        return self._cache.get(alarm_type)

    def source_path(self, alarm_type: str) -> Optional[Path]:
        """File the klaxon was decoded from (diagnostic only)."""
        return self._sources.get(alarm_type)

    def known_types(self) -> list[str]:
        """Alarm types with a loaded klaxon. AlarmAudio uses this to
        decide whether to do klaxon-first-then-TTS or TTS-only."""
        return sorted(self._cache.keys())

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _classify(name: str) -> Optional[str]:
        """Filename → alarm_type via fuzzy substring match."""
        lower = name.lower()
        for token, alarm_type in _NAME_TOKENS:
            if token in lower:
                return alarm_type
        return None

    @staticmethod
    def _decode_to_int16_mono(path: Path, target_rate: int) -> bytes:
        """Decode any PyAV-supported audio file → int16 mono PCM at
        `target_rate`. Stereo gets averaged to mono. Single-pass
        streaming decode + resample via PyAV's AudioResampler."""
        import av
        from av.audio.resampler import AudioResampler

        # s16 = signed 16-bit packed; 'mono' layout collapses channels.
        resampler = AudioResampler(
            format="s16", layout="mono", rate=target_rate,
        )
        chunks: list[np.ndarray] = []
        ctr = av.open(str(path))
        try:
            stream = ctr.streams.audio[0]
            for frame in ctr.decode(stream):
                # Resample returns a list of 0+ output frames per input
                # frame (rate conversion can produce uneven lengths).
                for out in resampler.resample(frame):
                    arr = out.to_ndarray()
                    # AudioResampler returns shape (1, N) for mono; flatten.
                    chunks.append(arr.reshape(-1).astype(np.int16, copy=False))
            # Flush the resampler — any tail samples not yet emitted.
            for out in resampler.resample(None):
                arr = out.to_ndarray()
                chunks.append(arr.reshape(-1).astype(np.int16, copy=False))
        finally:
            ctr.close()
        if not chunks:
            return b""
        pcm = np.concatenate(chunks)
        return pcm.tobytes()
