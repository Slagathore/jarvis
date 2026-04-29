"""
JARVIS — Ambient Home AI
========================
Mission: Low-level audio I/O utilities shared by the entire voice pipeline.
         Records from microphone until silence is detected, plays audio arrays
         and files, generates confirmation chimes, and enumerates devices.
         All audio is 16kHz mono float32 — the format Whisper and openWakeWord
         require. Conversion and normalization happen here so callers never
         need to think about formats.

Modules: modules/voice/audio_utils.py
Classes: (none — pure functions)
Functions:
    list_input_devices()          → List available microphone devices
    list_output_devices()         → List available speaker devices
    db_from_rms(rms)              → Convert RMS amplitude to dB
    record_until_silence(...)     → Blocking microphone capture
    record_until_silence_async(.) → Async wrapper (runs in thread)
    play_audio_array(audio, ...)  → Play numpy array (blocking)
    play_audio_array_async(...)   → Async wrapper
    play_audio_file(path, ...)    → Load and play audio file (blocking)
    play_audio_file_async(...)    → Async wrapper
    numpy_to_wav_bytes(audio)     → Encode array to WAV bytes
    wav_bytes_to_numpy(wav_bytes) → Decode WAV bytes to array
    play_chime(...)               → Synthesize and play a tone
    play_chime_async(...)         → Async wrapper

Variables:
    SAMPLE_RATE  — 16000 Hz (Whisper/openWakeWord requirement)
    CHANNELS     — 1 (mono)
    DTYPE        — np.float32
    CHUNK_FRAMES — 1024 frames per audio block (~64ms at 16kHz)

#todo: Add VAD (voice activity detection) pre-filter to skip non-speech audio
#todo: Support configurable input/output device by name (not just index)
#todo: Add audio normalization / gain control before transcription
#todo: Add noise suppression using noisereduce library
#todo: Expose a streaming generator version of record_until_silence for real-time processing
"""

import asyncio
import io
import time
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import sounddevice as sd
import soundfile as sf
from loguru import logger

from core.exceptions import AudioError

# ── Module-level audio constants ─────────────────────────────────────────────
SAMPLE_RATE = 16000      # 16kHz — required by Whisper and openWakeWord
CHANNELS = 1             # Mono
DTYPE = np.float32       # sounddevice native float
CHUNK_FRAMES = 1024      # Frames per read block (~64ms at 16kHz)


# ── Device Enumeration ───────────────────────────────────────────────────────

def list_input_devices() -> list[dict[str, Any]]:
    """
    Return all available audio input devices.
    Each entry: {"index": int, "name": str, "channels": int, "default_sr": float}
    """
    devices = sd.query_devices()
    result: list[dict[str, Any]] = []
    for i, device in enumerate(devices):
        d = cast(dict[str, Any], device)
        max_input_channels = int(d.get("max_input_channels", 0))
        if max_input_channels > 0:
            result.append({
                "index": i,
                "name": str(d.get("name", f"Device {i}")),
                "channels": max_input_channels,
                "default_sr": float(d.get("default_samplerate", 0.0)),
            })
    return result


def list_output_devices() -> list[dict[str, Any]]:
    """
    Return all available audio output devices.
    Each entry: {"index": int, "name": str, "channels": int}
    """
    devices = sd.query_devices()
    result: list[dict[str, Any]] = []
    for i, device in enumerate(devices):
        d = cast(dict[str, Any], device)
        max_output_channels = int(d.get("max_output_channels", 0))
        if max_output_channels > 0:
            result.append(
                {
                    "index": i,
                    "name": str(d.get("name", f"Device {i}")),
                    "channels": max_output_channels,
                }
            )
    return result


# ── Level Measurement ────────────────────────────────────────────────────────

def db_from_rms(rms: float) -> float:
    """
    Convert RMS amplitude (0.0–1.0 float) to decibels full scale (dBFS).
    Returns -100.0 for silence (rms near zero) to avoid log(0).
    """
    if rms < 1e-10:
        return -100.0
    return 20.0 * float(np.log10(rms))


# ── Recording ────────────────────────────────────────────────────────────────

def record_until_silence(
    silence_threshold_db: float = -40.0,
    silence_duration_ms: int = 800,
    max_duration_seconds: float = 30.0,
    speech_start_timeout_seconds: Optional[float] = None,
    device: Optional[int | str] = None,
) -> np.ndarray:
    """
    Record audio from the microphone until a sustained period of silence.

    Silence detection logic:
      - Audio below silence_threshold_db is considered silent.
      - Recording only stops after silence_duration_ms of consecutive silence.
      - Leading silence (before any speech) never triggers a stop — we wait
        for the user to start speaking first.
      - Hard limit: max_duration_seconds stops everything regardless.

    Args:
        silence_threshold_db:  Level below which a block is silent (dBFS).
        silence_duration_ms:   Consecutive ms of silence before stopping.
        max_duration_seconds:  Absolute recording time limit.
        speech_start_timeout_seconds:
                               Stop early if no speech starts within this time.
                               None waits the full max duration.
        device:                sounddevice device index or name. None = system default.

    Returns:
        Float32 numpy array, shape (N,), at SAMPLE_RATE Hz.

    Raises:
        AudioError: If no audio could be captured.
    """
    frames_collected: list[np.ndarray] = []

    # How many consecutive silent blocks trigger stop
    silence_blocks_needed = int(
        (silence_duration_ms / 1000.0) * SAMPLE_RATE / CHUNK_FRAMES
    )
    max_blocks = int(max_duration_seconds * SAMPLE_RATE / CHUNK_FRAMES)
    speech_start_timeout_blocks: Optional[int] = None
    if speech_start_timeout_seconds is not None:
        speech_start_timeout_blocks = max(
            1,
            int(speech_start_timeout_seconds * SAMPLE_RATE / CHUNK_FRAMES),
        )

    silence_block_count = 0
    speech_started = False  # Don't stop on pre-speech silence
    first_speech_frame = 0

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=CHUNK_FRAMES,
            device=device,
        ) as stream:
            logger.debug("[Audio] Recording started")
            block_count = 0

            while block_count < max_blocks:
                block, overflowed = stream.read(CHUNK_FRAMES)
                if overflowed:
                    logger.warning("[Audio] Input buffer overflow — audio gap possible")

                frames_collected.append(block.copy())
                block_count += 1

                # Measure energy level of this block
                rms = float(np.sqrt(np.mean(block ** 2)))
                level_db = db_from_rms(rms)

                if level_db > silence_threshold_db:
                    # Speech detected
                    if not speech_started:
                        # Keep ~250ms of pre-roll so initial consonants are not clipped.
                        first_speech_frame = max(0, (block_count - 4) * CHUNK_FRAMES)
                    speech_started = True
                    silence_block_count = 0
                elif speech_started:
                    # Silence after speech — count toward stop threshold
                    silence_block_count += 1
                    if silence_block_count >= silence_blocks_needed:
                        logger.debug("[Audio] Silence detected — stopping")
                        break
                elif (
                    speech_start_timeout_blocks is not None
                    and block_count >= speech_start_timeout_blocks
                ):
                    logger.debug("[Audio] No speech detected before timeout — stopping")
                    break

    except sd.PortAudioError as e:
        raise AudioError(f"PortAudio stream failed: {e}") from e

    if not frames_collected:
        raise AudioError("No audio frames were captured — check microphone")

    audio = np.concatenate(frames_collected, axis=0).flatten()
    if speech_started and first_speech_frame > 0:
        audio = audio[first_speech_frame:]
    duration = len(audio) / SAMPLE_RATE
    logger.debug(f"[Audio] Captured {duration:.2f}s ({len(frames_collected)} blocks)")
    return audio


async def record_until_silence_async(
    silence_threshold_db: float = -40.0,
    silence_duration_ms: int = 800,
    max_duration_seconds: float = 30.0,
    speech_start_timeout_seconds: Optional[float] = None,
    device: Optional[int | str] = None,
) -> np.ndarray:
    """
    Non-blocking wrapper for record_until_silence.
    Runs the blocking capture in a thread pool so the event loop stays alive.
    """
    return await asyncio.to_thread(
        record_until_silence,
        silence_threshold_db=silence_threshold_db,
        silence_duration_ms=silence_duration_ms,
        max_duration_seconds=max_duration_seconds,
        speech_start_timeout_seconds=speech_start_timeout_seconds,
        device=device,
    )


# ── Playback ─────────────────────────────────────────────────────────────────

def play_audio_array(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    device: Optional[int | str] = None,
) -> None:
    """
    Play a float32 numpy audio array through the output device.
    Blocks the calling thread until playback completes.

    Raises:
        AudioError: On PortAudio playback failure.
    """
    try:
        sd.play(audio, samplerate=sample_rate, device=device)
        sd.wait()
    except sd.PortAudioError as e:
        raise AudioError(f"Playback failed: {e}") from e


async def play_audio_array_async(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    device: Optional[int | str] = None,
) -> None:
    """Non-blocking wrapper for play_audio_array."""
    await asyncio.to_thread(play_audio_array, audio, sample_rate, device)


def play_audio_file(path: str | Path, device: Optional[int | str] = None) -> None:
    """
    Load an audio file (WAV, FLAC, OGG) with soundfile and play it.
    Handles mono/stereo conversion automatically.
    Blocks until playback is complete.

    Raises:
        AudioError: If the file cannot be read or playback fails.
    """
    try:
        data, samplerate = sf.read(str(path), dtype="float32")
        if data.ndim > 1:
            # Mix down stereo or multi-channel to mono
            data = data.mean(axis=1)
        sd.play(data, samplerate=samplerate, device=device)
        sd.wait()
    except Exception as e:
        raise AudioError(f"Failed to play '{path}': {e}") from e


async def play_audio_file_async(
    path: str | Path,
    device: Optional[int | str] = None,
) -> None:
    """Non-blocking wrapper for play_audio_file."""
    await asyncio.to_thread(play_audio_file, path, device)


# ── Format Conversion ────────────────────────────────────────────────────────

def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """
    Encode a float32 numpy audio array to WAV bytes in memory.
    Uses 16-bit PCM encoding (standard compatibility).
    """
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def wav_bytes_to_numpy(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Decode WAV bytes into a float32 numpy array.
    Returns (audio_array, sample_rate).
    Automatically mixes down multi-channel audio to mono.
    """
    buffer = io.BytesIO(wav_bytes)
    data, samplerate = sf.read(buffer, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, samplerate


# ── Chime ────────────────────────────────────────────────────────────────────

def play_chime(
    frequency: float = 880.0,
    duration_ms: int = 150,
    device: Optional[int | str] = None,
) -> None:
    """
    Synthesize a sine-wave tone and play it as the wake-word confirmation sound.
    Uses a 10ms fade in/out envelope to prevent clicking.
    No external file dependency — generated entirely in NumPy.

    Args:
        frequency:   Tone frequency in Hz. 880Hz = bright, non-jarring.
        duration_ms: Duration of the tone.
        device:      Output device index.
    """
    n_samples = int(SAMPLE_RATE * duration_ms / 1000.0)
    t = np.linspace(0.0, duration_ms / 1000.0, n_samples, dtype=np.float32)

    # 10ms fade in + fade out to prevent audible clicks
    fade_n = int(0.01 * SAMPLE_RATE)
    envelope = np.ones(n_samples, dtype=np.float32)
    envelope[:fade_n] = np.linspace(0.0, 1.0, fade_n)
    envelope[-fade_n:] = np.linspace(1.0, 0.0, fade_n)

    tone = (np.sin(2.0 * np.pi * frequency * t) * 0.3 * envelope)
    play_audio_array(tone, SAMPLE_RATE, device)


async def play_chime_async(
    frequency: float = 880.0,
    duration_ms: int = 150,
    device: Optional[int | str] = None,
) -> None:
    """Non-blocking wrapper for play_chime."""
    await asyncio.to_thread(play_chime, frequency, duration_ms, device)
