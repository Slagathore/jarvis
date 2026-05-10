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
DTYPE = np.float32       # exposed format for downstream callers (Whisper)
# Capture as int16 because some USB sound cards (Sound Blaster G5, etc.) ship
# zero-filled buffers when asked to capture as float32 directly even though
# PortAudio claims auto-conversion. Convert to float32 in Python after read.
CAPTURE_DTYPE = np.int16
INT16_TO_FLOAT = 1.0 / 32768.0
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
    mode: str = "silence",
    fixed_duration_seconds: float = 7.0,
    adaptive_noise_floor: bool = True,
    noise_floor_calibration_ms: int = 400,
    noise_floor_margin_db: float = 5.0,
) -> np.ndarray:
    """
    Record audio from the microphone.

    Two modes:
      "silence" — Wait for speech (block above silence_threshold_db), then stop
                  after silence_duration_ms of consecutive silence. Hard cap at
                  max_duration_seconds. Best when mic+voice reliably exceed the
                  threshold.
      "fixed"   — Record exactly fixed_duration_seconds regardless of audio
                  level. No threshold gating. Best when the mic/voice combo
                  doesn't reliably exceed any silence threshold so silence-mode
                  never starts capturing.

    Adaptive noise floor (silence mode only):
      The first noise_floor_calibration_ms of audio are treated as ambient and
      used to compute the room's actual noise floor. The effective silence
      threshold becomes max(silence_threshold_db, p75_calibration + margin) so
      mics in noisy rooms (HyperX SoloCast in an office with a fan running)
      don't read ambient noise as ongoing speech and burn the max_duration cap
      on every utterance. Set adaptive_noise_floor=False to use the static
      threshold only.

    Args:
        silence_threshold_db:        Lower bound for the silence threshold (dBFS).
        silence_duration_ms:         Consecutive ms of silence before stopping.
        max_duration_seconds:        Absolute recording time limit.
        speech_start_timeout_seconds:Stop early if no speech in this time.
        device:                      sounddevice device index or name. None = system default.
        mode:                        "silence" or "fixed".
        fixed_duration_seconds:      (fixed mode) How long to record.
        adaptive_noise_floor:        Auto-tune the silence threshold from ambient.
        noise_floor_calibration_ms:  How long to sample ambient before locking the threshold.
        noise_floor_margin_db:       Speech must exceed (calibrated_floor + margin).

    Returns:
        Float32 numpy array, shape (N,), at SAMPLE_RATE Hz.

    Raises:
        AudioError: If no audio could be captured.
    """
    frames_collected: list[np.ndarray] = []

    # Fixed-duration mode bypasses all silence/threshold logic — just record N
    # seconds. Use this when your mic doesn't reliably exceed any threshold so
    # silence-mode never even starts capturing.
    if mode == "fixed":
        fixed_blocks = max(1, int(fixed_duration_seconds * SAMPLE_RATE / CHUNK_FRAMES))
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=CAPTURE_DTYPE,
                blocksize=CHUNK_FRAMES,
                device=device,
            ) as stream:
                logger.debug(f"[Audio] Recording fixed {fixed_duration_seconds:.1f}s")
                for _ in range(fixed_blocks):
                    block_int16, overflowed = stream.read(CHUNK_FRAMES)
                    if overflowed:
                        logger.warning("[Audio] Input buffer overflow — audio gap possible")
                    frames_collected.append(
                        block_int16.astype(np.float32) * INT16_TO_FLOAT
                    )
        except sd.PortAudioError as e:
            raise AudioError(f"PortAudio stream failed: {e}") from e

        if not frames_collected:
            raise AudioError("No audio frames were captured — check microphone")
        audio = np.concatenate(frames_collected, axis=0).flatten()
        logger.debug(f"[Audio] Captured {len(audio) / SAMPLE_RATE:.2f}s ({len(frames_collected)} blocks, fixed)")
        return audio

    # Silence-detection mode (legacy behavior + adaptive noise floor)
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

    calibration_blocks = max(
        1, int(noise_floor_calibration_ms / 1000.0 * SAMPLE_RATE / CHUNK_FRAMES)
    )
    calibration_levels: list[float] = []
    effective_threshold_db: float = silence_threshold_db
    threshold_locked: bool = not adaptive_noise_floor

    silence_block_count = 0
    speech_started = False  # Don't stop on pre-speech silence
    first_speech_frame = 0

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=CAPTURE_DTYPE,   # int16 native — see CAPTURE_DTYPE comment
            blocksize=CHUNK_FRAMES,
            device=device,
        ) as stream:
            logger.debug("[Audio] Recording started")
            block_count = 0

            while block_count < max_blocks:
                block_int16, overflowed = stream.read(CHUNK_FRAMES)
                if overflowed:
                    logger.warning("[Audio] Input buffer overflow — audio gap possible")

                # Convert int16 to float32 [-1, 1] for downstream (Whisper expects float32)
                block = block_int16.astype(np.float32) * INT16_TO_FLOAT
                frames_collected.append(block)
                block_count += 1

                # Measure energy level of this block
                rms = float(np.sqrt(np.mean(block ** 2)))
                level_db = db_from_rms(rms)

                # Adaptive noise floor: collect ambient RMS during the calibration
                # window, then lock the threshold to (75th-percentile + margin).
                # 75th-percentile (instead of max) so a stray clack doesn't
                # poison the floor. Floor never goes BELOW the configured
                # silence_threshold_db, so this can only RAISE the bar.
                if not threshold_locked:
                    calibration_levels.append(level_db)
                    if block_count >= calibration_blocks:
                        ambient_p75 = float(
                            np.percentile(calibration_levels, 75)
                        )
                        adaptive_db = ambient_p75 + noise_floor_margin_db
                        effective_threshold_db = max(
                            silence_threshold_db, adaptive_db
                        )
                        threshold_locked = True
                        logger.debug(
                            f"[Audio] Noise floor calibrated: ambient_p75="
                            f"{ambient_p75:.1f} dBFS → threshold="
                            f"{effective_threshold_db:.1f} dBFS "
                            f"(static={silence_threshold_db:.1f}, "
                            f"adaptive={adaptive_db:.1f})"
                        )

                if level_db > effective_threshold_db:
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


async def record_until_silence_from_chunks(
    attach_tap,
    detach_tap,
    silence_threshold_db: float = -40.0,
    silence_duration_ms: int = 800,
    max_duration_seconds: float = 30.0,
    speech_start_timeout_seconds: Optional[float] = None,
    chunk_sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Capture audio from a callback-driven mic source until silence, mirroring
    `record_until_silence`'s semantics but consuming chunks via a tap instead
    of opening a sounddevice InputStream.

    Used by the orchestrator when wake fires in a non-office room — the
    Wyze RTSP mics deliver audio via callback (no sounddevice device index),
    so we install a tap on the room's wake adapter, buffer chunks here,
    apply the same RMS / silence detection logic, and return the captured
    float32 array at SAMPLE_RATE Hz.

    Args:
        attach_tap:   `def(callback) -> None`. The orchestrator passes
                      `adapter.attach_recording_tap`. Wires `callback` to
                      receive every PCM chunk the underlying mic emits.
        detach_tap:   `def() -> None`. The orchestrator passes
                      `adapter.detach_recording_tap`. Always called from a
                      `finally` so a stuck recording can't poison wake
                      detection forever.
        chunk_sample_rate: The rate the source emits at. WyzeRtspMicSource
                      resamples to 16 kHz which matches SAMPLE_RATE; if the
                      first chunk's reported rate disagrees we resample to
                      SAMPLE_RATE so downstream STT stays consistent.

    Returns:
        Float32 numpy array, shape (N,), at SAMPLE_RATE Hz.

    Raises:
        AudioError: If no audio could be captured.
    """
    queue: "asyncio.Queue[tuple[bytes, int]]" = asyncio.Queue(maxsize=256)

    async def _tap(pcm: bytes, sr: int) -> None:
        try:
            queue.put_nowait((pcm, sr))
        except asyncio.QueueFull:
            # Drop oldest. Audio capture is a real-time process; backing
            # up the queue means we're falling behind anyway, and the
            # newest packets matter more than the oldest for silence
            # detection at the tail of an utterance.
            try:
                _ = queue.get_nowait()
                queue.put_nowait((pcm, sr))
            except Exception:
                pass

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

    frames_collected: list[np.ndarray] = []
    silence_block_count = 0
    speech_started = False
    first_speech_frame = 0
    block_count = 0

    # Re-chunker: source chunks may not align with CHUNK_FRAMES, so we
    # buffer and slice. WyzeRtspMicSource emits 1280-sample chunks today
    # but we don't want to depend on that.
    pending = bytearray()
    bytes_per_block = CHUNK_FRAMES * 2  # int16 mono
    observed_rate = chunk_sample_rate

    attach_tap(_tap)
    try:
        deadline_per_chunk = max(0.5, max_duration_seconds + 1.0)
        while block_count < max_blocks:
            try:
                pcm, sr = await asyncio.wait_for(
                    queue.get(), timeout=deadline_per_chunk
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"[Audio] Tap-based recording timed out waiting for "
                    f"chunk after {deadline_per_chunk:.1f}s — bailing"
                )
                break
            if sr != observed_rate:
                observed_rate = sr
            pending.extend(pcm)
            while len(pending) >= bytes_per_block:
                block_int16 = np.frombuffer(
                    bytes(pending[:bytes_per_block]), dtype=np.int16
                )
                del pending[:bytes_per_block]
                if observed_rate != SAMPLE_RATE:
                    # Cheap linear resample to canonical rate. Sources
                    # SHOULD already deliver at SAMPLE_RATE; this is a
                    # safety net so a mis-configured source doesn't
                    # silently produce mis-sampled audio for STT.
                    factor = SAMPLE_RATE / float(observed_rate)
                    out_len = max(1, int(round(block_int16.size * factor)))
                    block_int16 = np.interp(
                        np.linspace(0.0, 1.0, num=out_len, endpoint=False),
                        np.linspace(
                            0.0, 1.0, num=block_int16.size, endpoint=False
                        ),
                        block_int16.astype(np.float64),
                    ).astype(np.int16)
                block = block_int16.astype(np.float32) * INT16_TO_FLOAT
                frames_collected.append(block)
                block_count += 1
                rms = float(np.sqrt(np.mean(block ** 2)))
                level_db = db_from_rms(rms)

                if level_db > silence_threshold_db:
                    if not speech_started:
                        first_speech_frame = max(
                            0, (block_count - 4) * CHUNK_FRAMES
                        )
                    speech_started = True
                    silence_block_count = 0
                elif speech_started:
                    silence_block_count += 1
                    if silence_block_count >= silence_blocks_needed:
                        logger.debug(
                            "[Audio] Silence detected — stopping (tap)"
                        )
                        break
                elif (
                    speech_start_timeout_blocks is not None
                    and block_count >= speech_start_timeout_blocks
                ):
                    logger.debug(
                        "[Audio] No speech detected before timeout — "
                        "stopping (tap)"
                    )
                    break
            else:
                continue
            # Inner break propagates out — breaks the outer while too.
            if speech_started and silence_block_count >= silence_blocks_needed:
                break
            if (
                not speech_started
                and speech_start_timeout_blocks is not None
                and block_count >= speech_start_timeout_blocks
            ):
                break
    finally:
        try:
            detach_tap()
        except Exception as e:
            logger.warning(f"[Audio] detach_tap raised: {e}")

    if not frames_collected:
        raise AudioError(
            "No audio frames were captured from chunk stream — tap may "
            "have detached before any chunk arrived"
        )
    audio = np.concatenate(frames_collected, axis=0).flatten()
    if speech_started and first_speech_frame > 0:
        audio = audio[first_speech_frame:]
    duration = len(audio) / SAMPLE_RATE
    logger.debug(
        f"[Audio] Tap-based capture: {duration:.2f}s "
        f"({len(frames_collected)} blocks)"
    )
    return audio


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


def chime_bytes(
    frequency: float = 880.0,
    duration_ms: int = 150,
) -> tuple[bytes, int]:
    """Return the wake-confirmation chime as raw int16 LE PCM bytes plus
    its sample rate. Used by the orchestrator to route the chime through
    SpeakerManager (and thus to the room where wake fired) instead of
    always playing via PC sounddevice. Same waveform as `play_chime` —
    same fade envelope, same amplitude — just emitted as bytes the
    SpeakerSink interface accepts.
    """
    n_samples = int(SAMPLE_RATE * duration_ms / 1000.0)
    t = np.linspace(0.0, duration_ms / 1000.0, n_samples, dtype=np.float32)
    fade_n = int(0.01 * SAMPLE_RATE)
    envelope = np.ones(n_samples, dtype=np.float32)
    envelope[:fade_n] = np.linspace(0.0, 1.0, fade_n)
    envelope[-fade_n:] = np.linspace(1.0, 0.0, fade_n)
    tone = np.sin(2.0 * np.pi * frequency * t) * 0.3 * envelope
    pcm_int16 = np.clip(tone * 32767.0, -32768, 32767).astype(np.int16)
    return pcm_int16.tobytes(), SAMPLE_RATE
