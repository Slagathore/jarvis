"""
JARVIS — Ambient Home AI
========================
Mission: Provide a centralized, typed exception hierarchy for all Jarvis subsystems.
         Every module raises a specific subclass so the orchestrator can handle
         each failure mode differently without catching bare Exception.

Hierarchy:
    JarvisError
    ├── ConfigError
    ├── VoiceError
    │   ├── AudioError
    │   ├── STTError
    │   ├── TTSError
    │   └── WakeWordError
    ├── LLMError
    ├── DatabaseError
    ├── MQTTError
    ├── VisionError
    ├── NodeError
    └── ActivityDetectionError

Modules: core/exceptions.py
Classes: JarvisError, ConfigError, VoiceError, AudioError, STTError, TTSError,
         WakeWordError, LLMError, DatabaseError, MQTTError, VisionError,
         NodeError, ActivityDetectionError

Variables: (none — this file is purely exception definitions)

#todo: Add retry-able mixin (RetryableError) for transient failures like network timeouts
#todo: Add error code enum for structured logging and dashboard error display
#todo: Consider adding context dict to base JarvisError for structured error payloads
"""


class JarvisError(Exception):
    """Base exception for all Jarvis errors. Never raise this directly — use a subclass."""
    pass


# ── Configuration ────────────────────────────────────────────────────────────

class ConfigError(JarvisError):
    """
    Raised when config.yaml is missing required fields, has invalid values,
    or cannot be parsed. Usually fatal at startup.
    """
    pass


# ── Voice Pipeline ───────────────────────────────────────────────────────────

class VoiceError(JarvisError):
    """Base class for all voice pipeline failures."""
    pass


class AudioError(VoiceError):
    """
    Raised when sounddevice operations fail: device not found,
    stream overflow, playback failure, or format mismatch.
    """
    pass


class STTError(VoiceError):
    """
    Raised when faster-whisper fails to load a model, transcribe audio,
    or when the audio input is in an unexpected format.
    """
    pass


class TTSError(VoiceError):
    """
    Raised when piper binary is missing, voice model not found,
    synthesis subprocess crashes, or SAPI fallback fails.
    """
    pass


class WakeWordError(VoiceError):
    """
    Raised when the openWakeWord model fails to load, the audio stream
    cannot be opened, or the detection loop crashes irrecoverably.
    """
    pass


# ── LLM / Brain ──────────────────────────────────────────────────────────────

class LLMError(JarvisError):
    """
    Raised when Ollama is unreachable, the requested model is not available,
    the request times out, or the response cannot be parsed.
    """
    pass


# ── Database ─────────────────────────────────────────────────────────────────

class DatabaseError(JarvisError):
    """
    Raised when SQLite operations fail: connection error, schema mismatch,
    query failure, or data corruption.
    """
    pass


# ── Network / MQTT ───────────────────────────────────────────────────────────

class MQTTError(JarvisError):
    """
    Raised when the MQTT broker is unreachable, authentication fails,
    or publish/subscribe operations fail after reconnect attempts are exhausted.
    """
    pass


class NodeError(JarvisError):
    """
    Raised when an ESP32-CAM node is unreachable, sends malformed data,
    or fails to respond within the expected heartbeat window.
    """
    pass


# ── Vision ───────────────────────────────────────────────────────────────────

class VisionError(JarvisError):
    """
    Raised when camera capture fails, OpenCV cannot open a device/stream,
    or a vision model (YOLO, MediaPipe) fails to initialize.
    """
    pass


# ── Activity Detection ───────────────────────────────────────────────────────

class ActivityDetectionError(JarvisError):
    """
    Raised when PC monitor cannot access process list (permissions),
    audio classification model fails, or state fusion receives contradictory signals.
    """
    pass
