"""
JARVIS — Ambient Home AI
========================
Mission: Typed configuration layer. Loads config.yaml, expands ${ENV_VAR}
         references against the loaded .env, and validates the per-room
         video / mic / speaker channel definitions via Pydantic discriminated
         unions. Bad input fails fast at boot with a clear message instead of
         exploding 20 minutes later inside a driver constructor.

         Lives ALONGSIDE the existing dict-based config flow in main.py — the
         orchestrator + most modules still receive a plain dict for backwards
         compatibility, but new code (CameraManager, MicManager,
         SpeakerManager) calls validate_rooms() to get typed RoomConfig
         models with autocomplete and exhaustive type matching.

         Why discriminated unions: the room schema has three "channel" slots
         (video / mic / speaker) and each slot can be one of several driver
         types. Pydantic's Field(discriminator="type") looks at the literal
         "type" field in the YAML to pick the right model. A typo like
         "type: wyze_rstp" produces a clean validation error pointing at the
         exact room — much friendlier than a KeyError ten layers down.

Modules: core/config.py
Classes:
    RootConfig          — Top-level wrapper, currently only validates rooms.
    RoomConfig          — Per-room video/mic/speaker triple.
    *VideoCfg / *MicCfg / *SpeakerCfg — Driver-specific config dataclasses.

Functions:
    expand_env_vars(obj) → recursively replace ${VAR} with os.environ[VAR].
    validate_rooms(cfg)  → return list[RoomConfig], raise ConfigError on bad input.

#todo: Migrate the rest of config.yaml (system, voice, vision, etc.) to Pydantic
       once the room migration proves out. Keep dict access alive for now to
       avoid a 50-file refactor in one push.
#todo: Add a `drivers:` defaults block validator so cross-driver tunables
       (buffer_size, chunk_ms, etc.) get the same fail-fast treatment.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from core.exceptions import ConfigError

# ── Env-var interpolation ────────────────────────────────────────────────────

# Matches ${VAR_NAME} — uppercase letters, digits, underscores. Anchored to
# avoid eating literal `$` followed by a brace expression in some other
# language (none in our YAML today, but cheap to be precise).
_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def expand_env_vars(obj: Any) -> Any:
    """Recursively replace ${VAR} tokens in any string leaves of a dict/list
    tree using os.environ. Non-strings pass through unchanged.

    Raises ConfigError if a referenced variable is unset — a missing password
    should fail at startup, not produce a broken-looking RTSP URL with the
    literal text "${WYZE_RTSP_PASSWORD}" inside it.
    """
    if isinstance(obj, str):
        def _sub(m: re.Match[str]) -> str:
            var = m.group(1)
            val = os.environ.get(var)
            if val is None:
                raise ConfigError(
                    f"Config references ${{{var}}} but env var is unset. "
                    f"Add it to your .env file or shell environment."
                )
            return val
        return _ENV_VAR_PATTERN.sub(_sub, obj)
    if isinstance(obj, dict):
        return {k: expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env_vars(v) for v in obj]
    return obj


# ── Video source configs ─────────────────────────────────────────────────────


class WyzeRtspVideoCfg(BaseModel):
    """Wyze V2 + wz_mini_hacks RTSP. URL pattern is wz_mini-build-specific —
    the original wz_mini used /unicast on port 554, newer builds use
    /video6_unicast on 8554. Keep the full URL configurable per-room so a
    future firmware change is a YAML edit, not a code change.
    """
    type: Literal["wyze_rtsp"]
    url: str
    transport: Literal["tcp", "udp"] = "tcp"


class Esp32HttpVideoCfg(BaseModel):
    """ESP32-CAM serving MJPEG/snapshots over HTTP. The existing CameraManager
    fetches one JPEG per request rather than holding a persistent stream —
    the ESPHome web server periodically drops long-lived connections.
    """
    type: Literal["esp32_http"]
    url: str


class UsbIndexVideoCfg(BaseModel):
    """USB webcam on the host. device_index is the cv2.VideoCapture int."""
    type: Literal["usb_index"]
    device_index: int = 0


class NoneVideoCfg(BaseModel):
    """Vision disabled in this room — grab_frame() returns None."""
    type: Literal["none"]


VideoSourceCfg = Annotated[
    Union[WyzeRtspVideoCfg, Esp32HttpVideoCfg, UsbIndexVideoCfg, NoneVideoCfg],
    Field(discriminator="type"),
]


# ── Mic source configs ───────────────────────────────────────────────────────


class WyzeRtspAudioCfg(BaseModel):
    """Demuxes audio from a Wyze RTSP stream via PyAV. URL is typically the
    same string as the video source — RTSP carries both tracks; PyAV picks
    out the audio stream and resamples to 16k mono int16 for openWakeWord
    and faster-whisper.
    """
    type: Literal["wyze_rtsp_audio"]
    url: str
    transport: Literal["tcp", "udp"] = "tcp"
    sample_rate_hz: int = 16000
    channels: int = 1


class Esp32MqttMicCfg(BaseModel):
    """ESP32 INMP441 mic publishing PCM frames to MQTT. Stub for now — needs
    the firmware audio-publish path to land before we can read it.
    """
    type: Literal["esp32_i2s_mic"]
    mqtt_topic: str


class UsbMicCfg(BaseModel):
    """sounddevice/PortAudio capture. device_name is matched as a substring
    against sd.query_devices(); device_index is exact. Provide one or the
    other.
    """
    type: Literal["usb_device_mic"]
    device_name: Optional[str] = None
    device_index: Optional[int] = None
    sample_rate_hz: int = 16000
    channels: int = 1


class NoneMicCfg(BaseModel):
    type: Literal["none"]


MicSourceCfg = Annotated[
    Union[WyzeRtspAudioCfg, Esp32MqttMicCfg, UsbMicCfg, NoneMicCfg],
    Field(discriminator="type"),
]


# ── Speaker sink configs ─────────────────────────────────────────────────────


class WyzeSshSpeakerCfg(BaseModel):
    """SSH → SFTP → audioplay_t20 against a Wyze cam. The built-in speaker
    is bad (~1cm cone, half-duplex with the mic, SCP latency adds 1-3s)
    but it means one device per room instead of three. Replace with
    usb_device_spk or esp32_i2s_spk when a real speaker is wired up.

    Note the cam's audio hardware is fixed at 8000 Hz / 16-bit / mono —
    higher rates get resampled host-side before SFTP.
    """
    type: Literal["wyze_ssh_aplay"]
    host: str
    ssh_user: str = "root"
    ssh_password: Optional[str] = None
    ssh_key_path: Optional[str] = None
    remote_play_path: str = "/tmp/jarvis_play.wav"
    # 0-100 — what audioplay_t20's volume arg accepts. Default 60 is
    # audible across a small room; the speaker distorts past ~80.
    volume: int = 60
    # Cam's native output rate. Don't change unless you've confirmed your
    # specific firmware build supports a different rate.
    sample_rate_hz: int = 8000
    connect_timeout_s: float = 5.0


class Esp32MqttSpeakerCfg(BaseModel):
    """ESP32 MAX98357A I2S speaker over MQTT. Stub for now."""
    type: Literal["esp32_i2s_spk"]
    mqtt_topic: str


class UsbSpeakerCfg(BaseModel):
    type: Literal["usb_device_spk"]
    device_name: Optional[str] = None
    device_index: Optional[int] = None
    sample_rate_hz: int = 22050
    channels: int = 1


class NoneSpeakerCfg(BaseModel):
    type: Literal["none"]


SpeakerSinkCfg = Annotated[
    Union[WyzeSshSpeakerCfg, Esp32MqttSpeakerCfg, UsbSpeakerCfg, NoneSpeakerCfg],
    Field(discriminator="type"),
]


# ── Room ─────────────────────────────────────────────────────────────────────


class RoomConfig(BaseModel):
    """One room = one ID + one optional channel of each kind. fps_active /
    fps_idle survive from the legacy schema because the ESP32 firmware reads
    them via NodeManager — Wyze cams ignore them.
    """
    id: str
    display_name: str
    video: VideoSourceCfg
    mic: MicSourceCfg
    speaker: SpeakerSinkCfg
    fps_active: Optional[int] = 5
    fps_idle: Optional[int] = 1


# ── Persona system ───────────────────────────────────────────────────────────


class PersonaConfig(BaseModel):
    """One persona = one system-prompt + behavior knob bundle. The active
    LLM model and TTS voice are NOT per-persona — both are sourced from
    the global config. Switching personas is purely a prompt-and-state
    change so it doesn't churn the LLM client or TTS subsystem.
    """
    system_prompt: str
    content_tier: Literal["standard", "unfiltered"] = "standard"
    response_style: str = "terse"
    # When False, this persona is hidden from the dashboard dropdown.
    # Activation still works via the command box — the only safety
    # property of "hidden" is that the user has to know the persona's
    # name, not that the network can't reach it.
    visible_in_ui: bool = True
    # When True, PersonaManager refuses to activate this persona unless
    # Cole is alone in his current room. Bypassed only by force=True.
    requires_privacy: bool = False
    # Optional pretty name for the dashboard dropdown. Defaults to the
    # config key (e.g. "default", "uwu") when not set.
    display_name: Optional[str] = None


class PersonaRevertCfg(BaseModel):
    """Knobs for the auto-revert behavior in PersonaManager. Defaults
    match the bootstrap doc and Cole's stated preferences.
    """
    away_timeout_s: int = 1800        # 30 min before away triggers revert
    phone_resume_window_s: int = 30   # how long to wait for "yes" after offering resume
    # Process names whose presence indicates an active phone/voice call.
    # Kept separate from process_activity_map so we can tune call detection
    # without altering activity classification.
    call_processes: list[str] = Field(default_factory=lambda: [
        "zoom.exe", "teams.exe", "slack.exe", "discord.exe",
    ])
    call_window_keywords: list[str] = Field(default_factory=lambda: [
        "Meeting", "Call", "Huddle",
    ])


def validate_personas(config: dict) -> tuple[dict[str, PersonaConfig], str, PersonaRevertCfg]:
    """Validate the personas / persona_overlay / persona_revert config.
    Returns (personas_dict, overlay_str, revert_cfg). Raises ConfigError
    on missing 'default' or any per-persona validation failure.

    The 'default' persona is mandatory because every revert path (manual,
    person-entry, away timeout, phone start) lands on it. A missing
    'default' would make those paths NoneType-crash at runtime; failing
    fast at boot is infinitely friendlier.
    """
    raw = config.get("personas", {})
    if not isinstance(raw, dict):
        raise ConfigError(
            f"config.yaml: 'personas' must be a mapping, got {type(raw).__name__}"
        )
    out: dict[str, PersonaConfig] = {}
    for name, raw_p in raw.items():
        if not isinstance(raw_p, dict):
            raise ConfigError(f"config.yaml: persona '{name}' must be a mapping")
        try:
            out[name] = PersonaConfig.model_validate(raw_p)
        except ValidationError as e:
            first = e.errors()[0]
            loc = ".".join(str(p) for p in first.get("loc", ()))
            msg = first.get("msg", "validation failed")
            raise ConfigError(
                f"config.yaml: persona '{name}' failed validation at "
                f"'{loc}': {msg}"
            ) from e
    if "default" not in out:
        raise ConfigError(
            "config.yaml: 'personas.default' is required — every revert path "
            "lands on the default persona, so it can't be missing"
        )
    overlay = config.get("persona_overlay", "")
    if not isinstance(overlay, str):
        raise ConfigError("config.yaml: 'persona_overlay' must be a string")
    revert_raw = config.get("persona_revert", {}) or {}
    if not isinstance(revert_raw, dict):
        raise ConfigError("config.yaml: 'persona_revert' must be a mapping")
    try:
        revert_cfg = PersonaRevertCfg.model_validate(revert_raw)
    except ValidationError as e:
        first = e.errors()[0]
        loc = ".".join(str(p) for p in first.get("loc", ()))
        raise ConfigError(
            f"config.yaml: persona_revert validation failed at '{loc}': "
            f"{first.get('msg', '')}"
        ) from e
    return out, overlay.strip(), revert_cfg


# ── Public entry points ──────────────────────────────────────────────────────


def validate_rooms(config: dict) -> list[RoomConfig]:
    """Validate the rooms[] block of an already env-expanded config dict.

    Returns the list of typed RoomConfig models on success. Raises ConfigError
    with a human-readable message on validation failure — the message includes
    which room (by id, when extractable) and which field broke, so you can
    open the YAML and fix it without reading a Pydantic stack trace.
    """
    raw_rooms = config.get("rooms", [])
    if not isinstance(raw_rooms, list):
        raise ConfigError(
            f"config.yaml: 'rooms' must be a list, got {type(raw_rooms).__name__}"
        )

    validated: list[RoomConfig] = []
    for idx, raw in enumerate(raw_rooms):
        if not isinstance(raw, dict):
            raise ConfigError(f"config.yaml: rooms[{idx}] must be a mapping")
        room_id = raw.get("id", f"<index {idx}>")
        try:
            validated.append(RoomConfig.model_validate(raw))
        except ValidationError as e:
            # Pydantic's default repr is dense — surface the first error
            # path/message clearly so a misplaced indent doesn't require
            # reading 30 lines of error context.
            first = e.errors()[0]
            loc = ".".join(str(p) for p in first.get("loc", ()))
            msg = first.get("msg", "validation failed")
            raise ConfigError(
                f"config.yaml: room '{room_id}' failed validation at "
                f"'{loc}': {msg}"
            ) from e
    return validated


def load_personas_overlay(overlay_path: Path) -> dict[str, Any]:
    """Read a private personas file and return its `personas` dict (or
    empty when missing/unparseable). The overlay file is gitignored;
    it's the user's escape valve for personas they don't want in the
    public config.

    Format: a YAML file with a top-level `personas:` key. Anything else
    in the file is ignored (so users can throw notes-to-self in there
    too without breaking the loader).

    Failure modes are non-fatal — a bad overlay logs a warning and
    boots without the private personas. The default + focus personas
    in config.yaml still work, so a syntax error in the overlay
    degrades the feature, not the whole system.
    """
    if not overlay_path.exists():
        return {}
    try:
        import yaml as _yaml
        with overlay_path.open("r", encoding="utf-8") as f:
            data = _yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(
            f"[Config] Persona overlay {overlay_path} unreadable ({e}); "
            "private personas not loaded"
        )
        return {}
    if not isinstance(data, dict):
        logger.warning(
            f"[Config] Persona overlay {overlay_path} root is not a mapping; ignoring"
        )
        return {}
    overlay_personas = data.get("personas", {})
    if not isinstance(overlay_personas, dict):
        logger.warning(
            f"[Config] Persona overlay {overlay_path}: 'personas' must be a mapping"
        )
        return {}
    if overlay_personas:
        logger.info(
            f"[Config] Loaded {len(overlay_personas)} persona(s) from "
            f"{overlay_path}: {sorted(overlay_personas.keys())}"
        )
    return overlay_personas


def expand_and_validate(config: dict) -> tuple[dict, list[RoomConfig]]:
    """One-call helper used by main.py at boot: expands env vars in-place,
    validates rooms AND personas, and returns the mutated dict + typed
    room list. Persona validation is non-blocking (logs and skips) when
    the personas section is absent — keeps existing configs without a
    persona section booting cleanly.

    Persona overlay: when `personas_overlay_file` is set, load that
    file's personas dict and merge it on top of the in-config personas.
    Overlay entries WIN on key collision — that's the intended override
    semantic (you can locally tweak a public persona without editing
    the shared config).
    """
    expanded = expand_env_vars(config)
    rooms = validate_rooms(expanded)
    # Validate personas if present — stash typed result on the dict so
    # PersonaManager can pick it up without re-parsing. Missing personas
    # section is OK on legacy configs; PersonaManager will create a
    # synthetic default-only setup at construct time.
    if "personas" in expanded or expanded.get("personas_overlay_file"):
        # Merge the overlay BEFORE validation so PersonaConfig validation
        # runs against the final composed dict — surfaces typos in the
        # overlay file with the same nice error messages.
        overlay_path_str = expanded.get("personas_overlay_file")
        if overlay_path_str:
            from pathlib import Path as _Path
            overlay_path = _Path(overlay_path_str)
            if not overlay_path.is_absolute():
                # Resolve relative to the project root (parent of core/)
                overlay_path = _Path(__file__).resolve().parents[1] / overlay_path
            overlay_personas = load_personas_overlay(overlay_path)
            if overlay_personas:
                # Deep-ish merge: per-persona dicts get replaced wholesale
                # rather than merged field-by-field. Keeps the mental model
                # simple — "the overlay defines this whole persona" vs
                # "the overlay tweaks one field of this persona."
                merged = dict(expanded.get("personas", {}))
                merged.update(overlay_personas)
                expanded["personas"] = merged
        personas, overlay, revert_cfg = validate_personas(expanded)
        expanded["_typed_personas"] = personas
        expanded["_persona_overlay"] = overlay
        expanded["_persona_revert_cfg"] = revert_cfg
    return expanded, rooms
