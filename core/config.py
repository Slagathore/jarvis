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
    """ESP32 INMP441 mic publishing PCM frames to MQTT. Host subscriber is
    fully wired (see Esp32MqttMicSource) — what's still pending is the
    firmware lambda on the ESPHome side that publishes captured I2S samples
    to this topic. Until that lands the subscription sits idle.
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
    """ESP32 MAX98357A I2S speaker over MQTT. Host publisher is fully wired
    (Esp32MqttSpeakerSink); the firmware-side MQTT-on_message → speaker.play
    path needs to be added to node_base.yaml for audio to actually reach
    the amp. Topic carries raw int16 LE PCM @ 16 kHz mono.
    """
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


# ── World Model config (§7, §22, §32) ────────────────────────────────────────


# Allowed values for various typed enums in the world_model config tree.
# Maintained here (not in the world_model module) because we want the
# typed validators in this file to be the single source of truth at
# startup; the runtime can still defensively accept extras at the
# raw-dict level if it has to, but config-loaded values must conform.
_AFFINITY_CONTEXTS: tuple[str, ...] = (
    "sleeping", "physical_contact", "rubbing",
    "proximity_general", "authority", "feeding",
)
# Note: affinity `strength` is enforced by Literal["low","medium","high"]
# on AffinityCfg directly — Pydantic raises on bad input there.
_KNOWN_TRACKED_SPECIES: tuple[str, ...] = ("cat", "dog")


class ExitDef(BaseModel):
    """One exit polygon in a room's world_model.exits list. Drives the
    bounded-house disappearance state machine (§17): bbox center inside
    a `to_room` polygon → TRANSITIONING; inside `exterior_exit` → DEPARTED;
    inside `to_unmonitored_zone` → IN_HOUSE_UNMONITORED."""
    kind: Literal["to_room", "to_unmonitored_zone", "exterior_exit"]
    polygon: list[list[float]]
    # to_room → next room id; exterior_exit → human-readable name (e.g.
    # "front_door"); to_unmonitored_zone → unmonitored room id.
    to: Optional[str] = None
    name: Optional[str] = None


class LandmarkDef(BaseModel):
    """A named region inside a room frame. §22.9 dwell-event firing
    (litterbox_visit etc.) hangs off these names."""
    name: str
    polygon: list[list[float]]


class RoomWorldModelCfg(BaseModel):
    """Per-room geometry block. Optional in YAML — rooms without it
    get a default `enabled=true` shell wired by orchestrator boot."""
    enabled: bool = True
    frame_width: int = 640
    frame_height: int = 480
    exits: list[ExitDef] = Field(default_factory=list)
    landmarks: list[LandmarkDef] = Field(default_factory=list)


class WorldModelNightlyCfg(BaseModel):
    """Knobs for the §22.6 nightly profile rebuild loop."""
    interval_hours: float = 24.0
    startup_grace_seconds: float = 60.0
    profile_days_back: int = 30


class WorldModelCfg(BaseModel):
    """Top-level world_model: block. Distinct from per-room blocks
    above — this is the household-wide configuration."""
    tracked_species: list[str] = Field(
        default_factory=lambda: list(_KNOWN_TRACKED_SPECIES)
    )
    visiting_animal_retention_minutes: int = 60
    nightly: WorldModelNightlyCfg = Field(
        default_factory=WorldModelNightlyCfg
    )


# ── §23 tracked_objects: ─────────────────────────────────────────────────────


class TrackedOpenVocabObject(BaseModel):
    """One open-vocab object the system should track via OWLv2 +
    CLIP. `description` is what gets fed to OWLv2 as the text query
    (more visual = better recall). `name` is the friendly handle
    used in events + queries."""
    name: str
    description: str
    typical_rooms: list[str] = Field(default_factory=list)


class DetectionIntervalCfg(BaseModel):
    open_vocabulary: float = 30.0
    closed_vocabulary: float = 0.0


class TrackedObjectsCfg(BaseModel):
    open_vocabulary: list[TrackedOpenVocabObject] = Field(
        default_factory=list,
    )
    closed_vocabulary: list[str] = Field(default_factory=list)
    detection_interval_seconds: DetectionIntervalCfg = Field(
        default_factory=DetectionIntervalCfg,
    )


class ResidentCfg(BaseModel):
    """One human resident — links a config-level token (cole/anna/jeff)
    to a display_name + primary_room. The token is what pets reference
    via household_owner / affinity.person."""
    id: str
    display_name: str
    primary_room: Optional[str] = None


class AffinityCfg(BaseModel):
    person: str
    strength: Literal["low", "medium", "high"]
    contexts: list[str] = Field(default_factory=list)


class PetCommonCfg(BaseModel):
    """Shared fields between PetCatCfg and PetDogCfg. Keeping them in
    one base avoids drift between the two species blocks."""
    name: str
    household_owner: Optional[str] = None
    expected_size: Optional[str] = None
    size_basis: Optional[str] = None
    color_class: Optional[str] = None
    personality: Optional[str] = None
    archived: bool = False
    affinities: list[AffinityCfg] = Field(default_factory=list)
    notes: Optional[str] = None


class PetCatCfg(PetCommonCfg):
    coat_length: Optional[str] = None
    coat_texture: Optional[str] = None
    home_room: Optional[str] = None
    cyclic_home_rooms: list[str] = Field(default_factory=list)
    unmonitored_home: Optional[str] = None
    preferred_perches: list[str] = Field(default_factory=list)
    preferred_landmarks: list[str] = Field(default_factory=list)
    conflicts_with: list[str] = Field(default_factory=list)
    hyper_alert_to: list[str] = Field(default_factory=list)
    distinctive_features: list[str] = Field(default_factory=list)
    sees_ghosts: bool = False
    age_state: Optional[str] = None


class PetDogCfg(PetCommonCfg):
    breed_class: Optional[str] = None
    home_rooms: list[str] = Field(default_factory=list)
    feeding_room: Optional[str] = None
    anxiety_triggers: list[str] = Field(default_factory=list)
    nicknames: list[str] = Field(default_factory=list)


class PetsCfg(BaseModel):
    cats: list[PetCatCfg] = Field(default_factory=list)
    dogs: list[PetDogCfg] = Field(default_factory=list)


def validate_tracked_objects(
    config: dict,
    rooms_known: set[str],
) -> Optional[TrackedObjectsCfg]:
    """Validate the §23 tracked_objects: block + its cross-references.
    Returns None when the block is absent (not all installs care).
    Cross-references enforced (HARD errors at boot):
      • open_vocabulary[*].typical_rooms ⊆ {rooms.id}
    Soft warnings:
      • Missing description (logged, but pydantic catches it as required).
    """
    raw = config.get("tracked_objects")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigError(
            "config.yaml: 'tracked_objects' must be a mapping, got "
            f"{type(raw).__name__}"
        )
    try:
        cfg = TrackedObjectsCfg.model_validate(raw)
    except ValidationError as e:
        first = e.errors()[0]
        loc = ".".join(str(p) for p in first.get("loc", ()))
        raise ConfigError(
            f"config.yaml: 'tracked_objects.{loc}' invalid: "
            f"{first.get('msg', 'validation failed')}"
        ) from e

    seen_names: set[str] = set()
    for obj in cfg.open_vocabulary:
        if obj.name in seen_names:
            raise ConfigError(
                f"config.yaml: duplicate tracked_objects open_vocabulary "
                f"name '{obj.name}'"
            )
        seen_names.add(obj.name)
        # Typical-rooms cross-reference — warn-only via the logger
        # rather than fail-fast. A typo here just means the room
        # prior won't apply for that object class; the tracking
        # still functions otherwise.
        unknown = [r for r in obj.typical_rooms if r not in rooms_known]
        if unknown:
            logger.warning(
                f"[Config] tracked_objects.{obj.name}.typical_rooms "
                f"references unknown room(s) {unknown} "
                f"(known: {sorted(rooms_known)})"
            )
    return cfg


def validate_world_model_config(
    config: dict,
) -> tuple[
    Optional[WorldModelCfg],
    list[ResidentCfg],
    Optional[PetsCfg],
    dict[str, RoomWorldModelCfg],
]:
    """Validate the world_model / residents / pets / per-room
    world_model blocks together. Cross-references enforced:
      • every pet's household_owner is a known resident.id
      • every affinity.person is a known resident.id
      • affinity contexts ∈ enum
      • exit.to_room refers to an existing rooms[].id
      • tracked_species and pets.* are consistent (warn-only on extras)

    Returns (wm_cfg, residents, pets, rooms_wm) — any of which may be
    None / empty when not declared. Raises ConfigError on cross-ref
    violations or pydantic field errors. Polygon issues (<3 points,
    out of frame) are logged as warnings only — they're geometry,
    not config schema, and the polygon viewer is the right place to
    fix them at runtime.
    """
    # ── Top-level world_model: ─────────────────────────────────────
    wm_cfg: Optional[WorldModelCfg] = None
    raw_wm = config.get("world_model")
    if raw_wm is not None:
        if not isinstance(raw_wm, dict):
            raise ConfigError(
                "config.yaml: 'world_model' must be a mapping, got "
                f"{type(raw_wm).__name__}"
            )
        try:
            wm_cfg = WorldModelCfg.model_validate(raw_wm)
        except ValidationError as e:
            first = e.errors()[0]
            loc = ".".join(str(p) for p in first.get("loc", ()))
            raise ConfigError(
                f"config.yaml: 'world_model.{loc}' invalid: "
                f"{first.get('msg', 'validation failed')}"
            ) from e

    # ── residents: ─────────────────────────────────────────────────
    residents: list[ResidentCfg] = []
    raw_residents = config.get("residents") or []
    if not isinstance(raw_residents, list):
        raise ConfigError(
            f"config.yaml: 'residents' must be a list, got "
            f"{type(raw_residents).__name__}"
        )
    seen_ids: set[str] = set()
    for idx, raw in enumerate(raw_residents):
        if not isinstance(raw, dict):
            raise ConfigError(
                f"config.yaml: residents[{idx}] must be a mapping"
            )
        try:
            r = ResidentCfg.model_validate(raw)
        except ValidationError as e:
            first = e.errors()[0]
            raise ConfigError(
                f"config.yaml: resident[{idx}] invalid at "
                f"'{'.'.join(str(p) for p in first.get('loc', ()))}': "
                f"{first.get('msg')}"
            ) from e
        if r.id in seen_ids:
            raise ConfigError(
                f"config.yaml: duplicate resident id '{r.id}'"
            )
        seen_ids.add(r.id)
        residents.append(r)

    # ── pets: ───────────────────────────────────────────────────────
    pets: Optional[PetsCfg] = None
    raw_pets = config.get("pets")
    if raw_pets is not None:
        if not isinstance(raw_pets, dict):
            raise ConfigError(
                f"config.yaml: 'pets' must be a mapping, got "
                f"{type(raw_pets).__name__}"
            )
        try:
            pets = PetsCfg.model_validate(raw_pets)
        except ValidationError as e:
            first = e.errors()[0]
            loc = ".".join(str(p) for p in first.get("loc", ()))
            raise ConfigError(
                f"config.yaml: 'pets.{loc}' invalid: "
                f"{first.get('msg', 'validation failed')}"
            ) from e

    # ── Cross-references: pets ↔ residents ─────────────────────────
    # Note: PetCatCfg / PetDogCfg both inherit PetCommonCfg, so iterating
    # them as a flat sequence is type-safe at runtime; we annotate the
    # collection as Sequence[PetCommonCfg] to satisfy invariant `list`
    # type-checker complaints.
    resident_ids = {r.id for r in residents}
    if pets is not None:
        from typing import Sequence
        all_pets: Sequence[PetCommonCfg] = [*pets.cats, *pets.dogs]
        for p in all_pets:
            if (p.household_owner is not None
                    and p.household_owner not in resident_ids):
                raise ConfigError(
                    f"config.yaml: pet '{p.name}' household_owner="
                    f"'{p.household_owner}' is not a declared resident "
                    f"(known: {sorted(resident_ids) or '[]'})"
                )
            for aff in p.affinities:
                if aff.person not in resident_ids:
                    raise ConfigError(
                        f"config.yaml: pet '{p.name}' affinity references "
                        f"unknown resident '{aff.person}' "
                        f"(known: {sorted(resident_ids) or '[]'})"
                    )
                bad_ctx = [
                    c for c in aff.contexts
                    if c not in _AFFINITY_CONTEXTS
                ]
                if bad_ctx:
                    raise ConfigError(
                        f"config.yaml: pet '{p.name}' affinity has invalid "
                        f"context(s) {bad_ctx} for resident '{aff.person}'. "
                        f"Allowed: {list(_AFFINITY_CONTEXTS)}"
                    )

    # ── Cross-references: tracked_species ↔ pets ───────────────────
    if wm_cfg is not None and pets is not None:
        if "cat" not in wm_cfg.tracked_species and pets.cats:
            logger.warning(
                f"[Config] {len(pets.cats)} cat(s) declared but 'cat' "
                "missing from world_model.tracked_species — they won't "
                "be tracked at runtime"
            )
        if "dog" not in wm_cfg.tracked_species and pets.dogs:
            logger.warning(
                f"[Config] {len(pets.dogs)} dog(s) declared but 'dog' "
                "missing from world_model.tracked_species"
            )

    # ── Per-room world_model: blocks ───────────────────────────────
    rooms_wm: dict[str, RoomWorldModelCfg] = {}
    room_ids: set[str] = set()
    for raw_room in config.get("rooms", []) or []:
        if not isinstance(raw_room, dict):
            continue
        rid = raw_room.get("id")
        if not rid:
            continue
        room_ids.add(rid)
        raw_block = raw_room.get("world_model")
        if raw_block is None:
            continue
        if not isinstance(raw_block, dict):
            raise ConfigError(
                f"config.yaml: rooms.{rid}.world_model must be a "
                f"mapping, got {type(raw_block).__name__}"
            )
        try:
            block = RoomWorldModelCfg.model_validate(raw_block)
        except ValidationError as e:
            first = e.errors()[0]
            loc = ".".join(str(p) for p in first.get("loc", ()))
            raise ConfigError(
                f"config.yaml: rooms.{rid}.world_model.{loc}: "
                f"{first.get('msg', 'validation failed')}"
            ) from e
        rooms_wm[rid] = block

    # ── Cross-references: exits.to_room → existing room id ─────────
    for rid, block in rooms_wm.items():
        for i, ex in enumerate(block.exits):
            # exterior_exit needs a name; to_room/to_unmonitored need `to`
            if ex.kind == "exterior_exit" and not ex.name:
                raise ConfigError(
                    f"config.yaml: rooms.{rid}.world_model.exits[{i}] "
                    "exterior_exit missing 'name' (used in alarm copy)"
                )
            if ex.kind in ("to_room", "to_unmonitored_zone") and not ex.to:
                raise ConfigError(
                    f"config.yaml: rooms.{rid}.world_model.exits[{i}] "
                    f"{ex.kind} missing 'to' field"
                )
            if ex.kind == "to_room" and ex.to not in room_ids:
                raise ConfigError(
                    f"config.yaml: rooms.{rid}.world_model.exits[{i}] "
                    f"to_room references unknown room '{ex.to}' "
                    f"(known: {sorted(room_ids)})"
                )
            # Polygon shape sanity — warn-only, geometry is hands-on
            # via the polygon viewer.
            if len(ex.polygon) < 3:
                logger.warning(
                    f"[Config] rooms.{rid}.world_model.exits[{i}] "
                    f"polygon has {len(ex.polygon)} points (need ≥3); "
                    "exit detection won't fire until fixed"
                )

    return wm_cfg, residents, pets, rooms_wm


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
    # World Model + residents + pets validation (§7, §22, §32 cross-refs).
    # Same pattern as personas: stash typed results on the dict so
    # downstream consumers can grab them without re-validating, and
    # cross-reference errors fail at boot instead of at observation time.
    try:
        wm_cfg, residents, pets, rooms_wm = validate_world_model_config(
            expanded
        )
        expanded["_typed_world_model"] = wm_cfg
        expanded["_typed_residents"] = residents
        expanded["_typed_pets"] = pets
        expanded["_typed_rooms_world_model"] = rooms_wm
    except ConfigError:
        # Re-raise so boot fails fast — but log the section first so the
        # error message in the user's terminal makes it obvious WHY.
        logger.error("[Config] world_model / residents / pets validation failed")
        raise
    # §23 tracked_objects validation. Independent of world_model so a
    # household can declare objects without having pet config, and
    # vice versa.
    try:
        rooms_known = {
            r.get("id") for r in expanded.get("rooms", [])
            if isinstance(r, dict) and r.get("id")
        }
        rooms_known.discard(None)
        tracked_objects_cfg = validate_tracked_objects(
            expanded, rooms_known,  # type: ignore[arg-type]
        )
        expanded["_typed_tracked_objects"] = tracked_objects_cfg
    except ConfigError:
        logger.error("[Config] tracked_objects validation failed")
        raise
    return expanded, rooms
