"""
JARVIS — Ambient Home AI
========================
Mission: Per-room microphone factory + lifecycle manager. Reads each room's
         `mic:` block from config.yaml, instantiates the matching MicSource
         driver (USB / Wyze RTSP / ESP32 MQTT / null), and exposes a single
         start_capture(room, callback) API.

         The factory dispatch is exhaustive over the schema's mic.type
         literal union — adding a new driver is one elif branch + one new
         file in modules/voice/sources/.

         This manager is purely glue. The drivers handle their own threads,
         locks, and reconnects. The manager just owns "which room maps to
         which driver" and shuts them all down on close().

Modules: modules/voice/mic_manager.py
Classes: MicManager

#todo: Add health monitoring — if a mic source goes silent for >Ns,
       restart it and surface a 'mic_stalled' event to the dashboard
#todo: Add per-room enable/disable so the dashboard can mute a room
       without editing YAML and restarting
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from modules.voice.sources.base import MicCallback, MicSource
from modules.voice.sources.esp32_mqtt import Esp32MqttMicSource
from modules.voice.sources.null_audio import NullMicSource
from modules.voice.sources.usb_mic import UsbMicSource
from modules.voice.sources.wyze_rtsp_mic import WyzeRtspMicSource


def _make_mic_source(room_id: str, mic_cfg: dict) -> Optional[MicSource]:
    """Return the right MicSource subclass for the room's mic.type, or None
    on an unknown type (the factory logs and skips so a typo can't crash the
    whole audio subsystem at boot).
    """
    mtype = mic_cfg.get("type", "none")
    if mtype == "usb_device_mic":
        return UsbMicSource(
            room=room_id,
            device_name=mic_cfg.get("device_name"),
            device_index=mic_cfg.get("device_index"),
            sample_rate_hz=int(mic_cfg.get("sample_rate_hz", 16000)),
            channels=int(mic_cfg.get("channels", 1)),
        )
    if mtype == "wyze_rtsp_audio":
        url = str(mic_cfg.get("url", "")).strip()
        if not url:
            logger.warning(f"[MicManager] Room '{room_id}' wyze_rtsp_audio has no url")
            return None
        return WyzeRtspMicSource(
            room=room_id,
            url=url,
            transport=str(mic_cfg.get("transport", "tcp")),
            sample_rate_hz=int(mic_cfg.get("sample_rate_hz", 16000)),
            channels=int(mic_cfg.get("channels", 1)),
        )
    if mtype == "esp32_i2s_mic":
        topic = str(mic_cfg.get("mqtt_topic", "")).strip()
        if not topic:
            logger.warning(f"[MicManager] Room '{room_id}' esp32_i2s_mic has no mqtt_topic")
            return None
        return Esp32MqttMicSource(room=room_id, mqtt_topic=topic)
    if mtype == "none":
        return NullMicSource(room=room_id)
    logger.warning(f"[MicManager] Room '{room_id}' has unknown mic.type '{mtype}'")
    return None


class MicManager:
    """Owns one MicSource per room. start_capture() begins streaming to a
    given callback for one room; close() tears them all down.
    """

    def __init__(self, config: dict) -> None:
        self._sources: dict[str, MicSource] = {}
        for room_cfg in config.get("rooms", []):
            room_id = room_cfg.get("id", "unknown")
            mic_cfg = room_cfg.get("mic") or {}
            if not isinstance(mic_cfg, dict):
                logger.warning(
                    f"[MicManager] Room '{room_id}' has malformed 'mic:' block"
                )
                continue
            src = _make_mic_source(room_id, mic_cfg)
            if src is not None:
                self._sources[room_id] = src

    def get_rooms(self) -> list[str]:
        """Rooms that have a non-null mic source instantiated."""
        return [
            r for r, s in self._sources.items()
            if not isinstance(s, NullMicSource)
        ]

    def attach_mqtt(self, mqtt_client) -> None:
        """Late-bind the MQTT client into any sources that need it (only
        the ESP32 MQTT mic today). MicManager is constructed during voice
        init before MQTTClient connects; the orchestrator calls this once
        the broker is up."""
        for src in self._sources.values():
            attach = getattr(src, "attach_mqtt", None)
            if callable(attach):
                attach(mqtt_client)

    async def start_capture(self, room: str, callback: MicCallback) -> bool:
        """Begin streaming for `room`. Returns False if no source is
        configured (caller can decide whether to log/raise).
        """
        src = self._sources.get(room)
        if src is None:
            logger.warning(f"[MicManager] No mic configured for room '{room}'")
            return False
        await src.start(callback)
        return True

    async def stop_capture(self, room: str) -> None:
        """Stop streaming for `room`. Safe to call on a stopped source."""
        src = self._sources.get(room)
        if src is not None:
            await src.stop()

    async def close(self) -> None:
        """Stop every active mic source. Failures per-room are logged but
        don't abort cleanup of the rest.
        """
        for room, src in self._sources.items():
            try:
                await src.stop()
            except Exception as e:
                logger.warning(f"[MicManager] stop('{room}') failed: {e}")
        self._sources.clear()
