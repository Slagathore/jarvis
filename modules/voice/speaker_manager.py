"""
JARVIS — Ambient Home AI
========================
Mission: Per-room speaker factory + lifecycle manager. Mirrors MicManager
         on the output side — reads each room's `speaker:` block and wires
         the matching SpeakerSink driver (USB / Wyze SSH / ESP32 MQTT /
         null), then exposes a single play(room, pcm, rate) call.

         The orchestrator's TTS path used to branch on a "local" vs "node"
         string in `speaker_sink`. With this manager, the branch lives in
         config — `speaker.type` selects the driver and the orchestrator
         just calls play() without caring which way the audio gets to the
         room.

Modules: modules/voice/speaker_manager.py
Classes: SpeakerManager

#todo: Add a fallback chain — e.g. if wyze_ssh fails 3 times in a row,
       auto-fall-back to a configured 'office' sink so the user still hears
       it. Today the orchestrator does that for "node" → "local" via
       _speaker_sink_for; we should fold that pattern in here.
#todo: Add a per-room cooldown so the speaker doesn't double-fire when
       two events arrive 50ms apart (both reach play() before the first
       returns). Today play() blocks until completion so this is moot,
       but a future async-mixer SpeakerSink would need it.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from modules.voice.sources.base import SpeakerSink
from modules.voice.sources.esp32_mqtt import Esp32MqttSpeakerSink
from modules.voice.sources.null_audio import NullSpeakerSink
from modules.voice.sources.usb_speaker import UsbSpeakerSink
from modules.voice.sources.wyze_ssh_speaker import WyzeSshSpeakerSink


def _make_speaker_sink(room_id: str, spk_cfg: dict) -> Optional[SpeakerSink]:
    stype = spk_cfg.get("type", "none")
    if stype == "usb_device_spk":
        return UsbSpeakerSink(
            room=room_id,
            device_name=spk_cfg.get("device_name"),
            device_index=spk_cfg.get("device_index"),
            sample_rate_hz=int(spk_cfg.get("sample_rate_hz", 22050)),
            channels=int(spk_cfg.get("channels", 1)),
        )
    if stype == "wyze_ssh_aplay":
        host = str(spk_cfg.get("host", "")).strip()
        if not host:
            logger.warning(f"[SpeakerManager] Room '{room_id}' wyze_ssh_aplay has no host")
            return None
        return WyzeSshSpeakerSink(
            room=room_id,
            host=host,
            ssh_user=str(spk_cfg.get("ssh_user", "root")),
            ssh_password=spk_cfg.get("ssh_password"),
            ssh_key_path=spk_cfg.get("ssh_key_path"),
            remote_play_path=str(spk_cfg.get("remote_play_path", "/tmp/jarvis_play.wav")),
            volume=int(spk_cfg.get("volume", 60)),
            sample_rate_hz=int(spk_cfg.get("sample_rate_hz", 8000)),
            connect_timeout_s=float(spk_cfg.get("connect_timeout_s", 5.0)),
        )
    if stype == "esp32_i2s_spk":
        topic = str(spk_cfg.get("mqtt_topic", "")).strip()
        if not topic:
            logger.warning(f"[SpeakerManager] Room '{room_id}' esp32_i2s_spk has no mqtt_topic")
            return None
        return Esp32MqttSpeakerSink(room=room_id, mqtt_topic=topic)
    if stype == "none":
        return NullSpeakerSink(room=room_id)
    logger.warning(f"[SpeakerManager] Room '{room_id}' unknown speaker.type '{stype}'")
    return None


class SpeakerManager:
    """One SpeakerSink per room. Orchestrator calls play(room, pcm, rate);
    the right driver handles the rest.
    """

    def __init__(self, config: dict) -> None:
        self._sinks: dict[str, SpeakerSink] = {}
        for room_cfg in config.get("rooms", []):
            room_id = room_cfg.get("id", "unknown")
            spk_cfg = room_cfg.get("speaker") or {}
            if not isinstance(spk_cfg, dict):
                logger.warning(
                    f"[SpeakerManager] Room '{room_id}' has malformed 'speaker:' block"
                )
                continue
            sink = _make_speaker_sink(room_id, spk_cfg)
            if sink is not None:
                self._sinks[room_id] = sink

    def get_rooms(self) -> list[str]:
        """Rooms with a non-null speaker sink."""
        return [
            r for r, s in self._sinks.items()
            if not isinstance(s, NullSpeakerSink)
        ]

    def get_speaker_type(self, room: str) -> str:
        """Return the driver type for a room — used by the orchestrator's
        legacy "is this a node-routed speaker?" check during the migration
        window. Returns "none" for unconfigured rooms.
        """
        sink = self._sinks.get(room)
        if sink is None:
            return "none"
        if isinstance(sink, WyzeSshSpeakerSink):
            return "wyze_ssh_aplay"
        if isinstance(sink, UsbSpeakerSink):
            return "usb_device_spk"
        if isinstance(sink, Esp32MqttSpeakerSink):
            return "esp32_i2s_spk"
        return "none"

    async def play(self, room: str, pcm: bytes, sample_rate: int) -> bool:
        """Send PCM to the room's speaker. Returns False if no sink is
        configured. Errors inside the driver are logged and swallowed —
        the orchestrator already has fallback logic and we don't want a
        single unreachable Wyze cam to crash the whole TTS path.
        """
        sink = self._sinks.get(room)
        if sink is None:
            logger.warning(f"[SpeakerManager] No speaker configured for room '{room}'")
            return False
        try:
            await sink.play(pcm, sample_rate)
            return True
        except Exception as e:
            logger.warning(f"[SpeakerManager] play('{room}') failed: {e}")
            return False

    async def close(self) -> None:
        for room, sink in self._sinks.items():
            try:
                await sink.close()
            except Exception as e:
                logger.warning(f"[SpeakerManager] close('{room}') failed: {e}")
        self._sinks.clear()
