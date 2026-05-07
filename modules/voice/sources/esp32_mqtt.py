"""
JARVIS — Ambient Home AI
========================
Mission: ESP32 I2S mic + MAX98357A speaker drivers over MQTT. Both are
         placeholder implementations until the firmware audio path lands —
         the existing ESPHome firmware ships camera + sensors but not the
         I2S audio interface yet.

         Today these classes exist so config validation passes when a room
         declares `mic.type: esp32_i2s_mic` (laundry_room currently does);
         the manager instantiates them but no audio actually flows. When
         the firmware ships, fill in the MQTT subscribe/publish wiring —
         the manager surface and config schema won't need to change.

         The shared NodeManager already maintains MQTT subscriptions for
         /jarvis/nodes/+/status; the audio topics will follow the same
         pattern once the firmware emits them.

Modules: modules/voice/sources/esp32_mqtt.py
Classes: Esp32MqttMicSource, Esp32MqttSpeakerSink

#todo: Wire jarvis/nodes/{room}/audio/in subscription so the mic publishes
       chunks in the same int16 PCM @ 16kHz format as the other sources.
       Firmware needs to chunk RTP-style with a small header so we can
       detect dropouts.
#todo: Speaker side needs to negotiate codec (raw PCM vs Opus) — Opus
       saves bandwidth on the laundry-room WiFi but adds firmware complexity.
"""

from __future__ import annotations

from loguru import logger

from modules.voice.sources.base import MicCallback, MicSource, SpeakerSink


class Esp32MqttMicSource(MicSource):
    """Stub — logs that audio would have flowed but firmware isn't ready."""

    def __init__(self, room: str, mqtt_topic: str) -> None:
        self._room = room
        self._topic = mqtt_topic

    @property
    def room(self) -> str:
        return self._room

    async def start(self, callback: MicCallback) -> None:
        logger.warning(
            f"[Esp32Mic:{self._room}] Stub driver — firmware audio path not yet "
            f"shipped; would subscribe to '{self._topic}' once it lands"
        )

    async def stop(self) -> None:
        return


class Esp32MqttSpeakerSink(SpeakerSink):
    """Stub — drops PCM with a one-time warning."""

    def __init__(self, room: str, mqtt_topic: str) -> None:
        self._room = room
        self._topic = mqtt_topic
        self._warned = False

    @property
    def room(self) -> str:
        return self._room

    async def play(self, pcm: bytes, sample_rate: int) -> None:
        if not self._warned:
            logger.warning(
                f"[Esp32Speaker:{self._room}] Stub driver — would publish "
                f"{len(pcm)} bytes @ {sample_rate}Hz to '{self._topic}' once "
                "firmware lands. (This message will not repeat.)"
            )
            self._warned = True

    async def close(self) -> None:
        return
