"""
JARVIS — Ambient Home AI
========================
Mission: ESP32 I2S mic + MAX98357A speaker drivers over MQTT. The host side
         is fully wired — audio in is a normal MQTT subscription that
         dispatches raw int16 PCM chunks into the MicSource callback, audio
         out publishes raw int16 PCM bytes to the room's audio/out topic.

         The firmware side of the bus is what's actually missing today: the
         ESPHome configs in hardware/esphome/ define the I2S microphone +
         MAX98357A speaker hardware, but the audio→MQTT publish path and
         MQTT→speaker.play subscribe path haven't been wired into
         node_base.yaml yet. Until that lands, the host subscriber sits
         attached but never receives data, and the host publisher emits to
         a topic no firmware is currently listening on. Both behaviors are
         what we want — once the firmware ships, audio flows without any
         host changes.

         Wire format (when firmware lands):
           - audio/in  : raw int16 little-endian PCM samples, 16 kHz mono.
                         Chunk size matches whatever ESPHome's on_data
                         emits (typically ~512 samples = 32 ms).
           - audio/out : same format, host → firmware. Speaker.play can
                         consume std::vector<uint8_t> directly so the
                         payload is fed straight to I2S.

         An optional Opus codec layer can be added later for bandwidth
         savings (raw PCM @ 16 kHz is 32 KB/s per node), but raw is fine
         for one or two nodes on a healthy LAN.

Modules: modules/voice/sources/esp32_mqtt.py
Classes: Esp32MqttMicSource, Esp32MqttSpeakerSink
"""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger

from modules.voice.sources.base import MicCallback, MicSource, SpeakerSink

# Sample rate the firmware is expected to publish at. Wake-word detection
# and Whisper STT both expect 16 kHz mono int16 — matching the convention
# the WyzeRtspMicSource emits. If the firmware ships at a different rate
# we'll resample host-side rather than retraining models.
DEFAULT_SAMPLE_RATE_HZ = 16000


class Esp32MqttMicSource(MicSource):
    """Subscribes to the room's audio/in MQTT topic and forwards binary
    payloads (raw int16 PCM) into the manager's callback. Until the
    firmware starts publishing the subscription is silent — same shape as
    a quiet room, no errors raised.
    """

    def __init__(
        self,
        room: str,
        mqtt_topic: str,
        sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
    ) -> None:
        self._room = room
        self._topic = mqtt_topic
        self._sample_rate = sample_rate_hz
        self._mqtt: Optional[Any] = None
        self._callback: Optional[MicCallback] = None
        self._subscribed: bool = False

    @property
    def room(self) -> str:
        return self._room

    def attach_mqtt(self, mqtt_client: Any) -> None:
        """Late-bound MQTT client wiring. MicManager constructs sources at
        boot before the MQTTClient exists; orchestrator calls this once MQTT
        connects so the subscription registers with a live client.
        """
        self._mqtt = mqtt_client

    async def start(self, callback: MicCallback) -> None:
        self._callback = callback
        if self._mqtt is None:
            logger.warning(
                f"[Esp32Mic:{self._room}] No MQTT client attached — "
                f"audio from '{self._topic}' won't reach the wake-word path. "
                "Orchestrator wiring bug?"
            )
            return
        if self._subscribed:
            return
        self._mqtt.subscribe(self._topic, self._on_message)
        self._subscribed = True
        logger.info(
            f"[Esp32Mic:{self._room}] Subscribed to '{self._topic}' "
            f"(awaiting firmware audio publish path)"
        )

    async def stop(self) -> None:
        # MQTTClient doesn't expose unsubscribe today; the stop signal is
        # implicit (callback set to None drops further dispatch).
        self._callback = None

    async def _on_message(self, _topic: str, payload: Any) -> None:
        """MQTT dispatch handler. Topic is fixed at subscribe time so we
        ignore the dispatched topic value. The MQTT client decodes UTF-8
        strings and JSON automatically; we only act on raw bytes (binary
        PCM) and skip anything that came in as text.
        """
        del _topic  # signature requirement; topic is fixed at subscribe time
        cb = self._callback
        if cb is None or not isinstance(payload, (bytes, bytearray)):
            return
        try:
            await cb(bytes(payload), self._sample_rate)
        except Exception as e:
            logger.warning(f"[Esp32Mic:{self._room}] callback raised: {e}")


class Esp32MqttSpeakerSink(SpeakerSink):
    """Publishes raw int16 PCM bytes to the room's audio/out topic. The
    firmware's MQTT on_message handler is expected to feed the payload to
    speaker.play, which accepts a std::vector<uint8_t> of int16 LE samples.
    """

    def __init__(
        self,
        room: str,
        mqtt_topic: str,
        sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
    ) -> None:
        self._room = room
        self._topic = mqtt_topic
        self._sample_rate = sample_rate_hz
        self._mqtt: Optional[Any] = None

    @property
    def room(self) -> str:
        return self._room

    def attach_mqtt(self, mqtt_client: Any) -> None:
        self._mqtt = mqtt_client

    async def play(self, pcm: bytes, sample_rate: int) -> None:
        if not pcm:
            return
        if self._mqtt is None:
            logger.warning(
                f"[Esp32Speaker:{self._room}] No MQTT client attached — "
                f"dropping {len(pcm)} bytes for '{self._topic}'"
            )
            return
        # Resample host-side if the caller's rate doesn't match what we
        # told the firmware to expect. Cheap linear resample is fine here —
        # the MAX98357A is a 1cm cone amp, not a hifi DAC.
        if sample_rate != self._sample_rate:
            pcm = self._resample_int16(pcm, sample_rate, self._sample_rate)
        # QoS 1 so a brief WiFi blip on the cam side doesn't silently lose
        # a TTS chunk; retain=False because audio is ephemeral.
        await self._mqtt.publish(self._topic, pcm, qos=1, retain=False)

    async def close(self) -> None:
        return

    @staticmethod
    def _resample_int16(pcm_in: bytes, in_rate: int, out_rate: int) -> bytes:
        """Linear-interp resample. Matches the resampler in wyze_ssh_speaker
        for consistency — the I2S amp won't reveal anything fancier."""
        if not pcm_in or in_rate == out_rate:
            return pcm_in
        import numpy as np  # local import: this path is rarely hit
        arr = np.frombuffer(pcm_in, dtype=np.int16)
        if arr.size == 0:
            return b""
        out_len = max(1, int(round(arr.size * out_rate / in_rate)))
        x_in = np.linspace(0.0, 1.0, num=arr.size, endpoint=False, dtype=np.float64)
        x_out = np.linspace(0.0, 1.0, num=out_len, endpoint=False, dtype=np.float64)
        out = np.interp(x_out, x_in, arr.astype(np.float64))
        return np.clip(out, -32768, 32767).astype(np.int16).tobytes()
