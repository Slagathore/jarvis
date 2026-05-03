"""
JARVIS — Ambient Home AI
========================
Mission: Continuously stream microphone audio and detect the "Hey Jarvis" wake word
         using openWakeWord. When the wake word is detected with sufficient confidence,
         publish a "voice.wake_detected" event to the bus and enforce a cooldown
         to prevent double-triggering.

         The audio stream runs in a background thread; detections are published
         to the async event bus from a thread-safe bridge.

Modules: modules/voice/wake_word.py
Classes: WakeWordDetector
Functions:
    WakeWordDetector.__init__(config, bus) — Store config and event bus reference
    WakeWordDetector.load()                — Load openWakeWord model (blocking)
    WakeWordDetector.listen_forever()      — Async coroutine, loops forever
    WakeWordDetector._stream_loop(loop)    — Blocking inner loop (runs in thread)
    WakeWordDetector.stop()                — Signal the stream to stop

Variables:
    WakeWordDetector.loaded     — bool
    WakeWordDetector._running   — bool, controls the stream loop
    WakeWordDetector._model     — openwakeword Model instance
    OWW_CHUNK_SIZE              — Audio chunk size for openWakeWord (1280 samples)

#todo: Support multiple wake words (hey_jarvis, hey_assistant, etc.) as aliases
#todo: Add per-room sensitivity configuration
#todo: Visualize wake word detection score in real time on dashboard
#todo: Add custom wake word training pipeline (record samples → train OWW model)
#todo: Emit confidence score to dashboard for tuning sensitivity
"""

import asyncio
import time
from typing import Any, Optional

import numpy as np
import sounddevice as sd
from loguru import logger

from core.event_bus import EventBus
from core.exceptions import WakeWordError

# openWakeWord expects 16kHz mono audio in chunks of exactly 1280 samples (80ms)
OWW_CHUNK_SIZE = 1280
OWW_SAMPLE_RATE = 16000


class WakeWordDetector:
    """
    Continuous microphone listener that publishes events when the wake word fires.

    The architecture:
      - A background thread (via asyncio.to_thread) reads from sounddevice and
        feeds chunks to the openWakeWord model.
      - When a detection occurs, the score + room info is published to the event
        bus using loop.call_soon_threadsafe so the publish is safe from the thread.
      - A cooldown timer prevents rapid re-triggering.

    Config keys (from config["voice"]["wake_word"]):
        model:             openWakeWord model name or path to .onnx file
        sensitivity:       Detection threshold 0.0–1.0
        cooldown_seconds:  Minimum seconds between detections
    """

    def __init__(
        self,
        config: dict,
        bus: EventBus,
        room: str = "office",
        device: Optional[int] = None,
    ) -> None:
        self._cfg = config["voice"]["wake_word"]
        voice_cfg = config.get("voice", {})
        self._bus = bus
        self._room = room
        self._device = (
            device
            if device is not None
            else self._cfg.get("device", voice_cfg.get("input_device"))
        )
        self.loaded: bool = False
        self._running: bool = False
        self._suspended: bool = False  # True while recording has the mic
        self._model: Optional[Any] = None
        self._last_detection: float = 0.0

    def load(self) -> None:
        """
        Load the openWakeWord model. Blocking — call during startup.
        Downloads the model if not cached (~first run only).

        Raises:
            WakeWordError: If openwakeword is not installed or model fails to load.
        """
        try:
            from openwakeword.model import Model
        except ImportError as e:
            raise WakeWordError(
                "openwakeword not installed. Run: pip install openwakeword"
            ) from e

        model_name = self._cfg.get("model", "hey_jarvis")
        logger.info(f"[WakeWord] Loading model '{model_name}'...")

        try:
            self._model = Model(wakeword_models=[model_name], inference_framework="onnx")
            self.loaded = True
            logger.info(f"[WakeWord] Model '{model_name}' ready")
        except Exception as e:
            raise WakeWordError(f"Failed to load wake word model '{model_name}': {e}") from e

        # Resolve and log the input device that wake-listening AND the follow-up
        # recording will both use — Cole needs to know if the system grabbed the
        # wrong mic (USB headset vs onboard vs webcam). If the configured device
        # is a string, resolve by substring match against device names.
        self._device = self._resolve_device(self._device)
        try:
            info = sd.query_devices(self._device, kind="input") if self._device is not None else sd.query_devices(kind="input")
            name = info.get("name", "unknown") if isinstance(info, dict) else "unknown"
            logger.info(f"[WakeWord] Using input device #{self._device}: {name!r}")
        except Exception as e:
            logger.warning(f"[WakeWord] Could not query input device {self._device!r}: {e}")

    def _resolve_device(self, device: Optional[int | str]) -> Optional[int]:
        """
        Resolve a configured device (None | int | str) to a sounddevice index.
        String inputs are matched as case-insensitive substring against device names.
        Returns None to mean "system default."
        """
        if device is None:
            return None
        if isinstance(device, int):
            return device
        if not isinstance(device, str) or not device.strip():
            return None
        needle = device.strip().lower()
        try:
            devices = sd.query_devices()
        except Exception as e:
            logger.warning(f"[WakeWord] Cannot enumerate devices: {e}")
            return None
        for i, d in enumerate(devices):
            d_dict = d if isinstance(d, dict) else {}
            if int(d_dict.get("max_input_channels", 0)) <= 0:
                continue
            if needle in str(d_dict.get("name", "")).lower():
                return i
        logger.warning(f"[WakeWord] No input device matches {device!r} — using default")
        return None

    async def listen_forever(self) -> None:
        """
        Main async entry point. Runs the blocking stream loop in a thread
        so the event loop stays alive. Restarts automatically on recoverable errors.
        """
        if not self.loaded:
            raise WakeWordError("Model not loaded — call load() first")

        self._running = True
        loop = asyncio.get_running_loop()
        logger.info(f"[WakeWord] Listening in room '{self._room}'...")

        while self._running:
            try:
                await asyncio.to_thread(self._stream_loop, loop)
            except asyncio.CancelledError:
                break
            except WakeWordError:
                raise
            except Exception as e:
                logger.error(f"[WakeWord] Stream error: {e} — restarting in 3s")
                await asyncio.sleep(3)

            # Stream exited cleanly due to suspend — wait until recording is done
            while self._running and self._suspended:
                await asyncio.sleep(0.05)

    def _stream_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Blocking audio stream loop. Runs in a thread pool via asyncio.to_thread.
        Reads audio chunks and passes them to openWakeWord for detection.

        Args:
            loop: The running event loop, used for thread-safe event publishing.
        """
        sensitivity = self._cfg.get("sensitivity", 0.5)
        cooldown = self._cfg.get("cooldown_seconds", 2)

        # Audio level meter — accumulate ~1s of chunks then emit averaged dBFS.
        # 80ms per chunk × 12 = 960ms. int16 range so divide by 32768 for normalized RMS.
        level_blocks_per_emit = 12
        level_buffer: list[float] = []

        try:
            with sd.InputStream(
                samplerate=OWW_SAMPLE_RATE,
                channels=1,
                dtype=np.int16,  # openWakeWord expects int16
                blocksize=OWW_CHUNK_SIZE,
                device=self._device,
            ) as stream:
                logger.debug("[WakeWord] Audio stream open")

                while self._running and not self._suspended:
                    block, overflowed = stream.read(OWW_CHUNK_SIZE)
                    if overflowed:
                        # Overflow is non-fatal — we just lose a chunk
                        logger.debug("[WakeWord] Audio overflow")

                    # openWakeWord expects shape (N,) int16
                    audio_chunk = block.flatten()

                    # Audio level meter — track normalized RMS for dashboard
                    chunk_norm = audio_chunk.astype(np.float32) / 32768.0
                    level_buffer.append(float(np.sqrt(np.mean(chunk_norm ** 2))))
                    if len(level_buffer) >= level_blocks_per_emit:
                        avg_rms = sum(level_buffer) / len(level_buffer)
                        level_buffer.clear()
                        db = 20.0 * float(np.log10(avg_rms)) if avg_rms > 1e-10 else -100.0
                        asyncio.run_coroutine_threadsafe(
                            self._bus.publish(
                                "audio.level",
                                {"room": self._room, "db": db},
                            ),
                            loop,
                        )

                    # Get prediction scores for all loaded models
                    model = self._model
                    if model is None:
                        raise WakeWordError("Wake word model was unloaded during streaming")
                    try:
                        predictions = model.predict(audio_chunk)
                    except Exception as e:
                        logger.debug(f"[WakeWord] Prediction error: {e}")
                        continue

                    if isinstance(predictions, tuple):
                        predictions = predictions[0]
                    if not isinstance(predictions, dict):
                        logger.debug("[WakeWord] Ignoring unexpected prediction payload")
                        continue

                    for model_name, score in predictions.items():
                        score_value = float(score)
                        if score_value >= sensitivity:
                            now = time.monotonic()
                            if (now - self._last_detection) < cooldown:
                                continue  # Still in cooldown window

                            self._last_detection = now
                            logger.info(
                                f"[WakeWord] Detected '{model_name}' "
                                f"(score={score_value:.3f}) in room '{self._room}'"
                            )

                            # Publish from thread — must use call_soon_threadsafe
                            payload = {
                                "room": self._room,
                                "confidence": score_value,
                                "model": model_name,
                            }
                            asyncio.run_coroutine_threadsafe(
                                self._bus.publish("voice.wake_detected", payload),
                                loop,
                            )

        except sd.PortAudioError as e:
            raise WakeWordError(f"Audio stream failed: {e}") from e

    def suspend(self) -> None:
        """
        Release the microphone so record_until_silence can open it.
        The listen_forever loop will reopen the stream after wakeup() is called.
        """
        self._suspended = True
        logger.debug("[WakeWord] Mic suspended for recording")

    def wakeup(self) -> None:
        """
        Re-enable microphone after recording is complete.
        Resets the cooldown timer so the freshly reopened stream cannot
        immediately re-fire on residual chime echo or the tail of the spoken
        request — without this, one wake utterance produces 2-3 detections.
        """
        self._last_detection = time.monotonic()
        self._suspended = False
        logger.debug("[WakeWord] Mic resumed (cooldown reset)")

    def stop(self) -> None:
        """Signal the stream loop to stop. Non-blocking."""
        self._running = False
        logger.debug("[WakeWord] Stop requested")

    @property
    def device(self) -> Optional[int | str]:
        """Configured input device for wake-listening and follow-up recording."""
        return self._device
