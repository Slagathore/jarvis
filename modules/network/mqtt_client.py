"""
JARVIS — Ambient Home AI
========================
Mission: Provide an async MQTT client for communicating with ESP32-CAM nodes
         over the home network. Handles connection, reconnection, topic subscription,
         message publishing, and message dispatch to event bus events. Uses
         aiomqtt for non-blocking async operation so MQTT never blocks
         the main event loop.

         MQTT is the primary communication channel between Jarvis (running on
         the home PC) and ESP32-CAM nodes deployed in each room. Nodes stream
         audio in, receive TTS audio out, and send status heartbeats.

Modules: modules/network/mqtt_client.py
Classes: MQTTClient
Functions:
    MQTTClient.__init__(config, event_bus) — Initialize with config
    MQTTClient.connect()                   — Establish async MQTT connection
    MQTTClient.disconnect()                — Clean shutdown
    MQTTClient.publish(topic, payload)     — Publish message (bytes or str)
    MQTTClient.subscribe(topic, handler)   — Register topic → async handler
    MQTTClient.listen_forever()            — Async loop dispatching incoming messages
    MQTTClient.is_connected                — Property: bool

Variables:
    MQTTClient._client          — aiomqtt.Client instance
    MQTTClient._event_bus       — EventBus reference for publishing node events
    MQTTClient._config          — mqtt config dict
    MQTTClient._subscriptions   — {topic_pattern: async_handler}
    MQTTClient._connected       — bool state flag

#todo: Add QoS level 1 for audio_out messages (at-least-once delivery)
#todo: Add retained messages for node discovery (nodes publish presence on connect)
#todo: Add TLS support for secure MQTT over internet
#todo: Add message rate limiting to prevent flooding
"""

import asyncio
import json
from typing import Any, Awaitable, Callable, Optional

from loguru import logger

AsyncMQTTClient = None
MqttError = Exception

try:
    from aiomqtt import Client as AsyncMQTTClient
    from aiomqtt import MqttError
    _ASYNCIO_MQTT_AVAILABLE = True
except ImportError:
    try:
        from asyncio_mqtt import Client as AsyncMQTTClient
        from asyncio_mqtt import MqttError
        _ASYNCIO_MQTT_AVAILABLE = True
    except ImportError:
        _ASYNCIO_MQTT_AVAILABLE = False
        logger.warning("[MQTT] aiomqtt/asyncio-mqtt not available — MQTT disabled")

MQTTMessageHandler = Callable[[str, Any], Awaitable[None]]


class MQTTClient:
    """
    Async MQTT client wrapping aiomqtt.

    Provides connection management, topic subscription dispatch, and safe
    reconnection. All incoming messages are dispatched to registered handlers
    and also forwarded to the event bus as `node.*` events.
    """

    def __init__(self, config: dict, event_bus) -> None:
        mqtt_cfg = config.get("mqtt", {})
        self._broker:   str = mqtt_cfg.get("broker", "localhost")
        self._port:     int = int(mqtt_cfg.get("port", 1883))
        self._username: Optional[str] = mqtt_cfg.get("username") or None
        self._password: Optional[str] = mqtt_cfg.get("password") or None
        self._keepalive: int = int(mqtt_cfg.get("keepalive", 60))
        self._reconnect_delay: float = float(
            mqtt_cfg.get("reconnect_delay_seconds", 5)
        )
        self._topics_cfg: dict = mqtt_cfg.get("topics", {})

        self._event_bus = event_bus
        self._client: Optional[Any] = None
        self._connected: bool = False
        self._subscriptions: dict[str, MQTTMessageHandler] = {}
        self._stop_event = asyncio.Event()

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> bool:
        """
        Establish MQTT connection. Returns True on success.
        Non-blocking — listen_forever() must be called separately to process messages.
        """
        if not _ASYNCIO_MQTT_AVAILABLE:
            logger.warning("[MQTT] MQTT async client unavailable — skipping connection")
            return False
        if AsyncMQTTClient is None:
            logger.warning("[MQTT] MQTT client class unavailable — skipping connection")
            return False

        try:
            kwargs: dict[str, Any] = {
                "hostname": self._broker,
                "port":     self._port,
                "keepalive": self._keepalive,
            }
            if self._username:
                kwargs["username"] = self._username
            if self._password:
                kwargs["password"] = self._password

            client = AsyncMQTTClient(**kwargs)
            await client.__aenter__()
            self._client = client
            self._connected = True
            logger.info(f"[MQTT] Connected to {self._broker}:{self._port}")
            return True

        except Exception as e:
            logger.error(f"[MQTT] Connection failed: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Gracefully close the MQTT connection."""
        self._stop_event.set()
        self._connected = False
        if self._client:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception:
                pass
            self._client = None
        logger.info("[MQTT] Disconnected")

    async def publish(
        self,
        topic: str,
        payload: Any,
        qos: int = 0,
        retain: bool = False,
    ) -> bool:
        """
        Publish a message to the given MQTT topic.

        Args:
            topic:   MQTT topic string.
            payload: str, bytes, or JSON-serializable dict.
            qos:     MQTT QoS level (0 or 1).
            retain:  Whether the broker should retain the message.

        Returns:
            True if published successfully, False otherwise.
        """
        if not self._connected or not self._client:
            logger.debug(f"[MQTT] Publish skipped (not connected): {topic}")
            return False

        try:
            client = self._client
            if client is None:
                return False

            if isinstance(payload, dict):
                payload = json.dumps(payload)
            if isinstance(payload, str):
                payload = payload.encode("utf-8")

            await client.publish(topic, payload, qos=qos, retain=retain)
            return True

        except Exception as e:
            logger.warning(f"[MQTT] Publish failed to '{topic}': {e}")
            return False

    def subscribe(self, topic: str, handler: MQTTMessageHandler) -> None:
        """
        Register an async handler for messages matching a topic pattern.
        Wildcard topics (+ and #) are supported.
        """
        self._subscriptions[topic] = handler
        logger.debug(f"[MQTT] Registered handler for topic '{topic}'")

    async def listen_forever(self) -> None:
        """
        Continuously listen for MQTT messages and dispatch to handlers.
        Reconnects automatically on connection loss.
        """
        if not _ASYNCIO_MQTT_AVAILABLE:
            return

        while not self._stop_event.is_set():
            try:
                if not self._connected:
                    success = await self.connect()
                    if not success:
                        await asyncio.sleep(self._reconnect_delay)
                        continue
                client = self._client
                if client is None:
                    self._connected = False
                    await asyncio.sleep(self._reconnect_delay)
                    continue

                # Subscribe to all registered topics
                for topic in self._subscriptions:
                    await client.subscribe(topic)
                    logger.debug(f"[MQTT] Subscribed to '{topic}'")

                # Message loop
                await self._consume_messages(client)

            except MqttError as e:
                logger.warning(f"[MQTT] Connection lost: {e} — reconnecting in {self._reconnect_delay}s")
                self._connected = False
                await asyncio.sleep(self._reconnect_delay)

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.error(f"[MQTT] Unexpected error in listen loop: {e}")
                self._connected = False
                await asyncio.sleep(self._reconnect_delay)

    async def _consume_messages(self, client: Any) -> None:
        """
        Consume MQTT messages across both aiomqtt and asyncio-mqtt APIs.

        aiomqtt exposes `client.messages` as a property returning an async
        iterator. Older asyncio-mqtt versions expose `client.messages()` as
        an async context manager.
        """
        messages_attr = getattr(client, "messages")
        messages = messages_attr() if callable(messages_attr) else messages_attr

        if hasattr(messages, "__aenter__") and hasattr(messages, "__aexit__"):
            async with messages as stream:
                await self._consume_message_stream(stream)
            return

        await self._consume_message_stream(messages)

    async def _consume_message_stream(self, messages: Any) -> None:
        """Dispatch messages from an async iterator until stopped."""
        async for msg in messages:
            if self._stop_event.is_set():
                break
            await self._dispatch(str(msg.topic), msg.payload)

    async def _dispatch(self, topic: str, payload: bytes) -> None:
        """
        Dispatch an incoming MQTT message to matching registered handlers
        and forward relevant node events to the event bus.
        """
        try:
            # Decode payload: prefer JSON, fall back to UTF-8 text, only keep raw
            # bytes for genuinely binary topics (audio streams). Without the text
            # fallback, plain string statuses like "online" arrive as bytes and
            # break downstream isinstance(data, str) checks.
            data: Any
            try:
                text = payload.decode("utf-8")
            except UnicodeDecodeError:
                data = payload
            else:
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    data = text

            # Dispatch to registered handlers
            for pattern, handler in self._subscriptions.items():
                if self._topic_matches(pattern, topic):
                    try:
                        await handler(topic, data)
                    except Exception as e:
                        logger.warning(
                            f"[MQTT] Handler error for '{topic}': {e}"
                        )

            # Forward node status messages to the event bus
            if "/status" in topic:
                room = self._extract_room(topic)
                await self._event_bus.publish(
                    "node.status",
                    {"room": room, "topic": topic, "data": data},
                )

            elif "/audio/in" in topic:
                room = self._extract_room(topic)
                await self._event_bus.publish(
                    "node.audio_received",
                    {"room": room, "audio": data},
                )

        except Exception as e:
            logger.debug(f"[MQTT] Dispatch error for '{topic}': {e}")

    @staticmethod
    def _topic_matches(pattern: str, topic: str) -> bool:
        """
        Check if a topic matches a subscription pattern.
        Supports '+' (single level wildcard) and '#' (multi-level wildcard).
        """
        if pattern == topic:
            return True

        p_parts = pattern.split("/")
        t_parts = topic.split("/")

        for i, (p, t) in enumerate(zip(p_parts, t_parts)):
            if p == "#":
                return True
            if p != "+" and p != t:
                return False

        return len(p_parts) == len(t_parts)

    @staticmethod
    def _extract_room(topic: str) -> str:
        """Extract room name from topic like 'jarvis/nodes/{room}/...'"""
        parts = topic.split("/")
        if len(parts) >= 3 and parts[1] == "nodes":
            return parts[2]
        return "unknown"
