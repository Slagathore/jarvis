"""
JARVIS — Ambient Home AI
========================
Mission: Track the online/offline status, last heartbeat time, and capabilities
         of all ESP32-CAM nodes deployed in the house. Provides a single source
         of truth for which rooms have active hardware nodes, enabling the
         orchestrator to route audio output to the correct room and know when
         nodes drop off the network.

Modules: modules/network/node_manager.py
Classes: NodeManager, NodeInfo (dataclass)
Functions:
    NodeManager.__init__(config, mqtt_client)  — Init with config and MQTT
    NodeManager.load()                          — Subscribe to MQTT topics
    NodeManager.get_node(room)                  — Get NodeInfo for a room
    NodeManager.get_online_rooms()              — List rooms with active nodes
    NodeManager.is_online(room)                 — True if node is alive
    NodeManager.send_audio(room, audio_bytes)   — Send TTS audio to a room's node
    NodeManager.get_status_summary()            — Dict of all node statuses
    NodeManager._on_status(topic, data)         — MQTT status handler
    NodeManager._check_stale_nodes()            — Background heartbeat monitor

Variables:
    NodeManager._nodes           — {room: NodeInfo}
    NodeManager._mqtt            — MQTTClient reference
    NodeManager._stale_seconds   — Threshold for marking node offline
    NodeInfo.room                — Room identifier
    NodeInfo.online              — bool, currently reachable
    NodeInfo.last_seen           — datetime of last heartbeat
    NodeInfo.ip_address          — Last known IP
    NodeInfo.firmware_version    — Reported firmware version string
    NodeInfo.has_camera          — bool
    NodeInfo.has_microphone      — bool

#todo: Add node OTA firmware update trigger via MQTT
#todo: Add node configuration push (wake word sensitivity, audio gain)
#todo: Add multi-node audio routing (speak in the room Cole is in)
#todo: Add node discovery via mDNS for zero-config setup
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from loguru import logger

# Seconds without heartbeat before a node is considered offline.
# ESPHome firmware publishes status every 15s via interval automation; the broker
# also fires the LWT instantly on ungraceful disconnect. 90s gives ~6 missed
# heartbeats of slack before we call it dead.
NODE_STALE_SECONDS: float = 90.0


@dataclass
class NodeInfo:
    """Status record for a single ESP32-CAM node."""
    room: str
    online: bool = False
    last_seen: Optional[datetime] = None
    ip_address: Optional[str] = None
    firmware_version: Optional[str] = None
    has_camera: bool = True
    has_microphone: bool = True


class NodeManager:
    """
    Tracks status of all ESP32-CAM nodes via MQTT heartbeats.

    Nodes publish to 'jarvis/nodes/{room}/status' every N seconds.
    If a node goes silent for NODE_STALE_SECONDS, it's marked offline.
    """

    def __init__(self, config: dict, mqtt_client, event_bus=None) -> None:
        self._config = config
        self._mqtt = mqtt_client
        # Optional bus reference — kept optional so unit tests that
        # construct a bare NodeManager without a full orchestrator
        # don't have to invent one. When present, transition publishes
        # go here; when absent, transitions only update local state.
        self._event_bus = event_bus
        self._stale_seconds: float = NODE_STALE_SECONDS

        # Initialize node records from rooms config. Under the new toggle
        # schema, "this room has an ESP32-CAM node" means at least one of
        # video / mic / speaker uses an esp32_* driver. We derive the IP
        # from the video URL when present; mic/speaker MQTT topics don't
        # carry IPs (they go through the broker).
        self._nodes: dict[str, NodeInfo] = {}
        # Per-room desired FPS, published on connect so the firmware's
        # set_idle_update_interval lambda picks it up.
        self._room_fps_idle: dict[str, int] = {}
        for room_cfg in config.get("rooms", []):
            room_id = room_cfg.get("id", "unknown")
            if self._room_has_esp32(room_cfg):
                self._nodes[room_id] = NodeInfo(
                    room=room_id,
                    ip_address=self._extract_esp32_ip(room_cfg),
                )
            fps_idle = room_cfg.get("fps_idle")
            if isinstance(fps_idle, (int, float)) and fps_idle > 0:
                self._room_fps_idle[room_id] = int(fps_idle)

        # Auxiliary nodes — ESP32 boxes physically present in rooms whose
        # primary mic/cam comes from a non-ESP source (USB, Wyze RTSP, etc.).
        # Without this section the NodeManager would auto-register them on
        # first MQTT heartbeat with a WARNING; declaring them here lets us
        # acknowledge the hardware exists without changing the room's primary
        # driver. Each entry: {room_id: <str>, fps_idle: <int> (optional)}.
        for node_cfg in config.get("nodes", []):
            if not isinstance(node_cfg, dict):
                continue
            aux_id = node_cfg.get("room_id")
            if not aux_id or aux_id in self._nodes:
                continue
            self._nodes[aux_id] = NodeInfo(room=aux_id)
            aux_fps = node_cfg.get("fps_idle")
            if isinstance(aux_fps, (int, float)) and aux_fps > 0:
                self._room_fps_idle[aux_id] = int(aux_fps)

    @staticmethod
    def _room_has_esp32(room_cfg: dict) -> bool:
        """True if any of the room's three channels uses an esp32_* driver."""
        for key in ("video", "mic", "speaker"):
            channel = room_cfg.get(key) or {}
            if isinstance(channel, dict):
                ctype = str(channel.get("type", ""))
                if ctype.startswith("esp32_"):
                    return True
        return False

    @staticmethod
    def _extract_esp32_ip(room_cfg: dict) -> Optional[str]:
        """Pull the ESP32 IP from the video URL when present. Returns None
        when only mic/speaker are ESP-routed (those go through MQTT topics,
        not direct IPs).
        """
        video_cfg = room_cfg.get("video") or {}
        if isinstance(video_cfg, dict) and video_cfg.get("type") == "esp32_http":
            url = str(video_cfg.get("url", ""))
            # Cheap parse: http://<host>:<port>/<path> → host
            try:
                from urllib.parse import urlparse
                return urlparse(url).hostname
            except Exception:
                return None
        return None

    async def load(self) -> None:
        """Register MQTT subscriptions for node status topics."""
        # Subscribe to status from all nodes (wildcard)
        self._mqtt.subscribe("jarvis/nodes/+/status", self._on_status)
        logger.info(
            f"[NodeManager] Tracking {len(self._nodes)} configured nodes: "
            + ", ".join(self._nodes.keys())
        )

    def get_node(self, room: str) -> Optional[NodeInfo]:
        """Return the NodeInfo for a specific room, or None if not configured."""
        return self._nodes.get(room)

    def get_online_rooms(self) -> list[str]:
        """Return list of room IDs with currently-online nodes."""
        return [
            room for room, info in self._nodes.items() if info.online
        ]

    def is_online(self, room: str) -> bool:
        """Return True if the node in the given room is currently online."""
        node = self._nodes.get(room)
        return node is not None and node.online

    async def send_audio(self, room: str, audio_bytes: bytes) -> bool:
        """
        Send TTS audio bytes to a room's ESP32-CAM node over MQTT.

        Args:
            room:        Target room identifier.
            audio_bytes: Raw PCM audio bytes to send.

        Returns:
            True if published successfully, False if node is offline or MQTT error.
        """
        if not self.is_online(room):
            logger.debug(f"[NodeManager] Cannot send audio — node '{room}' is offline")
            return False

        topic = f"jarvis/nodes/{room}/audio/out"
        success = await self._mqtt.publish(topic, audio_bytes, qos=1)
        if success:
            logger.debug(f"[NodeManager] Sent {len(audio_bytes)} audio bytes to '{room}'")
        return success

    def get_status_summary(self) -> dict[str, dict]:
        """Return serializable status dict for all known nodes."""
        return {
            room: {
                "online":            info.online,
                "last_seen":         info.last_seen.isoformat() if info.last_seen else None,
                "ip_address":        info.ip_address,
                "firmware_version":  info.firmware_version,
                "has_camera":        info.has_camera,
                "has_microphone":    info.has_microphone,
            }
            for room, info in self._nodes.items()
        }

    async def _on_status(self, topic: str, data: dict) -> None:
        """
        Handle incoming node status message.
        Updates the node's heartbeat time and online flag, and publishes
        a node.status event ONLY on a real transition (offline→online,
        or IP / firmware change). Heartbeats that confirm an already-
        online node don't fire an event — that was the ~15s-per-node
        log spam.
        """
        room = self._mqtt._extract_room(topic)
        now = datetime.now()

        if room not in self._nodes:
            # Auto-register unknown rooms that start reporting
            self._nodes[room] = NodeInfo(room=room)
            logger.warning(
                f"[NodeManager] Auto-registered unexpected node '{room}'. "
                "Check that the firmware room_id matches config.yaml."
            )

        node = self._nodes[room]
        was_online = node.online
        prev_ip = node.ip_address
        prev_fw = node.firmware_version

        node.online = True
        node.last_seen = now

        # Parse status payload fields
        if isinstance(data, dict):
            node.ip_address = data.get("ip", node.ip_address)
            node.firmware_version = data.get("fw", node.firmware_version)
            node.has_camera = data.get("cam", node.has_camera)
            node.has_microphone = data.get("mic", node.has_microphone)

        transition = not was_online
        metadata_changed = (
            node.ip_address != prev_ip
            or node.firmware_version != prev_fw
        )

        if transition:
            logger.info(f"[NodeManager] Node '{room}' came online (IP: {node.ip_address})")
            # Push the configured idle FPS to the node so its camera frame rate
            # matches what's in config.yaml (firmware default is 1fps idle).
            fps_idle = self._room_fps_idle.get(room)
            if fps_idle:
                fps_topic = f"jarvis/nodes/{room}/camera/fps"
                try:
                    await self._mqtt.publish(fps_topic, str(fps_idle), qos=0)
                    logger.info(
                        f"[NodeManager] Set '{room}' camera idle fps to {fps_idle}"
                    )
                except Exception as e:
                    logger.debug(f"[NodeManager] FPS publish to '{room}' failed: {e}")

        if (transition or metadata_changed) and self._event_bus is not None:
            await self._event_bus.publish(
                "node.status",
                {
                    "room": room,
                    "topic": topic,
                    "online": True,
                    "ip": node.ip_address,
                    "firmware_version": node.firmware_version,
                    "data": data,
                },
            )

    async def monitor_heartbeats(self) -> None:
        """
        Background task that checks for stale nodes every 10 seconds.
        Marks nodes offline if no heartbeat received within stale threshold,
        and publishes a node.status transition so the dashboard and
        orchestrator notice promptly.
        """
        while True:
            try:
                await asyncio.sleep(10)
                now = datetime.now()
                stale_rooms: list[tuple[str, NodeInfo, float]] = []
                for room, node in self._nodes.items():
                    if not node.online:
                        continue
                    if node.last_seen is None:
                        continue
                    age = (now - node.last_seen).total_seconds()
                    if age > self._stale_seconds:
                        node.online = False
                        stale_rooms.append((room, node, age))
                        logger.warning(
                            f"[NodeManager] Node '{room}' went offline "
                            f"(no heartbeat for {age:.0f}s)"
                        )
                if stale_rooms and self._event_bus is not None:
                    for room, node, _age in stale_rooms:
                        await self._event_bus.publish(
                            "node.status",
                            {
                                "room": room,
                                "online": False,
                                "ip": node.ip_address,
                                "firmware_version": node.firmware_version,
                            },
                        )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[NodeManager] Heartbeat monitor error: {e}")
