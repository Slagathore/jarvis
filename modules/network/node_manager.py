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

# Seconds without heartbeat before a node is considered offline
NODE_STALE_SECONDS: float = 30.0


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

    def __init__(self, config: dict, mqtt_client) -> None:
        self._config = config
        self._mqtt = mqtt_client
        self._stale_seconds: float = NODE_STALE_SECONDS

        # Initialize node records from rooms config
        self._nodes: dict[str, NodeInfo] = {}
        for room_cfg in config.get("rooms", []):
            room_id = room_cfg.get("id", "unknown")
            if room_cfg.get("has_node", False):
                self._nodes[room_id] = NodeInfo(
                    room=room_id,
                    ip_address=room_cfg.get("node_ip"),
                )

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
        Updates the node's heartbeat time and online flag.
        """
        room = self._mqtt._extract_room(topic)
        now = datetime.now()

        if room not in self._nodes:
            # Auto-register unknown rooms that start reporting
            self._nodes[room] = NodeInfo(room=room)
            logger.info(f"[NodeManager] Auto-registered new node: '{room}'")

        node = self._nodes[room]
        was_online = node.online

        node.online = True
        node.last_seen = now

        # Parse status payload fields
        if isinstance(data, dict):
            node.ip_address = data.get("ip", node.ip_address)
            node.firmware_version = data.get("fw", node.firmware_version)
            node.has_camera = data.get("cam", node.has_camera)
            node.has_microphone = data.get("mic", node.has_microphone)

        if not was_online:
            logger.info(f"[NodeManager] Node '{room}' came online (IP: {node.ip_address})")

    async def monitor_heartbeats(self) -> None:
        """
        Background task that checks for stale nodes every 10 seconds.
        Marks nodes offline if no heartbeat received within stale threshold.
        """
        while True:
            try:
                await asyncio.sleep(10)
                now = datetime.now()
                for room, node in self._nodes.items():
                    if not node.online:
                        continue
                    if node.last_seen is None:
                        continue
                    age = (now - node.last_seen).total_seconds()
                    if age > self._stale_seconds:
                        node.online = False
                        logger.warning(
                            f"[NodeManager] Node '{room}' went offline "
                            f"(no heartbeat for {age:.0f}s)"
                        )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[NodeManager] Heartbeat monitor error: {e}")
