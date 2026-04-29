"""
JARVIS — Ambient Home AI
========================
Mission: Manage all video input sources — local USB webcams via OpenCV and
         ESP32-CAM remote nodes via MJPEG-over-HTTP streams. Provides a unified
         async frame capture interface so the rest of the vision pipeline doesn't
         care whether the camera is local or remote.

         Supports per-room camera assignment, graceful fallback when cameras
         are unavailable, and frame buffering for downstream consumers.

Modules: modules/vision/camera_manager.py
Classes: CameraManager
Functions:
    CameraManager.__init__(config)          — Initialize camera sources from config
    CameraManager.load()                    — Open all configured cameras
    CameraManager.capture_frame(room)       — Capture a single frame for a room
    CameraManager.capture_frame_async(room) — Async wrapper
    CameraManager.get_available_rooms()     — List rooms that have a live camera
    CameraManager.close()                   — Release all OpenCV captures
    CameraManager._open_local(source)       — Open USB camera via cv2.VideoCapture
    CameraManager._capture_mjpeg(url)       — Fetch single JPEG from MJPEG stream

Variables:
    CameraManager._local_caps    — {room: cv2.VideoCapture}
    CameraManager._mjpeg_urls    — {room: str} for ESP32-CAM nodes
    CameraManager._config        — rooms config list

#todo: Add frame rate throttle to avoid saturating CPU during idle
#todo: Add MJPEG stream reconnect with exponential backoff
#todo: Add frame timestamp injection for latency measurement
#todo: Expose last_frame cache so vision loop can skip duplicate frames
#todo: Support RTSP streams in addition to MJPEG
"""

import asyncio
from typing import Any, Optional

import numpy as np
from loguru import logger

cv2 = None

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    logger.warning("[CameraManager] OpenCV not available — cameras disabled")


class CameraManager:
    """
    Unified camera manager for local USB and remote MJPEG cameras.

    Each room can have at most one camera source. Rooms with no camera
    return None from capture_frame and are excluded from get_available_rooms().

    Both local (int device index) and remote (HTTP MJPEG URL) cameras are
    opened via cv2.VideoCapture, which handles MJPEG streams natively.
    """

    def __init__(self, config: dict) -> None:
        self._rooms_config: list[dict] = config.get("rooms", [])
        self._caps: dict[str, Any] = {}   # room_id → cv2.VideoCapture
        self._last_frames: dict[str, np.ndarray] = {}

    async def load(self) -> None:
        """Open all camera sources defined in config."""
        if not _CV2_AVAILABLE or cv2 is None:
            logger.warning("[CameraManager] OpenCV not available — cameras disabled")
            return

        for room_cfg in self._rooms_config:
            room_id = room_cfg.get("id", "unknown")
            source = room_cfg.get("camera_source")

            if source is None and room_cfg.get("has_node", False):
                node_ip = room_cfg.get("node_ip")
                if isinstance(node_ip, str) and node_ip.strip():
                    source = f"http://{node_ip}:8080/"
                else:
                    logger.warning(
                        f"[CameraManager] Room '{room_id}' has_node=true but no node_ip configured"
                    )

            if source is None:
                continue

            label = str(source)
            logger.info(f"[CameraManager] Connecting to {label} for '{room_id}'...")
            try:
                # cv2.VideoCapture blocks (network connection for URLs) — run in thread
                # with a timeout so an offline node doesn't stall the whole startup.
                # Local cameras (int index) need more time on Windows DirectShow
                open_timeout = 20.0 if isinstance(source, int) else 8.0
                cap = await asyncio.wait_for(
                    asyncio.to_thread(cv2.VideoCapture, source),
                    timeout=open_timeout,
                )
                if cap.isOpened():
                    ok, frame = await asyncio.wait_for(
                        asyncio.to_thread(cap.read),
                        timeout=5.0,
                    )
                    if not ok or frame is None:
                        try:
                            cap.release()
                        except Exception:
                            pass
                        logger.warning(
                            f"[CameraManager] Opened {label} for '{room_id}' but could not read a frame"
                        )
                        continue
                    self._caps[room_id] = cap
                    self._last_frames[room_id] = frame
                    logger.info(f"[CameraManager] Opened {label} for '{room_id}'")
                else:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    logger.warning(f"[CameraManager] Could not open {label} for '{room_id}'")
            except asyncio.TimeoutError:
                logger.warning(
                    f"[CameraManager] Timed out connecting to {label} for '{room_id}' — "
                    "node may be offline"
                )
            except Exception as e:
                logger.warning(f"[CameraManager] Error opening {label} for '{room_id}': {e}")

        if not self._caps:
            logger.warning("[CameraManager] No cameras available")

    def get_available_rooms(self) -> list[str]:
        """Return list of room IDs that have an open camera."""
        return list(self._caps.keys())

    def capture_frame(self, room: str) -> Optional[np.ndarray]:
        """
        Blocking frame capture for a room.
        Returns numpy BGR array (H, W, 3) or None on failure.
        """
        return self._read_cap(room)

    async def capture_frame_async(self, room: str) -> Optional[np.ndarray]:
        """
        Async frame capture. Runs the blocking cv2.read() in a thread.
        Returns numpy BGR array (H, W, 3) or None on failure.
        """
        if room not in self._caps:
            return None
        return await asyncio.to_thread(self._read_cap, room)

    def _read_cap(self, room: str) -> Optional[np.ndarray]:
        """Read one frame from the VideoCapture for this room."""
        cap = self._caps.get(room)
        if cap is None:
            return None
        cached = self._last_frames.pop(room, None)
        if cached is not None:
            return cached
        try:
            ret, frame = cap.read()
            if ret:
                return frame
            logger.warning(f"[CameraManager] Empty frame from '{room}' — stream may have dropped")
            return None
        except Exception as e:
            logger.warning(f"[CameraManager] Capture error for '{room}': {e}")
            return None

    async def close(self) -> None:
        """Release all camera resources."""
        for room, cap in self._caps.items():
            try:
                cap.release()
            except Exception:
                pass
            logger.debug(f"[CameraManager] Released camera for '{room}'")
        self._caps.clear()
        self._last_frames.clear()
