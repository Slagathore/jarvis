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
import io
from typing import Any, Optional, Protocol

import numpy as np
from loguru import logger

cv2 = None

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    logger.warning("[CameraManager] OpenCV not available — local cameras disabled")

aiohttp = None

try:
    import aiohttp
    _AIOHTTP_AVAILABLE = True
except ImportError:
    _AIOHTTP_AVAILABLE = False
    logger.warning("[CameraManager] aiohttp not available — MJPEG streams disabled")


class VideoCaptureLike(Protocol):
    def isOpened(self) -> bool: ...
    def read(self) -> tuple[bool, np.ndarray]: ...
    def release(self) -> None: ...


class CameraManager:
    """
    Unified camera manager for local USB and remote MJPEG cameras.

    Each room can have at most one camera source. Rooms with no camera
    return None from capture_frame and are excluded from get_available_rooms().
    """

    def __init__(self, config: dict) -> None:
        self._rooms_config: list[dict] = config.get("rooms", [])
        self._local_caps: dict[str, VideoCaptureLike] = {}
        self._mjpeg_urls: dict[str, str] = {}      # room_id → URL string
        self._aiohttp_session: Optional[Any] = None

    async def load(self) -> None:
        """Open all camera sources defined in config."""
        if _AIOHTTP_AVAILABLE and aiohttp is not None:
            self._aiohttp_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            )

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

            if isinstance(source, int):
                # Local USB webcam
                if _CV2_AVAILABLE and cv2 is not None:
                    cap = cv2.VideoCapture(source)
                    if cap.isOpened():
                        self._local_caps[room_id] = cap
                        logger.info(
                            f"[CameraManager] Opened local camera {source} for '{room_id}'"
                        )
                    else:
                        logger.warning(
                            f"[CameraManager] Could not open camera {source} for '{room_id}'"
                        )

            elif isinstance(source, str) and source.startswith("http"):
                # Remote MJPEG stream (ESP32-CAM)
                self._mjpeg_urls[room_id] = source
                logger.info(
                    f"[CameraManager] Registered MJPEG stream for '{room_id}': {source}"
                )

        if not self._local_caps and not self._mjpeg_urls:
            logger.warning("[CameraManager] No cameras available")

    def get_available_rooms(self) -> list[str]:
        """Return list of room IDs that have an open camera."""
        return list(self._local_caps.keys()) + list(self._mjpeg_urls.keys())

    def capture_frame(self, room: str) -> Optional[np.ndarray]:
        """
        Blocking frame capture for a room.
        Returns numpy BGR array (H, W, 3) or None on failure.
        """
        if room in self._local_caps:
            return self._capture_local(room)
        # MJPEG frames are async-only
        return None

    async def capture_frame_async(self, room: str) -> Optional[np.ndarray]:
        """
        Async frame capture. Handles both local and remote cameras.
        Returns numpy BGR array (H, W, 3) or None on failure.
        """
        if room in self._local_caps:
            return await asyncio.to_thread(self._capture_local, room)
        elif room in self._mjpeg_urls:
            return await self._capture_mjpeg(self._mjpeg_urls[room])
        return None

    def _capture_local(self, room: str) -> Optional[np.ndarray]:
        """Read one frame from a local OpenCV VideoCapture."""
        cap = self._local_caps.get(room)
        if cap is None:
            return None
        try:
            ret, frame = cap.read()
            if ret:
                return frame
            logger.debug(f"[CameraManager] Empty frame from local camera '{room}'")
            return None
        except Exception as e:
            logger.warning(f"[CameraManager] Local capture error for '{room}': {e}")
            return None

    async def _capture_mjpeg(self, url: str) -> Optional[np.ndarray]:
        """
        Fetch a single JPEG frame from an MJPEG stream endpoint.
        Expects the URL to return a single JPEG image on GET.
        """
        session = self._aiohttp_session
        if not _AIOHTTP_AVAILABLE or session is None:
            return None
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.read()
            # Decode JPEG bytes to numpy array via OpenCV
            if _CV2_AVAILABLE and cv2 is not None:
                arr = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                return frame
            return None
        except Exception as e:
            logger.debug(f"[CameraManager] MJPEG fetch error ({url}): {e}")
            return None

    async def close(self) -> None:
        """Release all camera resources."""
        for room, cap in self._local_caps.items():
            try:
                cap.release()
            except Exception:
                pass
            logger.debug(f"[CameraManager] Released camera for '{room}'")

        self._local_caps.clear()

        session = self._aiohttp_session
        if session is not None:
            await session.close()
            self._aiohttp_session = None
