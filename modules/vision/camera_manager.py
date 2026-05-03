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

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    _HAS_CV2 = False
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
        # Source per room so we can reopen on stream failure (ESP32-CAM web server
        # routinely terminates MJPEG sessions; OpenCV does not retry on its own).
        self._sources: dict[str, Any] = {}
        self._fail_counts: dict[str, int] = {}
        self._reopen_threshold: int = 3
        # Per-camera read lock — without this, the dashboard snapshot endpoint
        # (every 2s) and the vision_loop scan (every 60s) race on cap.read(),
        # which corrupts the MJPEG demuxer state and forces a reconnect each
        # time the two paths overlap.
        self._read_locks: dict[str, asyncio.Lock] = {}

    async def load(self) -> None:
        """Open all camera sources defined in config."""
        if not _HAS_CV2:
            logger.warning("[CameraManager] OpenCV not available — cameras disabled")
            return
        assert cv2 is not None  # narrowing for pyright; guaranteed by _HAS_CV2

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
                    asyncio.to_thread(cv2.VideoCapture, source),  # type: ignore[arg-type]
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
                    self._sources[room_id] = source
                    self._fail_counts[room_id] = 0
                    self._last_frames[room_id] = frame
                    self._read_locks[room_id] = asyncio.Lock()
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
        Holds a per-room lock so concurrent callers (vision_loop + dashboard
        snapshot endpoint) don't corrupt OpenCV's MJPEG demuxer state.
        """
        if room not in self._caps:
            return None
        lock = self._read_locks.get(room)
        if lock is None:
            return await asyncio.to_thread(self._read_cap, room)
        async with lock:
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
            if ret and frame is not None:
                self._fail_counts[room] = 0
                return frame
        except Exception as e:
            logger.warning(f"[CameraManager] Capture error for '{room}': {e}")

        # Read failed — count consecutive failures and reopen the stream after a
        # few. ESP32-CAM HTTP streams routinely terminate; we have to reconnect.
        self._fail_counts[room] = self._fail_counts.get(room, 0) + 1
        logger.warning(
            f"[CameraManager] Empty frame from '{room}' "
            f"({self._fail_counts[room]}/{self._reopen_threshold})"
        )
        if self._fail_counts[room] >= self._reopen_threshold:
            self._reopen(room)
        return None

    def _reopen(self, room: str) -> None:
        """Close and reopen the VideoCapture for a room. Best-effort, may fail."""
        if not _HAS_CV2:
            return
        assert cv2 is not None  # narrowing for pyright; guaranteed by _HAS_CV2
        source = self._sources.get(room)
        if source is None:
            return
        old = self._caps.pop(room, None)
        if old is not None:
            try:
                old.release()
            except Exception:
                pass
        try:
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                self._caps[room] = cap
                self._fail_counts[room] = 0
                logger.info(f"[CameraManager] Reopened '{room}' ({source})")
            else:
                try:
                    cap.release()
                except Exception:
                    pass
                logger.warning(f"[CameraManager] Reopen failed for '{room}' — will retry")
        except Exception as e:
            logger.warning(f"[CameraManager] Reopen error for '{room}': {e}")

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
