"""
JARVIS — Ambient Home AI
========================
Mission: Manage all video input sources — local USB webcams via OpenCV and
         remote ESP32-CAM nodes via short-lived HTTP snapshot fetches. Local
         devices keep cv2.VideoCapture (works fine). Network cameras use
         per-frame httpx GETs against the ESP web server's :8081/snapshot
         endpoint instead of holding a persistent MJPEG TCP stream that
         periodically drops.

Modules: modules/vision/camera_manager.py
Classes: CameraManager
Functions:
    CameraManager.__init__(config)          — Initialize camera sources from config
    CameraManager.load()                    — Open all configured cameras
    CameraManager.capture_frame(room)       — Capture a single frame for a room (blocking)
    CameraManager.capture_frame_async(room) — Async frame capture
    CameraManager.get_available_rooms()     — List rooms that have a live camera
    CameraManager.close()                   — Release all camera resources

Variables:
    CameraManager._caps          — {room: cv2.VideoCapture} for LOCAL devices only
    CameraManager._http_urls     — {room: str} snapshot URL for HTTP cameras
    CameraManager._sources       — {room: original_source} for diagnostics + reopen
    CameraManager._read_locks    — {room: asyncio.Lock} so concurrent reads serialize

#todo: Add frame rate throttle to avoid saturating CPU during idle
#todo: Add frame timestamp injection for latency measurement
#todo: Expose last_frame cache so vision loop can skip duplicate frames
#todo: Support RTSP streams in addition to MJPEG
"""

import asyncio
from typing import Any, Optional

import httpx
import numpy as np
from loguru import logger

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    _HAS_CV2 = False
    logger.warning("[CameraManager] OpenCV not available — cameras disabled")

# ESP32-CAM snapshot endpoint port (matches esp32_camera_web_server: snapshot mode)
_DEFAULT_SNAPSHOT_PORT = 8081
_HTTP_TIMEOUT_SECONDS  = 5.0


class CameraManager:
    """
    Unified camera manager for local USB devices and remote HTTP snapshot endpoints.

    Each room has at most one camera. Local devices (int) use cv2.VideoCapture;
    HTTP URLs use per-frame httpx.get() against an ESP32-CAM /snapshot endpoint.
    The HTTP path makes one short request per frame instead of holding a
    persistent MJPEG stream — that's far more stable on ESPHome's web server,
    which routinely terminated long-held streams about once per minute.
    """

    def __init__(self, config: dict) -> None:
        self._rooms_config: list[dict] = config.get("rooms", [])
        # Local USB devices — cv2 captures
        self._caps: dict[str, Any] = {}
        # HTTP snapshot URLs (ESP32-CAM nodes) — fetched per-frame, no persistent state
        self._http_urls: dict[str, str] = {}
        # Original source for diagnostics + reopen
        self._sources: dict[str, Any] = {}
        self._last_frames: dict[str, np.ndarray] = {}
        self._fail_counts: dict[str, int] = {}
        self._reopen_threshold: int = 3
        # Per-camera read lock so concurrent callers (dashboard 2s polling +
        # vision_loop 60s scan) don't race on the same cv2.VideoCapture or
        # double up HTTP fetches in flight.
        self._read_locks: dict[str, asyncio.Lock] = {}
        # Reusable HTTP client for snapshot fetches (connection pooling)
        self._http_client: Optional[httpx.AsyncClient] = None

    async def load(self) -> None:
        """Open all camera sources defined in config."""
        if not _HAS_CV2:
            logger.warning("[CameraManager] OpenCV not available — cameras disabled")
            return
        assert cv2 is not None  # narrowing for pyright; guaranteed by _HAS_CV2

        # Lazy create HTTP client now that an event loop exists
        self._http_client = httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS)

        for room_cfg in self._rooms_config:
            room_id = room_cfg.get("id", "unknown")
            source = room_cfg.get("camera_source")

            # ESP32-CAM nodes get the snapshot URL on port 8081
            if source is None and room_cfg.get("has_node", False):
                node_ip = room_cfg.get("node_ip")
                if isinstance(node_ip, str) and node_ip.strip():
                    source = f"http://{node_ip}:{_DEFAULT_SNAPSHOT_PORT}/"
                else:
                    logger.warning(
                        f"[CameraManager] Room '{room_id}' has_node=true but no node_ip configured"
                    )

            if source is None:
                continue

            if isinstance(source, str):
                # HTTP snapshot endpoint — verify it serves a JPEG and stash the URL
                if await self._verify_http_source(room_id, source):
                    self._http_urls[room_id] = source
                    self._sources[room_id] = source
                    self._fail_counts[room_id] = 0
                    self._read_locks[room_id] = asyncio.Lock()
                continue

            # Local USB device (int) — keep cv2.VideoCapture path
            await self._open_local_device(room_id, source)

        if not self._caps and not self._http_urls:
            logger.warning("[CameraManager] No cameras available")

    async def _verify_http_source(self, room_id: str, url: str) -> bool:
        """Fetch one frame from the URL to confirm the endpoint is alive."""
        logger.info(f"[CameraManager] Probing snapshot URL {url} for '{room_id}'...")
        frame = await self._fetch_http_frame(url)
        if frame is None:
            logger.warning(
                f"[CameraManager] Could not fetch initial frame from {url} for '{room_id}'"
            )
            return False
        self._last_frames[room_id] = frame
        logger.info(f"[CameraManager] Connected snapshot source {url} for '{room_id}'")
        return True

    async def _open_local_device(self, room_id: str, source: int) -> None:
        """Open a USB webcam via cv2.VideoCapture with retries (DirectShow flakiness)."""
        assert cv2 is not None
        label = str(source)
        attempts = 3
        opened = False
        for attempt in range(1, attempts + 1):
            logger.info(
                f"[CameraManager] Connecting to {label} for '{room_id}' "
                f"(attempt {attempt}/{attempts})..."
            )
            try:
                # Explicit DirectShow on Windows — auto-select hangs for ~20s
                # on Sound BlasterX G5 capture path.
                cap = await asyncio.wait_for(
                    asyncio.to_thread(cv2.VideoCapture, source, cv2.CAP_DSHOW),  # type: ignore[arg-type]
                    timeout=20.0,
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
                    else:
                        self._caps[room_id] = cap
                        self._sources[room_id] = source
                        self._fail_counts[room_id] = 0
                        self._last_frames[room_id] = frame
                        self._read_locks[room_id] = asyncio.Lock()
                        logger.info(f"[CameraManager] Opened {label} for '{room_id}'")
                        opened = True
                        break
                else:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    logger.warning(f"[CameraManager] Could not open {label} for '{room_id}'")
            except asyncio.TimeoutError:
                logger.warning(
                    f"[CameraManager] Timed out connecting to {label} for '{room_id}'"
                )
            except Exception as e:
                logger.warning(f"[CameraManager] Error opening {label} for '{room_id}': {e}")

            if attempt < attempts:
                await asyncio.sleep(1.5)

        if not opened:
            logger.warning(
                f"[CameraManager] Gave up on '{room_id}' after {attempts} attempt(s)"
            )

    async def _fetch_http_frame(self, url: str) -> Optional[np.ndarray]:
        """One HTTP GET → JPEG bytes → decoded BGR frame. None on failure."""
        if cv2 is None:
            return None
        client = self._http_client or httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS)
        try:
            resp = await client.get(url)
            resp.raise_for_status()
        except Exception as e:
            logger.debug(f"[CameraManager] HTTP fetch {url} failed: {e}")
            return None
        try:
            arr = np.frombuffer(resp.content, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return None
            return frame
        except Exception as e:
            logger.debug(f"[CameraManager] JPEG decode {url} failed: {e}")
            return None

    def get_available_rooms(self) -> list[str]:
        """Return list of room IDs that have an open camera."""
        return list(self._caps.keys()) + list(self._http_urls.keys())

    def capture_frame(self, room: str) -> Optional[np.ndarray]:
        """
        Blocking frame capture for a room.
        Returns numpy BGR array (H, W, 3) or None on failure.
        Note: HTTP snapshot rooms can't be served by this sync path; use
        capture_frame_async() instead.
        """
        return self._read_cap(room)

    async def capture_frame_async(self, room: str) -> Optional[np.ndarray]:
        """
        Async frame capture. For HTTP snapshot sources this is a single HTTP
        GET; for local devices it's cv2.read() in a thread. Per-room lock
        prevents concurrent callers (dashboard 2s polling + vision_loop 60s
        scan) from doubling up.
        """
        if room not in self._caps and room not in self._http_urls:
            return None
        lock = self._read_locks.get(room)
        if lock is not None:
            async with lock:
                return await self._capture_locked(room)
        return await self._capture_locked(room)

    async def _capture_locked(self, room: str) -> Optional[np.ndarray]:
        """Inner capture, lock already held. Routes to HTTP or cv2 path."""
        # Cached first frame from open/probe — consume once
        cached = self._last_frames.pop(room, None)
        if cached is not None:
            return cached

        url = self._http_urls.get(room)
        if url is not None:
            frame = await self._fetch_http_frame(url)
            if frame is not None:
                self._fail_counts[room] = 0
                return frame
            # Failures here are transient — ESP web server might be busy.
            # No reopen needed since each request is independent.
            self._fail_counts[room] = self._fail_counts.get(room, 0) + 1
            logger.debug(
                f"[CameraManager] HTTP fetch failed for '{room}' "
                f"({self._fail_counts[room]} consecutive)"
            )
            return None

        # Local cv2 device path
        return await asyncio.to_thread(self._read_cap, room)

    def _read_cap(self, room: str) -> Optional[np.ndarray]:
        """Read one frame from the local cv2.VideoCapture for this room."""
        cap = self._caps.get(room)
        if cap is None:
            return None
        try:
            ret, frame = cap.read()
            if ret and frame is not None:
                self._fail_counts[room] = 0
                return frame
        except Exception as e:
            logger.warning(f"[CameraManager] Capture error for '{room}': {e}")

        self._fail_counts[room] = self._fail_counts.get(room, 0) + 1
        logger.warning(
            f"[CameraManager] Empty frame from '{room}' "
            f"({self._fail_counts[room]}/{self._reopen_threshold})"
        )
        if self._fail_counts[room] >= self._reopen_threshold:
            self._reopen(room)
        return None

    def _reopen(self, room: str) -> None:
        """Close and reopen a local cv2.VideoCapture. HTTP sources don't need this."""
        if not _HAS_CV2:
            return
        assert cv2 is not None
        source = self._sources.get(room)
        if source is None or not isinstance(source, int):
            return
        old = self._caps.pop(room, None)
        if old is not None:
            try:
                old.release()
            except Exception:
                pass
        try:
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
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
        self._http_urls.clear()
        self._last_frames.clear()
        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception:
                pass
            self._http_client = None
