"""
JARVIS — Ambient Home AI
========================
Mission: Manage all video input sources — local USB webcams, ESP32-CAM HTTP
         snapshots, and Wyze V2 RTSP streams (via wz_mini_hacks). Each room
         declares its `video.type` in config.yaml; this manager dispatches
         to the right open/read code path. The rest of Jarvis just calls
         capture_frame_async(room) and gets a BGR ndarray back regardless of
         whether the bytes came from a webcam, a local HTTP fetch, or an
         RTSP demuxer.

         Driver selection is an internal dispatch on `video.type`:
           - "usb_index"   → cv2.VideoCapture(int, CAP_DSHOW)
           - "esp32_http"  → per-frame httpx GET against a /snapshot endpoint
           - "wyze_rtsp"   → cv2.VideoCapture(url, CAP_FFMPEG) with the
                             three Wyze-specific touches: FFmpeg backend,
                             buffer size 1, and stale-frame draining before
                             every read.
           - "none"        → returns None, no resources held

         Each room also gets an asyncio.Lock so the dashboard's polling and
         the vision_loop's scheduled scan don't race on the same capture.

Modules: modules/vision/camera_manager.py
Classes: CameraManager
Functions:
    CameraManager.__init__(config)          — Wire room configs from config.yaml
    CameraManager.load()                    — Open every camera defined in rooms[]
    CameraManager.capture_frame(room)       — Sync read (USB only)
    CameraManager.capture_frame_async(room) — Async read (all source types)
    CameraManager.get_available_rooms()     — Rooms with a live camera
    CameraManager.close()                   — Release every capture cleanly

Variables:
    CameraManager._caps          — {room: cv2.VideoCapture} for USB + RTSP
    CameraManager._http_urls     — {room: str} HTTP snapshot URL
    CameraManager._video_kinds   — {room: "usb"|"http"|"rtsp"} for routing
    CameraManager._read_locks    — Per-room asyncio.Lock for serialized reads
    CameraManager._wyze_drain    — Stale-frame drain count for RTSP rooms

#todo: Add per-frame timestamp injection so vision_loop can detect stalled streams
#todo: Expose a last_frame cache so duplicate-frame detection can short-circuit
       expensive scans (YOLO + MediaPipe)
#todo: Move HTTP snapshot port (8081) into the room.video.url instead of
       constructing it here — currently we trust the URL has the port baked in
#todo: SHARED-RTSP-CONTAINER for Wyze: cv2.VideoCapture and the WyzeRtspMicSource
       both open separate RTSP connections to the same /video6_unicast path on
       the cam. wz_mini_hacks accepts the second client but only feeds audio
       to one of them — the second silently gets zero packets. Right fix:
       have the Wyze video path use PyAV (same as the mic) and share one
       container per cam, with two consumers (a video-frame queue feeding
       capture_frame_async, and an audio-chunk queue feeding the mic
       callback). Until then, smoke-test Wyze rooms one channel at a time
       and don't run vision + STT against the same Wyze cam in production.
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

_HTTP_TIMEOUT_SECONDS = 5.0
# Default Wyze RTSP open timeout. cv2's FFmpeg backend reads ~5MB of stream
# data before declaring the capture open; on Wyze's low-bitrate H.264 that
# can take 15-25s on cold connect. PyAV is much faster (~1s) for the same
# URL, so the mic side opens quickly while the video side trails. 30s
# gives enough slack to avoid spurious "cam offline" reports while still
# bailing on a genuinely-dead host. Override via
# config.drivers.wyze_rtsp_video.open_timeout_s.
_DEFAULT_RTSP_OPEN_TIMEOUT_SECONDS = 30.0


class CameraManager:
    """
    Multi-source video manager — USB webcam, ESP32-CAM HTTP snapshot, or
    Wyze V2 RTSP. Reads the per-room `video:` block from config.yaml and
    dispatches each room to the right driver.

    Each room has at most one camera. Per-room asyncio.Lock prevents the
    dashboard's snapshot polling from racing with the vision_loop's scheduled
    YOLO+MediaPipe scan on the same cv2.VideoCapture handle.
    """

    def __init__(self, config: dict) -> None:
        self._rooms_config: list[dict] = config.get("rooms", [])
        # cv2.VideoCapture pool — used by both USB ("usb") and RTSP ("rtsp") rooms
        self._caps: dict[str, Any] = {}
        # HTTP snapshot URLs — fetched per-frame, no persistent state
        self._http_urls: dict[str, str] = {}
        # Routing: which read code path to use for each room
        self._video_kinds: dict[str, str] = {}  # "usb" | "http" | "rtsp"
        # Last source spec (int / url) for diagnostics + reopen
        self._sources: dict[str, Any] = {}
        # First-frame cache from probe — consumed once per room
        self._last_frames: dict[str, np.ndarray] = {}
        # Consecutive read-failure count → triggers reopen at threshold
        self._fail_counts: dict[str, int] = {}
        self._reopen_threshold: int = 3
        self._read_locks: dict[str, asyncio.Lock] = {}
        # Pull tunables from config.drivers; fall back to sensible defaults.
        # The drain count is the magic Wyze touch — the camera sends a
        # buffered backlog when you connect, and we want the freshest frame.
        drivers = config.get("drivers", {}) or {}
        wyze_v = drivers.get("wyze_rtsp_video", {}) or {}
        self._wyze_drain: int = int(wyze_v.get("drain_stale_frames", 2))
        self._wyze_buffer_size: int = int(wyze_v.get("buffer_size", 1))
        self._wyze_reconnect_delay: float = float(wyze_v.get("reconnect_delay_s", 3.0))
        self._wyze_open_timeout: float = float(
            wyze_v.get("open_timeout_s", _DEFAULT_RTSP_OPEN_TIMEOUT_SECONDS)
        )
        # Reusable HTTP client for snapshot fetches (connection pooling). Lazy
        # because httpx's client requires an event loop on init for some
        # transport configs.
        self._http_client: Optional[httpx.AsyncClient] = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def load(self) -> None:
        """Open every camera defined in config.rooms[].video. Tolerant of
        per-room failures — a Wyze cam offline at boot logs and skips that
        room without aborting the rest.
        """
        if not _HAS_CV2:
            logger.warning("[CameraManager] OpenCV not available — cameras disabled")
            return
        assert cv2 is not None  # narrowing for pyright

        self._http_client = httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS)

        for room_cfg in self._rooms_config:
            room_id = room_cfg.get("id", "unknown")
            video_cfg = room_cfg.get("video") or {}
            if not isinstance(video_cfg, dict):
                logger.warning(
                    f"[CameraManager] Room '{room_id}' has malformed 'video:' block; skipping"
                )
                continue
            vtype = video_cfg.get("type", "none")
            if vtype == "none":
                continue
            try:
                await self._open_one(room_id, vtype, video_cfg, room_cfg)
            except Exception as e:
                logger.warning(
                    f"[CameraManager] Failed to open '{room_id}' ({vtype}): {e}"
                )

        if not self._caps and not self._http_urls:
            logger.warning("[CameraManager] No cameras available")

    async def _open_one(
        self,
        room_id: str,
        vtype: str,
        video_cfg: dict,
        room_cfg: dict,
    ) -> None:
        """Dispatch to the right opener based on video.type."""
        if vtype == "usb_index":
            await self._open_usb(
                room_id,
                int(video_cfg.get("device_index", 0)),
                room_cfg.get("fps_active"),
            )
        elif vtype == "esp32_http":
            url = str(video_cfg.get("url", "")).strip()
            if not url:
                logger.warning(f"[CameraManager] Room '{room_id}' esp32_http has no url")
                return
            await self._open_http(room_id, url)
        elif vtype == "wyze_rtsp":
            url = str(video_cfg.get("url", "")).strip()
            transport = str(video_cfg.get("transport", "tcp")).lower()
            if not url:
                logger.warning(f"[CameraManager] Room '{room_id}' wyze_rtsp has no url")
                return
            await self._open_wyze_rtsp(room_id, url, transport)
        else:
            logger.warning(
                f"[CameraManager] Room '{room_id}' has unknown video.type '{vtype}'"
            )

    # ── HTTP snapshot path (ESP32-CAM) ───────────────────────────────────────

    async def _open_http(self, room_id: str, url: str) -> None:
        logger.info(f"[CameraManager] Probing snapshot URL {url} for '{room_id}'...")
        frame = await self._fetch_http_frame(url)
        if frame is None:
            logger.warning(
                f"[CameraManager] Could not fetch initial frame from {url} for '{room_id}'"
            )
            return
        self._http_urls[room_id] = url
        self._video_kinds[room_id] = "http"
        self._sources[room_id] = url
        self._fail_counts[room_id] = 0
        self._last_frames[room_id] = frame
        self._read_locks[room_id] = asyncio.Lock()
        logger.info(f"[CameraManager] Connected snapshot source {url} for '{room_id}'")

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

    # ── USB webcam path ──────────────────────────────────────────────────────

    async def _open_usb(
        self,
        room_id: str,
        device_index: int,
        fps_active: Optional[int],
    ) -> None:
        """Open a USB webcam via cv2.VideoCapture with retries (DirectShow flakiness)."""
        assert cv2 is not None
        attempts = 3
        for attempt in range(1, attempts + 1):
            logger.info(
                f"[CameraManager] Connecting to USB device {device_index} for "
                f"'{room_id}' (attempt {attempt}/{attempts})..."
            )
            try:
                # Explicit DirectShow on Windows — auto-select hangs ~20s on
                # Cole's Sound BlasterX G5 capture path (it enumerates as a
                # webcam with no available frames).
                cap = await asyncio.wait_for(
                    asyncio.to_thread(cv2.VideoCapture, device_index, cv2.CAP_DSHOW),
                    timeout=20.0,
                )
                if not cap.isOpened():
                    self._safe_release(cap)
                    logger.warning(
                        f"[CameraManager] Could not open device {device_index} for '{room_id}'"
                    )
                else:
                    if fps_active is not None:
                        try:
                            await asyncio.to_thread(
                                cap.set, cv2.CAP_PROP_FPS, float(fps_active)
                            )
                            actual = await asyncio.to_thread(cap.get, cv2.CAP_PROP_FPS)
                            logger.info(
                                f"[CameraManager] '{room_id}' requested {fps_active} fps, "
                                f"driver reports {actual:.1f}"
                            )
                        except Exception as e:
                            logger.debug(
                                f"[CameraManager] Could not set fps for '{room_id}': {e}"
                            )
                    ok, frame = await asyncio.wait_for(
                        asyncio.to_thread(cap.read), timeout=5.0
                    )
                    if not ok or frame is None:
                        self._safe_release(cap)
                        logger.warning(
                            f"[CameraManager] Opened device {device_index} for "
                            f"'{room_id}' but could not read a frame"
                        )
                    else:
                        self._caps[room_id] = cap
                        self._video_kinds[room_id] = "usb"
                        self._sources[room_id] = device_index
                        self._fail_counts[room_id] = 0
                        self._last_frames[room_id] = frame
                        self._read_locks[room_id] = asyncio.Lock()
                        logger.info(
                            f"[CameraManager] Opened device {device_index} for '{room_id}'"
                        )
                        return
            except asyncio.TimeoutError:
                logger.warning(
                    f"[CameraManager] Timed out connecting to device {device_index} "
                    f"for '{room_id}'"
                )
            except Exception as e:
                logger.warning(
                    f"[CameraManager] Error opening device {device_index} for "
                    f"'{room_id}': {e}"
                )

            if attempt < attempts:
                await asyncio.sleep(1.5)

        logger.warning(
            f"[CameraManager] Gave up on '{room_id}' after {attempts} attempt(s)"
        )

    # ── Wyze RTSP path ───────────────────────────────────────────────────────

    async def _open_wyze_rtsp(self, room_id: str, url: str, transport: str) -> None:
        """Open a Wyze RTSP stream with FFmpeg backend + buffer size 1.

        The Wyze-specific touches:
          1. CAP_FFMPEG backend — the default backend is platform-dependent
             and on Windows often picks a native DShow path that doesn't
             know what to do with rtsp://.
          2. CAP_PROP_BUFFERSIZE=1 — OpenCV defaults to ~5 frames buffered;
             Wyze streams ~30s of latency by the time you grab a frame
             without this. With buffer=1 + the stale-drain in capture, we
             get ~150ms latency.
          3. OPENCV_FFMPEG_CAPTURE_OPTIONS sets the RTSP transport
             (typically tcp; udp is faster but flakier on busy WiFi).
          4. analyzeduration + probesize tuned WAY down (default is ~5s /
             5MB, which on Wyze's low-bitrate stream takes 15-25s before
             cv2.VideoCapture's constructor returns). Cutting both to
             ~500ms / 100KB lets the open call finish in 2-4s — still
             enough probe data for FFmpeg to lock the codec, fast enough
             that test_wyze.py doesn't appear to hang.
        """
        assert cv2 is not None
        # OpenCV reads FFmpeg options from this env var at capture-open
        # time. There's no first-class API to pass them. Reset it after open
        # so the next room can use a different transport without surprise.
        import os as _os
        prev = _os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
        # Pipe-separated key;value pairs is FFmpeg's option syntax for the
        # OpenCV bridge. stimeout is the socket-level read timeout in
        # microseconds — without it, a half-broken cam wedges the open call
        # past our async timeout and the orphaned thread holds GIL contention.
        ffmpeg_opts = (
            f"rtsp_transport;{transport}"
            "|stimeout;5000000"
            "|analyzeduration;500000"
            "|probesize;100000"
        )
        _os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ffmpeg_opts
        try:
            cap = await asyncio.wait_for(
                asyncio.to_thread(cv2.VideoCapture, url, cv2.CAP_FFMPEG),
                timeout=self._wyze_open_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"[CameraManager] RTSP open timed out for '{room_id}' ({url})"
            )
            return
        except Exception as e:
            logger.warning(f"[CameraManager] RTSP open error for '{room_id}': {e}")
            return
        finally:
            # Restore the env var so other RTSP opens in the same process
            # don't accidentally inherit this transport choice.
            if prev is None:
                _os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
            else:
                _os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = prev

        if not cap.isOpened():
            self._safe_release(cap)
            logger.warning(
                f"[CameraManager] RTSP could not open '{room_id}' ({url}) — "
                "check that wz_mini_hacks is running and the IP/port are reachable"
            )
            return

        try:
            await asyncio.to_thread(cap.set, cv2.CAP_PROP_BUFFERSIZE, self._wyze_buffer_size)
        except Exception as e:
            logger.debug(f"[CameraManager] Couldn't set buffer size for '{room_id}': {e}")

        # Probe one frame so we know the stream's actually flowing, not just
        # that the TCP socket opened.
        try:
            ok, frame = await asyncio.wait_for(
                asyncio.to_thread(cap.read), timeout=8.0
            )
        except asyncio.TimeoutError:
            self._safe_release(cap)
            logger.warning(
                f"[CameraManager] RTSP probe read timed out for '{room_id}'"
            )
            return
        except Exception as e:
            self._safe_release(cap)
            logger.warning(f"[CameraManager] RTSP probe read error for '{room_id}': {e}")
            return

        if not ok or frame is None:
            self._safe_release(cap)
            logger.warning(
                f"[CameraManager] RTSP opened '{room_id}' but no frame — stream may "
                "be advertised but not yet producing video"
            )
            return

        self._caps[room_id] = cap
        self._video_kinds[room_id] = "rtsp"
        self._sources[room_id] = url
        self._fail_counts[room_id] = 0
        self._last_frames[room_id] = frame
        self._read_locks[room_id] = asyncio.Lock()
        logger.info(f"[CameraManager] RTSP connected for '{room_id}' ({url})")

    # ── Public read API ──────────────────────────────────────────────────────

    def get_available_rooms(self) -> list[str]:
        """Return list of room IDs that have an open camera (any kind)."""
        return list(self._caps.keys()) + list(self._http_urls.keys())

    def capture_frame(self, room: str) -> Optional[np.ndarray]:
        """
        Blocking frame capture for a room.
        Returns numpy BGR array (H, W, 3) or None on failure.
        Note: HTTP snapshot rooms can't be served by this sync path — use
        capture_frame_async() for those.
        """
        kind = self._video_kinds.get(room)
        if kind == "http":
            # Sync HTTP fetch isn't worth implementing — every caller of the
            # sync path is in a thread already, and async-from-sync is awkward.
            logger.debug(
                f"[CameraManager] capture_frame() can't serve HTTP room '{room}'; "
                "use capture_frame_async()"
            )
            return None
        return self._read_cap(room)

    async def capture_frame_async(self, room: str) -> Optional[np.ndarray]:
        """
        Async frame capture. Routes by source kind:
          - http: single httpx GET against the snapshot URL
          - usb / rtsp: cv2.read() in a thread, with stale-frame draining
            for RTSP so we always get the freshest frame
        Per-room lock prevents concurrent callers from doubling up.
        """
        if room not in self._caps and room not in self._http_urls:
            return None
        lock = self._read_locks.get(room)
        if lock is not None:
            async with lock:
                return await self._capture_locked(room)
        return await self._capture_locked(room)

    async def _capture_locked(self, room: str) -> Optional[np.ndarray]:
        """Inner capture, lock already held. Routes to the right source kind."""
        # Cached first frame from open/probe — consume once, then fall through
        # to a real read on the next call.
        cached = self._last_frames.pop(room, None)
        if cached is not None:
            return cached

        kind = self._video_kinds.get(room)
        if kind == "http":
            url = self._http_urls.get(room)
            if url is None:
                return None
            frame = await self._fetch_http_frame(url)
            if frame is not None:
                self._fail_counts[room] = 0
                return frame
            self._fail_counts[room] = self._fail_counts.get(room, 0) + 1
            logger.debug(
                f"[CameraManager] HTTP fetch failed for '{room}' "
                f"({self._fail_counts[room]} consecutive)"
            )
            return None

        if kind == "rtsp":
            return await asyncio.to_thread(self._read_cap_rtsp, room)

        # USB
        return await asyncio.to_thread(self._read_cap, room)

    # ── cv2.VideoCapture readers ─────────────────────────────────────────────

    def _read_cap(self, room: str) -> Optional[np.ndarray]:
        """USB webcam: one cv2.read(). Reopen on threshold-many failures."""
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
            self._reopen_usb(room)
        return None

    def _read_cap_rtsp(self, room: str) -> Optional[np.ndarray]:
        """RTSP read with stale-frame draining. cap.grab() N times to flush
        the FFmpeg internal buffer, then cap.read() to actually decode the
        next frame. Without this, we read a frame from ~30s ago.
        """
        cap = self._caps.get(room)
        if cap is None:
            return None
        try:
            for _ in range(self._wyze_drain):
                cap.grab()
            ret, frame = cap.read()
            if ret and frame is not None:
                self._fail_counts[room] = 0
                return frame
        except Exception as e:
            logger.warning(f"[CameraManager] RTSP read error for '{room}': {e}")

        self._fail_counts[room] = self._fail_counts.get(room, 0) + 1
        logger.warning(
            f"[CameraManager] RTSP empty/error frame from '{room}' "
            f"({self._fail_counts[room]}/{self._reopen_threshold})"
        )
        if self._fail_counts[room] >= self._reopen_threshold:
            self._reopen_rtsp(room)
        return None

    # ── Reopen + cleanup ─────────────────────────────────────────────────────

    def _reopen_usb(self, room: str) -> None:
        """Close and reopen a USB cv2.VideoCapture."""
        if not _HAS_CV2:
            return
        assert cv2 is not None
        source = self._sources.get(room)
        if source is None or not isinstance(source, int):
            return
        old = self._caps.pop(room, None)
        if old is not None:
            self._safe_release(old)
        try:
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if cap.isOpened():
                self._caps[room] = cap
                self._fail_counts[room] = 0
                logger.info(f"[CameraManager] Reopened USB '{room}' ({source})")
            else:
                self._safe_release(cap)
                logger.warning(
                    f"[CameraManager] USB reopen failed for '{room}' — will retry"
                )
        except Exception as e:
            logger.warning(f"[CameraManager] USB reopen error for '{room}': {e}")

    def _reopen_rtsp(self, room: str) -> None:
        """Close and reopen a Wyze RTSP cv2.VideoCapture. Sleeps the
        configured reconnect_delay first — hammering wz_mini_hacks while it's
        already in trouble (cam rebooting, WiFi blip) just makes it worse.
        """
        if not _HAS_CV2:
            return
        assert cv2 is not None
        source = self._sources.get(room)
        if source is None or not isinstance(source, str):
            return
        old = self._caps.pop(room, None)
        if old is not None:
            self._safe_release(old)
        # Sync sleep — we're already inside a thread (called from
        # _read_cap_rtsp via asyncio.to_thread). Don't add asyncio overhead.
        import time as _time
        _time.sleep(self._wyze_reconnect_delay)
        try:
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, self._wyze_buffer_size)
                self._caps[room] = cap
                self._fail_counts[room] = 0
                logger.info(f"[CameraManager] Reopened RTSP '{room}'")
            else:
                self._safe_release(cap)
                logger.warning(
                    f"[CameraManager] RTSP reopen failed for '{room}' — will retry"
                )
        except Exception as e:
            logger.warning(f"[CameraManager] RTSP reopen error for '{room}': {e}")

    @staticmethod
    def _safe_release(cap: Any) -> None:
        try:
            cap.release()
        except Exception:
            pass

    async def close(self) -> None:
        """Release every cv2.VideoCapture and the shared httpx client."""
        for room, cap in self._caps.items():
            self._safe_release(cap)
            logger.debug(f"[CameraManager] Released camera for '{room}'")
        self._caps.clear()
        self._http_urls.clear()
        self._video_kinds.clear()
        self._last_frames.clear()
        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception:
                pass
            self._http_client = None
