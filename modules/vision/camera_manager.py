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
import threading
from typing import Any, Optional

import httpx
import numpy as np
from loguru import logger


class _RTSPFrameDrainer:
    """Background-thread RTSP frame drainer.

    Why this exists: cv2.VideoCapture against an RTSP source on the FFmpeg
    backend buffers frames internally. CAP_PROP_BUFFERSIZE=1 is a hint
    the FFmpeg backend on Windows often ignores in practice. When we read
    at 5 fps from a 15 fps source, the cv2/FFmpeg pipeline silently
    accumulates frames between our reads — a small backlog at first, then
    seconds of lag, then tens of seconds. Result: the dashboard shows
    Cole in the bedroom 30 s after he physically left.

    This class drains as fast as the source delivers. Each successful
    read is stored in a thread-safe slot, overwriting whatever was there.
    `latest()` returns whatever is currently in the slot — always the most
    recent frame the cv2 cap has produced. Anything in the FFmpeg backlog
    is consumed (and discarded) by the drainer thread, so it never piles
    up.

    Owned 1:1 with a cv2.VideoCapture. CameraManager creates a drainer
    when it opens an RTSP cap and stops the drainer before releasing the
    cap (drainer-then-cap teardown order matters: a cap.release() while
    the drainer thread is mid-read causes a crash inside FFmpeg).
    """

    def __init__(self, cap: Any, room: str) -> None:
        self._cap = cap
        self._room = room
        self._latest: Optional[np.ndarray] = None
        self._latest_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"rtsp-drainer-{room}",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        # Tight read loop. cv2.read() blocks until the next frame arrives,
        # so we naturally pace at the source's frame rate. On read failure
        # we briefly back off to avoid pegging the CPU when the cam has
        # disconnected — the manager's reopen path will replace the cap
        # entirely so this thread will be told to stop.
        consecutive_fails = 0
        while not self._stop_event.is_set():
            try:
                ret, frame = self._cap.read()
            except Exception:
                ret, frame = False, None
            if ret and frame is not None:
                consecutive_fails = 0
                with self._latest_lock:
                    self._latest = frame
            else:
                consecutive_fails += 1
                # 50 ms backoff on failure; if the manager is going to
                # replace the cap it'll signal stop quickly so we won't
                # idle here for long.
                self._stop_event.wait(0.05)
                if consecutive_fails >= 200:
                    # 10s of solid failures — the cap is dead. Stop;
                    # CameraManager's read path will see no frame and
                    # trigger a reopen via the existing throttled path.
                    logger.debug(
                        f"[RTSPDrainer:{self._room}] 10s of read failures, "
                        "exiting thread"
                    )
                    return

    def latest(self) -> Optional[np.ndarray]:
        """Return the most recent frame (or None if no frame has arrived
        yet). Returned array is a SHALLOW reference to the drainer's slot —
        callers shouldn't mutate it; if they need to keep it past the next
        write, they should copy.
        """
        with self._latest_lock:
            return self._latest

    def stop(self, timeout_s: float = 1.0) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout_s)

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

    def __init__(self, config: dict, room_settings: Any = None) -> None:
        self._rooms_config: list[dict] = config.get("rooms", [])
        # RoomSettings is optional — when None (e.g. smoke test, unit test),
        # capture_frame_async returns the raw frame untransformed. When
        # supplied (orchestrator wiring), per-room rotate/flip/brightness/
        # contrast tweaks are applied on every frame so YOLO + MediaPipe
        # see the corrected orientation, not just the dashboard preview.
        self._room_settings = room_settings
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
        # Throttled-retry timestamps for RTSP self-reconnect. When a cap
        # is None (cam rebooted, transient drop), don't hammer the cam
        # — wait at least _wyze_reconnect_delay between attempts.
        self._next_reopen_attempt: dict[str, float] = {}
        # Per-room background drainer threads for RTSP caps. cv2's FFmpeg
        # backend buffers frames internally; without a drainer constantly
        # reading, our 5 fps reads against a 15 fps source accumulate
        # multi-second lag. The drainer keeps the cap drained at source
        # rate and exposes only the latest frame. Owned 1:1 with a
        # cv2.VideoCapture in self._caps[room]; lifetime tied to the cap.
        self._drainers: dict[str, _RTSPFrameDrainer] = {}
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
        """Open a USB webcam via cv2.VideoCapture with retries.

        Per attempt: try DirectShow first, then Media Foundation as a
        fallback. The two backends use different driver paths, so when
        DirectShow returns not-opened (e.g. another process has the
        device, or Windows hasn't released the handle from a prior
        Jarvis instance), CAP_MSMF often succeeds. Avoiding
        auto-backend-select intentionally — on Cole's machine that
        path tries to enumerate his Sound BlasterX G5 as a webcam and
        hangs ~20s.

        Wait between attempts is exponential-ish (1.5 → 3 → 5s) so a
        prior process has time to release the device handle on a clean
        Jarvis restart.
        """
        assert cv2 is not None
        backends = [("CAP_DSHOW", cv2.CAP_DSHOW), ("CAP_MSMF", cv2.CAP_MSMF)]
        between_waits = [1.5, 3.0, 5.0]
        attempts = len(between_waits)

        for attempt in range(1, attempts + 1):
            cap = None
            for backend_label, backend_const in backends:
                logger.info(
                    f"[CameraManager] Connecting to USB device {device_index} for "
                    f"'{room_id}' (attempt {attempt}/{attempts}, {backend_label})..."
                )
                try:
                    candidate = await asyncio.wait_for(
                        asyncio.to_thread(cv2.VideoCapture, device_index, backend_const),
                        timeout=20.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"[CameraManager] {backend_label} timed out for device "
                        f"{device_index}"
                    )
                    continue
                except Exception as e:
                    logger.warning(
                        f"[CameraManager] {backend_label} error opening device "
                        f"{device_index}: {e}"
                    )
                    continue
                if not candidate.isOpened():
                    self._safe_release(candidate)
                    logger.warning(
                        f"[CameraManager] {backend_label} could not open device "
                        f"{device_index} for '{room_id}'"
                    )
                    continue
                cap = candidate
                break  # got a working cap

            if cap is not None:
                # We have an open cap; configure FPS and verify with one read.
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
                try:
                    ok, frame = await asyncio.wait_for(
                        asyncio.to_thread(cap.read), timeout=5.0
                    )
                except asyncio.TimeoutError:
                    self._safe_release(cap)
                    logger.warning(
                        f"[CameraManager] Read timed out for '{room_id}' after open"
                    )
                else:
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
                            f"[CameraManager] Opened device {device_index} for "
                            f"'{room_id}'"
                        )
                        return

            if attempt < attempts:
                wait = between_waits[attempt - 1]
                logger.debug(
                    f"[CameraManager] Will retry '{room_id}' in {wait:.1f}s "
                    "(another process may be holding the device)"
                )
                await asyncio.sleep(wait)

        logger.warning(
            f"[CameraManager] Gave up on '{room_id}' after {attempts} attempt(s) "
            "— check for another app holding the camera (Discord call, browser "
            "with webcam permission, OBS, Camera app)"
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
        # Stop any previous drainer for this room (reopen path) before
        # spawning a new one — same teardown invariant as caps.
        self._stop_drainer(room_id)
        self._drainers[room_id] = _RTSPFrameDrainer(cap, room_id)
        logger.info(f"[CameraManager] RTSP connected for '{room_id}' ({url})")

    # ── Public read API ──────────────────────────────────────────────────────

    def get_available_rooms(self) -> list[str]:
        """Return list of room IDs that have an open camera (any kind)."""
        return list(self._caps.keys()) + list(self._http_urls.keys())

    def get_configured_rooms(self) -> list[dict]:
        """Return every room that has a camera *configured* (regardless of
        whether the underlying capture is currently open).

        Used by the dashboard to decide which rooms get a reconnect button:
        get_available_rooms() reflects only currently-streaming cams, which
        is the wrong signal because a stuck/dropped cam disappears from
        that list precisely when we want the button to appear. Each item
        is `{"room": <id>, "kind": "usb"|"http"|"rtsp"}`.
        """
        return [
            {"room": room, "kind": kind}
            for room, kind in self._video_kinds.items()
        ]

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
            return self._apply_room_tweaks(room, cached)

        kind = self._video_kinds.get(room)
        if kind == "http":
            url = self._http_urls.get(room)
            if url is None:
                return None
            frame = await self._fetch_http_frame(url)
            if frame is not None:
                self._fail_counts[room] = 0
                return self._apply_room_tweaks(room, frame)
            self._fail_counts[room] = self._fail_counts.get(room, 0) + 1
            logger.debug(
                f"[CameraManager] HTTP fetch failed for '{room}' "
                f"({self._fail_counts[room]} consecutive)"
            )
            return None

        if kind == "rtsp":
            frame = await asyncio.to_thread(self._read_cap_rtsp, room)
            return self._apply_room_tweaks(room, frame) if frame is not None else None

        # USB
        frame = await asyncio.to_thread(self._read_cap, room)
        return self._apply_room_tweaks(room, frame) if frame is not None else None

    # ── Per-room post-processing ─────────────────────────────────────────────

    def _apply_room_tweaks(self, room: str, frame: np.ndarray) -> np.ndarray:
        """Apply rotate/flip/brightness/contrast from RoomSettings. Returns
        the original frame untouched when no settings or no RoomSettings
        instance — keeps the hot path cheap when the dashboard hasn't
        configured anything for this room.
        """
        if self._room_settings is None or frame is None:
            return frame
        try:
            tweaks = self._room_settings.get(room)
        except Exception:
            return frame
        if not tweaks:
            return frame

        out = frame
        rot = tweaks.get("rotation")
        if rot in (90, 180, 270) and cv2 is not None:
            # cv2 rotate constants: ROTATE_90_CLOCKWISE=0, ROTATE_180=1,
            # ROTATE_90_COUNTERCLOCKWISE=2. We define rotation as clockwise
            # degrees so 90→0, 180→1, 270→2.
            rot_map = {90: cv2.ROTATE_90_CLOCKWISE,
                       180: cv2.ROTATE_180,
                       270: cv2.ROTATE_90_COUNTERCLOCKWISE}
            try:
                out = cv2.rotate(out, rot_map[rot])
            except Exception as e:
                logger.debug(f"[CameraManager] rotate failed for '{room}': {e}")

        flip_h = bool(tweaks.get("flip_h", False))
        flip_v = bool(tweaks.get("flip_v", False))
        if (flip_h or flip_v) and cv2 is not None:
            # cv2.flip flipCode: 0=v, 1=h, -1=both
            code = (-1 if (flip_h and flip_v) else (1 if flip_h else 0))
            try:
                out = cv2.flip(out, code)
            except Exception as e:
                logger.debug(f"[CameraManager] flip failed for '{room}': {e}")

        # Brightness + contrast in one convertScaleAbs call: out = α·in + β.
        # Skip if both are at defaults to avoid the per-pixel pass.
        brightness = float(tweaks.get("brightness", 1.0))
        contrast = float(tweaks.get("contrast", 1.0))
        if (brightness != 1.0 or contrast != 1.0) and cv2 is not None:
            # Map: contrast → α (multiplier); brightness → β (additive offset).
            # Brightness 1.0 = no shift; >1 brightens; <1 darkens. Range
            # ±50 luminance is a reasonable feel for the slider.
            alpha = max(0.1, contrast)
            beta = (brightness - 1.0) * 50.0
            try:
                out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
            except Exception as e:
                logger.debug(
                    f"[CameraManager] brightness/contrast failed for '{room}': {e}"
                )

        return out

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
            # Connection lost — cam rebooted, network blip, or a previous
            # reopen attempt failed (we previously stranded the room with
            # no path back short of a Jarvis restart). Try a throttled
            # reconnect; the next read attempt will use the new cap if it
            # came up. Returns None for THIS read either way.
            self._try_reopen_rtsp_throttled(room)
            return None
        # Read from the drainer's latest-frame slot rather than calling
        # cap.read() directly. The drainer has been pulling at source
        # rate in its own thread, so its slot always holds the freshest
        # frame the cv2/FFmpeg pipeline has produced — no buffered
        # backlog, no multi-second lag. Falls back to direct read if no
        # drainer is registered (shouldn't happen for RTSP rooms but
        # better than crashing).
        drainer = self._drainers.get(room)
        if drainer is not None:
            frame = drainer.latest()
            if frame is not None:
                self._fail_counts[room] = 0
                return frame
            # Drainer hasn't captured anything yet OR the thread died.
            # Fall through to fail-count tracking below so the existing
            # reopen path triggers when a stream genuinely stalls.
        else:
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
        """Close any existing cap then attempt a fresh open. Called from
        _read_cap_rtsp when a still-open cap accumulates _reopen_threshold
        read failures (slow death, packet loss, cam degrading). Drops the
        old cap explicitly because the FFmpeg context is in a bad state
        and reusing it just produces more bad frames.
        """
        # Stop drainer FIRST so its thread isn't reading the cap while
        # we release it. _safe_release on a cap mid-read crashes inside
        # FFmpeg.
        self._stop_drainer(room)
        old = self._caps.pop(room, None)
        if old is not None:
            self._safe_release(old)
        self._try_reopen_rtsp_throttled(room, force=True)

    def _try_reopen_rtsp_throttled(self, room: str, force: bool = False) -> None:
        """Attempt to (re)connect a Wyze RTSP stream, throttled by
        _wyze_reconnect_delay so we don't hammer the cam during a reboot
        or network outage. force=True bypasses the throttle for the
        slow-death reopen path where we already burned _reopen_threshold
        failed reads (the throttle window has effectively elapsed).

        Runs synchronously — callers (_read_cap_rtsp) are already inside
        asyncio.to_thread, and the throttle is the wait, not a sleep.
        Returns nothing; the next read will see the new cap if successful.
        """
        if not _HAS_CV2:
            return
        assert cv2 is not None
        source = self._sources.get(room)
        if source is None or not isinstance(source, str):
            return
        import time as _time
        now = _time.monotonic()
        if not force:
            next_ok = self._next_reopen_attempt.get(room, 0.0)
            if now < next_ok:
                return  # still throttled; quietly skip this attempt
        # Set throttle BEFORE the attempt so a slow / hung VideoCapture
        # constructor can't get retried in parallel.
        self._next_reopen_attempt[room] = now + self._wyze_reconnect_delay

        # Set the same FFmpeg options as the initial open (stimeout +
        # analyzeduration + probesize) so the reconnect doesn't fall
        # back to FFmpeg defaults that take 15-25s on Wyze's bitrate.
        # transport defaults to tcp; if the room originally used udp we
        # don't have that here, but TCP-on-reconnect is a safe choice
        # (slightly slower but more reliable, and the open succeeds).
        import os as _os
        prev_opts = _os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
        _os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp"
            "|stimeout;5000000"
            "|analyzeduration;500000"
            "|probesize;100000"
        )
        try:
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, self._wyze_buffer_size)
                self._caps[room] = cap
                self._fail_counts[room] = 0
                # Spin up a fresh drainer for the new cap. The previous
                # drainer (if any) was already stopped by the path that
                # got us here (_reopen_rtsp via _stop_drainer, or the
                # initial-open path which calls _stop_drainer before
                # registering a new one).
                self._drainers[room] = _RTSPFrameDrainer(cap, room)
                logger.info(f"[CameraManager] RTSP reconnected '{room}'")
            else:
                self._safe_release(cap)
                logger.debug(
                    f"[CameraManager] RTSP reconnect '{room}' not yet ready "
                    f"(retry in {self._wyze_reconnect_delay:.0f}s)"
                )
        except Exception as e:
            logger.debug(f"[CameraManager] RTSP reconnect error for '{room}': {e}")
        finally:
            if prev_opts is None:
                _os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
            else:
                _os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = prev_opts

    @staticmethod
    def _safe_release(cap: Any) -> None:
        try:
            cap.release()
        except Exception:
            pass

    def _stop_drainer(self, room: str) -> None:
        """Stop and remove the drainer for `room` if one exists. Safe to
        call even when no drainer is registered. MUST be called BEFORE
        the underlying cv2.VideoCapture is released — the drainer thread
        is mid-read into the cap, and releasing the cap from another
        thread while it's reading crashes inside FFmpeg.
        """
        drainer = self._drainers.pop(room, None)
        if drainer is not None:
            try:
                drainer.stop(timeout_s=1.0)
            except Exception as e:
                logger.debug(f"[CameraManager] drainer stop for '{room}' raised: {e}")

    async def close(self) -> None:
        """Release every cv2.VideoCapture and the shared httpx client."""
        # Drainers first, then caps — see _stop_drainer for why.
        for room in list(self._drainers.keys()):
            self._stop_drainer(room)
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
