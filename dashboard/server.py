"""
JARVIS — Ambient Home AI
========================
Mission: Real-time web dashboard for monitoring Jarvis's state. Provides a live
         view of activity detection, room states, appliance tracking, conversation
         log, and system health. Uses FastAPI + WebSocket to push events to the
         browser as they happen.

         The orchestrator calls dashboard.broadcast(event_dict) whenever anything
         changes. The dashboard caches the latest state in memory so new browser
         connections immediately get the current picture.

Modules: dashboard/server.py
Classes: DashboardServer
Functions:
    DashboardServer.__init__(host, port)    — Create FastAPI app, init state
    DashboardServer._default_state()        — Initial state before signals arrive
    DashboardServer._setup_routes()         — Register all FastAPI endpoints
    DashboardServer.broadcast(event)        — Push event to all WS clients + update cache
    DashboardServer._update_state(event)    — Update internal state cache from event
    DashboardServer.run()                   — Start uvicorn server as async task

Variables:
    DashboardServer.app          — FastAPI application
    DashboardServer._clients     — List of connected WebSocket clients
    DashboardServer._state       — In-memory state cache (dict)
    DashboardServer._conversation — Last 50 conversation entries

Endpoints:
    GET  /           → serves index.html
    GET  /static/*   → serves CSS, JS
    WS   /ws         → real-time event stream to browser
    GET  /api/state  → current full state snapshot
    GET  /api/health → liveness check

#todo: Add authentication (simple token header) to prevent unauthorized dashboard access
#todo: Add event replay buffer — allow catching up on missed events after reconnect
#todo: Add REST API to manually set Jarvis DND mode from the dashboard
"""

import asyncio
import json
import os
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, Optional, cast

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from loguru import logger

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    _CV2_AVAILABLE = False

STATIC_DIR = Path(__file__).parent / "static"


class DashboardServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 7070):
        self.host = host
        self.port = port
        self.app = FastAPI(title="Jarvis Dashboard", docs_url=None, redoc_url=None)
        self._clients: list[WebSocket] = []
        self._state: dict = self._default_state()
        self._conversation: list[dict] = []  # Last 50 messages
        self._max_conversation = 50
        self._chat_handler = None   # Callable[[str, str], Awaitable] registered by orchestrator
        self._voice_handler = None  # Callable[[str], Awaitable] for runtime voice switching
        self._available_voices: list = []
        self._active_voice: str = ""
        self._camera_manager = None  # Set by orchestrator via register_camera_manager()
        self._camera_jpeg_quality = 70
        self._mic_manager = None     # Set by orchestrator via register_mic_manager()
        self._speaker_manager = None # Set by orchestrator via register_speaker_manager()
        self._room_settings = None   # Set by orchestrator via register_room_settings()
        self._wyze_cam_controls: dict = {}  # Set by orchestrator via register_wyze_cam_controls()
        self._persona = None  # Set by orchestrator via register_persona()
        self._reminders_store = None  # Set by orchestrator via register_reminders_store()
        self._calendar = None         # Set by orchestrator via register_calendar()
        self._interruptibility = None # Set by orchestrator via register_interruptibility()
        self._orchestrator = None     # Set by orchestrator via register_orchestrator()
        self._speaker_id = None       # Set by orchestrator via register_speaker_id()
        self._face_recognizer = None  # Set by orchestrator via register_face_recognizer()
        self._identity = None         # Set by orchestrator via register_identity() — Identity v2
        self._notifications = None    # Set by orchestrator via register_notifications()
        self._model_registry = None   # Set by orchestrator via register_model_registry()
        self._memory = None           # Set by orchestrator via register_memory()
        self._computer = None         # Set by orchestrator via register_computer()
        self._selfedit = None         # Set by orchestrator via register_selfedit()
        self._webhook_manager = None  # Set by orchestrator via register_webhook_manager()
        self._wake_calibration: dict[str, dict] = {}

        self._setup_routes()

    def _default_state(self) -> dict:
        """Initial state before any signals arrive."""
        return {
            "activity": "unknown",
            "location": "unknown",
            "interruptibility": 0.5,
            "confidence": 0.0,
            "signals": [],
            "context": {},
            "rooms": {},
            "appliances": {
                "washer":     {"status": "idle", "runtime_minutes": None},
                "dryer":      {"status": "idle", "runtime_minutes": None},
                "dishwasher": {"status": "idle", "runtime_minutes": None},
            },
            "system": {
                "ollama":    {"online": False, "model": ""},
                "mqtt":      {"online": False, "broker": ""},
                "whisper":   {"loaded": False, "model": ""},
                "nodes":     {},
            },
            "last_speech": None,
            "wake_calibration": {},
            "updated_at": datetime.now().isoformat(),
        }

    def set_room_ids(self, room_ids: list) -> None:
        """Pre-populate the rooms state so the dashboard shows all rooms from startup."""
        for room_id in room_ids:
            if room_id not in self._state["rooms"]:
                self._state["rooms"][room_id] = {}

    def register_chat_handler(self, handler) -> None:
        """Register the coroutine function the orchestrator uses to handle typed messages."""
        self._chat_handler = handler

    def register_voice_handler(self, handler, voices: list, active: str) -> None:
        """Register voice-switch handler and store available voices for the UI."""
        self._voice_handler = handler
        self._available_voices = voices
        self._active_voice = active

    def register_camera_manager(self, camera_manager) -> None:
        """Wire the orchestrator's CameraManager so /api/camera/{room}/snapshot.jpg works."""
        self._camera_manager = camera_manager
        rooms = camera_manager.get_available_rooms() if camera_manager else []
        for room_id in rooms:
            self._state["rooms"].setdefault(room_id, {})
            self._state["rooms"][room_id]["has_camera"] = True

    def register_mic_manager(self, mic_manager) -> None:
        """Wire the orchestrator's MicManager so /api/mic/{room}/* endpoints
        (and the dashboard's per-room mic indicator) know which rooms have
        a live mic source. Per-room state gets a `has_mic` flag."""
        self._mic_manager = mic_manager
        rooms = mic_manager.get_rooms() if mic_manager else []
        for room_id in rooms:
            self._state["rooms"].setdefault(room_id, {})
            self._state["rooms"][room_id]["has_mic"] = True

    def register_speaker_manager(self, speaker_manager) -> None:
        """Wire the orchestrator's SpeakerManager so /api/speaker/{room}/test
        can play a verification phrase in any room with a configured sink."""
        self._speaker_manager = speaker_manager
        rooms = speaker_manager.get_rooms() if speaker_manager else []
        for room_id in rooms:
            self._state["rooms"].setdefault(room_id, {})
            self._state["rooms"][room_id]["has_speaker"] = True
            self._state["rooms"][room_id]["speaker_type"] = speaker_manager.get_speaker_type(room_id)

    def register_room_settings(self, room_settings) -> None:
        """Wire the per-room runtime tweak store so the dashboard's per-feed
        cog modal can read/write rotation, flip, brightness, contrast,
        volume, and mute. The store is shared with CameraManager and
        SpeakerManager so changes take effect on the very next frame /
        play() call without a restart.
        """
        self._room_settings = room_settings

    def register_persona(self, persona_manager) -> None:
        """Wire the PersonaManager so the dashboard's persona dropdown +
        command box can list/switch personas. Hidden personas are
        excluded from the dropdown listing (enforced at /api/persona/list)
        — the only way to activate them is to type the name into the
        command box.
        """
        self._persona = persona_manager

    def register_wyze_cam_controls(self, controls: dict) -> None:
        """Wire per-Wyze-room hardware controls (night vision, IR LEDs,
        status LED, reboot). Dict is {room_id: WyzeCamControl}. The
        modal's WYZE CAMERA section is hidden for rooms not in this dict,
        so non-Wyze rooms (USB office, ESP laundry) don't get useless
        toggles."""
        self._wyze_cam_controls = controls or {}
        for room_id in self._wyze_cam_controls:
            self._state["rooms"].setdefault(room_id, {})
            self._state["rooms"][room_id]["wyze_cam"] = True

    def register_reminders_store(self, store) -> None:
        """Wire the orchestrator's RemindersStore so /api/reminders endpoints work."""
        self._reminders_store = store

    def register_calendar(self, calendar) -> None:
        """Wire the orchestrator's GoogleCalendar so /api/calendar endpoints work."""
        self._calendar = calendar

    def register_interruptibility(self, manager) -> None:
        """Wire InterruptibilityManager so /api/dnd endpoints can toggle DND."""
        self._interruptibility = manager

    def register_orchestrator(self, orchestrator) -> None:
        """Wire the orchestrator so endpoints can poke its state (enrollment
        flag). Also subscribe the dashboard's broadcast pipe to the bus's
        world topics — previously both `world.entity_event` and
        `world.state_snapshot` had no subscribers and the bus dropped
        them on every publish (visible as 'No subscribers for ...' DEBUG
        spam). Forwarding them gives the World Events and Interactions
        panels live updates instead of relying solely on the 5–10s REST
        poll, and silences the dropped-event noise.
        """
        self._orchestrator = orchestrator
        bus = getattr(orchestrator, "bus", None)
        if bus is None:
            return

        async def _on_entity_event(payload: dict) -> None:
            await self.broadcast({
                "type": "world.entity_event",
                **payload,
            })

        async def _on_state_snapshot(payload: dict) -> None:
            # Snapshot is big-ish (every entity); throttle to no more than
            # one broadcast per ~5s so a 30s-cadence publisher doesn't
            # accidentally turn into a firehose under future tuning.
            now = datetime.now().timestamp()
            last = getattr(self, "_last_snapshot_broadcast", 0.0)
            if now - last < 5.0:
                return
            self._last_snapshot_broadcast = now  # type: ignore[attr-defined]
            await self.broadcast({
                "type": "world.state_snapshot",
                **payload,
            })

        try:
            bus.subscribe("world.entity_event", _on_entity_event)
            bus.subscribe("world.state_snapshot", _on_state_snapshot)
        except Exception as e:
            logger.warning(f"[Dashboard] world bus subscribe failed: {e}")

    def register_speaker_id(self, speaker_id) -> None:
        """Wire SpeakerIdentifier so /api/speakers endpoints can list/delete enrollments."""
        self._speaker_id = speaker_id

    def register_face_recognizer(self, face_recognizer) -> None:
        """Wire FaceRecognizer so /api/faces endpoints can list/delete enrollments."""
        self._face_recognizer = face_recognizer

    def register_identity(self, identity_manager) -> None:
        """Wire the Identity v2 manager so /api/identity endpoints can drive
        cross-modal enrollment and the pending-review queue."""
        self._identity = identity_manager

    def register_notifications(self, manager) -> None:
        """Wire NotificationManager so /api/notifications endpoints work."""
        self._notifications = manager

    def register_model_registry(self, registry) -> None:
        """Wire ModelRegistry so /api/models endpoints work."""
        self._model_registry = registry

    def register_memory(self, memory_store) -> None:
        """Wire MemoryStore so /api/memory endpoints work."""
        self._memory = memory_store

    def register_computer(self, computer_control) -> None:
        """Wire ComputerControl so /api/computer endpoints work."""
        self._computer = computer_control

    def register_selfedit(self, selfedit_control) -> None:
        """Wire SelfEditControl so /api/selfedit endpoints work."""
        self._selfedit = selfedit_control

    def register_webhook_manager(self, webhooks) -> None:
        """Wire WebhookManager so /api/webhook/{name} endpoints can dispatch inbound calls."""
        self._webhook_manager = webhooks

    def _wake_room_state(self, room: str) -> dict:
        item = self._wake_calibration.setdefault(room, {
            "room": room,
            "rms_db": None,
            "peak_db": None,
            "wake_score": 0.0,
            "wake_model": "",
            "sensitivity": 0.5,
            "false_positive_count": 0,
            "suggested_sensitivity": 0.5,
            "updated_at": None,
        })
        return item

    def _suggest_wake_sensitivity(self, item: dict) -> float:
        current = float(item.get("sensitivity") or 0.5)
        false_positives = int(item.get("false_positive_count") or 0)
        score = float(item.get("wake_score") or 0.0)
        if false_positives >= 3:
            suggested = current + 0.10
        elif false_positives >= 1:
            suggested = current + 0.05
        elif score > 0 and score < max(0.05, current * 0.45):
            suggested = current - 0.05
        else:
            suggested = current
        return round(min(0.95, max(0.05, suggested)), 2)

    def _capability_status(self) -> dict:
        """Return best-effort degraded-mode state for the dashboard."""
        orch = self._orchestrator

        def _item(name: str, status: str, detail: str = "") -> dict:
            return {"name": name, "status": status, "detail": detail}

        if orch is None:
            return {
                "overall": "degraded",
                "items": [_item("orchestrator", "error", "not registered")],
            }

        items: list[dict] = []
        wake = getattr(orch, "wake", None)
        items.append(_item(
            "wake word",
            "loaded" if getattr(wake, "loaded", False) else "error",
            str(getattr(wake, "device", "")),
        ))
        stt = getattr(orch, "stt", None)
        items.append(_item(
            "stt",
            "loaded" if getattr(stt, "loaded", False) else "error",
            str(getattr(stt, "_model_size", "")),
        ))
        tts = getattr(orch, "tts", None)
        items.append(_item(
            "tts",
            "loaded" if getattr(tts, "loaded", False) else "error",
            str(getattr(tts, "_active_voice", "")),
        ))
        llm = getattr(orch, "llm", None)
        items.append(_item(
            "llm",
            "loaded" if llm is not None else "disabled",
            str(getattr(llm, "model", "")),
        ))
        mqtt = getattr(orch, "mqtt", None)
        mqtt_online = bool(getattr(mqtt, "_connected", False))
        items.append(_item("mqtt", "loaded" if mqtt_online else "degraded"))
        cal = getattr(orch, "calendar", None)
        items.append(_item(
            "calendar",
            "loaded" if getattr(cal, "is_authenticated", False) else "disabled",
        ))
        cameras = getattr(orch, "cameras", None)
        cam_rooms = cameras.get_available_rooms() if cameras is not None else []
        items.append(_item(
            "cameras",
            "loaded" if cam_rooms else "degraded",
            f"{len(cam_rooms)} online",
        ))
        items.append(_item(
            "world model",
            "loaded" if getattr(orch, "world_model", None) is not None else "disabled",
        ))
        items.append(_item(
            "face id",
            "loaded" if getattr(orch, "face_recognizer", None) is not None else "disabled",
        ))
        items.append(_item(
            "speaker id",
            "loaded" if getattr(orch, "speaker_id", None) is not None else "disabled",
        ))
        ob = getattr(orch, "observation_builder", None)
        openvocab = getattr(ob, "openvocab_detector", None)
        items.append(_item(
            "open-vocab objects",
            "loaded" if openvocab is not None else "disabled",
        ))
        integrations = getattr(orch, "integrations", None)
        for plugin in integrations.status() if integrations is not None else []:
            items.append(_item(
                f"integration:{plugin.get('name', 'unknown')}",
                str(plugin.get("status", "unknown")),
                str(plugin.get("detail", plugin.get("error", ""))),
            ))

        bad = [i for i in items if i["status"] in {"error", "degraded"}]
        overall = "ok" if not bad else "degraded"
        return {"overall": overall, "items": items}

    def _setup_routes(self):
        app = self.app

        # Serve static files. Subclass StaticFiles so every response gets
        # `Cache-Control: no-store` — without this, Chrome aggressively
        # caches app.js / style.css and a hot-reloaded dashboard shows
        # stale UI until the user manually hard-refreshes (Ctrl+F5).
        # We're a single-user dev tool, not a CDN; bandwidth from
        # re-fetching ~50KB of static is irrelevant compared to "the
        # new feature I just shipped doesn't appear" frustration.
        class NoCacheStatic(StaticFiles):
            async def get_response(self, path, scope):
                resp = await super().get_response(path, scope)
                resp.headers["Cache-Control"] = "no-store, max-age=0"
                return resp

        if STATIC_DIR.exists():
            app.mount("/static", NoCacheStatic(directory=str(STATIC_DIR)), name="static")

        @app.get("/", response_class=HTMLResponse)
        async def index():
            html_path = STATIC_DIR / "index.html"
            if html_path.exists():
                # Same no-cache treatment for the HTML shell — otherwise
                # an updated index.html (new <script> tags, new section
                # markup) won't show up either.
                return HTMLResponse(
                    content=html_path.read_text(encoding="utf-8"),
                    headers={"Cache-Control": "no-store, max-age=0"},
                )
            return HTMLResponse(content="<h1>Dashboard loading...</h1>")

        @app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket):
            await ws.accept()
            self._clients.append(ws)
            logger.debug(f"[Dashboard] Client connected ({len(self._clients)} total)")

            try:
                # Send current full state immediately on connect.
                await ws.send_json({
                    "type": "full_state",
                    "state": self._state,
                    "conversation": self._conversation,
                })
                while True:
                    # Keep connection alive, receive pings from client
                    await ws.receive_text()
            except WebSocketDisconnect:
                pass
            except Exception as e:
                logger.debug(f"[Dashboard] WebSocket client error: {e}")
            finally:
                if ws in self._clients:
                    self._clients.remove(ws)
                logger.debug(
                    f"[Dashboard] Client disconnected ({len(self._clients)} remaining)"
                )

        @app.websocket("/ws/logs")
        async def logs_endpoint(ws: WebSocket):
            """Stream loguru records to the dashboard Logs tab. The
            client may send a JSON filter once on connect:
              {"min_level": "INFO", "include": ["modules.world_model"]}
            (include is a prefix list; empty = all). The server installs
            a per-client loguru sink so filters apply at source, not on
            the wire."""
            await ws.accept()
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue(maxsize=2000)
            filt = {"min_level_no": 0, "include": []}

            def _enqueue_log(payload: dict) -> None:
                try:
                    queue.put_nowait(payload)
                except asyncio.QueueFull:
                    # Drop the oldest record so a paused Logs tab cannot keep
                    # stale log entries alive indefinitely.
                    with suppress(Exception):
                        queue.get_nowait()
                    with suppress(Exception):
                        queue.put_nowait(payload)

            def _sink(message):
                try:
                    r = message.record
                    if r["level"].no < filt["min_level_no"]:
                        return
                    name = r["name"] or ""
                    inc = filt["include"]
                    if inc and not any(name.startswith(p) for p in inc):
                        return
                    payload = {
                        "ts": r["time"].isoformat(),
                        "level": r["level"].name,
                        "name": name,
                        "function": r["function"],
                        "line": r["line"],
                        "message": r["message"],
                    }
                    if not loop.is_closed():
                        loop.call_soon_threadsafe(_enqueue_log, payload)
                except Exception:
                    pass

            sink_id = logger.add(_sink, level="DEBUG", enqueue=False)

            async def _drain():
                while True:
                    item = await queue.get()
                    await ws.send_json({"type": "log", "record": item})

            sender = asyncio.create_task(_drain())
            try:
                while True:
                    raw = await ws.receive_text()
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue
                    lv = str(msg.get("min_level", "DEBUG")).upper()
                    name_to_no = {
                        "TRACE": 5, "DEBUG": 10, "INFO": 20,
                        "SUCCESS": 25, "WARNING": 30, "ERROR": 40,
                        "CRITICAL": 50,
                    }
                    filt["min_level_no"] = name_to_no.get(lv, 0)
                    inc = msg.get("include") or []
                    filt["include"] = [str(p) for p in inc if isinstance(p, str)]
            except WebSocketDisconnect:
                pass
            finally:
                sender.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await sender
                try:
                    logger.remove(sink_id)
                except Exception:
                    pass

        @app.get("/api/state")
        async def get_state():
            return JSONResponse({
                "state": self._state,
                "conversation": self._conversation,
            })

        @app.get("/api/health")
        async def health():
            return JSONResponse({
                "status": "ok",
                "clients": len(self._clients),
                "updated_at": self._state.get("updated_at"),
            })

        @app.get("/api/degraded")
        async def degraded_status():
            return JSONResponse(self._capability_status())

        @app.get("/api/wake_calibration")
        async def wake_calibration_get():
            return JSONResponse({
                "rooms": sorted(
                    self._wake_calibration.values(),
                    key=lambda r: str(r.get("room", "")),
                )
            })

        @app.post("/api/wake_calibration/{room}/false_positive")
        async def wake_calibration_false_positive(room: str):
            item = self._wake_room_state(room)
            item["false_positive_count"] = int(item.get("false_positive_count") or 0) + 1
            item["suggested_sensitivity"] = self._suggest_wake_sensitivity(item)
            item["updated_at"] = datetime.now().isoformat()
            await self.broadcast({
                "type": "wake_calibration",
                "room": room,
                **item,
            })
            return JSONResponse(item)

        @app.post("/api/chat")
        async def chat_endpoint(request: Request):
            body = await request.json()
            text = str(body.get("text", "")).strip()
            room = str(body.get("room", "office"))
            if text and self._chat_handler:
                asyncio.create_task(self._chat_handler(text, room))
            return JSONResponse({"ok": True})

        @app.get("/api/voices")
        async def get_voices():
            return JSONResponse({
                "voices": self._available_voices,
                "active": self._active_voice,
            })

        @app.post("/api/voice")
        async def set_voice(request: Request):
            body = await request.json()
            voice = str(body.get("voice", "")).strip()
            if voice and self._voice_handler:
                asyncio.create_task(self._voice_handler(voice))
                self._active_voice = voice
            return JSONResponse({"ok": True, "voice": voice})

        @app.get("/api/reminders")
        async def list_reminders():
            store = self._reminders_store
            if store is None:
                return JSONResponse({"reminders": []})
            try:
                items = await store.list_pending()
            except Exception as e:
                logger.warning(f"[Dashboard] list_reminders failed: {e}")
                return JSONResponse({"reminders": []})
            return JSONResponse({"reminders": items})

        @app.post("/api/reminders")
        async def create_reminder(request: Request):
            store = self._reminders_store
            if store is None:
                raise HTTPException(status_code=503, detail="Reminders not available")
            body = await request.json()
            message = str(body.get("message", "")).strip()
            trigger_iso = str(body.get("trigger_time", "")).strip()
            if not message or not trigger_iso:
                raise HTTPException(status_code=400, detail="message and trigger_time required")
            try:
                trigger_time = datetime.fromisoformat(trigger_iso)
            except ValueError:
                raise HTTPException(status_code=400, detail="trigger_time must be ISO 8601")
            recurrence_seconds = body.get("recurrence_seconds")
            if recurrence_seconds is not None:
                try:
                    recurrence_seconds = int(recurrence_seconds)
                    if recurrence_seconds <= 0:
                        recurrence_seconds = None
                except (TypeError, ValueError):
                    recurrence_seconds = None
            rid = await store.add(message, trigger_time, recurrence_seconds=recurrence_seconds)
            await self.broadcast({
                "type":               "reminder_added",
                "id":                 rid,
                "message":            message,
                "trigger_time":       trigger_time.isoformat(),
                "recurrence_seconds": recurrence_seconds,
            })
            return JSONResponse({"id": rid, "ok": True})

        @app.delete("/api/reminders/{reminder_id}")
        async def delete_reminder(reminder_id: int):
            store = self._reminders_store
            if store is None:
                raise HTTPException(status_code=503, detail="Reminders not available")
            await store.delete(reminder_id)
            await self.broadcast({"type": "reminder_deleted", "id": reminder_id})
            return JSONResponse({"ok": True})

        @app.get("/api/calendar/upcoming")
        async def calendar_upcoming(hours: float = 24.0):
            cal = self._calendar
            if cal is None or not getattr(cal, "is_authenticated", False):
                return JSONResponse({"events": [], "authenticated": False})
            events = await cal.upcoming_events(hours=hours)
            return JSONResponse({"events": events, "authenticated": True})

        @app.post("/api/calendar")
        async def calendar_create(request: Request):
            cal = self._calendar
            if cal is None or not getattr(cal, "is_authenticated", False):
                raise HTTPException(status_code=503, detail="Calendar not available")
            body = await request.json()
            title = str(body.get("title", "")).strip()
            start_iso = str(body.get("start", "")).strip()
            end_iso = str(body.get("end", "")).strip()
            if not title or not start_iso:
                raise HTTPException(status_code=400, detail="title and start required")
            try:
                start = datetime.fromisoformat(start_iso)
                end = datetime.fromisoformat(end_iso) if end_iso else None
            except ValueError:
                raise HTTPException(status_code=400, detail="start/end must be ISO 8601")
            event = await cal.add_event(title=title, start=start, end=end)
            if event is None:
                raise HTTPException(status_code=502, detail="Calendar API rejected event")
            await self.broadcast({"type": "calendar_added", "event": event})
            return JSONResponse(event)

        @app.delete("/api/calendar/{event_id}")
        async def calendar_delete(event_id: str):
            cal = self._calendar
            if cal is None or not getattr(cal, "is_authenticated", False):
                raise HTTPException(status_code=503, detail="Calendar not available")
            ok = await cal.delete_event(event_id)
            if ok:
                await self.broadcast({"type": "calendar_deleted", "id": event_id})
            return JSONResponse({"ok": ok})

        @app.get("/api/dnd")
        async def dnd_status():
            mgr = self._interruptibility
            if mgr is None:
                return JSONResponse({"active": False, "until": None})
            until = mgr.dnd_until() if mgr.is_dnd() else None
            return JSONResponse({
                "active": mgr.is_dnd(),
                "until":  until.isoformat() if until else None,
            })

        @app.get("/api/speakers")
        async def list_speakers():
            sid = self._speaker_id
            if sid is None:
                return JSONResponse({"speakers": []})
            return JSONResponse({"speakers": await sid.list_enrolled()})

        @app.post("/api/speakers/enroll")
        async def enroll_speaker(request: Request):
            """
            Arms the next wake-word capture to be enrolled as the given name.
            User then says "Hey Jarvis" and a normal greeting; the audio gets
            routed to enrollment instead of LLM processing.
            """
            orch = self._orchestrator
            if orch is None:
                raise HTTPException(status_code=503, detail="Orchestrator not registered")
            body = await request.json()
            name = str(body.get("name", "")).strip()
            if not name:
                raise HTTPException(status_code=400, detail="name required")
            # Default to a single 'wake' prompt for the legacy single-button
            # ARM. Identity v2's 3-sentence enrollment uses /api/identity/voice/arm
            # which lets the dashboard pass an explicit prompt_id.
            orch._pending_speaker_enrollment = (name, "wake")
            await self.broadcast({"type": "speaker_enrollment_armed", "name": name})
            return JSONResponse({"armed": True, "name": name})

        @app.delete("/api/speakers/{name}")
        async def delete_speaker(name: str):
            sid = self._speaker_id
            if sid is None:
                raise HTTPException(status_code=503, detail="Speaker ID not available")
            ok = await sid.delete(name)
            await self.broadcast({"type": "speaker_deleted", "name": name})
            return JSONResponse({"ok": ok})

        @app.get("/api/faces")
        async def list_faces():
            fr = self._face_recognizer
            if fr is None:
                return JSONResponse({"faces": []})
            return JSONResponse({"faces": await fr.list_enrolled()})

        @app.post("/api/faces/enroll")
        async def enroll_face(request: Request):
            """
            Capture the current frame from the office_cam and enroll the
            largest detected face as the given name.
            """
            fr = self._face_recognizer
            cm = self._camera_manager
            if fr is None or cm is None:
                raise HTTPException(status_code=503, detail="Face recognition not available")
            body = await request.json()
            name = str(body.get("name", "")).strip()
            room = str(body.get("room", "office_cam"))
            if not name:
                raise HTTPException(status_code=400, detail="name required")
            frame = await cm.capture_frame_async(room)
            if frame is None:
                raise HTTPException(status_code=502, detail=f"Could not capture frame from '{room}'")
            ok = await fr.enroll(name, frame)
            if not ok:
                raise HTTPException(status_code=422, detail="No face detected in frame or enrollment failed")
            await self.broadcast({"type": "face_enrolled", "name": name})
            return JSONResponse({"ok": True, "name": name})

        @app.post("/api/webhook/{name}")
        async def webhook_inbound(name: str, request: Request):
            """
            External services (IFTTT, Home Assistant, Zapier, etc.) POST here
            with a JSON payload + X-Webhook-Token header. We dispatch a
            'webhook.{name}' event onto the internal bus. Token must match
            the value configured under webhooks.inbound_tokens.{name}.
            """
            wm = self._webhook_manager
            if wm is None:
                raise HTTPException(status_code=503, detail="Webhooks not registered")
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            token = request.headers.get("X-Webhook-Token", "")
            try:
                await wm.trigger_inbound(name, token, payload)
            except KeyError:
                raise HTTPException(status_code=404, detail=f"Unknown webhook '{name}'")
            except PermissionError:
                raise HTTPException(status_code=401, detail="Invalid webhook token")
            return JSONResponse({"ok": True})

        @app.get("/api/webhooks")
        async def webhooks_status():
            """Return registered webhook names + outbound subscription map (no tokens)."""
            wm = self._webhook_manager
            if wm is None:
                return JSONResponse({"inbound": [], "outbound": {}})
            return JSONResponse({
                "inbound":  wm.list_inbound_names(),
                "outbound": wm.list_outbound_routes(),
            })

        @app.get("/api/config")
        async def get_config_yaml():
            """Return the raw config.yaml text so the dashboard can edit it."""
            from pathlib import Path as _Path
            cfg_path = _Path(__file__).parent.parent / "config.yaml"
            if not cfg_path.exists():
                raise HTTPException(status_code=404, detail="config.yaml not found")
            try:
                return JSONResponse({"path": str(cfg_path), "yaml": cfg_path.read_text(encoding="utf-8")})
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"read failed: {e}")

        @app.post("/api/config")
        async def save_config_yaml(request: Request):
            """
            Persist the dashboard-edited config.yaml back to disk. Most config
            changes only take effect on next restart — we don't try to hot-reload
            because that would require careful re-init of every module.
            Validates that the YAML parses before writing; on parse error we
            return 400 and don't touch the file.
            """
            from pathlib import Path as _Path
            import yaml as _yaml
            body = await request.json()
            new_yaml = str(body.get("yaml", ""))
            if not new_yaml.strip():
                raise HTTPException(status_code=400, detail="empty yaml")
            try:
                parsed = _yaml.safe_load(new_yaml)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"yaml parse error: {e}")
            if not isinstance(parsed, dict):
                raise HTTPException(status_code=400, detail="root must be a mapping")
            cfg_path = _Path(__file__).parent.parent / "config.yaml"
            # Backup the current file before overwrite — easy rollback if
            # the new YAML breaks startup.
            try:
                if cfg_path.exists():
                    backup = cfg_path.with_suffix(".yaml.bak")
                    backup.write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception as e:
                logger.warning(f"[Dashboard] config backup failed: {e}")
            try:
                cfg_path.write_text(new_yaml, encoding="utf-8")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"write failed: {e}")
            await self.broadcast({"type": "config_saved"})
            return JSONResponse({"ok": True, "restart_required": True})

        @app.delete("/api/faces/{name}")
        async def delete_face(name: str):
            fr = self._face_recognizer
            if fr is None:
                raise HTTPException(status_code=503, detail="Face recognition not available")
            ok = await fr.delete(name)
            await self.broadcast({"type": "face_deleted", "name": name})
            return JSONResponse({"ok": ok})

        # ── Identity v2 (cross-modal persons) ────────────────────────────────

        @app.get("/api/identity/persons")
        async def identity_list_persons():
            ident = self._identity
            if ident is None:
                return JSONResponse({"persons": []})
            return JSONResponse({"persons": await ident.list_persons()})

        @app.post("/api/identity/face/enroll")
        async def identity_enroll_face(request: Request):
            """Capture a frame from the given room and enroll it as a face
            sample for `name` with the given pose label (one of:
            center, left, right, up, down)."""
            ident = self._identity
            cm = self._camera_manager
            if ident is None or cm is None:
                raise HTTPException(status_code=503, detail="Identity not available")
            body = await request.json()
            name = str(body.get("name", "")).strip()
            pose = str(body.get("pose", "center")).strip().lower()
            room = str(body.get("room", "office"))
            if not name:
                raise HTTPException(status_code=400, detail="name required")
            if pose not in ("center", "left", "right", "up", "down", "candid"):
                raise HTTPException(status_code=400, detail="invalid pose")
            frame = await cm.capture_frame_async(room)
            if frame is None:
                raise HTTPException(
                    status_code=502, detail=f"Could not capture frame from '{room}'"
                )
            sample_id = await ident.enroll_face(name, frame, pose=pose)
            if sample_id is None:
                raise HTTPException(status_code=422, detail="No face detected in frame")
            await self.broadcast(
                {"type": "identity_face_enrolled", "name": name, "pose": pose}
            )
            return JSONResponse({"ok": True, "name": name, "pose": pose, "sample_id": sample_id})

        @app.post("/api/identity/voice/arm")
        async def identity_arm_voice(request: Request):
            """Arm the next wake-word capture to be saved as a voice sample
            for `name` with the given prompt_id (e.g. 'sentence_1')."""
            orch = self._orchestrator
            if orch is None:
                raise HTTPException(status_code=503, detail="Orchestrator not registered")
            body = await request.json()
            name = str(body.get("name", "")).strip()
            prompt_id = str(body.get("prompt_id", "wake")).strip() or "wake"
            if not name:
                raise HTTPException(status_code=400, detail="name required")
            orch._pending_speaker_enrollment = (name, prompt_id)
            await self.broadcast(
                {
                    "type": "identity_voice_armed",
                    "name": name,
                    "prompt_id": prompt_id,
                }
            )
            return JSONResponse({"armed": True, "name": name, "prompt_id": prompt_id})

        @app.delete("/api/identity/persons/{person_id}")
        async def identity_delete_person(person_id: int):
            ident = self._identity
            if ident is None:
                raise HTTPException(status_code=503, detail="Identity not available")
            ok = await ident.delete_person(person_id)
            await self.broadcast({"type": "identity_person_deleted", "person_id": person_id})
            return JSONResponse({"ok": ok})

        @app.get("/api/identity/persons/{person_id}/thumbnail.jpg")
        async def identity_person_thumbnail(person_id: int):
            from fastapi.responses import Response
            ident = self._identity
            if ident is None:
                raise HTTPException(status_code=503, detail="Identity not available")
            data = await ident.get_person_thumbnail(person_id)
            if data is None:
                raise HTTPException(status_code=404, detail="No thumbnail")
            return Response(content=data, media_type="image/jpeg")

        @app.get("/api/identity/persons/{person_id}/samples")
        async def identity_person_samples(person_id: int):
            ident = self._identity
            if ident is None:
                return JSONResponse({"face": [], "voice": []})
            return JSONResponse({
                "face":  await ident.list_face_samples(person_id),
                "voice": await ident.list_voice_samples(person_id),
            })

        @app.get("/api/identity/face_samples/{sample_id}/image.jpg")
        async def identity_face_sample_image(sample_id: int):
            from fastapi.responses import Response
            ident = self._identity
            if ident is None:
                raise HTTPException(status_code=503, detail="Identity not available")
            data = await ident.get_face_sample_image(sample_id)
            if data is None:
                raise HTTPException(status_code=404, detail="No image")
            return Response(content=data, media_type="image/jpeg")

        @app.delete("/api/identity/face_samples/{sample_id}")
        async def identity_delete_face_sample(sample_id: int):
            ident = self._identity
            if ident is None:
                raise HTTPException(status_code=503, detail="Identity not available")
            ok = await ident.delete_face_sample(sample_id)
            await self.broadcast({"type": "identity_sample_deleted", "modality": "face", "id": sample_id})
            return JSONResponse({"ok": ok})

        @app.delete("/api/identity/voice_samples/{sample_id}")
        async def identity_delete_voice_sample(sample_id: int):
            ident = self._identity
            if ident is None:
                raise HTTPException(status_code=503, detail="Identity not available")
            ok = await ident.delete_voice_sample(sample_id)
            await self.broadcast({"type": "identity_sample_deleted", "modality": "voice", "id": sample_id})
            return JSONResponse({"ok": ok})

        @app.post("/api/identity/persons/{person_id}/rename")
        async def identity_rename_person(person_id: int, request: Request):
            ident = self._identity
            if ident is None:
                raise HTTPException(status_code=503, detail="Identity not available")
            body = await request.json()
            new_name = str(body.get("name", "")).strip()
            if not new_name:
                raise HTTPException(status_code=400, detail="name required")
            ok = await ident.rename_person(person_id, new_name)
            await self.broadcast(
                {"type": "identity_person_renamed", "person_id": person_id, "name": new_name}
            )
            return JSONResponse({"ok": ok, "name": new_name})

        @app.get("/api/identity/pending")
        async def identity_list_pending():
            ident = self._identity
            if ident is None:
                return JSONResponse({"pending": []})
            return JSONResponse({"pending": await ident.list_pending()})

        @app.get("/api/identity/pending/{pending_id}/image.jpg")
        async def identity_pending_image(pending_id: int):
            from fastapi.responses import Response
            ident = self._identity
            if ident is None:
                raise HTTPException(status_code=503, detail="Identity not available")
            data = await ident.get_pending_image(pending_id)
            if data is None:
                raise HTTPException(status_code=404, detail="No image for this pending row")
            return Response(content=data, media_type="image/jpeg")

        @app.get("/api/identity/pending/{pending_id}/audio.wav")
        async def identity_pending_audio(pending_id: int):
            from fastapi.responses import Response
            import struct
            ident = self._identity
            if ident is None:
                raise HTTPException(status_code=503, detail="Identity not available")
            pcm = await ident.get_pending_audio(pending_id)
            if pcm is None:
                raise HTTPException(status_code=404, detail="No audio for this pending row")
            # Wrap raw PCM16 16kHz mono in a WAV header so <audio> can play it.
            sample_rate = 16000
            byte_rate = sample_rate * 2
            data_size = len(pcm)
            header = b"RIFF" + struct.pack("<I", 36 + data_size) + b"WAVE"
            header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, byte_rate, 2, 16)
            header += b"data" + struct.pack("<I", data_size)
            return Response(content=header + pcm, media_type="audio/wav")

        @app.post("/api/identity/pending/{pending_id}/resolve")
        async def identity_resolve_pending(pending_id: int, request: Request):
            ident = self._identity
            if ident is None:
                raise HTTPException(status_code=503, detail="Identity not available")
            body = await request.json()
            action = str(body.get("action", "")).strip()
            target_name = body.get("target_name")
            if action not in ("confirm", "assign", "reject"):
                raise HTTPException(status_code=400, detail="action must be confirm|assign|reject")
            ok = await ident.resolve_pending(
                pending_id, action, target_name=target_name
            )
            await self.broadcast(
                {"type": "identity_pending_resolved", "pending_id": pending_id, "action": action}
            )
            return JSONResponse({"ok": ok})

        @app.post("/api/identity/pending/collapse")
        async def identity_pending_collapse(request: Request):
            """Maintenance: collapse near-duplicate unresolved pending
            rows into single cluster representatives. Body:
              {modality: "face"|"voice", min_sim: float=0.35}
            Returns {kept, rejected, scanned}."""
            ident = self._identity
            if ident is None:
                raise HTTPException(status_code=503, detail="Identity not available")
            try:
                body = await request.json()
            except Exception:
                body = {}
            modality = str(body.get("modality", "face")).strip() or "face"
            try:
                min_sim = float(body.get("min_sim", 0.35))
            except (TypeError, ValueError):
                min_sim = 0.35
            result = await ident.collapse_pending_duplicates(
                modality=modality, min_sim=min_sim,
            )
            await self.broadcast({
                "type": "identity_pending_collapsed",
                **result,
            })
            return JSONResponse(result)

        @app.post("/api/identity/pending/reject_all")
        async def identity_pending_reject_all():
            """Maintenance: nuke every unresolved pending row. Returns
            {rejected: int}. Use when the queue has gotten out of hand
            and you'd rather start clean than triage."""
            ident = self._identity
            if ident is None:
                raise HTTPException(status_code=503, detail="Identity not available")
            n = await ident.reject_all_unresolved_pending()
            await self.broadcast({
                "type": "identity_pending_bulk_resolved",
                "count": 0, "action": "reject_all",
                "rejected": n,
            })
            return JSONResponse({"rejected": n})

        @app.post("/api/identity/pending/bulk")
        async def identity_resolve_pending_bulk(request: Request):
            """Bulk resolve. Body:
              {ids: [1,2,3], action: "assign|reject", target_name: "Cole"}
            Returns {ok, skipped_quality, failed, ids}. Used by the
            Pending Reviews tab to drain a backlog in one click."""
            ident = self._identity
            if ident is None:
                raise HTTPException(status_code=503, detail="Identity not available")
            body = await request.json()
            ids_raw = body.get("ids") or []
            action = str(body.get("action", "")).strip()
            target_name = body.get("target_name")
            if action not in ("assign", "reject"):
                raise HTTPException(
                    status_code=400, detail="action must be assign|reject",
                )
            if action == "assign" and not target_name:
                raise HTTPException(
                    status_code=400, detail="target_name required for assign",
                )
            try:
                ids = [int(x) for x in ids_raw]
            except (ValueError, TypeError) as e:
                raise HTTPException(status_code=400, detail=f"bad ids: {e}") from e
            if not ids:
                return JSONResponse({"ok": 0, "skipped_quality": 0, "failed": 0, "ids": []})
            result = await ident.resolve_pending_bulk(ids, action, target_name)
            await self.broadcast({
                "type": "identity_pending_bulk_resolved",
                "count": result["ok"], "action": action,
            })
            return JSONResponse(result)

        @app.post("/api/identity/bank_prune")
        async def identity_bank_prune(request: Request):
            """Harm-based eviction pass: drop near-duplicate samples
            from every person's face (and optionally voice) bank. The
            standard add-path keeps the cap at 60 face / 40 voice;
            this endpoint sweeps existing rows that snuck in over the
            cap (resolved before the cap-evict path landed) or that
            converged on each other after enrollment.

            Body: {modality?: "face"|"voice"|"both" = "face",
                   person_id?: int, threshold?: float}
            Returns a per-person count of dropped rows.
            """
            ident = self._identity
            if ident is None:
                raise HTTPException(status_code=503, detail="Identity not available")
            try:
                body = await request.json()
            except Exception:
                body = {}
            modality = str(body.get("modality") or "face")
            pid = body.get("person_id")
            try:
                pid = int(pid) if pid is not None else None
            except (TypeError, ValueError):
                pid = None
            threshold = body.get("threshold")
            try:
                threshold = float(threshold) if threshold is not None else None
            except (TypeError, ValueError):
                threshold = None
            modalities = (
                ["face", "voice"] if modality == "both" else [modality]
            )
            results: list[dict] = []
            for m in modalities:
                if m not in ("face", "voice"):
                    continue
                results.append(await ident.prune_bank_redundancy(
                    person_id=pid, modality=m, threshold=threshold,
                ))
            await self.broadcast({"type": "identity_bank_pruned"})
            return JSONResponse({"results": results})

        @app.get("/api/identity/bank_stats")
        async def identity_bank_stats():
            """Per-person sample count + dimensional health so the
            Pending Reviews tab can show 'Cole: 235 ArcFace samples'
            and the user knows how big the centroid bank is for each
            resident."""
            ident = self._identity
            if ident is None:
                return JSONResponse({"persons": [], "available": False})
            persons = await ident.list_persons()
            face_counts = {
                pid: len(samples) for pid, samples in
                ident._face_samples.items()
            }
            voice_counts = {
                pid: len(samples) for pid, samples in
                ident._voice_samples.items()
            }
            for p in persons:
                p["face_samples"] = face_counts.get(int(p["id"]), 0)
                p["voice_samples"] = voice_counts.get(int(p["id"]), 0)
            return JSONResponse({"persons": persons, "available": True})

        # ── Notifications (bell + dropdown) ──────────────────────────────────

        @app.get("/api/notifications")
        async def notifications_list(only_unread: bool = False, limit: int = 50):
            mgr = self._notifications
            if mgr is None:
                return JSONResponse({"items": [], "unread": 0})
            items = await mgr.list_recent(limit=limit, only_unread=only_unread)
            unread = await mgr.unread_count()
            return JSONResponse({"items": items, "unread": unread})

        @app.post("/api/notifications/{notification_id}/read")
        async def notifications_mark_read(notification_id: int):
            mgr = self._notifications
            if mgr is None:
                raise HTTPException(status_code=503, detail="Notifications not available")
            ok = await mgr.mark_read(notification_id)
            return JSONResponse({"ok": ok})

        @app.post("/api/notifications/read_all")
        async def notifications_mark_all_read():
            mgr = self._notifications
            if mgr is None:
                raise HTTPException(status_code=503, detail="Notifications not available")
            ok = await mgr.mark_all_read()
            return JSONResponse({"ok": ok})

        @app.delete("/api/notifications/{notification_id}")
        async def notifications_delete(notification_id: int):
            mgr = self._notifications
            if mgr is None:
                raise HTTPException(status_code=503, detail="Notifications not available")
            ok = await mgr.delete(notification_id)
            return JSONResponse({"ok": ok})

        # ── Computer control (kill-switch + pending-action review) ──────────

        @app.get("/api/computer/status")
        async def computer_status():
            c = self._computer
            if c is None:
                return JSONResponse({"available": False, "enabled": False, "pending": []})
            return JSONResponse({
                "available": True,
                **c.status(),
                "pending":  c.list_pending(),
            })

        @app.post("/api/computer/enable")
        async def computer_enable(request: Request):
            c = self._computer
            if c is None:
                raise HTTPException(status_code=503, detail="Computer control not available")
            body = await request.json()
            value = bool(body.get("enabled", False))
            c.set_enabled(value)
            await self.broadcast({"type": "computer.toggled", "enabled": value})
            return JSONResponse({"ok": True, "enabled": value})

        @app.post("/api/computer/pending/{action_id}/approve")
        async def computer_approve(action_id: int):
            c = self._computer
            if c is None:
                raise HTTPException(status_code=503, detail="Computer control not available")
            result = await c.approve(action_id)
            return JSONResponse(result)

        @app.post("/api/computer/pending/{action_id}/reject")
        async def computer_reject(action_id: int):
            c = self._computer
            if c is None:
                raise HTTPException(status_code=503, detail="Computer control not available")
            ok = await c.reject(action_id)
            return JSONResponse({"ok": ok})

        # ── System (restart / shutdown) ─────────────────────────────────────

        @app.post("/api/system/restart")
        async def system_restart():
            """Exit with code 43 — supervisor (if running) will relaunch
            with NO heartbeat enforcement and NO git-revert path. This is
            distinct from self-edit's restart_self (exit 42), which DOES
            enable the heartbeat-or-revert dance. Dashboard restart is
            unconditionally safe for any commit Cole authored."""
            import asyncio as _asyncio, os as _os
            await self.broadcast({"type": "system.restarting"})
            # call_later schedules a callback on the loop's timer; unlike
            # create_task() the result isn't a Task object that uvicorn's
            # graceful-shutdown can cancel. So even if uvicorn starts
            # tearing down right after the response goes out, this still
            # fires and os._exit takes the whole process with it.
            loop = _asyncio.get_event_loop()
            loop.call_later(0.5, lambda: (logger.warning("[System] Restart — exit 43"), _os._exit(43)))
            return JSONResponse({"ok": True, "action": "restart"})

        @app.post("/api/system/shutdown")
        async def system_shutdown():
            """Clean exit — supervisor will stop the loop on a non-42 exit.
            Uses loop.call_later for the same reason restart does — the
            create_task pattern was racy with uvicorn shutdown and would
            sometimes only kill the dashboard.
            """
            import asyncio as _asyncio, os as _os
            await self.broadcast({"type": "system.shutting_down"})
            loop = _asyncio.get_event_loop()
            loop.call_later(0.5, lambda: (logger.warning("[System] Shutdown — exit 0"), _os._exit(0)))
            return JSONResponse({"ok": True, "action": "shutdown"})

        # ── Persona system ──────────────────────────────────────────────────

        @app.get("/api/personas")
        async def personas_list():
            """Return only personas with visible_in_ui=true. Hidden ones
            (uwu) MUST NOT appear here — that's the user-facing safety
            property of 'hidden': the persona doesn't exist as far as
            the dropdown knows. Activating it requires typing the name
            into the command box, i.e., knowing it exists."""
            p = self._persona
            if p is None:
                return JSONResponse({"personas": [], "active": None, "locked": False})
            return JSONResponse({
                "personas": p.list_visible(),
                "active": p.current_name(),
                "locked": p.is_locked(),
            })

        @app.get("/api/persona/current")
        async def persona_current():
            """Live status — what's active, lock state, and any pending
            phone-resume offer the dashboard should show as a prompt."""
            p = self._persona
            if p is None:
                return JSONResponse({"active": None, "locked": False, "pending_resume": None})
            st = p.state()
            return JSONResponse({
                "active": st.active,
                "locked": st.locked,
                "pending_resume": st.pending_resume,
                "last_change_ts": st.last_change_ts,
            })

        @app.post("/api/persona/set")
        async def persona_set(request: Request):
            """Switch personas. Body: {name, lock?, force?}.
            - name: persona key from config.yaml (visible OR hidden — the
              listing endpoint hides them but SET accepts any known name).
            - lock: when true, blocks the time/phone auto-revert paths.
              Person-entry revert still fires regardless.
            - force: bypass the privacy gate (use sparingly — intended for
              CLI / debug scenarios, not the normal UI).
            """
            from core.exceptions import PersonaError as _PersonaError
            p = self._persona
            if p is None:
                raise HTTPException(status_code=503, detail="Persona system not configured")
            try:
                body = await request.json()
            except Exception:
                body = {}
            name = str(body.get("name", "")).strip()
            if not name:
                raise HTTPException(status_code=400, detail="name required")
            try:
                await p.set(
                    name,
                    lock=bool(body.get("lock", False)),
                    force=bool(body.get("force", False)),
                )
            except _PersonaError as e:
                # 409 = conflict (privacy gate or unknown persona). Lets
                # the frontend distinguish "wrong name" / "not allowed
                # right now" from generic 500s.
                raise HTTPException(status_code=409, detail=str(e))
            return JSONResponse({"ok": True, "active": p.current_name(), "locked": p.is_locked()})

        @app.post("/api/persona/command")
        async def persona_command(request: Request):
            """Free-text command from the hidden command box. Recognized:
              uwu / focus / default / <any persona name> → set
              revert | normal                            → snap to default
              lock | unlock                              → toggle lock
              resume                                     → accept pending phone-resume
              set <name>                                 → explicit set form
            Returns the new active state plus a `recognized` flag so the
            frontend can tell "did anything happen" vs "typo".
            """
            from core.exceptions import PersonaError as _PersonaError
            p = self._persona
            if p is None:
                raise HTTPException(status_code=503, detail="Persona system not configured")
            try:
                body = await request.json()
            except Exception:
                body = {}
            cmd = str(body.get("text", "")).strip().lower()
            if not cmd:
                raise HTTPException(status_code=400, detail="text required")
            recognized = True
            try:
                if cmd in ("revert", "normal"):
                    await p.revert(reason="dashboard_command")
                elif cmd == "lock":
                    p.set_lock(True)
                elif cmd == "unlock":
                    p.set_lock(False)
                elif cmd == "resume":
                    ok = await p.accept_pending_resume()
                    if not ok:
                        raise HTTPException(
                            status_code=409, detail="No pending resume to accept"
                        )
                elif cmd.startswith("set "):
                    name = cmd.split(maxsplit=1)[1].strip()
                    await p.set(name)
                else:
                    # Bare persona name — try to set it
                    await p.set(cmd)
            except _PersonaError as e:
                raise HTTPException(status_code=409, detail=str(e))
            except HTTPException:
                raise
            except Exception:
                recognized = False
                raise HTTPException(status_code=400, detail=f"Unknown command: {cmd}")
            return JSONResponse({
                "ok": True,
                "active": p.current_name(),
                "locked": p.is_locked(),
                "recognized": recognized,
            })

        # ── Self-edit (kill switch + pending review + revert) ───────────────

        @app.get("/api/selfedit/status")
        async def selfedit_status():
            s = self._selfedit
            if s is None:
                return JSONResponse({"available": False, "enabled": False, "pending": []})
            return JSONResponse({
                "available": True, **s.status(),
                "pending":   s.list_pending(),
            })

        @app.post("/api/selfedit/enable")
        async def selfedit_enable(request: Request):
            s = self._selfedit
            if s is None:
                raise HTTPException(status_code=503, detail="Self-edit not available")
            body = await request.json()
            value = bool(body.get("enabled", False))
            s.set_enabled(value)
            await self.broadcast({"type": "selfedit.toggled", "enabled": value})
            return JSONResponse({"ok": True, "enabled": value})

        @app.post("/api/selfedit/pending/{edit_id}/approve")
        async def selfedit_approve(edit_id: int):
            s = self._selfedit
            if s is None:
                raise HTTPException(status_code=503, detail="Self-edit not available")
            return JSONResponse(await s.approve_pending(edit_id))

        @app.post("/api/selfedit/pending/{edit_id}/reject")
        async def selfedit_reject(edit_id: int):
            s = self._selfedit
            if s is None:
                raise HTTPException(status_code=503, detail="Self-edit not available")
            ok = await s.reject_pending(edit_id)
            return JSONResponse({"ok": ok})

        @app.post("/api/selfedit/revert")
        async def selfedit_revert():
            s = self._selfedit
            if s is None:
                raise HTTPException(status_code=503, detail="Self-edit not available")
            return JSONResponse(await s.revert_last())

        # ── Memory v2 ───────────────────────────────────────────────────────

        @app.get("/api/memory")
        async def memory_list(kind: Optional[str] = None, limit: int = 100):
            mem = self._memory
            if mem is None:
                return JSONResponse({"items": []})
            return JSONResponse({"items": await mem.list_recent(limit=limit, kind=kind)})

        @app.get("/api/memory/search")
        async def memory_search(q: str, k: int = 10):
            mem = self._memory
            if mem is None:
                return JSONResponse({"items": []})
            return JSONResponse({"items": await mem.retrieve(q, k=int(k))})

        @app.post("/api/memory")
        async def memory_add(request: Request):
            mem = self._memory
            if mem is None:
                raise HTTPException(status_code=503, detail="Memory store not available")
            body = await request.json()
            content = str(body.get("content", "")).strip()
            if not content:
                raise HTTPException(status_code=400, detail="content required")
            mid = await mem.add(
                kind=str(body.get("kind", "fact")).lower(),
                content=content,
                subject=body.get("subject"),
                importance=float(body.get("importance", 0.5)),
                source_kind="manual",
            )
            # MemoryStore.add() handles the broadcast internally now —
            # don't double-fire from here.
            return JSONResponse({"ok": mid is not None, "id": mid})

        @app.post("/api/memory/{memory_id}")
        async def memory_update(memory_id: int, request: Request):
            mem = self._memory
            if mem is None:
                raise HTTPException(status_code=503, detail="Memory store not available")
            body = await request.json()
            ok = await mem.update(
                memory_id,
                content=body.get("content"),
                importance=body.get("importance"),
                kind=body.get("kind"),
                subject=body.get("subject"),
            )
            # mem.update() broadcasts memory.updated internally on success.
            return JSONResponse({"ok": ok})

        @app.delete("/api/memory/{memory_id}")
        async def memory_delete(memory_id: int):
            mem = self._memory
            if mem is None:
                raise HTTPException(status_code=503, detail="Memory store not available")
            ok = await mem.delete(memory_id)
            # mem.delete() broadcasts memory.deleted internally on success.
            return JSONResponse({"ok": ok})

        # ── LLM model selector ──────────────────────────────────────────────

        @app.get("/api/models")
        async def models_list():
            reg = self._model_registry
            if reg is None:
                return JSONResponse({"installed": [], "catalog": []})
            return JSONResponse({
                "installed": await reg.list_installed(),
                "catalog":   reg.list_known_recommendations(),
            })

        @app.post("/api/models/active")
        async def models_set_active(request: Request):
            reg = self._model_registry
            if reg is None:
                raise HTTPException(status_code=503, detail="Model registry not available")
            body = await request.json()
            name = str(body.get("name", "")).strip()
            kind = str(body.get("kind", "chat")).strip().lower()
            if not name or kind not in ("chat", "vision", "action"):
                raise HTTPException(status_code=400, detail="name + kind required")
            ok = await reg.set_active(name, kind=kind)
            await self.broadcast({"type": "model.activated", "name": name, "kind": kind})
            return JSONResponse({"ok": ok, "name": name, "kind": kind})

        @app.get("/api/models/presets")
        async def models_presets():
            reg = self._model_registry
            if reg is None:
                return JSONResponse({"presets": []})
            return JSONResponse({"presets": reg.list_presets()})

        @app.get("/api/models/{name:path}/settings")
        async def models_get_settings(name: str):
            reg = self._model_registry
            if reg is None:
                raise HTTPException(status_code=503, detail="Model registry not available")
            return JSONResponse({"settings": await reg.get_settings(name)})

        @app.post("/api/models/{name:path}/settings")
        async def models_set_settings(name: str, request: Request):
            reg = self._model_registry
            if reg is None:
                raise HTTPException(status_code=503, detail="Model registry not available")
            body = await request.json()
            ok = await reg.set_settings(name, body or {})
            await self.broadcast({"type": "model.settings_updated", "name": name})
            return JSONResponse({"ok": ok})

        @app.delete("/api/models/{name:path}/settings")
        async def models_clear_settings(name: str):
            reg = self._model_registry
            if reg is None:
                raise HTTPException(status_code=503, detail="Model registry not available")
            ok = await reg.clear_settings(name)
            await self.broadcast({"type": "model.settings_updated", "name": name})
            return JSONResponse({"ok": ok})

        @app.post("/api/models/notes")
        async def models_set_notes(request: Request):
            reg = self._model_registry
            if reg is None:
                raise HTTPException(status_code=503, detail="Model registry not available")
            body = await request.json()
            name = str(body.get("name", "")).strip()
            notes = str(body.get("notes", ""))
            if not name:
                raise HTTPException(status_code=400, detail="name required")
            ok = await reg.set_user_notes(name, notes)
            return JSONResponse({"ok": ok})

        @app.delete("/api/models/{name:path}")
        async def models_delete(name: str):
            reg = self._model_registry
            if reg is None:
                raise HTTPException(status_code=503, detail="Model registry not available")
            ok = await reg.delete(name)
            await self.broadcast({"type": "model.deleted", "name": name})
            return JSONResponse({"ok": ok})

        @app.post("/api/models/pull")
        async def models_pull(request: Request):
            """Streams Ollama pull progress as Server-Sent Events. The
            dashboard subscribes via fetch + ReadableStream and re-renders
            the progress bar per chunk."""
            from fastapi.responses import StreamingResponse
            import json as _json
            reg = self._model_registry
            if reg is None:
                raise HTTPException(status_code=503, detail="Model registry not available")
            body = await request.json()
            name = str(body.get("name", "")).strip()
            if not name:
                raise HTTPException(status_code=400, detail="name required")

            async def _stream():
                async for chunk in reg.pull(name):
                    yield f"data: {_json.dumps(chunk)}\n\n"
                yield "data: {\"status\": \"done\"}\n\n"
                # Tell the UI to refresh the installed-list once the pull lands
                try:
                    await self.broadcast({"type": "model.pulled", "name": name})
                except Exception:
                    pass

            return StreamingResponse(_stream(), media_type="text/event-stream")

        @app.post("/api/dnd")
        async def dnd_set(request: Request):
            mgr = self._interruptibility
            if mgr is None:
                raise HTTPException(status_code=503, detail="Interruptibility not available")
            body = await request.json()
            minutes = body.get("minutes")
            try:
                minutes_f = float(minutes) if minutes is not None else 0.0
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="minutes must be a number")
            if minutes_f <= 0:
                mgr.clear_dnd()
                await self.broadcast({"type": "dnd", "active": False, "until": None})
                return JSONResponse({"active": False, "until": None})
            until = mgr.set_dnd(minutes_f)
            await self.broadcast({
                "type":   "dnd",
                "active": True,
                "until":  until.isoformat(),
                "minutes": minutes_f,
            })
            return JSONResponse({"active": True, "until": until.isoformat()})

        @app.get("/api/camera/{room}/snapshot.jpg")
        async def camera_snapshot(room: str):
            """Single-frame JPEG snapshot of a room's camera. Browser polls this."""
            if not _CV2_AVAILABLE or cv2 is None:
                raise HTTPException(status_code=503, detail="OpenCV not available")
            cm = self._camera_manager
            if cm is None:
                raise HTTPException(status_code=503, detail="Camera manager not registered")
            if room not in cm.get_available_rooms():
                raise HTTPException(status_code=404, detail=f"No camera for '{room}'")
            frame = await cm.capture_frame_async(room)
            if frame is None:
                raise HTTPException(status_code=502, detail="Frame capture failed")
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._camera_jpeg_quality])
            if not ok:
                raise HTTPException(status_code=500, detail="JPEG encode failed")
            return Response(
                content=buf.tobytes(),
                media_type="image/jpeg",
                headers={"Cache-Control": "no-store"},
            )

        @app.get("/api/cameras")
        async def cameras_list():
            """List of every camera-configured room, regardless of whether
            the capture is currently open. The dashboard uses this to
            decide where to render the reconnect button — gating on
            "currently streaming" is wrong because a stuck cam disappears
            from that list at exactly the moment the user needs the button.
            """
            cm = self._camera_manager
            if cm is None:
                return JSONResponse({"cameras": []})
            return JSONResponse({"cameras": cm.get_configured_rooms()})

        @app.get("/api/world_model/rooms")
        async def world_model_rooms():
            """List rooms with a world_model: block enabled. Used by the
            polygon viewer dropdown."""
            orch = self._orchestrator
            if orch is None:
                return JSONResponse({"rooms": []})
            out = []
            for r in (getattr(orch, "config", {}) or {}).get("rooms", []):
                wm = r.get("world_model") or {}
                if wm.get("enabled", True) is False:
                    continue
                if not wm:
                    continue
                out.append({"id": r.get("id"), "display_name": r.get("display_name")})
            return JSONResponse({"rooms": out})

        @app.get("/api/world_model/rooms/{room}/polygons")
        async def world_model_polygons(room: str):
            """Return the configured exits + landmarks for a room — JSON
            payload the polygon viewer overlays on the live snapshot.

            Returns the LIVE topology from the running WorldModel, which
            already merges runtime overrides (from
            data/polygon_overrides.json) over config.yaml. So the editor
            sees its own saved edits on reload, not the YAML stubs.
            """
            orch = self._orchestrator
            if orch is None:
                raise HTTPException(status_code=503, detail="Orchestrator not registered")
            wm_inst = getattr(orch, "world_model", None)
            if wm_inst is not None and room in wm_inst.cameras:
                cam = wm_inst.cameras[room]
                room_cfg = None
                for r in (getattr(orch, "config", {}) or {}).get("rooms", []):
                    if r.get("id") == room:
                        room_cfg = r
                        break
                return JSONResponse({
                    "room": room,
                    "display_name": (room_cfg or {}).get("display_name"),
                    "enabled": True,
                    "frame_width": int(cam.get("frame_width", 640)),
                    "frame_height": int(cam.get("frame_height", 480)),
                    "exits": list(cam.get("exits", [])),
                    "landmarks": list(cam.get("landmarks", [])),
                })
            # Fall back to config.yaml read when the world model isn't
            # initialized yet (boot timing).
            for r in (getattr(orch, "config", {}) or {}).get("rooms", []):
                if r.get("id") != room:
                    continue
                wm = r.get("world_model") or {}
                return JSONResponse({
                    "room": room,
                    "display_name": r.get("display_name"),
                    "enabled": bool(wm.get("enabled", True)),
                    "frame_width": int(wm.get("frame_width", 640)),
                    "frame_height": int(wm.get("frame_height", 480)),
                    "exits": list(wm.get("exits", [])),
                    "landmarks": list(wm.get("landmarks", [])),
                })
            raise HTTPException(status_code=404, detail=f"No room '{room}' in config")

        @app.post("/api/world_model/rooms/{room}/polygons")
        async def world_model_polygons_save(room: str, request: Request):
            """Persist the polygon editor's edits for one room. Body:
              {frame_width, frame_height, exits, landmarks}
            Writes data/polygon_overrides.json (atomically) and asks
            the live WorldModel to reload its camera topology so the
            edits take effect without a restart."""
            from modules.world_model.world_model import (
                _load_polygon_overrides, save_polygon_overrides,
            )
            orch = self._orchestrator
            if orch is None:
                raise HTTPException(status_code=503, detail="Orchestrator not registered")
            body = await request.json()
            fw = int(body.get("frame_width") or 0)
            fh = int(body.get("frame_height") or 0)
            exits_raw = body.get("exits") or []
            landmarks_raw = body.get("landmarks") or []
            if fw <= 0 or fh <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="frame_width and frame_height required",
                )
            # Light validation — each polygon should be a list of >=3
            # [x,y] pairs. Reject malformed entries so we never persist
            # garbage that could crash _classify_exit downstream.
            def _validate(plist):
                out = []
                for p in plist:
                    poly = p.get("polygon") if isinstance(p, dict) else None
                    if not isinstance(poly, list) or len(poly) < 3:
                        continue
                    cleaned = []
                    ok = True
                    for pt in poly:
                        if (isinstance(pt, (list, tuple))
                                and len(pt) == 2
                                and all(isinstance(c, (int, float)) for c in pt)):
                            cleaned.append([int(pt[0]), int(pt[1])])
                        else:
                            ok = False
                            break
                    if not ok:
                        continue
                    new_entry = dict(p)
                    new_entry["polygon"] = cleaned
                    out.append(new_entry)
                return out
            exits = _validate(exits_raw)
            landmarks = _validate(landmarks_raw)
            overrides = _load_polygon_overrides()
            overrides[room] = {
                "frame_width": fw,
                "frame_height": fh,
                "exits": exits,
                "landmarks": landmarks,
            }
            save_polygon_overrides(overrides)
            # Live-reload so the world model picks up the edits without
            # a restart. Failures here are non-fatal (file is on disk).
            wm_inst = getattr(orch, "world_model", None)
            if wm_inst is not None:
                try:
                    wm_inst.reload_polygons()
                except Exception as e:
                    logger.warning(f"[polygons] reload failed: {e}")
            await self.broadcast({
                "type": "world_model_polygons_saved",
                "room": room,
                "exits": len(exits),
                "landmarks": len(landmarks),
            })
            return JSONResponse({
                "ok": True, "room": room,
                "exits": len(exits), "landmarks": len(landmarks),
            })

        @app.get("/polygons", response_class=HTMLResponse)
        async def polygon_viewer_page():
            """Serve the polygon-viewer SPA page. Static asset; the JS
            inside fetches snapshot + polygons via the APIs above."""
            page = STATIC_DIR / "polygon_viewer.html"
            if not page.exists():
                raise HTTPException(status_code=404, detail="polygon_viewer.html missing")
            return HTMLResponse(page.read_text(encoding="utf-8"))

        @app.get("/api/world_model/pets")
        async def world_model_pets():
            """Resident pets — every cat + dog from §22 bootstrap, with
            current state, last_seen_room, owner, and a per-pet care
            summary (last food/litterbox/water/leash event timestamps).
            Returns {} when world model is offline."""
            orch = self._orchestrator
            wq = getattr(orch, "world_query_tools", None) if orch else None
            if wq is None:
                return JSONResponse({"pets": [], "available": False})
            try:
                base = await wq.list_pets()
                # Enrich each pet with where_is_pet (likely_room +
                # unmonitored fallback) and care_summary (last 24h
                # by interaction_kind). Both are cheap reads off the
                # in-memory model + recent event index.
                pets_out = []
                for p in base:
                    name = p.get("name")
                    if not name:
                        continue
                    try:
                        where = await wq.where_is_pet(name)
                    except Exception as e:
                        logger.warning(
                            f"[/api/world_model/pets] where_is_pet({name!r}) failed: {e}"
                        )
                        where = {
                            "state": p.get("state"),
                            "last_seen_room": p.get("last_seen_room"),
                            "likely_room": p.get("last_seen_room"),
                            "likely_room_inferred": False,
                            "unmonitored_home": p.get("unmonitored_home"),
                            "duration_in_state_seconds": None,
                        }
                    try:
                        care = await wq.pet_care_summary(name, hours_ago=24)
                    except Exception as e:
                        logger.warning(
                            f"[/api/world_model/pets] pet_care_summary({name!r}) failed: {e}"
                        )
                        care = {"by_kind": {}}
                    pets_out.append({
                        "name": name,
                        "species": p.get("species"),
                        "owner_person_id": p.get("owner_person_id"),
                        "state": where.get("state"),
                        "last_seen_room": where.get("last_seen_room"),
                        "last_seen_landmark": where.get("last_seen_landmark"),
                        "last_seen_ts": where.get("last_seen_ts"),
                        "likely_room": where.get("likely_room"),
                        "likely_room_inferred": where.get(
                            "likely_room_inferred", False
                        ),
                        "unmonitored_home": where.get("unmonitored_home"),
                        "duration_in_state_seconds": where.get(
                            "duration_in_state_seconds"
                        ),
                        "care": care.get("by_kind", {}),
                        # Lore card data — seed metadata from config
                        # (color, coat, personality, notes, etc.).
                        "seed": p.get("seed") or {},
                    })
                return JSONResponse({
                    "pets": pets_out, "available": True,
                })
            except Exception as e:
                logger.warning(f"[/api/world_model/pets] {e}")
                raise HTTPException(status_code=500, detail=str(e)) from e

        @app.post("/api/world_model/yolo_now")
        async def world_model_yolo_now(request: Request):
            """Run YOLO on a fresh snapshot from `room`, filter to
            cat/dog (or whatever species[] the body asks for), and
            return bboxes for the live pet-tagging modal. Distinct
            from /recent_animal_detections — that one reads the
            event log; this one fires YOLO on demand, so a sitting
            cat that hasn't moved enough to log a recent event still
            shows up.

            Body: {room: str, species: ["cat","dog"]}
            """
            orch = self._orchestrator
            if orch is None:
                raise HTTPException(status_code=503, detail="No orchestrator")
            body = await request.json()
            room = (body.get("room") or "").strip()
            species = body.get("species") or ["cat", "dog"]
            if not room:
                raise HTTPException(status_code=400, detail="room required")
            cm = getattr(orch, "cameras", None)
            detector = getattr(orch, "object_detector", None)
            if cm is None or detector is None:
                raise HTTPException(
                    status_code=503,
                    detail="Camera manager or YOLO not initialized",
                )
            frame = await cm.capture_frame_async(room)
            if frame is None:
                raise HTTPException(
                    status_code=503,
                    detail=f"No frame from camera '{room}'",
                )
            detections = await detector.detect_async(frame)
            wanted = set(species)
            out: list[dict] = []
            for d in detections:
                cls = d.get("class")
                if cls not in wanted:
                    continue
                box = d.get("box") or []
                if len(box) != 4:
                    continue
                out.append({
                    "species": cls,
                    "bbox": box,
                    "confidence": d.get("confidence", 0.0),
                    "label": d.get("label"),
                })
            return JSONResponse({
                "detections": out,
                "frame_width": int(frame.shape[1]),
                "frame_height": int(frame.shape[0]),
                "room": room,
            })

        @app.post("/api/world_model/yolo_region")
        async def world_model_yolo_region(request: Request):
            """Rerun YOLO on a user-selected rectangular region of a
            fresh snapshot from `room`, with a much looser confidence
            threshold than the regular pipeline (default 0.10 vs the
            standard ~0.20). Useful when you see a pet in the frame
            but the live overlay doesn't — the standard pass on a
            full frame is tuned to suppress false positives, this
            region pass trades that for sensitivity.

            Body: {room, bbox: [x1,y1,x2,y2], conf?=0.10, padding?=0.15}

            Returns detections with bboxes already mapped back into
            full-frame coordinates so they overlay correctly on the
            modal's snapshot."""
            orch = self._orchestrator
            if orch is None:
                raise HTTPException(status_code=503, detail="No orchestrator")
            body = await request.json()
            room = (body.get("room") or "").strip()
            click_bbox = body.get("bbox") or []
            conf = float(body.get("conf") or 0.10)
            padding = float(body.get("padding") or 0.15)
            if not room or len(click_bbox) != 4:
                raise HTTPException(
                    status_code=400,
                    detail="room and bbox=[x1,y1,x2,y2] required",
                )
            cm = getattr(orch, "cameras", None)
            detector = getattr(orch, "object_detector", None)
            if cm is None or detector is None:
                raise HTTPException(
                    status_code=503,
                    detail="Camera manager or YOLO not initialized",
                )
            frame = await cm.capture_frame_async(room)
            if frame is None:
                raise HTTPException(
                    status_code=503,
                    detail=f"No frame from camera '{room}'",
                )
            fh, fw = int(frame.shape[0]), int(frame.shape[1])
            cx1, cy1, cx2, cy2 = (float(c) for c in click_bbox)
            # Pad the ROI so YOLO has context around the object.
            w = max(1.0, cx2 - cx1)
            h = max(1.0, cy2 - cy1)
            px = int(round(w * padding))
            py = int(round(h * padding))
            rx1 = max(0, int(cx1) - px)
            ry1 = max(0, int(cy1) - py)
            rx2 = min(fw, int(cx2) + px)
            ry2 = min(fh, int(cy2) + py)
            if rx2 <= rx1 or ry2 <= ry1:
                return JSONResponse({"detections": [], "roi": [rx1, ry1, rx2, ry2]})
            crop = frame[ry1:ry2, rx1:rx2]
            if not hasattr(detector, "detect_with_threshold_async"):
                # Older detector without ROI helper — fall back to
                # standard detect_async on the crop. Slightly worse
                # sensitivity but functional.
                dets = await detector.detect_async(crop)
            else:
                dets = await detector.detect_with_threshold_async(crop, conf)
            out: list[dict] = []
            for d in dets:
                box = d.get("box") or []
                if len(box) != 4:
                    continue
                # Map crop coords back into full-frame coords.
                bx1, by1, bx2, by2 = (int(c) for c in box)
                out.append({
                    "class": d.get("class"),
                    "confidence": d.get("confidence", 0.0),
                    "label": d.get("label"),
                    "box": [rx1 + bx1, ry1 + by1, rx1 + bx2, ry1 + by2],
                })
            return JSONResponse({
                "detections": out,
                "roi": [rx1, ry1, rx2, ry2],
                "frame_width": fw,
                "frame_height": fh,
                "conf": conf,
            })

        @app.get("/api/world_model/recent_animal_detections")
        async def world_model_recent_animal_detections(
            room: str, seconds: int = 30, species: Optional[str] = None,
        ):
            """Recent cat/dog detections in a room — used by the live
            tag-pet modal on camera tiles. Returns rows with bbox
            already JSON-parsed so the frontend can draw clickable
            overlays straight on the snapshot."""
            orch = self._orchestrator
            ws = getattr(orch, "world_store", None) if orch else None
            if ws is None:
                return JSONResponse({"detections": [], "available": False})
            try:
                seconds = max(1, min(int(seconds), 600))
                from datetime import datetime as _dt, timezone as _tz, timedelta as _td
                cutoff = (_dt.now(_tz.utc) - _td(seconds=seconds)).isoformat()
                species_filter = ""
                args: list = [room, cutoff]
                if species in ("cat", "dog"):
                    species_filter = "AND entity_type = ? "
                    args.append(species)
                else:
                    species_filter = "AND entity_type IN ('cat','dog') "
                rows = await ws.db.fetchall(
                    "SELECT id, ts, entity_id, entity_name, entity_type, "
                    "room, bbox, snapshot_path, event_type "
                    "FROM world_entity_events "
                    f"WHERE room = ? AND ts >= ? {species_filter} "
                    "ORDER BY ts DESC LIMIT 60",
                    args,
                )
                import json as _json
                out: list[dict] = []
                for r in rows:
                    d = dict(r)
                    raw = d.get("bbox")
                    if isinstance(raw, str) and raw:
                        try:
                            d["bbox"] = _json.loads(raw)
                        except Exception:
                            d["bbox"] = None
                    out.append(d)
                return JSONResponse({
                    "detections": out, "available": True,
                    "room": room, "seconds": seconds,
                })
            except Exception as e:
                logger.warning(
                    f"[/api/world_model/recent_animal_detections] {e}"
                )
                raise HTTPException(status_code=500, detail=str(e)) from e

        @app.post("/api/world_model/not_an_animal")
        async def world_model_not_an_animal(request: Request):
            """Negative-reinforcement: the user clicked a bbox that
            YOLO/the world model thought was a cat or dog, but it isn't —
            it's a box around nothing (shadow, plush, plant, etc).
            Body: {room, bbox: [x1,y1,x2,y2], seconds=30}

            Action: delete every cat/dog event whose bbox overlaps the
            click (IoU >= 0.3) in the last `seconds`, AND register the
            region as a false-positive zone for that room so the
            object-detector pipeline can soft-suppress similar
            detections at low confidence. The region cache lives on the
            orchestrator (in-memory, expires after 6h) so a restart
            clears it — false-positive sources tend to be transient
            (a hung-up jacket, a plush on the couch) and we'd rather
            re-learn than persist a stale suppression mask.
            """
            orch = self._orchestrator
            ws = getattr(orch, "world_store", None) if orch else None
            if ws is None:
                raise HTTPException(status_code=503, detail="World model offline")
            body = await request.json()
            room = (body.get("room") or "").strip()
            click_bbox = body.get("bbox") or []
            seconds = int(body.get("seconds") or 30)
            if not room or len(click_bbox) != 4:
                raise HTTPException(
                    status_code=400,
                    detail="room and bbox=[x1,y1,x2,y2] required",
                )
            from datetime import datetime as _dt, timezone as _tz, timedelta as _td
            now = _dt.now(_tz.utc)
            cutoff = (now - _td(seconds=seconds)).isoformat()
            rows = await ws.db.fetchall(
                "SELECT id, bbox FROM world_entity_events "
                "WHERE room = ? AND ts >= ? "
                "AND entity_type IN ('cat', 'dog') "
                "AND bbox IS NOT NULL",
                (room, cutoff),
            )
            import json as _json
            cx1, cy1, cx2, cy2 = (float(c) for c in click_bbox)
            click_w = max(0.0, cx2 - cx1)
            click_h = max(0.0, cy2 - cy1)
            click_area = click_w * click_h
            to_delete: list[str] = []
            for r in rows:
                try:
                    rb = _json.loads(r["bbox"])
                    if len(rb) != 4:
                        continue
                    rx1, ry1, rx2, ry2 = (float(c) for c in rb)
                except Exception:
                    continue
                ix1, iy1 = max(rx1, cx1), max(ry1, cy1)
                ix2, iy2 = min(rx2, cx2), min(ry2, cy2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                ra = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
                union = click_area + ra - inter
                iou = inter / union if union > 0 else 0.0
                if iou >= 0.3:
                    to_delete.append(r["id"])
            if to_delete:
                placeholders = ",".join("?" for _ in to_delete)
                await ws.db.execute(
                    f"DELETE FROM world_entity_events WHERE id IN ({placeholders})",
                    to_delete,
                )
            # Register a soft-suppression region on the orchestrator so
            # the next pipeline tick can downweight low-confidence
            # detections in this bbox. Expiry is 6h — the false-positive
            # source (shadow, plush, plant) usually moves or lights
            # change by then; persistent ones will get re-flagged.
            assert orch is not None
            negatives = getattr(orch, "_vision_negative_regions", None)
            if negatives is None:
                negatives = []
                orch._vision_negative_regions = negatives  # type: ignore[attr-defined]
            negatives.append({
                "room": room,
                "bbox": [cx1, cy1, cx2, cy2],
                "registered_at": now.isoformat(),
                "expires_at": (now + _td(hours=6)).isoformat(),
            })
            # Trim expired entries so the list doesn't grow forever.
            now_iso = now.isoformat()
            orch._vision_negative_regions = [  # type: ignore[attr-defined]
                n for n in negatives
                if n.get("expires_at", "") > now_iso
            ]
            await self.broadcast({
                "type": "world_negative_region_added",
                "room": room,
                "deleted_events": len(to_delete),
            })
            logger.info(
                f"[NotAnAnimal] '{room}': deleted {len(to_delete)} bogus "
                f"cat/dog event(s); region marked false-positive for 6h"
            )
            return JSONResponse({
                "ok": True,
                "deleted_events": len(to_delete),
                "room": room,
            })

        @app.post("/api/world_model/tag_in_frame")
        async def world_model_tag_in_frame(request: Request):
            """Bulk-relabel recent cat/dog events for the given (room,
            species) whose bbox overlaps the user's clicked bbox.
            Body: {room, pet_name, bbox: [x1,y1,x2,y2], seconds=30}.
            Uses IoU > 0.3 as the match criterion — generous enough to
            survive small frame-to-frame motion but tight enough to
            avoid relabeling unrelated detections.
            """
            orch = self._orchestrator
            wq = getattr(orch, "world_query_tools", None) if orch else None
            ws = getattr(orch, "world_store", None) if orch else None
            if wq is None or ws is None:
                raise HTTPException(status_code=503, detail="World model offline")
            body = await request.json()
            room = (body.get("room") or "").strip()
            pet_name = (body.get("pet_name") or "").strip()
            seconds = int(body.get("seconds") or 30)
            click_bbox = body.get("bbox") or []
            if not room or not pet_name or len(click_bbox) != 4:
                raise HTTPException(
                    status_code=400,
                    detail="room, pet_name, and bbox=[x1,y1,x2,y2] required",
                )
            target = wq.world.find_entity_by_name(pet_name)
            if target is None or target.entity_type not in ("cat", "dog"):
                raise HTTPException(
                    status_code=404,
                    detail=f"no resident pet named '{pet_name}'",
                )
            from datetime import datetime as _dt, timezone as _tz, timedelta as _td
            cutoff = (_dt.now(_tz.utc) - _td(seconds=seconds)).isoformat()
            # Query BOTH cat and dog rows — the user may be correcting a
            # cross-species misattribution (e.g. clicking the cat on the
            # table that the cost function tagged as Dalila the dog).
            # The frontend now lets the user pick any pet regardless of
            # the detected species, so the backend must follow suit.
            rows = await ws.db.fetchall(
                "SELECT id, bbox, entity_type FROM world_entity_events "
                "WHERE room = ? AND ts >= ? "
                "AND entity_type IN ('cat', 'dog') "
                "AND bbox IS NOT NULL",
                (room, cutoff),
            )
            import json as _json
            cx1, cy1, cx2, cy2 = (float(c) for c in click_bbox)
            click_w = max(0.0, cx2 - cx1)
            click_h = max(0.0, cy2 - cy1)
            click_area = click_w * click_h
            relabeled: list[str] = []
            cross_species = 0
            for r in rows:
                try:
                    rb = _json.loads(r["bbox"])
                    if len(rb) != 4:
                        continue
                    rx1, ry1, rx2, ry2 = (float(c) for c in rb)
                except Exception:
                    continue
                # IoU
                ix1, iy1 = max(rx1, cx1), max(ry1, cy1)
                ix2, iy2 = min(rx2, cx2), min(ry2, cy2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                ra = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
                union = click_area + ra - inter
                iou = inter / union if union > 0 else 0.0
                if iou >= 0.3:
                    relabeled.append(r["id"])
                    if r["entity_type"] != target.entity_type:
                        cross_species += 1
            for evt_id in relabeled:
                # Also update entity_type so cross-species corrections
                # (Dalila/dog -> Spooky/cat) flip the row's species.
                # Otherwise downstream filters like list_pets / care
                # summaries would still see the old class.
                await ws.db.execute(
                    "UPDATE world_entity_events SET entity_id = ?, "
                    "entity_name = ?, entity_type = ? WHERE id = ?",
                    (target.id, target.display_name,
                     target.entity_type, evt_id),
                )
            if cross_species:
                logger.info(
                    f"[Tag] {cross_species} cross-species correction(s) "
                    f"to '{target.display_name}' ({target.entity_type}) "
                    f"in '{room}' — YOLO or cost-function misclassification"
                )
            await self.broadcast({
                "type": "world_pet_tagged_in_frame",
                "room": room,
                "pet_name": target.display_name,
                "count": len(relabeled),
            })
            return JSONResponse({
                "ok": True,
                "relabeled": len(relabeled),
                "pet_name": target.display_name,
            })

        @app.post("/api/world_model/events/{event_id}/relabel")
        async def world_model_event_relabel(
            event_id: str, request: Request,
        ):
            """Reattribute a single world event to a named resident pet
            (or to a different named resident). The body is
            {"pet_name": "Sneaky"}. Used by the dashboard's "click a
            cat/dog event in the live feed to tell Jarvis what it is"
            workflow. The event's entity_id + entity_name are updated;
            future cost-function decisions for the affected entity are
            unchanged (this is event-history relabel, not centroid
            retrain — that comes from cluster-build).
            """
            orch = self._orchestrator
            wq = getattr(orch, "world_query_tools", None) if orch else None
            ws = getattr(orch, "world_store", None) if orch else None
            if wq is None or ws is None:
                raise HTTPException(status_code=503, detail="World model offline")
            body = await request.json()
            pet_name = (body.get("pet_name") or "").strip()
            if not pet_name:
                raise HTTPException(status_code=400, detail="pet_name required")
            target = wq.world.find_entity_by_name(pet_name)
            if target is None or target.entity_type not in ("cat", "dog"):
                raise HTTPException(
                    status_code=404,
                    detail=f"no resident pet named '{pet_name}'",
                )
            row = await ws.db.fetchone(
                "SELECT entity_type FROM world_entity_events WHERE id = ?",
                (event_id,),
            )
            if row is None:
                raise HTTPException(status_code=404, detail="event not found")
            if row["entity_type"] not in ("cat", "dog"):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "only cat/dog events can be relabeled here; "
                        "use the identity flow for people"
                    ),
                )
            await ws.db.execute(
                "UPDATE world_entity_events "
                "SET entity_id = ?, entity_name = ? WHERE id = ?",
                (target.id, target.display_name, event_id),
            )
            await self.broadcast({
                "type": "world_event_relabeled",
                "event_id": event_id,
                "pet_name": target.display_name,
            })
            return JSONResponse({
                "ok": True,
                "event_id": event_id,
                "pet_name": target.display_name,
                "entity_id": target.id,
            })

        @app.get("/api/world_model/pets/{name}/thumbnails")
        async def world_model_pet_thumbnails(
            name: str, limit: int = 8, hours_ago: int = 168,
        ):
            """Return up to `limit` recent event ids for a named pet
            whose row has a snapshot_path on disk. The lore-card modal
            renders each via /api/world_model/cluster/event/{id}/image.jpg
            so the user sees what the pet actually looks like, not just
            a glyph + lore text."""
            orch = self._orchestrator
            wq = getattr(orch, "world_query_tools", None) if orch else None
            ws = getattr(orch, "world_store", None) if orch else None
            if wq is None or ws is None:
                return JSONResponse({"thumbnails": [], "available": False})
            try:
                limit = max(1, min(int(limit), 30))
                hours_ago = max(1, min(int(hours_ago), 24 * 30))
                ent = wq.world.find_entity_by_name(name)
                if ent is None:
                    return JSONResponse({"thumbnails": [], "available": True})
                from datetime import datetime as _dt, timezone as _tz, timedelta as _td
                cutoff = (
                    _dt.now(_tz.utc) - _td(hours=hours_ago)
                ).isoformat()
                rows = await ws.db.fetchall(
                    "SELECT id, ts, room, event_type "
                    "FROM world_entity_events "
                    "WHERE entity_id = ? AND ts >= ? "
                    "AND snapshot_path IS NOT NULL "
                    "ORDER BY ts DESC LIMIT ?",
                    (ent.id, cutoff, limit),
                )
                out = [
                    {
                        "event_id": r["id"],
                        "ts": r["ts"],
                        "room": r["room"],
                        "event_type": r["event_type"],
                        "url": (
                            "/api/world_model/cluster/event/"
                            f"{r['id']}/image.jpg"
                        ),
                    }
                    for r in rows
                ]
                return JSONResponse({"thumbnails": out, "available": True})
            except Exception as e:
                logger.warning(
                    f"[/api/world_model/pets/{name}/thumbnails] {e}"
                )
                raise HTTPException(status_code=500, detail=str(e)) from e

        @app.get("/api/world_model/pets/{name}/events")
        async def world_model_pet_events(
            name: str, limit: int = 30, hours_ago: int = 168
        ):
            """Recent events for a single pet — drives the lore-card
            modal in the dashboard so the user can see the pet's
            recent activity (where it's been, what landmarks it
            interacted with, etc.)."""
            orch = self._orchestrator
            wq = getattr(orch, "world_query_tools", None) if orch else None
            if wq is None:
                return JSONResponse({"events": [], "available": False})
            try:
                limit = max(1, min(int(limit), 200))
                hours_ago = max(1, min(int(hours_ago), 24 * 30))
                rows = await wq.search_recent_events(
                    entity_name=name,
                    hours_ago=hours_ago,
                    limit=limit,
                )
                import json as _json
                out: list[dict] = []
                for r in rows:
                    d = dict(r)
                    raw = d.get("metadata")
                    if isinstance(raw, str) and raw:
                        try:
                            d["metadata"] = _json.loads(raw)
                        except Exception:
                            d["metadata"] = {}
                    elif raw is None:
                        d["metadata"] = {}
                    out.append(d)
                return JSONResponse({"events": out, "available": True})
            except Exception as e:
                logger.warning(
                    f"[/api/world_model/pets/{name}/events] {e}"
                )
                raise HTTPException(status_code=500, detail=str(e)) from e

        # ── §29.8 Clown alarm (v4.1) ──────────────────────────────────────

        def _clown_alarm():
            """Resolve the active ClownAlarm instance via the alarm
            dispatcher, or None when not wired."""
            orch = self._orchestrator
            disp = getattr(orch, "alarm_dispatcher", None) if orch else None
            if disp is None:
                return None
            return disp.alarms.get("clown")

        @app.get("/api/clown/status")
        async def clown_status():
            """State + cooldown + recent generation events for the
            dashboard clown card."""
            ca = _clown_alarm()
            if ca is None:
                return JSONResponse({"available": False})
            return JSONResponse({
                "available": True,
                "state": ca.state.value,
                "cooldown_remaining_seconds": ca._cooldown_remaining(),
                "cooldown_reason": ca._cooldown_reason,
                "recent_improv_events": ca.recent_improv_events()[-10:],
                "pool_size": len(ca._pool),
                "pool_improv_slots": sum(
                    1 for r in ca._pool if r.generate
                ),
            })

        @app.post("/api/clown/test_fire")
        async def clown_test_fire(payload: Optional[dict] = None):
            """§29.8.5 — synthesize a clown.detected event so Cole can
            confirm the audio sequence end-to-end after editing the
            YAML or replacing audio files."""
            orch = self._orchestrator
            if orch is None:
                raise HTTPException(status_code=503, detail="No orchestrator")
            from datetime import timezone as _tz
            await orch.bus.publish("clown.detected", {
                "trigger": "dashboard_test",
                "evidence": "dashboard test fire button",
                "confidence": 1.0,
                "ts": datetime.now(_tz.utc).isoformat(),
                **(payload or {}),
            })
            return JSONResponse({"fired": True})

        @app.post("/api/clown/reload_pool")
        async def clown_reload_pool():
            """Hot-reload the response YAML after Cole edits it."""
            ca = _clown_alarm()
            if ca is None:
                raise HTTPException(status_code=503, detail="Clown alarm unavailable")
            count = ca.reload_pool()
            return JSONResponse({"loaded": count})

        @app.get("/api/clown/pool")
        async def clown_pool():
            """Browse the response pool (id / tone / text / generate flag)."""
            ca = _clown_alarm()
            if ca is None:
                return JSONResponse({"available": False, "responses": []})
            return JSONResponse({
                "available": True,
                "responses": ca.pool_summary(),
            })

        @app.post("/api/clown/cooldown")
        async def clown_cooldown(payload: dict):
            """Voice-equivalent cooldown set from the dashboard. Body:
              {phrase: 'for an hour'} → suppress_for_seconds via
              parse_cooldown_phrase, OR
              {seconds: 1800, reason: '...'} → direct seconds."""
            ca = _clown_alarm()
            if ca is None:
                raise HTTPException(status_code=503, detail="Clown alarm unavailable")
            phrase = (payload or {}).get("phrase")
            if phrase:
                from modules.safety.alarms import parse_cooldown_phrase
                seconds, reason = parse_cooldown_phrase(str(phrase))
                if seconds < 0:
                    ca.suppress_indefinitely(reason=reason)
                    return JSONResponse({"suppressed": True, "indefinite": True})
                if seconds == 0:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Could not parse phrase: {phrase!r}",
                    )
                ca.suppress_for_seconds(seconds, reason=reason)
                return JSONResponse({
                    "suppressed": True, "seconds": seconds,
                    "reason": reason,
                })
            seconds = float((payload or {}).get("seconds", 0))
            reason = (payload or {}).get("reason")
            if seconds <= 0:
                raise HTTPException(
                    status_code=400, detail="seconds must be > 0",
                )
            ca.suppress_for_seconds(seconds, reason=reason)
            return JSONResponse({
                "suppressed": True, "seconds": seconds, "reason": reason,
            })

        @app.post("/api/clown/reenable")
        async def clown_reenable():
            ca = _clown_alarm()
            if ca is None:
                raise HTTPException(status_code=503, detail="Clown alarm unavailable")
            ca.reenable()
            return JSONResponse({"reenabled": True})

        @app.get("/api/world_model/interactions")
        async def world_model_interactions(
            limit: int = 30, hours_ago: int = 24,
        ):
            """§24.6 — focused tail of interaction events
            (INTERACTED_WITH / PICKED_UP / PLACED_DOWN / HANDED_OFF)
            so the dashboard can render a narrative timeline. Same
            shape as /api/world_model/events but pre-filtered + with
            crop_path resolved into an absolute snapshot URL the
            browser can load directly."""
            orch = self._orchestrator
            wq = getattr(orch, "world_query_tools", None) if orch else None
            if wq is None:
                return JSONResponse({"events": [], "available": False})
            try:
                limit = max(1, min(int(limit), 200))
                hours_ago = max(1, min(int(hours_ago), 168))
                rows = await wq.search_recent_events(
                    hours_ago=hours_ago,
                    limit=limit,
                    event_types=[
                        "interacted_with", "picked_up",
                        "placed_down", "handed_off",
                    ],
                )
                import json as _json
                clean: list[dict] = []
                for r in rows:
                    d = dict(r)
                    raw = d.get("metadata")
                    if isinstance(raw, str) and raw:
                        try:
                            d["metadata"] = _json.loads(raw)
                        except Exception:
                            d["metadata"] = {}
                    elif raw is None:
                        d["metadata"] = {}
                    # If a snapshot_path exists, expose its
                    # cluster-event URL so the JS can lazy-load.
                    if d.get("snapshot_path"):
                        d["thumbnail_url"] = (
                            "/api/world_model/cluster/event/"
                            f"{d['id']}/image.jpg"
                        )
                    clean.append(d)
                return JSONResponse({
                    "events": clean, "available": True,
                })
            except Exception as e:
                logger.warning(f"[/api/world_model/interactions] {e}")
                raise HTTPException(status_code=500, detail=str(e)) from e

        @app.get("/api/world_model/events")
        async def world_model_events(limit: int = 20, hours_ago: int = 12):
            """Recent world entity events — used for a live tail panel
            in the dashboard so the user can watch §22.9 landmark
            interactions and state transitions fire as they happen."""
            orch = self._orchestrator
            wq = getattr(orch, "world_query_tools", None) if orch else None
            if wq is None:
                return JSONResponse({"events": [], "available": False})
            try:
                limit = max(1, min(int(limit), 200))
                hours_ago = max(1, min(int(hours_ago), 168))
                rows = await wq.search_recent_events(
                    hours_ago=hours_ago,
                    limit=min(limit * 5, 500),
                )
                # search_events serializes metadata as a JSON string;
                # decode here so the browser doesn't double-parse.
                import json as _json
                clean: list[dict] = []
                for r in rows:
                    d = dict(r)
                    raw = d.get("metadata")
                    if isinstance(raw, str) and raw:
                        try:
                            d["metadata"] = _json.loads(raw)
                        except Exception:
                            d["metadata"] = {}
                    elif raw is None:
                        d["metadata"] = {}
                    meta = d.get("metadata") or {}
                    name = str(d.get("entity_name") or "")
                    generic_object_churn = (
                        d.get("entity_type") == "object"
                        and d.get("event_type") in {
                            "lost_visibility", "reappeared",
                        }
                        and name.startswith("unknown_")
                        and meta.get("source", "yolo") == "yolo"
                    )
                    if generic_object_churn:
                        continue
                    clean.append(d)
                    if len(clean) >= limit:
                        break
                return JSONResponse({
                    "events": clean, "available": True,
                })
            except Exception as e:
                logger.warning(f"[/api/world_model/events] {e}")
                raise HTTPException(status_code=500, detail=str(e)) from e

        # ── Live tunables (Settings tab) ─────────────────────────────────
        # GET  /api/tunables       → return all tunable groups
        # PATCH /api/tunables      → merge updates into runtime state
        #
        # Distinct from /api/config which dumps the full YAML. This one
        # is the curated, hot-reloadable knob surface that the dashboard
        # Settings tab edits. Changes are runtime-only and revert on
        # restart; persistent edits should go through the YAML directly.
        _LOG_FILTER_MODULES_ATTR = "_console_debug_blacklist"

        def _gather_config_state() -> dict:
            orch = self._orchestrator
            wm = getattr(orch, "world_model", None) if orch else None
            ob = getattr(orch, "observation_builder", None) if orch else None
            wm_cfg = dict(wm.cfg) if wm and isinstance(wm.cfg, dict) else {}
            snap_intervals = (
                dict(ob._snapshot_min_interval_s)
                if ob is not None
                else {}
            )
            try:
                from main import _CONSOLE_DEBUG_BLACKLIST as _blk
                log_blacklist = sorted(_blk)
            except Exception:
                log_blacklist = []
            return {
                "world_model": {
                    "visibility_grace_seconds": float(wm_cfg.get(
                        "visibility_grace_seconds", 3.0
                    )),
                    "visibility_window_seconds": float(wm_cfg.get(
                        "visibility_window_seconds", 6.0
                    )),
                    "visibility_min_samples": int(wm_cfg.get(
                        "visibility_min_samples", 4
                    )),
                    "visibility_seen_fraction_floor": float(wm_cfg.get(
                        "visibility_seen_fraction_floor", 0.25
                    )),
                    "movement_jitter_threshold": float(wm_cfg.get(
                        "movement_jitter_threshold", 0.15
                    )),
                    "posture_debounce_frames": int(wm_cfg.get(
                        "posture_debounce_frames", 3
                    )),
                    "interaction_debounce_frames": int(wm_cfg.get(
                        "interaction_debounce_frames", 3
                    )),
                    "landmark_dwell_frames": int(wm_cfg.get(
                        "landmark_dwell_frames", 3
                    )),
                    "T_handoff_seconds": float(wm_cfg.get(
                        "T_handoff_seconds", 8.0
                    )),
                    "stationary_long_minutes": float(wm_cfg.get(
                        "stationary_long_minutes", 5.0
                    )),
                    "person_continuity_seconds": float(wm_cfg.get(
                        "person_continuity_seconds", 5.0
                    )),
                },
                "snapshots": snap_intervals,
                "logs": {"console_debug_blacklist": log_blacklist},
            }

        @app.get("/api/tunables")
        async def tunables_get():
            return JSONResponse(_gather_config_state())

        @app.get("/api/perf")
        async def perf_get():
            """Snapshot the per-component timing tracker so the Perf tab
            can show where the lag is. Cheap aggregation — just walks
            the rolling deques."""
            from modules.context.perf_tracker import perf as _perf
            return JSONResponse(_perf().snapshot())

        @app.patch("/api/tunables")
        async def tunables_patch(request: Request):
            body = await request.json()
            orch = self._orchestrator
            wm = getattr(orch, "world_model", None) if orch else None
            ob = getattr(orch, "observation_builder", None) if orch else None
            applied: dict = {"world_model": {}, "snapshots": {}, "logs": {}}
            errors: list[str] = []

            wm_updates = body.get("world_model") or {}
            if wm is not None and isinstance(wm_updates, dict):
                for k, v in wm_updates.items():
                    # Allow-list the keys we expose. Avoids accidental
                    # blast-radius on unrelated cfg entries.
                    if k not in {
                        "visibility_grace_seconds",
                        "visibility_window_seconds",
                        "visibility_min_samples",
                        "visibility_seen_fraction_floor",
                        "movement_jitter_threshold", "posture_debounce_frames",
                        "interaction_debounce_frames", "landmark_dwell_frames",
                        "T_handoff_seconds", "stationary_long_minutes",
                        "person_continuity_seconds",
                    }:
                        errors.append(f"unknown world_model key: {k}")
                        continue
                    try:
                        if k.endswith("_samples") or k.endswith("_frames"):
                            wm.cfg[k] = int(v)
                        else:
                            wm.cfg[k] = float(v)
                        applied["world_model"][k] = wm.cfg[k]
                    except (TypeError, ValueError) as e:
                        errors.append(f"world_model.{k}: {e}")

            snap_updates = body.get("snapshots") or {}
            if ob is not None and isinstance(snap_updates, dict):
                for k, v in snap_updates.items():
                    if k not in ob._snapshot_min_interval_s:
                        errors.append(f"unknown snapshot kind: {k}")
                        continue
                    try:
                        ob._snapshot_min_interval_s[k] = max(0.0, float(v))
                        applied["snapshots"][k] = ob._snapshot_min_interval_s[k]
                    except (TypeError, ValueError) as e:
                        errors.append(f"snapshots.{k}: {e}")

            log_updates = body.get("logs") or {}
            if isinstance(log_updates, dict):
                bl = log_updates.get("console_debug_blacklist")
                if isinstance(bl, list):
                    try:
                        from main import _CONSOLE_DEBUG_BLACKLIST as _blk
                        _blk.clear()
                        for m in bl:
                            if isinstance(m, str) and m:
                                _blk.add(m)
                        applied["logs"]["console_debug_blacklist"] = sorted(_blk)
                    except Exception as e:
                        errors.append(f"logs.console_debug_blacklist: {e}")

            return JSONResponse({
                "applied": applied, "errors": errors,
                "state": _gather_config_state(),
            })

        # ── §22.5 cluster builder UI ──────────────────────────────────────
        # Workflow:
        #   GET  /api/world_model/cluster/known_pets?species=cat → dropdown
        #   POST /api/world_model/cluster/build {species}        → run K-means
        #   GET  /api/world_model/cluster/event/{event_id}/image.jpg
        #   POST /api/world_model/cluster/apply {species, clusters, labels}
        #   GET  /clusters                                       → SPA page

        @app.get("/api/world_model/cluster/known_pets")
        async def cluster_known_pets(species: str = "cat"):
            """List of resident pet display_names for `species`. Used by
            the cluster-viewer dropdown so the user can label a cluster
            as e.g. 'Spooky' without having to type."""
            orch = self._orchestrator
            wm = getattr(orch, "world_model", None) if orch else None
            if wm is None:
                return JSONResponse({"pets": [], "available": False})
            try:
                names = sorted({
                    e.display_name for e in wm.entities.values()
                    if e.entity_type == species
                    and e.is_resident
                    and e.archived_at is None
                    and e.display_name
                })
                return JSONResponse({
                    "pets": names, "species": species, "available": True,
                })
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e)) from e

        @app.post("/api/world_model/cluster/build")
        async def cluster_build(payload: dict):
            """Run AnimalClusterBuilder.cluster_unattributed on demand.
            Threshold gate is honored — too few unattributed events
            returns {clusters: {}, ready: false}. When clusters DO come
            back, we attach a per-cluster sample of event ids + their
            snapshot_path so the UI can render thumbnails."""
            orch = self._orchestrator
            ws = getattr(orch, "world_store", None) if orch else None
            wm = getattr(orch, "world_model", None) if orch else None
            if ws is None or wm is None:
                raise HTTPException(
                    status_code=503, detail="World model not registered"
                )
            assert orch is not None  # narrowed by the ws/wm check above
            species = str(payload.get("species") or "cat")
            n_clusters = payload.get("n_clusters")
            days_back = int(payload.get("days_back", 7))
            from modules.world_model.cluster_builder import (
                AnimalClusterBuilder,
            )
            cluster_cfg = (orch.config.get("world_model") or {}).get(
                "cluster_builder", {}
            )
            builder = AnimalClusterBuilder(ws, cluster_cfg)
            try:
                clusters = await builder.cluster_unattributed(
                    species=species,
                    n_clusters=int(n_clusters) if n_clusters is not None else None,
                    days_back=days_back,
                )
            except Exception as e:
                logger.warning(f"[cluster/build] {e}")
                raise HTTPException(status_code=500, detail=str(e)) from e
            if not clusters:
                threshold = int(cluster_cfg.get(
                    "cluster_min_observations", 200,
                ))
                # Give the UI useful telemetry on the gate.
                rows = await ws.search_events(
                    event_types=["first_seen", "moved_within_room",
                                 "reappeared", "moved_to"],
                    limit=threshold,
                )
                count_unattrib = sum(
                    1 for r in rows
                    if r.get("entity_type") == species
                    and (r.get("entity_name") is None
                         or str(r.get("entity_name", "")).startswith("unknown_"))
                )
                return JSONResponse({
                    "ready": False,
                    "species": species,
                    "unattributed_count": count_unattrib,
                    "threshold": threshold,
                    "clusters": {},
                })
            # Hydrate per-cluster samples (event_id + snapshot_path).
            # Pull the events so we can include thumbnail paths.
            sample_per_cluster = int(payload.get("samples_per_cluster", 9))
            event_id_set: set[str] = set()
            for ids in clusters.values():
                for eid in ids[:sample_per_cluster]:
                    event_id_set.add(eid)
            # No bulk-by-id helper on WorldStore; iterate the recent
            # window and filter. Cheap on day-1 data sizes.
            from datetime import datetime, timedelta
            since = datetime.utcnow() - timedelta(days=max(days_back, 1) + 1)
            recent = await ws.search_events(
                event_types=["first_seen", "moved_within_room",
                             "reappeared", "moved_to"],
                since=since,
                limit=20000,
            )
            by_id: dict[str, dict] = {
                r["id"]: r for r in recent if r["id"] in event_id_set
            }
            # Decode metadata once.
            import json as _json
            for r in by_id.values():
                raw = r.get("metadata")
                if isinstance(raw, str) and raw:
                    try:
                        r["metadata"] = _json.loads(raw)
                    except Exception:
                        r["metadata"] = {}
                elif raw is None:
                    r["metadata"] = {}
            cluster_payload: dict[int, dict] = {}
            for cluster_id, event_ids in clusters.items():
                samples: list[dict] = []
                for eid in event_ids[:sample_per_cluster]:
                    row = by_id.get(eid)
                    if row is None:
                        continue
                    meta = row.get("metadata") or {}
                    samples.append({
                        "event_id": eid,
                        "room": row.get("room"),
                        "ts": row.get("ts"),
                        "color_class": meta.get("color_class"),
                        "size_normalized": meta.get("size_normalized"),
                        "snapshot_path": row.get("snapshot_path"),
                    })
                cluster_payload[cluster_id] = {
                    "size": len(event_ids),
                    "event_ids": list(event_ids),
                    "samples": samples,
                }
            return JSONResponse({
                "ready": True,
                "species": species,
                "clusters": cluster_payload,
            })

        @app.get("/api/world_model/cluster/event/{event_id}/image.jpg")
        async def cluster_event_image(event_id: str):
            """Serve the snapshot JPEG for one event, by id. Resolves
            the path stored on the event row + a couple of safety
            checks so we don't serve random files."""
            orch = self._orchestrator
            ws = getattr(orch, "world_store", None) if orch else None
            if ws is None:
                raise HTTPException(status_code=503, detail="No world store")
            row = await ws.db.fetchone(
                "SELECT snapshot_path FROM world_entity_events WHERE id = ?",
                (event_id,),
            )
            if row is None or not row["snapshot_path"]:
                raise HTTPException(status_code=404, detail="No snapshot")
            assert orch is not None  # narrowed by the ws check above
            from pathlib import Path as _Path
            path = _Path(row["snapshot_path"])
            # Constrain to the configured snapshot dir to avoid path
            # traversal — only serve files under `data/world_snapshots`.
            data_dir = _Path(
                (orch.config.get("system") or {}).get("data_dir", "data")
            ).resolve()
            try:
                path = path.resolve()
                if data_dir not in path.parents:
                    raise HTTPException(
                        status_code=403,
                        detail="snapshot path outside data_dir",
                    )
            except OSError as e:
                raise HTTPException(status_code=404, detail=str(e)) from e
            if not path.exists():
                raise HTTPException(status_code=404, detail="File missing")
            return Response(
                content=path.read_bytes(),
                media_type="image/jpeg",
                headers={"Cache-Control": "max-age=3600"},
            )

        @app.post("/api/world_model/cluster/event/{event_id}/delete")
        async def cluster_event_delete(event_id: str):
            """Drop a single world_entity_events row — used by the
            cluster viewer's "remove from cluster" action when the
            user spots a row that doesn't belong with the rest (a
            cat misclassified as a dog, a stray detection on a
            stuffed animal, etc). The row is gone permanently;
            future cluster builds won't re-pull it.
            """
            orch = self._orchestrator
            ws = getattr(orch, "world_store", None) if orch else None
            if ws is None:
                raise HTTPException(status_code=503, detail="World store unavailable")
            row = await ws.db.fetchone(
                "SELECT id FROM world_entity_events WHERE id = ?",
                (event_id,),
            )
            if row is None:
                raise HTTPException(status_code=404, detail="Event not found")
            await ws.db.execute(
                "DELETE FROM world_entity_events WHERE id = ?",
                (event_id,),
            )
            await self.broadcast({
                "type": "world_event_deleted",
                "event_id": event_id,
            })
            return JSONResponse({"ok": True, "event_id": event_id})

        @app.post("/api/world_model/cluster/apply")
        async def cluster_apply(payload: dict):
            """Submit cluster labels — body shape:
              {species: 'cat',
               clusters: {0: [event_id, ...], 1: [...]},
               labels:   {0: 'Spooky', 1: 'Velcro'}}
            Empty / missing labels are skipped. Triggers a profile
            rebuild after re-attribution so the dashboard reflects the
            new attribution immediately."""
            orch = self._orchestrator
            ws = getattr(orch, "world_store", None) if orch else None
            wm = getattr(orch, "world_model", None) if orch else None
            if ws is None or wm is None:
                raise HTTPException(
                    status_code=503, detail="World model not registered"
                )
            species = str(payload.get("species") or "cat")
            clusters_raw = payload.get("clusters") or {}
            labels_raw = payload.get("labels") or {}
            # Coerce keys back to int — JSON only sends string keys.
            clusters: dict[int, list[str]] = {}
            for k, v in clusters_raw.items():
                try:
                    clusters[int(k)] = list(v)
                except (TypeError, ValueError):
                    continue
            labels: dict[int, str] = {}
            for k, v in labels_raw.items():
                try:
                    if v:
                        labels[int(k)] = str(v)
                except (TypeError, ValueError):
                    continue
            if not labels:
                return JSONResponse({"updated": 0, "labels_applied": 0})
            from modules.world_model.cluster_builder import (
                apply_cluster_labels,
            )
            try:
                updated = await apply_cluster_labels(
                    ws, labels, clusters, species=species,
                )
            except Exception as e:
                logger.warning(f"[cluster/apply] {e}")
                raise HTTPException(status_code=500, detail=str(e)) from e
            # Trigger a fresh profile build for each newly-attributed
            # pet so the dashboard reflects the new data immediately.
            from modules.world_model.pets import BehavioralProfileBuilder
            builder = BehavioralProfileBuilder()
            for pet_name in set(labels.values()):
                ent = wm.find_entity_by_name(pet_name)
                if ent is None:
                    continue
                try:
                    await builder.rebuild_for(wm, ent, days_back=30)
                except Exception as e:
                    logger.debug(
                        f"[cluster/apply] profile rebuild failed for "
                        f"'{pet_name}': {e}"
                    )
            return JSONResponse({
                "updated": updated, "labels_applied": len(labels),
            })

        @app.get("/clusters", response_class=HTMLResponse)
        async def cluster_viewer_page():
            """Serve the cluster-viewer SPA page. Static asset; the JS
            inside fetches everything via the APIs above."""
            page = STATIC_DIR / "cluster_viewer.html"
            if not page.exists():
                raise HTTPException(
                    status_code=404, detail="cluster_viewer.html missing"
                )
            return HTMLResponse(page.read_text(encoding="utf-8"))

        @app.post("/api/camera/{room}/reconnect")
        async def camera_reconnect(room: str):
            """Force-reopen a room's RTSP capture. The orchestrator's
            CameraManager auto-reconnects on read failures with throttled
            backoff, but a long-dropped stream (cam reboot, WiFi drop) can
            leave the throttle window open with no live cap to retry on.
            This endpoint bypasses the throttle and tries a fresh open
            immediately, on demand from the dashboard's per-cam button.
            """
            cm = self._camera_manager
            if cm is None:
                raise HTTPException(status_code=503, detail="Camera manager not registered")
            # All cameras live in _video_kinds; available_rooms only contains
            # ones that opened successfully at boot, so we can't filter on it
            # here — that would refuse exactly the case this endpoint is for.
            kind = getattr(cm, "_video_kinds", {}).get(room)
            if kind is None:
                raise HTTPException(status_code=404, detail=f"No camera configured for '{room}'")
            if kind != "rtsp":
                # USB / HTTP cams have their own reopen paths the orchestrator
                # already drives; only RTSP exposes the manual force-reopen
                # because that's where the throttle-stuck failure mode lives.
                raise HTTPException(
                    status_code=400,
                    detail=f"Reconnect only supported for RTSP cameras, '{room}' is {kind}",
                )
            try:
                reconnect = getattr(cm, "reconnect_camera", None)
                if callable(reconnect):
                    reconnect_camera = cast(
                        Callable[[str], Awaitable[bool]], reconnect
                    )
                    now_open = bool(await reconnect_camera(room))
                else:
                    await asyncio.to_thread(cm._reopen_rtsp, room)
                    now_open = room in cm.get_available_rooms()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Reconnect failed: {e}") from e
            return JSONResponse({"room": room, "reconnected": now_open})

        @app.get("/stream/{room}")
        async def camera_mjpeg_stream(room: str):
            """Multipart MJPEG stream — efficient live view that the browser
            consumes as <img src="/stream/bedroom">. Holds the connection
            open and pushes one JPEG per frame instead of the snapshot
            endpoint's poll-per-frame round trip. Useful for the dashboard's
            full-room view.

            Capped at ~5 fps on the server side because the JPEG encoder is
            the bottleneck for cv2.imencode on large RTSP frames; bump if
            that turns out to be CPU-cheap on Cole's GPU box.
            """
            from fastapi.responses import StreamingResponse
            if not _CV2_AVAILABLE or cv2 is None:
                raise HTTPException(status_code=503, detail="OpenCV not available")
            cm = self._camera_manager
            if cm is None:
                raise HTTPException(status_code=503, detail="Camera manager not registered")
            if room not in cm.get_available_rooms():
                raise HTTPException(status_code=404, detail=f"No camera for '{room}'")

            stream_fps_cap = 5.0
            frame_interval = 1.0 / stream_fps_cap

            async def gen():
                # Re-narrow inside the closure — Pylance doesn't carry the
                # outer `if cv2 is None` check through a generator boundary.
                assert cv2 is not None
                while True:
                    frame = await cm.capture_frame_async(room)
                    if frame is None:
                        # Don't tear the stream down on a transient miss —
                        # Wyze RTSP burps occasionally and recovers.
                        await asyncio.sleep(0.5)
                        continue
                    ok, buf = cv2.imencode(
                        ".jpg",
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, self._camera_jpeg_quality],
                    )
                    if not ok:
                        await asyncio.sleep(frame_interval)
                        continue
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + buf.tobytes()
                        + b"\r\n"
                    )
                    await asyncio.sleep(frame_interval)

            return StreamingResponse(
                gen(),
                media_type="multipart/x-mixed-replace; boundary=frame",
                headers={"Cache-Control": "no-store"},
            )

        # ── Per-room speaker / mic test endpoints ────────────────────────

        @app.post("/api/speaker/{room}/test")
        async def speaker_test(room: str, request: Request):
            """Play a short test phrase in the given room's speaker.
            Used for verifying a Wyze cam (or any other sink) is wired
            up correctly without having to wait for an organic TTS event.

            Body (optional): {"text": "custom phrase"}; defaults to a
            self-identifying phrase that names the room, so Cole can tell
            which cam actually beeped if multiple are within earshot.
            """
            sm = self._speaker_manager
            orch = self._orchestrator
            if sm is None or orch is None:
                raise HTTPException(
                    status_code=503,
                    detail="Speaker manager / orchestrator not registered",
                )
            if room not in sm.get_rooms():
                raise HTTPException(
                    status_code=404, detail=f"No speaker configured for '{room}'"
                )
            body = {}
            try:
                body = await request.json()
            except Exception:
                pass  # empty body is fine — use the default phrase
            text = str(body.get("text", "")).strip() or (
                f"Speaker test from {room.replace('_', ' ')}."
            )
            # Route through the orchestrator's full _speak path so the
            # per-room sink dispatch (Wyze SSH vs ESP MQTT vs PC) is the
            # same path organic TTS goes through. priority=oneway to
            # suppress the post-speech listen window on a manual test.
            try:
                await orch._speak(text, room=room, priority="oneway")
            except Exception as e:
                logger.warning(f"[Dashboard] speaker test for '{room}' failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
            await self.broadcast(
                {"type": "speaker_test", "room": room, "text": text}
            )
            return JSONResponse({"ok": True, "room": room, "text": text})

        @app.get("/api/mic/{room}/status")
        async def mic_status(room: str):
            """Report whether the given room has a configured mic source.
            Doesn't probe the source live — the dashboard polls this for
            the per-room mic indicator dot. Live mic-level data flows
            through the existing 'audio_level' broadcast events.
            """
            mm = self._mic_manager
            sm = self._speaker_manager
            return JSONResponse({
                "room": room,
                "has_mic": (mm is not None and room in mm.get_rooms()),
                "has_speaker": (sm is not None and room in sm.get_rooms()),
                "speaker_type": (
                    sm.get_speaker_type(room) if sm is not None else "none"
                ),
            })

        # ── Per-room runtime settings (rotate / flip / volume / mute) ────

        @app.get("/api/room/{room}/settings")
        async def room_settings_get(room: str):
            """Return the room's current runtime tweaks. Empty dict means
            'all defaults'. Used by the per-feed cog modal to populate
            the form when it opens."""
            rs = self._room_settings
            if rs is None:
                return JSONResponse({"settings": {}})
            return JSONResponse({"settings": rs.get(room)})

        @app.post("/api/room/{room}/settings")
        async def room_settings_set(room: str, request: Request):
            """Patch the room's runtime tweaks. Body is a partial dict —
            only keys present get updated; pass null to clear a key.
            Recognized keys: rotation (0/90/180/270), flip_h, flip_v,
            brightness, contrast, volume, muted. Unknown keys are
            silently dropped after normalize().
            """
            rs = self._room_settings
            if rs is None:
                raise HTTPException(status_code=503, detail="RoomSettings not registered")
            try:
                body = await request.json()
            except Exception:
                body = {}
            if not isinstance(body, dict):
                raise HTTPException(status_code=400, detail="body must be an object")
            from core.room_settings import RoomSettings as _RS
            cleaned = _RS.normalize(body)
            updated = await rs.update(room, **cleaned)
            await self.broadcast(
                {"type": "room_settings_changed", "room": room, "settings": updated}
            )
            return JSONResponse({"ok": True, "room": room, "settings": updated})

        @app.delete("/api/room/{room}/settings")
        async def room_settings_clear(room: str):
            """Wipe all runtime tweaks for the room — reverts to config.yaml defaults."""
            rs = self._room_settings
            if rs is None:
                raise HTTPException(status_code=503, detail="RoomSettings not registered")
            await rs.clear_room(room)
            await self.broadcast(
                {"type": "room_settings_changed", "room": room, "settings": {}}
            )
            return JSONResponse({"ok": True})

        @app.post("/api/room/{room}/speak")
        async def room_speak(room: str, request: Request):
            """Speak arbitrary text in this room's speaker. Routes through
            the orchestrator's full _speak path so the per-room sink
            dispatch (Wyze SSH vs ESP MQTT vs PC) is identical to organic
            TTS. Used by the per-feed cog modal's 'Send TTS' field.
            """
            orch = self._orchestrator
            if orch is None:
                raise HTTPException(status_code=503, detail="Orchestrator not registered")
            try:
                body = await request.json()
            except Exception:
                body = {}
            text = str(body.get("text", "")).strip()
            if not text:
                raise HTTPException(status_code=400, detail="text required")
            # priority=oneway suppresses the post-speech listen window —
            # this is a manual broadcast, not an organic conversation turn.
            try:
                await orch._speak(text, room=room, priority="oneway")
            except Exception as e:
                logger.warning(f"[Dashboard] room_speak('{room}') failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
            return JSONResponse({"ok": True, "room": room, "text": text})

        # ── Wyze cam hardware controls (per-cam SSH-driven) ─────────────

        @app.get("/api/wyze/{room}/cam")
        async def wyze_cam_get(room: str):
            """Return the cam's current /configs/.parameters values for
            the keys we know about (night vision, IR LEDs, status LED,
            etc.) plus the WYZE_PARAMS spec so the dashboard can render
            the correct option labels without hardcoding them. Missing
            keys come back as null = "cam is using default."
            """
            ctrl = self._wyze_cam_controls.get(room)
            if ctrl is None:
                raise HTTPException(
                    status_code=404, detail=f"Room '{room}' is not a Wyze cam"
                )
            from modules.vision.wyze_cam_control import WYZE_PARAMS
            values = await ctrl.get_all()
            spec = {
                k: {
                    "label": v["label"],
                    "options": [
                        {"value": opt, "label": v["labels"].get(opt, str(opt))}
                        for opt in v["valid"]
                    ],
                    "reboot_required": v["reboot_required"],
                }
                for k, v in WYZE_PARAMS.items()
            }
            return JSONResponse({"room": room, "values": values, "spec": spec})

        @app.post("/api/wyze/{room}/cam")
        async def wyze_cam_set(room: str, request: Request):
            """Set one or more cam params. Body: {key: value, ...}. Each
            key is validated against WYZE_PARAMS; bad keys are ignored
            and reported in the response so the dashboard can highlight
            what didn't take.
            """
            ctrl = self._wyze_cam_controls.get(room)
            if ctrl is None:
                raise HTTPException(
                    status_code=404, detail=f"Room '{room}' is not a Wyze cam"
                )
            try:
                body = await request.json()
            except Exception:
                body = {}
            if not isinstance(body, dict):
                raise HTTPException(status_code=400, detail="body must be an object")
            results = {}
            for k, v in body.items():
                results[k] = await ctrl.set_param(k, v)
            await self.broadcast(
                {"type": "wyze_cam_changed", "room": room, "results": results}
            )
            return JSONResponse({"ok": all(results.values()), "results": results})

        @app.post("/api/wyze/{room}/reboot")
        async def wyze_cam_reboot(room: str):
            """Reboot the Wyze cam — necessary for some param changes
            (the reboot_required flag in the spec). Cam comes back
            online ~25-40s later. RTSP/SSH will fail in the interim;
            CameraManager's reopen logic + SSH lazy-reconnect handle
            the recovery transparently.
            """
            ctrl = self._wyze_cam_controls.get(room)
            if ctrl is None:
                raise HTTPException(
                    status_code=404, detail=f"Room '{room}' is not a Wyze cam"
                )
            ok = await ctrl.reboot()
            if not ok:
                raise HTTPException(status_code=502, detail="reboot SSH command failed")
            await self.broadcast({"type": "wyze_cam_rebooting", "room": room})
            return JSONResponse({"ok": True, "room": room})

        @app.post("/api/room/{room}/play_pcm")
        async def room_play_pcm(room: str, request: Request):
            """Talkback: client uploads raw int16 PCM bytes (16kHz mono
            assumed unless `?rate=N` query is supplied) and we route it
            through SpeakerManager. Used by the per-feed cog modal's
            'Push from your mic' button — browser captures audio via
            getUserMedia, downsamples to int16, POSTs the buffer here.
            """
            sm = self._speaker_manager
            if sm is None:
                raise HTTPException(status_code=503, detail="SpeakerManager not registered")
            if room not in sm.get_rooms():
                raise HTTPException(status_code=404, detail=f"No speaker for '{room}'")
            try:
                pcm = await request.body()
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"body read failed: {e}")
            if not pcm:
                raise HTTPException(status_code=400, detail="empty body")
            try:
                rate = int(request.query_params.get("rate", "16000"))
            except ValueError:
                rate = 16000
            ok = await sm.play(room, pcm, rate)
            if not ok:
                raise HTTPException(status_code=502, detail="speaker rejected playback")
            return JSONResponse({"ok": True, "room": room, "bytes": len(pcm), "rate": rate})

    async def broadcast(self, event: dict):
        """
        Push an event to all connected browser clients.
        Called by the orchestrator whenever anything changes.

        event types:
          "state_update"    → activity state changed
          "speech"          → Jarvis said something
          "user_speech"     → Cole said something
          "node_status"     → ESP32 node came online/offline
          "appliance"       → appliance state changed
          "system_health"   → Ollama/MQTT status changed
          "vision"          → room camera update
        """
        event["timestamp"] = datetime.now().isoformat()

        # Update internal state cache
        self._update_state(event)

        # Track conversation
        if event.get("type") in ("speech", "user_speech"):
            self._conversation.append({
                "role": "jarvis" if event["type"] == "speech" else "cole",
                "text": event.get("text", ""),
                "room": event.get("room", ""),
                "timestamp": event["timestamp"],
            })
            # Trim to max
            if len(self._conversation) > self._max_conversation:
                self._conversation = self._conversation[-self._max_conversation:]

        # Push to all connected clients
        dead = []
        for client in list(self._clients):
            try:
                await asyncio.wait_for(
                    client.send_json({"type": "event", "event": event}),
                    timeout=2.0,
                )
            except Exception:
                dead.append(client)

        for d in dead:
            if d in self._clients:
                self._clients.remove(d)

    def _update_state(self, event: dict):
        """Update the internal state cache based on incoming event."""
        etype = event.get("type")
        self._state["updated_at"] = event.get("timestamp", datetime.now().isoformat())

        if etype == "state_update":
            self._state.update({
                "activity":         event.get("activity", self._state["activity"]),
                "location":         event.get("location", self._state["location"]),
                "interruptibility": event.get("interruptibility", self._state["interruptibility"]),
                "confidence":       event.get("confidence", self._state["confidence"]),
                "signals":          event.get("signals", self._state["signals"]),
                "context":          event.get("context", self._state["context"]),
            })

        elif etype == "speech":
            self._state["last_speech"] = {
                "text":      event.get("text"),
                "room":      event.get("room"),
                "priority":  event.get("priority"),
                "timestamp": event.get("timestamp"),
            }

        elif etype == "node_status":
            room = event.get("room")
            if room:
                self._state["system"]["nodes"][room] = {
                    "online":     event.get("online", False),
                    "ip":         event.get("ip"),
                    "updated_at": event.get("timestamp"),
                }

        elif etype == "appliance":
            name = event.get("appliance")
            if name and name in self._state["appliances"]:
                self._state["appliances"][name].update({
                    "status":          event.get("status"),
                    "runtime_minutes": event.get("runtime_minutes"),
                })

        elif etype == "system_health":
            self._state["system"].update(event.get("health", {}))

        elif etype == "vision":
            room = event.get("room")
            if room:
                if room not in self._state["rooms"]:
                    self._state["rooms"][room] = {}
                self._state["rooms"][room].update({
                    "lights_on":      event.get("lights_on"),
                    "person_present": event.get("person_present"),
                    "description":    event.get("description"),
                    "updated_at":     event.get("timestamp"),
                    # Vision events only fire for rooms with cameras, so this
                    # is a reliable signal even if register_camera_manager
                    # hadn't run yet when the dashboard's full_state was sent.
                    "has_camera":     True,
                })

        elif etype == "audio_level":
            room = event.get("room")
            if room:
                if room not in self._state["rooms"]:
                    self._state["rooms"][room] = {}
                self._state["rooms"][room]["audio_db"] = event.get("db")
                if "peak_db" in event:
                    self._state["rooms"][room]["audio_peak_db"] = event.get("peak_db")
                item = self._wake_room_state(room)
                item["rms_db"] = event.get("db")
                item["peak_db"] = event.get("peak_db", item.get("peak_db"))
                item["updated_at"] = event.get("timestamp")
                item["suggested_sensitivity"] = self._suggest_wake_sensitivity(item)
                self._state["wake_calibration"] = self._wake_calibration

        elif etype == "wake_score":
            room = event.get("room")
            if room:
                item = self._wake_room_state(room)
                item["wake_score"] = event.get("score", 0.0)
                item["wake_model"] = event.get("model", "")
                item["sensitivity"] = event.get("sensitivity", item.get("sensitivity", 0.5))
                item["updated_at"] = event.get("timestamp")
                item["suggested_sensitivity"] = self._suggest_wake_sensitivity(item)
                if room not in self._state["rooms"]:
                    self._state["rooms"][room] = {}
                self._state["rooms"][room]["wake_score"] = item["wake_score"]
                self._state["rooms"][room]["wake_sensitivity"] = item["sensitivity"]
                self._state["wake_calibration"] = self._wake_calibration

        elif etype == "wake_calibration":
            room = event.get("room")
            if room:
                item = self._wake_room_state(room)
                item.update({k: v for k, v in event.items() if k in item})
                item["suggested_sensitivity"] = self._suggest_wake_sensitivity(item)
                self._state["wake_calibration"] = self._wake_calibration

    async def run(self):
        """Start the dashboard server. Run as a background asyncio task.

        Disabling uvicorn's signal handlers is critical: by default it
        intercepts SIGINT/SIGTERM and gracefully shuts itself down. When
        the dashboard is one of many tasks inside the orchestrator,
        Ctrl+C ends up only killing uvicorn — the rest of the orchestrator
        keeps running. Letting the orchestrator's own KeyboardInterrupt
        handling do the right thing for the whole process is what we want.

        uvicorn 0.30+ exposes `install_signal_handlers` as a Server method
        to override; older versions accepted it as a Config kwarg. Override
        the method directly so we don't have to version-sniff.
        """
        import uvicorn
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning",  # Suppress uvicorn's own logs
            access_log=False,
        )
        server = uvicorn.Server(config)
        # No-op the method uvicorn calls inside serve() to set up its own
        # SIGINT/SIGTERM hooks. Lambda discards self, accepts no args.
        server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
        logger.info(f"[Dashboard] Running at http://{self.host}:{self.port}")
        await server.serve()
