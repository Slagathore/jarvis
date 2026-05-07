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
from datetime import datetime
from pathlib import Path
from typing import Optional

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
        """Wire the orchestrator so endpoints can poke its state (enrollment flag)."""
        self._orchestrator = orchestrator

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

            # Send current full state immediately on connect
            await ws.send_json({
                "type": "full_state",
                "state": self._state,
                "conversation": self._conversation,
            })

            try:
                while True:
                    # Keep connection alive, receive pings from client
                    await ws.receive_text()
            except WebSocketDisconnect:
                if ws in self._clients:
                    self._clients.remove(ws)
                logger.debug(
                    f"[Dashboard] Client disconnected ({len(self._clients)} remaining)"
                )

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
        for client in self._clients:
            try:
                await client.send_json({"type": "event", "event": event})
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
