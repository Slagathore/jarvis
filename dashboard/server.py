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
        self._reminders_store = None  # Set by orchestrator via register_reminders_store()
        self._calendar = None         # Set by orchestrator via register_calendar()
        self._interruptibility = None # Set by orchestrator via register_interruptibility()
        self._orchestrator = None     # Set by orchestrator via register_orchestrator()
        self._speaker_id = None       # Set by orchestrator via register_speaker_id()
        self._face_recognizer = None  # Set by orchestrator via register_face_recognizer()

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

    def _setup_routes(self):
        app = self.app

        # Serve static files
        if STATIC_DIR.exists():
            app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/", response_class=HTMLResponse)
        async def index():
            html_path = STATIC_DIR / "index.html"
            if html_path.exists():
                return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
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
            orch._pending_speaker_enrollment = name
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

        @app.delete("/api/faces/{name}")
        async def delete_face(name: str):
            fr = self._face_recognizer
            if fr is None:
                raise HTTPException(status_code=503, detail="Face recognition not available")
            ok = await fr.delete(name)
            await self.broadcast({"type": "face_deleted", "name": name})
            return JSONResponse({"ok": ok})

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
        """Start the dashboard server. Run as a background asyncio task."""
        import uvicorn
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning",  # Suppress uvicorn's own logs
            access_log=False,
        )
        server = uvicorn.Server(config)
        logger.info(f"[Dashboard] Running at http://{self.host}:{self.port}")
        await server.serve()
