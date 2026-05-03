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
                })

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
