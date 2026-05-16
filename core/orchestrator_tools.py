"""
JARVIS — Ambient Home AI
========================
Mission: ToolsMixin — the LLM tool layer extracted from core/orchestrator.py
         (audit roadmap D6, decomposition step 1).

         These are the `_tool_*` implementations the local model can call,
         plus the per-domain handler-registry builders and
         `_build_tool_registry` that assembles them. They were ~1,100 lines
         inside the 5,800-line Orchestrator god-object; pulling them into a
         mixin keeps every `self.*` reference working (Orchestrator inherits
         ToolsMixin) while making both files navigable.

         The full orchestrator import block is duplicated below on purpose:
         these methods reference many of the same names, and over-importing
         is harmless — it guarantees no missing-name surprises at call time.

Modules: core/orchestrator_tools.py
Classes: ToolsMixin
"""

import asyncio
import base64
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
from loguru import logger

from core.task_supervisor import TaskPolicy, TaskSupervisor
from core.event_bus import EventBus
from core.exceptions import JarvisError
from dashboard.server import DashboardServer
from modules.integrations import IntegrationContext, IntegrationRegistry
from modules.activity.appliance_tracker import ApplianceTracker
from modules.activity.audio_classifier import AudioClassifier
from modules.activity.pc_monitor import PCMonitor
from modules.brain.llm import OllamaLLM
from modules.brain.model_registry import ModelRegistry
from modules.brain.ask_claude import ClaudeClient
from modules.computer.control import ComputerControl
from modules.selfedit.edit import SelfEditControl
from modules.brain.prompt_builder import PromptBuilder
from modules.brain.session import SessionManager
from modules.context.activity_history import ActivityHistory
from modules.context.curiosity import CuriosityEngine
from modules.context.interruptibility import InterruptibilityManager
from modules.context.sleep_tracker import SleepTracker
from modules.context.state import UNKNOWN_STATE, ActivityState
from modules.context.state_fusion import StateFusion
from modules.memory.database import DatabaseManager
from modules.memory.event_log import EventLogger
from modules.memory.memory_v2 import MemoryStore
from modules.notifications import NotificationDispatcher, NotificationManager
from modules.notifications.channels import build_channels_from_config
from modules.memory.room_baselines import RoomBaselines
from modules.agenda import GoogleCalendar
from modules.network.mqtt_client import MQTTClient
from modules.network.node_manager import NodeManager
from modules.network.webhooks import WebhookManager
from modules.reminders import ReminderScheduler, RemindersStore, parse_reminder
from modules.vision.anomaly_detector import AnomalyDetector
from modules.vision.camera_manager import CameraManager
from modules.vision.face_recognizer import FaceRecognizer
from modules.vision.light_detector import LightDetector
from modules.vision.mess_detector import MessDetector
from modules.vision.object_detector import ObjectDetector
from modules.vision.posture_analyzer import PostureAnalyzer
from modules.vision.observation_builder import ObservationBuilder
from modules.layout import DoorMap
from modules.world_model import (
    BehavioralProfileBuilder,
    InteractionMonitor,
    WorldModel,
    WorldQueryTools,
    WorldStore,
    bootstrap_pets_from_config,
)
from modules.world_model.belief import BeliefResolver
from modules.vision.scene_analyzer import SceneAnalyzer
from modules.vision.wyze_cam_control import WyzeCamControl
from core.room_settings import RoomSettings
from modules.brain.persona_manager import PersonaManager
from modules.voice.audio_focus import AudioFocus
from modules.voice.intents import parse_dnd
from modules.voice.mic_manager import MicManager
from modules.voice.speaker_manager import SpeakerManager
from modules.voice.speaker_id import SpeakerIdentifier
from modules.identity.identity_manager import IdentityManager
from modules.voice.sources.null_audio import NullMicSource, NullSpeakerSink
from modules.voice.sources.usb_mic import UsbMicSource
from modules.voice.sources.wake_adapter import MicSourceWakeAdapter
from modules.voice.sources.wyze_ssh_speaker import WyzeSshSpeakerSink
from modules.voice.stt import WhisperSTT
from modules.voice.tts import PiperTTS
from modules.voice.wake_word import WakeWordDetector
from modules.voice.wake_source import WakeSourceManager


# Per-room echo suppression: how long after Jarvis finishes speaking
# in a room we ignore that room's wake events. Tunes the trade-off
# between "Jarvis hears himself and re-wakes" (too short) and "Cole
# can't immediately follow up after a reply" (too long). 1.5s is
# enough for the speaker tail + room reverb to settle on the cam mic
# without noticeably gating real follow-ups (the in-conversation
# follow-up listener is already open separately for that case).
_ECHO_SUPPRESSION_TAIL_S: float = 1.5


class ToolsMixin:
    """LLM tool implementations + tool-registry assembly.

    Mixed into Orchestrator — all `self.*` attributes resolve
    against the concrete Orchestrator instance at runtime.
    """

    # ── Host attributes ──────────────────────────────────────────────────────
    # These live on the concrete Orchestrator (set in its __init__, or a
    # class constant). Declared here under TYPE_CHECKING so the type checker
    # knows the mixin's `self.*` references resolve — no runtime effect.
    if TYPE_CHECKING:
        config: dict
        calendar: Optional[Any]
        memory: Optional[Any]
        computer: Optional[Any]
        selfedit: Optional[Any]
        cameras: Optional[Any]
        llm: Optional[Any]
        world_query_tools: Optional[Any]
        _claude_client: Optional[Any]
        _CALENDAR_TOOLS: list[dict]
        _broadcast: Callable[[dict], Any]

    async def _tool_calendar_list_events(
        self,
        start_iso: Optional[str] = None,
        end_iso: Optional[str] = None,
    ) -> dict:
        """Tool handler: list calendar events in an arbitrary time range."""
        if self.calendar is None or not self.calendar.is_authenticated:
            return {"error": "calendar not available"}
        from datetime import timedelta as _td
        now = datetime.now()
        try:
            start = datetime.fromisoformat(start_iso) if start_iso else now
        except ValueError:
            return {"error": f"start_iso not parseable: {start_iso!r}"}
        try:
            end = datetime.fromisoformat(end_iso) if end_iso else (now + _td(days=7))
        except ValueError:
            return {"error": f"end_iso not parseable: {end_iso!r}"}

        # GoogleCalendar.upcoming_events takes hours-from-now; fall back to a
        # direct service.events().list call for an arbitrary range.
        events = await self._calendar_list_in_range(start, end)
        return {"start": start.isoformat(), "end": end.isoformat(), "events": events}

    async def _calendar_list_in_range(self, start: datetime, end: datetime) -> list[dict]:
        """List calendar events between two arbitrary datetimes."""
        cal = self.calendar
        if cal is None or cal._service is None:
            return []
        from datetime import timezone as _tz
        # Google requires ISO with tz; assume local if naive.
        if start.tzinfo is None:
            start = start.astimezone()
        if end.tzinfo is None:
            end = end.astimezone()
        service = cal._service

        def _list() -> list[dict]:
            resp = service.events().list(
                calendarId=cal._calendar_id,
                timeMin=start.isoformat(),
                timeMax=end.isoformat(),
                maxResults=50,
                singleEvents=True,
                orderBy="startTime",
            ).execute()
            return resp.get("items", [])

        try:
            raw = await asyncio.to_thread(_list)
        except Exception as e:
            logger.warning(f"[Calendar] tool list_in_range failed: {e}")
            return []
        return [cal._normalize_event(e) for e in raw]

    async def _tool_calendar_create_event(
        self,
        title: str,
        start_iso: str,
        end_iso: Optional[str] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
    ) -> dict:
        """Tool handler: create a new event."""
        if self.calendar is None or not self.calendar.is_authenticated:
            return {"error": "calendar not available"}
        try:
            start = datetime.fromisoformat(start_iso)
        except ValueError:
            return {"error": f"start_iso not parseable: {start_iso!r}"}
        end: Optional[datetime] = None
        if end_iso:
            try:
                end = datetime.fromisoformat(end_iso)
            except ValueError:
                return {"error": f"end_iso not parseable: {end_iso!r}"}
        event = await self.calendar.add_event(
            title=title, start=start, end=end,
            description=description, location=location,
        )
        if event is None:
            return {"error": "calendar API rejected create"}
        await self._broadcast({"type": "calendar_added", "event": event})
        return event

    async def _tool_calendar_delete_event(self, event_id: str) -> dict:
        """Tool handler: delete an event by ID."""
        if self.calendar is None or not self.calendar.is_authenticated:
            return {"error": "calendar not available"}
        ok = await self.calendar.delete_event(event_id)
        if ok:
            await self._broadcast({"type": "calendar_deleted", "id": event_id})
        return {"ok": ok, "event_id": event_id}

    async def _tool_calendar_update_event(
        self,
        event_id: str,
        title: Optional[str] = None,
        start_iso: Optional[str] = None,
        end_iso: Optional[str] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
    ) -> dict:
        """Tool handler: patch an existing event."""
        if self.calendar is None or not self.calendar.is_authenticated:
            return {"error": "calendar not available"}
        start = end = None
        if start_iso:
            try:
                start = datetime.fromisoformat(start_iso)
            except ValueError:
                return {"error": f"start_iso not parseable: {start_iso!r}"}
        if end_iso:
            try:
                end = datetime.fromisoformat(end_iso)
            except ValueError:
                return {"error": f"end_iso not parseable: {end_iso!r}"}
        event = await self.calendar.update_event(
            event_id=event_id, title=title, start=start, end=end,
            description=description, location=location,
        )
        if event is None:
            return {"error": "calendar API rejected update"}
        await self._broadcast({"type": "calendar_updated", "event": event})
        return event

    def _calendar_tool_handlers(self) -> dict:
        """Map tool names to bound async handlers."""
        return {
            "calendar_list_events":   self._tool_calendar_list_events,
            "calendar_create_event":  self._tool_calendar_create_event,
            "calendar_delete_event":  self._tool_calendar_delete_event,
            "calendar_update_event":  self._tool_calendar_update_event,
        }

    # ── ask_claude tool ──────────────────────────────────────────────────────
    # Lets the local LLM escalate hard questions to Claude (a stronger
    # reasoning model). Useful when Jarvis is on a smaller local model and
    # hits a wall on a tricky debug, design choice, or code question.
    _ASK_CLAUDE_TOOL: dict = {
        "type": "function",
        "function": {
            "name": "ask_claude",
            "description": (
                "Ask Anthropic's Claude (a strong reasoning model) a question. "
                "Use this when you need help debugging code, reasoning about a "
                "tricky design, or getting a second opinion. Pass the user's "
                "question or your own. Optionally include a code snippet, "
                "stack trace, or other context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask Claude. Be specific.",
                    },
                    "context": {
                        "type": "string",
                        "description": (
                            "Optional context: code snippet, error message, file "
                            "contents, etc. Goes after the question."
                        ),
                    },
                },
                "required": ["question"],
            },
        },
    }

    async def _tool_ask_claude(self, question: str, context: Optional[str] = None) -> dict:
        if self._claude_client is None:
            return {"error": "ask_claude unavailable — ANTHROPIC_API_KEY not set"}
        if not self._claude_client.has_key:
            return {"error": "ask_claude unavailable — ANTHROPIC_API_KEY not set"}
        answer = await self._claude_client.ask(question, context=context)
        return {"answer": answer}

    # ── Memory tools ────────────────────────────────────────────────────────
    # Let the LLM explicitly save and search long-term memories. Auto-extraction
    # already runs after every turn, but these let the model take direct
    # action when it decides "this specifically should be remembered" or
    # "I need to look something up before answering".
    _MEMORY_TOOLS: list[dict] = [
        {
            "type": "function",
            "function": {
                "name": "remember",
                "description": (
                    "Save a fact, preference, or instruction to long-term memory. "
                    "Use when the user shares something specific worth remembering "
                    "verbatim, or when you've decided something matters enough to "
                    "outlive the current conversation. Auto-extraction already runs "
                    "after every turn — only call this if you want explicit control."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content":    {"type": "string", "description": "The memory as a single declarative sentence."},
                        "kind":       {"type": "string", "description": "fact | preference | event | instruction"},
                        "subject":    {"type": "string", "description": "Person, room, or topic this relates to. Optional."},
                        "importance": {"type": "number", "description": "0.0-1.0. 0.9+ for load-bearing personal info."},
                    },
                    "required": ["content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "recall",
                "description": (
                    "Search long-term memory for facts/preferences/events relevant "
                    "to a query. Use when you suspect Cole has told you something "
                    "before and you want to confirm before guessing. Returns up to "
                    "8 best matches scored by semantic similarity × importance × recency."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What you're looking for."},
                        "k":     {"type": "integer", "description": "Max results (default 8)."},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "record_thought",
                "description": (
                    "Save one of your own reflections or realizations. Use when "
                    "you've noticed something interesting about a pattern, "
                    "drawn a conclusion worth keeping, or want to write down a "
                    "passing observation. These are SEARCHABLE later via recall()."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The thought, written naturally."},
                        "subject": {"type": "string", "description": "What it's about. Optional."},
                    },
                    "required": ["content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "record_question",
                "description": (
                    "Save a question you want answered later — by Cole when he's "
                    "next available, or by Claude via ask_claude. Useful when "
                    "you're curious about something but don't want to interrupt "
                    "the current conversation. The dashboard surfaces unanswered "
                    "questions for review."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content":    {"type": "string", "description": "The question."},
                        "subject":    {"type": "string", "description": "What it's about. Optional."},
                        "importance": {"type": "number", "description": "0.0-1.0. Default 0.5."},
                    },
                    "required": ["content"],
                },
            },
        },
    ]

    async def _tool_remember(
        self, content: str, kind: str = "fact",
        subject: Optional[str] = None, importance: float = 0.7,
    ) -> dict:
        if self.memory is None:
            return {"error": "memory store unavailable"}
        mid = await self.memory.add(
            kind=kind, content=content, subject=subject,
            importance=importance, source_kind="manual",
        )
        return {"ok": mid is not None, "memory_id": mid}

    async def _tool_recall(self, query: str, k: int = 8) -> dict:
        if self.memory is None:
            return {"error": "memory store unavailable"}
        hits = await self.memory.retrieve(query, k=int(k))
        return {"matches": hits}

    async def _tool_record_thought(
        self, content: str, subject: Optional[str] = None,
    ) -> dict:
        if self.memory is None:
            return {"error": "memory store unavailable"}
        mid = await self.memory.record_thought(content, subject=subject)
        return {"ok": mid is not None, "memory_id": mid}

    async def _tool_record_question(
        self, content: str, subject: Optional[str] = None,
        importance: float = 0.5,
    ) -> dict:
        if self.memory is None:
            return {"error": "memory store unavailable"}
        mid = await self.memory.record_question(
            content, subject=subject, importance=importance,
        )
        return {"ok": mid is not None, "memory_id": mid}

    # ── Computer control tools ──────────────────────────────────────────────
    # Mouse + keyboard + screenshot. Gated by the kill switch on
    # self.computer; tools only register when computer.enabled is True so the
    # LLM doesn't even know they exist when control is off.
    _COMPUTER_TOOLS: list[dict] = [
        {
            "type": "function",
            "function": {
                "name": "screenshot",
                "description": "Capture the current desktop. Returns a base64-encoded JPEG plus screen size. Use to see what's on screen before deciding where to click.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "screen_size",
                "description": "Return the screen resolution as {width, height}. Useful for computing relative click positions.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "mouse_click",
                "description": "Click at absolute screen pixel (x, y). Take a screenshot first to find the right coordinates.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                        "button": {"type": "string", "description": "left | right | middle"},
                        "clicks": {"type": "integer", "description": "1 for click, 2 for double-click"},
                    },
                    "required": ["x", "y"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "mouse_move",
                "description": "Move the cursor to (x, y) without clicking.",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                    "required": ["x", "y"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "keyboard_type",
                "description": "Type a string at the current keyboard focus. Refuses dangerous patterns (rm -rf, format, shutdown, etc.); some patterns require confirmation.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "keyboard_hotkey",
                "description": "Send a key combination, e.g. ['ctrl', 'c']. Refuses dangerous combos (Ctrl+Alt+Del, Win+L, Alt+F4).",
                "parameters": {
                    "type": "object",
                    "properties": {"keys": {"type": "array", "items": {"type": "string"}}},
                    "required": ["keys"],
                },
            },
        },
    ]

    def _computer_tool_handlers(self) -> dict:
        c = self.computer
        if c is None:
            return {}
        return {
            "screenshot":      lambda: c.screenshot(),
            "screen_size":     lambda: c.screen_size(),
            "mouse_click":     c.mouse_click,
            "mouse_move":      c.mouse_move,
            "keyboard_type":   c.keyboard_type,
            "keyboard_hotkey": c.keyboard_hotkey,
        }

    # ── Self-edit tools ─────────────────────────────────────────────────────
    # Read tools always available so the LLM can analyze its own code even
    # when write is disabled. Write/restart tools only register when the
    # selfedit kill switch is ON.
    _SELFEDIT_READ_TOOLS: list[dict] = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from the project. Path is relative to the project root.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files matching a glob pattern relative to the project root.",
                "parameters": {
                    "type": "object",
                    "properties": {"glob": {"type": "string", "description": "Default: **/*"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "grep_files",
                "description": "Search file contents for a regex pattern. Returns up to 100 matches.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "glob": {"type": "string", "description": "Default: **/*.py"},
                    },
                    "required": ["pattern"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "git_log",
                "description": "Show the last N commits.",
                "parameters": {
                    "type": "object",
                    "properties": {"n": {"type": "integer", "description": "Default 20"}},
                },
            },
        },
    ]
    _SELFEDIT_WRITE_TOOLS: list[dict] = [
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write a complete file. Auto-commits before writing so the change is one git reset away. Some files require dashboard confirmation; the protected list (this file, orchestrator, .env, jarvis.db) is blocked outright.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Find-and-replace exactly one occurrence of old_string with new_string. Auto-commits before editing.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                    },
                    "required": ["path", "old_string", "new_string"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "restart_self",
                "description": "Schedule a graceful restart so just-applied edits take effect. Pairs with the supervisor wrapper for auto-revert if the new instance fails to start within 10 seconds.",
                "parameters": {
                    "type": "object",
                    "properties": {"reason": {"type": "string"}},
                    "required": ["reason"],
                },
            },
        },
    ]

    def _selfedit_handlers(self) -> dict:
        s = self.selfedit
        if s is None:
            return {}
        return {
            "read_file":     s.read_file,
            "list_files":    s.list_files,
            "grep_files":    s.grep_files,
            "git_log":       s.git_log,
            "write_file":    s.write_file,
            "edit_file":     s.edit_file,
            "restart_self":  s.restart_self,
        }

    # ── Pattern D: action tools that trigger mid-loop model swap ────────────
    # When the LLM calls any of these tool names, OllamaLLM.chat_with_tools
    # swaps the rest of the loop onto the configured action_model (typically
    # a cheap local model like qwen3.5:4b) so the rate-limit-burning
    # mechanical execution doesn't keep hitting the cloud chat model.
    _ACTION_TOOL_NAMES: set = {
        # Computer control
        "screenshot", "screen_size",
        "mouse_click", "mouse_move",
        "keyboard_type", "keyboard_hotkey",
        # Self-edit (read tools count too — file exploration was a major
        # rate-limit burn pattern Cole observed)
        "read_file", "list_files", "grep_files", "git_log",
        "write_file", "edit_file", "restart_self",
    }

    def _build_tool_registry(self) -> tuple[list[dict], dict]:
        """Aggregate every tool currently available to the LLM.

        Returns (tools_schema, name->handler dict). Each module that adds
        tools (calendar, claude, memory, computer, self-edit) appends here.
        Empty result triggers a plain chat (no-tools) call upstream.
        """
        tools: list[dict] = []
        handlers: dict = {}
        if self.calendar is not None and self.calendar.is_authenticated:
            tools.extend(self._CALENDAR_TOOLS)
            handlers.update(self._calendar_tool_handlers())
        if self._claude_client is not None and self._claude_client.has_key:
            tools.append(self._ASK_CLAUDE_TOOL)
            handlers["ask_claude"] = self._tool_ask_claude
        if self.memory is not None:
            tools.extend(self._MEMORY_TOOLS)
            handlers["remember"]        = self._tool_remember
            handlers["recall"]          = self._tool_recall
            handlers["record_thought"]  = self._tool_record_thought
            handlers["record_question"] = self._tool_record_question
        # Computer tools only appear when the kill switch is ON. Removing
        # them entirely (rather than always-present-with-error-stub) means
        # the LLM doesn't know to attempt them when disabled.
        if self.computer is not None and self.computer.enabled:
            tools.extend(self._COMPUTER_TOOLS)
            handlers.update(self._computer_tool_handlers())
        # Self-edit: read tools always available so the LLM can analyze
        # its own code regardless of the write switch. Write/restart only
        # when the switch is on.
        if self.selfedit is not None:
            tools.extend(self._SELFEDIT_READ_TOOLS)
            handlers.update({
                k: v for k, v in self._selfedit_handlers().items()
                if k in {"read_file", "list_files", "grep_files", "git_log"}
            })
            if self.selfedit.enabled:
                tools.extend(self._SELFEDIT_WRITE_TOOLS)
                handlers.update({
                    k: v for k, v in self._selfedit_handlers().items()
                    if k in {"write_file", "edit_file", "restart_self"}
                })
        # Live vision tool — fresh snapshot of any camera-equipped room.
        # The wake-room frame is auto-attached to the user message in
        # _process_user_text already; this tool lets the LLM pull a frame
        # from a DIFFERENT room mid-conversation (e.g. "is the dog still
        # in the living room?" while Cole is in the office).
        if self.cameras is not None:
            tools.append(self._GET_SNAPSHOT_TOOL)
            handlers["get_room_snapshot"] = self._tool_get_room_snapshot
        # World Model query layer — persistent presence + bounded-house
        # state. The LLM picks these for "where is X", "who's home",
        # "what happened in <room>" patterns. Only registered when
        # WorldModel actually came up successfully.
        if self.world_query_tools is not None:
            tools.extend(self._WORLD_TOOLS)
            handlers.update(self._world_tool_handlers())
        return tools, handlers

    # ── World Model query tools (Phase 3.2) ─────────────────────────────────
    # Schemas the LLM sees. Descriptions are tuned to nudge tool-selection:
    # "where is X" → get_entity_status, "anyone home" → who_is_home,
    # "who's in <room>" → list_entities_in_room, narrative recall →
    # search_recent_events.
    _WORLD_TOOLS: list[dict] = [
        {
            "type": "function",
            "function": {
                "name": "get_entity_status",
                "description": (
                    "Look up where a specific PERSON, CAT, or named OBJECT is "
                    "right now from the persistent world model. Use this for "
                    "any 'where is X', 'is X home', 'is X still in the kitchen' "
                    "question. Returns last-seen room/landmark/timestamp + "
                    "current state (present, in_room_unseen, in_house_unmonitored, "
                    "departed, transitioning). PREFER this over guessing or "
                    "asking for a fresh snapshot — the world model is the "
                    "ground truth for presence."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": (
                                "Display name of the entity to look up. "
                                "Case-insensitive. Examples: 'Cole', 'Anna', "
                                "'Spooky', 'wallet'."
                            ),
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_entities_in_room",
                "description": (
                    "Return everyone (people, cats, named objects) currently "
                    "PRESENT in the given room. Use for 'who's in the "
                    "bedroom', 'is anyone in the kitchen'. Excludes entities "
                    "that have left or are hidden — only counts live "
                    "observations."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["room"],
                    "properties": {
                        "room": {
                            "type": "string",
                            "description": (
                                "Room ID — one of the configured rooms "
                                "(office, bedroom, kitchen, living_room, "
                                "laundry_room)."
                            ),
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "who_is_home",
                "description": (
                    "Return the residents currently considered 'home' — any "
                    "in-house state including under-desk hiding or in an "
                    "unmonitored bedroom. Use for 'is anyone home', 'who's "
                    "around', 'who else is in the house'."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_recent_events",
                "description": (
                    "Search the world model's event log for recent state "
                    "changes (entered/left/moved/disappeared/etc.). Use for "
                    "'when did Cole get home', 'has anyone been in the "
                    "kitchen tonight', 'what happened in the bedroom this "
                    "afternoon'. All filters are optional — combine them to "
                    "narrow the result."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_name": {
                            "type": "string",
                            "description": (
                                "Optional display name. Filters to events "
                                "for this entity."
                            ),
                        },
                        "room": {
                            "type": "string",
                            "description": (
                                "Optional room ID. Filters to events that "
                                "occurred in this room."
                            ),
                        },
                        "event_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional event type filter. Common values: "
                                "first_seen, name_linked, moved_to, "
                                "moved_within_room, lost_visibility, "
                                "reappeared, departed, entered_unmonitored, "
                                "stationary_long."
                            ),
                        },
                        "hours_ago": {
                            "type": "integer",
                            "description": (
                                "Search window in hours back from now. "
                                "Default 24."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max rows. Default 20.",
                        },
                    },
                },
            },
        },
        # ── Pet-aware tools (§22) ────────────────────────────────────────
        {
            "type": "function",
            "function": {
                "name": "where_is_pet",
                "description": (
                    "Pet-flavored 'where is X'. Use for 'where's Spooky', "
                    "'is Velcro in Jeff's room?', 'has Summer been outside?'. "
                    "Returns last-seen room/landmark + a `likely_room` field "
                    "that falls back to the pet's unmonitored_home (e.g. "
                    "Velcro → jeff_room) when no recent observation exists. "
                    "Phrase 'likely_room_inferred=true' results as a hedge "
                    "('probably in Jeff's room')."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": (
                                "Pet name. Case-insensitive. Examples: "
                                "'Spooky', 'Velcro', 'Summer', 'Dalila'."
                            ),
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_pets",
                "description": (
                    "List the resident pets the system knows about. Use for "
                    "'how many cats do I have?', 'what pets live here?'. "
                    "Optional `species` filter ('cat' or 'dog')."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "species": {
                            "type": "string",
                            "description": (
                                "Optional filter — 'cat' or 'dog'. Omit "
                                "for both."
                            ),
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "pet_care_summary",
                "description": (
                    "Per-pet summary of recent care-relevant events. Use for "
                    "'has Spooky used the litterbox today', 'when did Summer "
                    "last eat', 'is anyone going on a walk soon' (LEASH_INTERACTION). "
                    "Returns counts + last-seen timestamp grouped by "
                    "interaction_kind: litterbox_visit, food_dish_visit, "
                    "dog_food_visit, dog_water_visit, leash_interaction."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Pet name. Case-insensitive.",
                        },
                        "hours_ago": {
                            "type": "integer",
                            "description": (
                                "Lookback window. Default 24 hours."
                            ),
                        },
                    },
                },
            },
        },
        # ── Interaction tools (§24.5) ────────────────────────────────────
        {
            "type": "function",
            "function": {
                "name": "what_did_someone_do_with",
                "description": (
                    "Chronological list of interaction events involving a "
                    "person — INTERACTED_WITH, PICKED_UP, PLACED_DOWN. Use "
                    "for 'what did Cole do with the wallet today', 'has "
                    "Anna touched the keys'. Optional `object_name` filter "
                    "narrows to one object. Returned oldest-first so the "
                    "LLM can phrase it as a narrative."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["person_name"],
                    "properties": {
                        "person_name": {
                            "type": "string",
                            "description": "Resident name, case-insensitive.",
                        },
                        "object_name": {
                            "type": "string",
                            "description": (
                                "Optional object filter. Substring-matches "
                                "metadata.object_name when no entity exists."
                            ),
                        },
                        "hours_ago": {
                            "type": "integer",
                            "description": "Lookback window. Default 24h.",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "who_last_touched",
                "description": (
                    "Most recent interaction event for a tracked object. "
                    "Use for 'who last touched my wallet', 'who put this "
                    "down here'. Returns event_type + ts + person_name + "
                    "room. Returns {found: false} when the object isn't "
                    "in the world model's entity registry."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["object_name"],
                    "properties": {
                        "object_name": {
                            "type": "string",
                            "description": "Object's display name.",
                        },
                    },
                },
            },
        },
        # ── Object find (§23.7) ─────────────────────────────────────────
        {
            "type": "function",
            "function": {
                "name": "find_object",
                "description": (
                    "Find a tracked object by free-text description "
                    "(visual + semantic match via CLIP). Use for "
                    "'where's my wallet', 'find my keys', 'where did I "
                    "leave the leash', 'where's the brown box that came "
                    "yesterday'. Description can be specific "
                    "('a small leather wallet') or generic ('keys'). "
                    "Returns the best-match entity + alternatives. When "
                    "the response carries `hedge: true`, phrase the "
                    "answer as a guess ('I think it's...', 'might be in "
                    "the office') rather than a definite assertion."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["description"],
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": (
                                "Free-text describing the object. Best "
                                "results are concrete + visual: 'orange "
                                "pill bottle', 'brown leather wallet'."
                            ),
                        },
                        "k": {
                            "type": "integer",
                            "description": (
                                "How many alternatives to return "
                                "alongside the primary match. Default 3."
                            ),
                        },
                    },
                },
            },
        },
    ]

    def _world_tool_handlers(self) -> dict:
        """Bind each schema name to a coroutine that calls into
        WorldQueryTools. Tiny wrappers so the LLM tool-call dict shape
        (kwargs) lines up with the keyword params on each method.
        """
        wq = self.world_query_tools
        if wq is None:
            return {}

        async def _entity(name: str) -> dict:
            return await wq.get_entity_status(name)

        async def _in_room(room: str) -> list[dict]:
            return await wq.list_entities_in_room(room)

        async def _home() -> list[dict]:
            return await wq.who_is_home()

        async def _events(
            entity_name: Optional[str] = None,
            room: Optional[str] = None,
            event_types: Optional[list[str]] = None,
            hours_ago: int = 24,
            limit: int = 20,
        ) -> list[dict]:
            return await wq.search_recent_events(
                entity_name=entity_name, room=room,
                event_types=event_types, hours_ago=hours_ago, limit=limit,
            )

        async def _where_pet(name: str) -> dict:
            return await wq.where_is_pet(name)

        async def _list_pets(species: Optional[str] = None) -> list[dict]:
            return await wq.list_pets(species=species)

        async def _pet_care(name: str, hours_ago: int = 24) -> dict:
            return await wq.pet_care_summary(name, hours_ago=hours_ago)

        async def _did_with(
            person_name: str,
            object_name: Optional[str] = None,
            hours_ago: int = 24,
        ) -> list[dict]:
            return await wq.what_did_someone_do_with(
                person_name=person_name,
                object_name=object_name,
                hours_ago=hours_ago,
            )

        async def _last_touched(object_name: str) -> dict:
            return await wq.who_last_touched(object_name)

        async def _find_object(description: str, k: int = 3) -> dict:
            return await wq.find_object(description, k=k)

        return {
            "get_entity_status":         _entity,
            "list_entities_in_room":     _in_room,
            "who_is_home":               _home,
            "search_recent_events":      _events,
            "where_is_pet":              _where_pet,
            "list_pets":                 _list_pets,
            "pet_care_summary":          _pet_care,
            "what_did_someone_do_with":  _did_with,
            "who_last_touched":          _last_touched,
            "find_object":               _find_object,
        }

    _GET_SNAPSHOT_TOOL = {
        "type": "function",
        "function": {
            "name": "get_room_snapshot",
            "description": (
                "Capture a FRESH snapshot from a specific room's camera and "
                "return a description of what is currently visible there. "
                "Use this when you need up-to-the-second info about a room "
                "OTHER than the one the user is in (the user's own room "
                "frame is already attached to their message). Examples: "
                "checking on a pet in another room, verifying whether a "
                "person is still where you last saw them, looking for an "
                "object the user mentioned. Don't call this for the user's "
                "current room — that frame is already in the prompt."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "room": {
                        "type": "string",
                        "description": (
                            "Room ID to snapshot. Must be one of the "
                            "configured camera-equipped rooms (e.g. "
                            "'office', 'bedroom', 'kitchen', "
                            "'living_room', 'laundry_room')."
                        ),
                    },
                    "question": {
                        "type": "string",
                        "description": (
                            "Optional specific question to ask about the "
                            "image (e.g. 'is the cat on the couch?'). "
                            "Defaults to a general scene description."
                        ),
                    },
                },
                "required": ["room"],
            },
        },
    }

    async def _attach_room_snapshot(
        self, messages: list[dict], room: str
    ) -> None:
        """Mutate `messages` in place: capture a fresh JPEG from `room`'s
        camera and attach it to the LAST user message via the `images`
        field (Ollama-style multimodal). No-op if the room has no live
        camera, or if capture fails — voice still works without vision.

        Why per-turn rather than from the periodic vision loop: the loop
        runs on a 60s cadence and its output is a TEXT description. When
        the user actually speaks, we want the LLM to see the SCENE AT
        THAT MOMENT — pose, objects in hand, where the gaze is — none of
        which the periodic description can capture in time.
        """
        if self.cameras is None:
            return
        if room not in self.cameras.get_available_rooms():
            return
        try:
            frame = await self.cameras.capture_frame_async(room)
        except Exception as e:
            logger.debug(f"[Vision/prompt] capture for '{room}' raised: {e}")
            return
        if frame is None:
            return
        try:
            import cv2
            ok, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75]
            )
            if not ok:
                return
            img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        except Exception as e:
            logger.debug(f"[Vision/prompt] encode for '{room}' raised: {e}")
            return
        # Find the last user message and attach. Walking from the tail
        # because the prompt builder may have added trailing context
        # messages with other roles.
        for msg in reversed(messages):
            if msg.get("role") == "user":
                existing = msg.get("images") or []
                msg["images"] = list(existing) + [img_b64]
                logger.debug(
                    f"[Vision/prompt] attached fresh '{room}' frame "
                    f"({len(buf)} bytes) to user msg"
                )
                return

    async def _tool_get_room_snapshot(
        self, room: str, question: Optional[str] = None
    ) -> dict:
        """Tool handler for get_room_snapshot. Captures a fresh frame from
        the given room and runs a vision query against it. Returns a dict
        the LLM tool-loop will JSON-serialize for its next turn.
        """
        if self.cameras is None:
            return {"error": "camera manager unavailable"}
        if self.llm is None:
            return {"error": "LLM unavailable for vision query"}
        if room not in self.cameras.get_available_rooms():
            return {
                "error": f"room '{room}' has no live camera",
                "available_rooms": self.cameras.get_available_rooms(),
            }
        try:
            frame = await self.cameras.capture_frame_async(room)
        except Exception as e:
            return {"error": f"frame capture failed: {e}"}
        if frame is None:
            return {"error": f"camera in '{room}' returned no frame"}
        prompt = question or (
            "Describe what is currently visible in this scene. Be concise "
            "but include people, pets, notable objects, and activity."
        )
        try:
            description = await self.llm.vision_query(frame, prompt)
        except Exception as e:
            return {"error": f"vision query failed: {e}"}
        return {
            "room": room,
            "description": description,
            "question": prompt,
        }
