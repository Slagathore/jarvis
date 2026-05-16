"""
JARVIS — Ambient Home AI
========================
Mission: GeminiLiveSession — a client for Google's Gemini Live API
         (bidiGenerateContent over a WebSocket). This is a *speech-to-
         speech* path: stream microphone PCM in, receive synthesized
         audio + transcripts out, with server-side VAD and barge-in.

         WHY THIS EXISTS:
         `gemini-3.1-flash-preview` answers text/audio via REST
         (gemini_direct.py). `gemini-3.1-flash-live-preview` is a
         *different* model — it ONLY supports `bidiGenerateContent`,
         i.e. the WebSocket Live API. It does its own VAD, turn
         detection, and interruption, and returns native audio. On
         Google's free tier the Live API is unusually generous
         (unlimited RPM/RPD, 65k TPM at the time of writing) — but it
         is a *preview* surface that may change or be withdrawn, so
         every caller must be prepared to fall back to the local
         wake → STT → LLM → TTS stack (see modules/brain/voice_route).

         PROTOCOL (probed + confirmed 2026-05-16):
           setup            → {"setup": {...}}            → {"setupComplete": {}}
           text turn        → {"clientContent": {...}}
           audio in (live)  → {"realtimeInput": {"audio": {...}}}
           server out       → {"serverContent": {...}}    (audio / transcripts)
           tool call        → {"toolCall": {...}}
           session resume   → {"sessionResumptionUpdate": {...}}

         This module is transport only — it speaks the protocol and
         emits structured events. Persona (systemInstruction), voice
         (speechConfig), and tools are passed in by the caller; pipeline
         wiring lives elsewhere.

Modules: modules/brain/gemini_live.py
Classes: GeminiLiveSession, LiveEvent
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError:  # pragma: no cover - dependency guard
    websockets = None  # type: ignore
    ConnectionClosed = Exception  # type: ignore


_LIVE_HOST = "generativelanguage.googleapis.com"
_LIVE_PATH = (
    "/ws/google.ai.generativelanguage.v1beta"
    ".GenerativeService.BidiGenerateContent"
)
# Live API audio contract: input is 16 kHz mono PCM16, output is 24 kHz
# mono PCM16. Callers resample to/from these rates at the mic/speaker edge.
LIVE_INPUT_RATE = 16000
LIVE_OUTPUT_RATE = 24000


@dataclass
class LiveEvent:
    """One structured event from the Live session. `kind` is one of:

    audio              — `pcm` carries 24 kHz PCM16 output audio
    input_transcript   — `text` is a chunk of the user's transcribed speech
    output_transcript  — `text` is a chunk of the model's spoken reply
    tool_call          — `tool_calls` is a list of {id, name, args}
    interrupted        — the user barged in; stop playback immediately
    turn_complete      — the model finished its turn
    go_away            — the server is about to close (preview/quota); reconnect
    error              — `text` describes a transport/protocol failure
    closed             — the WebSocket closed
    """

    kind: str
    text: str = ""
    pcm: bytes = b""
    tool_calls: list[dict] = field(default_factory=list)


class GeminiLiveSession:
    """One Gemini Live conversation over a WebSocket.

    Lifecycle: construct → connect() → send_audio()/send_text() while
    consuming events() → close(). A session is single-conversation; the
    fallback router opens a fresh one per wake. Session-resumption tokens
    are surfaced so a long conversation can survive a server `goAway`.
    """

    def __init__(
        self,
        *,
        model: str = "models/gemini-3.1-flash-live-preview",
        api_key: Optional[str] = None,
        system_instruction: Optional[str] = None,
        voice: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        response_modality: str = "AUDIO",
        connect_timeout: float = 15.0,
    ) -> None:
        self._model = model if model.startswith("models/") else f"models/{model}"
        self._api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self._system_instruction = system_instruction
        self._voice = voice
        self._tools = tools or []
        self._response_modality = response_modality
        self._connect_timeout = connect_timeout
        self._ws: Any = None
        self._resumption_handle: Optional[str] = None
        self._closed = False

    @property
    def available(self) -> bool:
        """True if this session can even be attempted (deps + key present)."""
        return websockets is not None and bool(self._api_key)

    @property
    def resumption_handle(self) -> Optional[str]:
        """Latest session-resumption token, if the server has issued one."""
        return self._resumption_handle

    # ── Setup ────────────────────────────────────────────────────────────────

    def _build_setup(self) -> dict:
        gen_cfg: dict[str, Any] = {"responseModalities": [self._response_modality]}
        if self._voice:
            gen_cfg["speechConfig"] = {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": self._voice}
                }
            }
        setup: dict[str, Any] = {
            "model": self._model,
            "generationConfig": gen_cfg,
            # Always ask for both transcripts: the input transcript is the
            # free STT we route to memory; the output transcript is what
            # the dashboard / logs show the model said.
            "inputAudioTranscription": {},
            "outputAudioTranscription": {},
            # Let the server own VAD + barge-in detection.
            "realtimeInputConfig": {
                "automaticActivityDetection": {},
            },
            # Ask the server for resumption handles so a `goAway` mid-
            # conversation can be recovered instead of dropped.
            "sessionResumption": (
                {"handle": self._resumption_handle}
                if self._resumption_handle else {}
            ),
        }
        if self._system_instruction:
            setup["systemInstruction"] = {
                "parts": [{"text": self._system_instruction}]
            }
        if self._tools:
            setup["tools"] = [{"functionDeclarations": self._tools}]
        return {"setup": setup}

    async def connect(self) -> bool:
        """Open the WebSocket and complete the setup handshake.
        Returns False (rather than raising) on any failure so the caller
        can fall straight through to the backup route."""
        if not self.available:
            logger.warning(
                "[GeminiLive] unavailable — "
                f"websockets={'ok' if websockets else 'MISSING'}, "
                f"key={'set' if self._api_key else 'MISSING'}"
            )
            return False
        url = f"wss://{_LIVE_HOST}{_LIVE_PATH}?key={self._api_key}"
        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(url, max_size=16 * 1024 * 1024),
                timeout=self._connect_timeout,
            )
            await self._ws.send(json.dumps(self._build_setup()))
            raw = await asyncio.wait_for(self._ws.recv(), timeout=20.0)
            data = json.loads(_as_text(raw))
            if "setupComplete" not in data:
                logger.warning(f"[GeminiLive] no setupComplete: {data}")
                await self.close()
                return False
            logger.info(f"[GeminiLive] session up ({self._model})")
            return True
        except Exception as e:
            logger.warning(f"[GeminiLive] connect failed: {type(e).__name__}: {e}")
            await self.close()
            return False

    # ── Send ─────────────────────────────────────────────────────────────────

    async def send_audio(self, pcm16_16k: bytes) -> None:
        """Stream a chunk of 16 kHz mono PCM16 mic audio to the session."""
        if self._ws is None or self._closed:
            return
        b64 = base64.b64encode(pcm16_16k).decode("ascii")
        await self._ws.send(json.dumps({"realtimeInput": {
            "audio": {"data": b64, "mimeType": f"audio/pcm;rate={LIVE_INPUT_RATE}"},
        }}))

    async def send_text(self, text: str, *, end_turn: bool = True) -> None:
        """Send a text turn (used for tests and text-mode interactions)."""
        if self._ws is None or self._closed:
            return
        await self._ws.send(json.dumps({"clientContent": {
            "turns": [{"role": "user", "parts": [{"text": text}]}],
            "turnComplete": end_turn,
        }}))

    async def send_tool_response(self, responses: list[dict]) -> None:
        """Return tool results: each item {id, name, response:{...}}."""
        if self._ws is None or self._closed:
            return
        await self._ws.send(json.dumps({"toolResponse": {
            "functionResponses": responses,
        }}))

    # ── Receive ──────────────────────────────────────────────────────────────

    async def events(self):
        """Async-iterate structured LiveEvents until the session closes.
        Never raises — a transport failure is surfaced as an `error`
        event followed by `closed`, so the caller's loop ends cleanly.

        Note: the terminal `error`/`closed` events are yielded from the
        main body, never from a `finally` — yielding while a
        GeneratorExit is unwinding (which happens the moment a consumer
        `break`s out of the loop) is illegal and raises
        "async generator ignored GeneratorExit"."""
        if self._ws is None:
            yield LiveEvent(kind="closed")
            return
        err: Optional[str] = None
        try:
            async for raw in self._ws:
                data = json.loads(_as_text(raw))
                for ev in self._parse(data):
                    yield ev
        except ConnectionClosed as e:
            err = f"connection closed: {e}"
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        self._closed = True
        if err:
            yield LiveEvent(kind="error", text=err)
        yield LiveEvent(kind="closed")

    def _parse(self, data: dict) -> list[LiveEvent]:
        out: list[LiveEvent] = []
        # Session-resumption handle — store it; not a caller-facing event.
        sru = data.get("sessionResumptionUpdate")
        if sru and sru.get("resumable") and sru.get("newHandle"):
            self._resumption_handle = sru["newHandle"]

        if "goAway" in data:
            out.append(LiveEvent(kind="go_away"))

        sc = data.get("serverContent") or {}
        if sc.get("interrupted"):
            out.append(LiveEvent(kind="interrupted"))
        it = sc.get("inputTranscription") or {}
        if it.get("text"):
            out.append(LiveEvent(kind="input_transcript", text=it["text"]))
        ot = sc.get("outputTranscription") or {}
        if ot.get("text"):
            out.append(LiveEvent(kind="output_transcript", text=ot["text"]))
        for p in (sc.get("modelTurn", {}).get("parts") or []):
            inl = p.get("inlineData") or {}
            if inl.get("data"):
                try:
                    out.append(LiveEvent(
                        kind="audio", pcm=base64.b64decode(inl["data"])))
                except Exception:
                    pass
        if sc.get("turnComplete"):
            out.append(LiveEvent(kind="turn_complete"))

        tc = data.get("toolCall") or {}
        calls = tc.get("functionCalls") or []
        if calls:
            out.append(LiveEvent(kind="tool_call", tool_calls=[
                {"id": c.get("id"), "name": c.get("name"),
                 "args": c.get("args") or {}}
                for c in calls
            ]))
        return out

    # ── Teardown ─────────────────────────────────────────────────────────────

    async def close(self) -> None:
        self._closed = True
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None


def _as_text(raw: Any) -> str:
    return raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
