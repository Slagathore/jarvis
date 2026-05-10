"""
JARVIS — Safety
===============
Mission: v4.1 §29.8 — Clown Alarm. Non-safety alarm; exists for morale.
         Triggers on `clown.detected` events (verbal STT-intent OR
         visual CLIP-zero-shot publishers, both bus-published).

         Architecturally identical to the other §29 alarms — same
         state machine, same audio dispatcher, same persistence. The
         differences are content (3-stage horn → TTS → calliope
         sequence with a curated YAML response pool + LLM improv
         slots) and priority (lowest of the four — audio-suppressed
         by ANY other alarm).

         Termination paths (per §29.8.4):
            1. Calliope finishes naturally → RESOLVED (primary)
            2. "Jarvis, stop alarm" → MUTED (5-min mute timer)
            3. "Jarvis, no clown alarms for 30 min" → SUPPRESSED
               (voice-requested cooldown via suppress_for_seconds)
            4. Higher-priority alarm preempts → audio cancelled, alarm
               stays in FIRING_AUDIO; resumes when audio frees up

         No automatic cooldown (per Cole's preference). No phone
         broadcast on fire (would be obnoxious; tunable via
         send_phone_notification, default false).

         Improv generation lives here (not a separate module) because
         it's tightly coupled to the response-pool dataclass + alarm
         lifecycle. The patch-patch's 3-layer fallback is implemented
         in `_generate_improv`.

Modules: modules/safety/alarms/clown.py
Classes: ClownResponse, ClownAlarm
Spec:    new 2.md §29.8 (v4.1 micropatch + patch-patch corrections).

#todo: Persona-flavor responses. Per §29.8 closing note, a `tone:
       glados` block in the YAML is the right path if persona-flavored
       improv is desired — branching in alarm logic is the wrong path.
       No code work needed here; just YAML edits.
"""
from __future__ import annotations

import asyncio
import random
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from modules.safety.alarms.alarm import Alarm
from modules.safety.alarms.state import AlarmState, AlarmType


# Paths (assets directory under the project root). Override via
# config.alarms.clown.* if Cole wants to point at a different YAML /
# prompt template.
_DEFAULT_RESPONSES_YAML = Path("assets/clown_responses.yaml")
_DEFAULT_PROMPT_STANDARD = Path("prompts/clown_improv.md")
_DEFAULT_PROMPT_SUPPLEMENT = Path("prompts/clown_improv_supplement.md")

# Patch-patch tunables — surface via §28.5g if Cole wants to retune.
MIN_EXAMPLES_FOR_GENERATION = 3   # below this, supplement from other styles
TARGET_EXAMPLE_COUNT = 5          # ideal example count in the prompt


@dataclass
class ClownResponse:
    """One entry from the response pool. `generate=True` entries are
    LLM improv slots; the actual text is generated at fire time using
    `style_seed` to pick few-shot examples from the curated pool."""
    id: str
    tone: str
    text: Optional[str] = None
    generate: bool = False
    style_seed: Optional[str] = None


@dataclass
class ImprovEvent:
    """One improv-generation event for the dashboard recent-fires card.
    Persisted in-memory; not in the alarm_fires table — that table is
    for state-machine fires, not the per-fire content generation."""
    ts: datetime
    style_seed: str
    outcome: str   # generated_standard | generated_with_supplement |
                   # fallback_zero_examples | fallback_generation_failed
    examples_used_count: int = 0
    cross_style_supplement_count: int = 0
    error: Optional[str] = None
    final_text: Optional[str] = None


class ClownAlarm(Alarm):
    """Lowest-priority §29.8 alarm. Subscribes to `clown.detected` and
    plays a horns-x3 → TTS → calliope sequence."""

    PRIORITY = 40   # lowest; fire=10, cat=20, door=30, clown=40
    ALARM_TYPE = AlarmType.CLOWN

    def __init__(
        self,
        bus: Any,
        audio: Any,                              # AlarmAudio instance
        notifier: Optional[Any] = None,
        store: Optional[Any] = None,
        llm: Optional[Any] = None,               # LLM switcher / Ollama client
        responses_yaml: Optional[Path] = None,
        prompt_template_path: Optional[Path] = None,
        prompt_supplement_path: Optional[Path] = None,
        recent_queue_depth: int = 10,
        improv_cache_ttl_seconds: float = 60.0,
        improv_generation_timeout_seconds: float = 4.0,
        improv_temperature: float = 0.9,
        horn_loop_count: int = 3,
        # Mute window for "Jarvis, stop alarm". Independent of
        # voice-requested cooldown which uses suppress_for_seconds.
        mute_seconds: float = 300.0,
    ) -> None:
        super().__init__(
            bus=bus, notifier=notifier, store=store,
            mute_seconds=mute_seconds,
        )
        self._audio = audio
        self._llm = llm
        self._responses_yaml = (
            responses_yaml or _DEFAULT_RESPONSES_YAML
        )
        self._prompt_path = (
            prompt_template_path or _DEFAULT_PROMPT_STANDARD
        )
        self._prompt_supplement_path = (
            prompt_supplement_path or _DEFAULT_PROMPT_SUPPLEMENT
        )
        self._horn_loop_count = int(horn_loop_count)
        self._recent_queue: deque[str] = deque(maxlen=int(recent_queue_depth))
        self._improv_cache_ttl = float(improv_cache_ttl_seconds)
        self._improv_timeout = float(improv_generation_timeout_seconds)
        self._improv_temperature = float(improv_temperature)
        # Cache of recent generations: style_seed → (ts, text). Lets a
        # rapid re-fire reuse the same generated joke without burning
        # an LLM call.
        self._improv_cache: dict[str, tuple[float, str]] = {}
        # In-memory tail of generation events for the dashboard.
        self._improv_events: deque[ImprovEvent] = deque(maxlen=50)
        # Voice-requested cooldown end-time (monotonic). Zero = no cooldown.
        self._cooldown_until: float = 0.0
        self._cooldown_reason: Optional[str] = None
        # Loaded pool — populated lazily / via reload_pool().
        self._pool: list[ClownResponse] = []
        self._last_response: Optional[ClownResponse] = None
        self._last_response_text: Optional[str] = None

    async def start(self) -> None:
        await self.bus.subscribe("clown.detected", self._on_detected)
        self.reload_pool()
        logger.info(
            f"[ClownAlarm] watching clown.detected; pool size="
            f"{len(self._pool)}, recent_queue_depth="
            f"{self._recent_queue.maxlen}"
        )

    # ── Pool loading ──────────────────────────────────────────────────────

    def reload_pool(self) -> int:
        """Read clown_responses.yaml from disk into self._pool. Returns
        count loaded. Failures log + leave the existing pool intact —
        the alarm should never be left with no responses available."""
        try:
            import yaml as _yaml
            text = self._responses_yaml.read_text(encoding="utf-8")
            data = _yaml.safe_load(text) or {}
        except Exception as e:
            logger.warning(
                f"[ClownAlarm] could not read response pool from "
                f"{self._responses_yaml}: {e}"
            )
            return len(self._pool)
        raw = data.get("responses") or []
        new_pool: list[ClownResponse] = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            try:
                new_pool.append(ClownResponse(
                    id=str(entry["id"]),
                    tone=str(entry.get("tone") or "unknown"),
                    text=entry.get("text"),
                    generate=bool(entry.get("generate", False)),
                    style_seed=entry.get("style_seed"),
                ))
            except KeyError as e:
                logger.warning(
                    f"[ClownAlarm] skipping malformed pool entry "
                    f"(missing {e})"
                )
        if new_pool:
            self._pool = new_pool
            logger.info(
                f"[ClownAlarm] reloaded pool: {len(new_pool)} entries "
                f"({sum(1 for r in new_pool if r.generate)} improv slots)"
            )
        return len(self._pool)

    # ── Bus handler ───────────────────────────────────────────────────────

    async def _on_detected(self, payload: dict) -> None:
        """`clown.detected` payload shape:
            {trigger: 'verbal'|'visual',
             confidence: float,
             evidence: str | None   (the STT phrase / visual prompt),
             room: str | None,
             ts: ISO}"""
        # Voice-requested cooldown check.
        if self._is_in_cooldown():
            logger.info(
                f"[ClownAlarm] suppressed (cooldown until "
                f"{self._cooldown_remaining():.0f}s; reason="
                f"{self._cooldown_reason or 'unspecified'})"
            )
            return
        # Pick a response BEFORE firing — generation can run in
        # parallel with the horns playback per §29.8.3.
        response_text = await self._pick_and_render_response()
        # Stash so dashboard / state row metadata can surface it.
        self._last_response_text = response_text
        await self.fire({
            "trigger": payload.get("trigger"),
            "evidence": payload.get("evidence"),
            "confidence": payload.get("confidence"),
            "room": payload.get("room"),
            "response_text": response_text,
        })
        # Kick off the audio sequence directly. The §29.6 dispatcher
        # priority logic is layered ON TOP — if a higher alarm is
        # active, on_alarm_state_change won't grant audio to the clown
        # alarm, and play_clown_sequence won't run. That's handled by
        # AlarmDispatcher; here we just request playback.
        if self._dispatcher is not None and self._dispatcher.audio_owner() == self.ALARM_TYPE:
            await self._audio.play_clown_sequence(
                announcement=response_text,
                on_complete=lambda: self.condition_cleared(
                    {"reason": "calliope_finished"}
                ),
                horn_loop_count=self._horn_loop_count,
            )

    # ── Voice cooldown (§29.8.4 termination path 3) ───────────────────────

    def suppress_for_seconds(
        self, seconds: float, reason: Optional[str] = None,
    ) -> None:
        """'Jarvis, no clown alarms for 30 min' → set a cooldown end-
        timestamp. Detections during the window are logged but don't
        fire. NOT the same as `suppress()` (which is the framework's
        SUPPRESSED-state primitive); cooldown is a softer rate-limit."""
        self._cooldown_until = time.monotonic() + max(0.0, float(seconds))
        self._cooldown_reason = reason
        logger.info(
            f"[ClownAlarm] cooldown set for {seconds:.0f}s "
            f"({reason or 'unspecified'})"
        )

    def suppress_indefinitely(self, reason: Optional[str] = None) -> None:
        """'Until I say so' / 'indefinitely' → very long cooldown that
        only `reenable()` clears. Implemented as a far-future timestamp
        so the same `is_in_cooldown` path handles both finite and
        infinite suppressions uniformly."""
        # ~10 years of monotonic time — effectively forever.
        self._cooldown_until = time.monotonic() + 10 * 365 * 24 * 3600
        self._cooldown_reason = reason or "indefinite"
        logger.info("[ClownAlarm] suppressed indefinitely")

    def reenable(self) -> None:
        """'Jarvis, re-enable clown alarm' → clear any active cooldown."""
        self._cooldown_until = 0.0
        self._cooldown_reason = None
        logger.info("[ClownAlarm] re-enabled")

    def _is_in_cooldown(self) -> bool:
        return self._cooldown_until > time.monotonic()

    def _cooldown_remaining(self) -> float:
        return max(0.0, self._cooldown_until - time.monotonic())

    # ── Response selection + improv generation ────────────────────────────

    async def _pick_and_render_response(self) -> str:
        """Random-with-dedup selection. If the pick is an improv slot,
        run the 3-layer fallback generation."""
        if not self._pool:
            # Same hardcoded last-resort message _select_curated_fallback
            # would emit — keeps the alarm from ever silently no-op'ing.
            return (
                "Clown detected. The response pool is empty. "
                "This is somehow more concerning than the clown."
            )
        candidates = [r for r in self._pool if r.id not in self._recent_queue]
        if not candidates:
            # Pool fully exhausted by recent use — reset the queue.
            self._recent_queue.clear()
            candidates = list(self._pool)
        chosen = random.choice(candidates)
        self._recent_queue.append(chosen.id)
        self._last_response = chosen

        if not chosen.generate:
            return chosen.text or ""

        # Improv slot → check cache first, otherwise generate.
        style_seed = chosen.style_seed or "meta"
        cached = self._improv_cache.get(style_seed)
        if cached is not None:
            cached_ts, cached_text = cached
            if (time.monotonic() - cached_ts) <= self._improv_cache_ttl:
                logger.debug(
                    f"[ClownAlarm] reusing cached improv for '{style_seed}'"
                )
                return cached_text
        text = await self._generate_improv(style_seed)
        self._improv_cache[style_seed] = (time.monotonic(), text)
        return text

    async def _generate_improv(self, style_seed: str) -> str:
        """LLM-improv generation with the patch-patch's 3-layer fallback.

        Layer 1 (≥ MIN_EXAMPLES_FOR_GENERATION in-style examples):
            standard few-shot generation.
        Layer 2 (1-2 in-style examples):
            supplement with cross-style examples up to
            TARGET_EXAMPLE_COUNT, mark them distinctly in the prompt
            so the model anchors on the target style.
        Layer 3 (0 in-style examples):
            log warning + fall back to a curated entry. LLM generation
            without style anchoring produces tonally drifting output;
            better to play an existing joke.
        """
        in_style = [
            r for r in self._pool
            if r.tone == style_seed and not r.generate and r.text
        ]
        in_style_count = len(in_style)

        # Layer 3: bail to curated.
        if in_style_count == 0:
            logger.warning(
                f"[ClownAlarm] improv requested for style '{style_seed}' "
                "but pool has zero curated entries in that style. "
                "Falling back to curated entry."
            )
            self._record_improv_event(
                style_seed=style_seed,
                outcome="fallback_zero_examples",
            )
            return self._select_curated_fallback(prefer_style=None).text or ""

        # Layer 1 vs Layer 2 path-fork.
        if in_style_count >= MIN_EXAMPLES_FOR_GENERATION:
            examples = random.sample(
                in_style, k=min(TARGET_EXAMPLE_COUNT, in_style_count),
            )
            prompt = self._build_improv_prompt(style_seed, examples)
            outcome_branch = "generated_standard"
            supplement_count = 0
            examples_count = len(examples)
        else:
            cross_style = [
                r for r in self._pool
                if r.tone != style_seed
                and not r.generate
                and r.tone != "improv"
                and r.text
            ]
            supplement_count = min(
                TARGET_EXAMPLE_COUNT - in_style_count, len(cross_style),
            )
            supplement = (
                random.sample(cross_style, k=supplement_count)
                if supplement_count > 0 else []
            )
            prompt = self._build_improv_prompt_with_supplement(
                style_seed=style_seed,
                in_style_examples=in_style,
                cross_style_examples=supplement,
            )
            outcome_branch = "generated_with_supplement"
            examples_count = in_style_count

        # LLM call with explicit error handling — original micropatch
        # documented timeout fallback but didn't actually try/except.
        try:
            text = await self._call_llm(prompt)
            text = (text or "").strip() or (
                self._select_curated_fallback(
                    prefer_style=style_seed,
                ).text or ""
            )
            self._record_improv_event(
                style_seed=style_seed,
                outcome=outcome_branch,
                examples_used_count=examples_count,
                cross_style_supplement_count=supplement_count,
                final_text=text,
            )
            return text
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(
                f"[ClownAlarm] improv generation failed for style "
                f"'{style_seed}': {e}. Falling back to curated."
            )
            self._record_improv_event(
                style_seed=style_seed,
                outcome="fallback_generation_failed",
                examples_used_count=examples_count,
                cross_style_supplement_count=supplement_count,
                error=str(e),
            )
            return self._select_curated_fallback(
                prefer_style=style_seed,
            ).text or ""

    async def _call_llm(self, prompt: str) -> str:
        """Route through whatever LLM client was injected. Tolerates
        a few different shapes (Ollama AsyncClient, our own wrapper,
        a raw `complete(prompt=, ...)` callable). Falls through to
        empty string if the client is None — caller treats that as a
        failure and bails to curated."""
        if self._llm is None:
            raise RuntimeError("no LLM client wired")

        # Preferred: dashboard llm_switcher with a `complete()` coro.
        # The cast keeps the type-checker happy — getattr returns
        # `object | None`, but at runtime we expect a coroutine
        # function, and asyncio.wait_for will raise if it isn't.
        from typing import Awaitable, cast
        complete = getattr(self._llm, "complete", None)
        if callable(complete):
            coro = cast(
                Awaitable[str],
                complete(
                    prompt=prompt,
                    max_tokens=80,
                    temperature=self._improv_temperature,
                ),
            )
            return await asyncio.wait_for(coro, timeout=self._improv_timeout)

        # Fallback: Ollama AsyncClient. The repo's existing chat path.
        chat = getattr(self._llm, "chat", None)
        if callable(chat):
            messages = [{"role": "user", "content": prompt}]
            chat_coro = cast(Awaitable[Any], chat(messages))
            result = await asyncio.wait_for(
                chat_coro, timeout=self._improv_timeout,
            )
            # Ollama AsyncClient.chat returns dict-ish; our wrapper
            # may return a plain string. Tolerate both.
            if isinstance(result, str):
                return result
            if isinstance(result, dict):
                msg = result.get("message") or {}
                if isinstance(msg, dict):
                    return str(msg.get("content", ""))
                return str(result.get("content", ""))
            return str(result)

        raise RuntimeError(
            f"unsupported LLM client interface: {type(self._llm).__name__}"
        )

    def _select_curated_fallback(
        self, prefer_style: Optional[str],
    ) -> ClownResponse:
        """Pick a curated (non-improv) entry, preferring `prefer_style`
        when any exist, otherwise any curated entry. Hardcoded last
        resort response if the pool is empty so the alarm never
        silently fails to produce output."""
        curated = [r for r in self._pool if not r.generate and r.text]
        if not curated:
            return ClownResponse(
                id="hardcoded_fallback",
                tone="meta",
                text=(
                    "Clown detected. The response pool is empty. "
                    "This is somehow more concerning than the clown."
                ),
                generate=False,
            )
        if prefer_style:
            in_style = [r for r in curated if r.tone == prefer_style]
            if in_style:
                return random.choice(in_style)
        return random.choice(curated)

    def _record_improv_event(
        self,
        style_seed: str,
        outcome: str,
        examples_used_count: int = 0,
        cross_style_supplement_count: int = 0,
        error: Optional[str] = None,
        final_text: Optional[str] = None,
    ) -> None:
        """Append an improv generation event to the in-memory tail.
        Surfaced by the dashboard's clown alarm card so when
        generations get weird (zero examples / thin pool / failures)
        Cole can see it without grepping logs."""
        self._improv_events.append(ImprovEvent(
            ts=datetime.now(timezone.utc),
            style_seed=style_seed,
            outcome=outcome,
            examples_used_count=examples_used_count,
            cross_style_supplement_count=cross_style_supplement_count,
            error=error,
            final_text=final_text,
        ))

    def recent_improv_events(self) -> list[dict]:
        """Dashboard read API — return the in-memory tail as dicts."""
        return [
            {
                "ts": e.ts.isoformat(),
                "style_seed": e.style_seed,
                "outcome": e.outcome,
                "examples_used_count": e.examples_used_count,
                "cross_style_supplement_count":
                    e.cross_style_supplement_count,
                "error": e.error,
                "final_text": e.final_text,
            }
            for e in self._improv_events
        ]

    def pool_summary(self) -> list[dict]:
        """Dashboard read API — list of pool entries with id/tone/text."""
        return [
            {
                "id": r.id,
                "tone": r.tone,
                "generate": r.generate,
                "style_seed": r.style_seed,
                "text": r.text,
            }
            for r in self._pool
        ]

    # ── Prompt building ───────────────────────────────────────────────────

    def _build_improv_prompt(
        self, style_seed: str, examples: list[ClownResponse],
    ) -> str:
        try:
            template = self._prompt_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(
                f"[ClownAlarm] could not read prompt template "
                f"{self._prompt_path}: {e}; using inline fallback"
            )
            template = (
                "Generate one over-the-top alarm announcement for a "
                "household AI. Style: {style_seed}. Examples:\n"
                "{examples}\nOutput only the response text."
            )
        examples_text = "\n".join(
            f"- {e.text}" for e in examples if e.text
        )
        return template.replace(
            "{style_seed}", style_seed
        ).replace(
            "{examples}", examples_text
        )

    def _build_improv_prompt_with_supplement(
        self,
        style_seed: str,
        in_style_examples: list[ClownResponse],
        cross_style_examples: list[ClownResponse],
    ) -> str:
        try:
            template = self._prompt_supplement_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(
                f"[ClownAlarm] could not read supplement template "
                f"{self._prompt_supplement_path}: {e}; falling back to "
                "standard prompt template"
            )
            return self._build_improv_prompt(
                style_seed, in_style_examples + cross_style_examples,
            )
        in_text = "\n".join(
            f"- {e.text}" for e in in_style_examples if e.text
        )
        cross_text = "\n".join(
            f"- ({e.tone}) {e.text}" for e in cross_style_examples if e.text
        )
        return template.replace(
            "{style_seed}", style_seed
        ).replace(
            "{in_style_examples}", in_text
        ).replace(
            "{cross_style_examples}", cross_text
        )

    # ── Alarm hooks ───────────────────────────────────────────────────────

    def _announcement(self, context: dict) -> tuple[str, str]:
        """Dispatcher consults this for the title-line shown on the
        speakers + the phone-alert body. The actual response text is
        already rendered in `context['response_text']`; if absent, fall
        back to the most-recent rendered text."""
        body_text = (
            context.get("response_text")
            or self._last_response_text
            or "Clown detected."
        )
        return (body_text, body_text)

    async def _condition_still_true(self) -> bool:
        """Mute-rearm decision. The clown alarm's "condition" is
        bounded by the audio file's duration — once calliope finishes,
        the condition has cleared. So if we hit the mute timer
        boundary and never received a calliope-finished signal, we're
        still mid-fire and should rearm. In practice the calliope
        completes well before the 5-min mute timer expires, so this
        almost always returns False."""
        return self.fire_id is not None and self.state != AlarmState.RESOLVED


# ── helpers (module-level so tests can call without instance) ──────────────


def parse_cooldown_phrase(phrase: str) -> tuple[float, str]:
    """Parse a natural-language cooldown phrase into (seconds, reason).

    Handles the §29.8.4 examples:
       'for an hour'              → 3600
       'for 30 minutes'           → 1800
       'for the rest of the day'  → seconds until midnight local
       'until tomorrow'           → seconds until tomorrow 00:00 local
       'until tomorrow morning'   → seconds until tomorrow 06:00 local
       'until I say so' / 'indefinitely' → -1.0 (sentinel; caller
       should call suppress_indefinitely instead of suppress_for_seconds)

    Returns (-1.0, reason) for the indefinite case so callers can
    branch cleanly. Caller is responsible for invoking the right
    suppress_* method."""
    import re
    p = phrase.lower().strip()
    if "indefinit" in p or "until i say" in p or "until cole say" in p:
        return -1.0, "indefinite"
    # Hours / minutes / seconds — match flat numbers + words.
    # Allow plural `s` and a trailing word boundary so "30 minutes"
    # parses identically to "30 minute" / "30 min".
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*(hours?|minutes?|seconds?|mins?|secs?|h|m|s)\b",
        p,
    )
    if m:
        n = float(m.group(1))
        unit = m.group(2).rstrip("s")
        if unit.startswith("h"):
            return n * 3600, f"for {n:g} hour(s)"
        if unit.startswith("min") or unit == "m":
            return n * 60, f"for {n:g} minute(s)"
        if unit.startswith("sec") or unit == "s":
            return n, f"for {n:g} second(s)"
    if "an hour" in p:
        return 3600.0, "for an hour"
    if "rest of the day" in p or "rest of today" in p:
        now = datetime.now()
        midnight = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0,
        )
        return (midnight - now).total_seconds(), "until midnight"
    if "until tomorrow morning" in p:
        # Default morning offset = 6 AM (configurable per §28.5g).
        now = datetime.now()
        target = (now + timedelta(days=1)).replace(
            hour=6, minute=0, second=0, microsecond=0,
        )
        return (target - now).total_seconds(), "until tomorrow morning"
    if "until tomorrow" in p:
        now = datetime.now()
        target = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0,
        )
        return (target - now).total_seconds(), "until tomorrow"
    # Couldn't parse — caller should fall back to a default duration.
    return 0.0, "unparsable"
