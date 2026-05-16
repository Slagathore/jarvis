"""
JARVIS — Ambient Home AI
========================
Mission: OrchestratorMixin — shared base for the Orchestrator concern-mixins
         (audit roadmap D6 decomposition).

         core/orchestrator.py was a 5,879-line god-object. It is now split
         into concern-mixins — ToolsMixin, InitMixin, ConversationMixin,
         LoopsMixin — that Orchestrator inherits. Every method still uses
         `self.*` against the one concrete Orchestrator instance, so runtime
         behavior is identical.

         A type checker, looking at a mixin in isolation, cannot see that
         the mixins are only ever combined into Orchestrator. Two hints fix
         that, both TYPE_CHECKING-only (zero runtime effect — a real typo
         still raises AttributeError):

           1. Every Orchestrator.__init__ instance attribute is declared
              here as `Any`. A bare `__getattr__` is not enough: a mixin
              that *assigns* an attribute (e.g. `self._current_state = x`)
              makes the checker infer a narrow local type and skip
              __getattr__ entirely — which previously produced spurious
              "Never is not iterable" / "attribute of None" errors.
           2. `__getattr__` covers everything else — chiefly methods
              defined on sibling mixins.

         This list is generated from Orchestrator.__init__; if you add an
         instance attribute there, add it here too (or regenerate).

Modules: core/orchestrator_base.py
Classes: OrchestratorMixin
"""

from typing import TYPE_CHECKING, Any


class OrchestratorMixin:
    """Base for the Orchestrator concern-mixins. Carries no behavior — only
    the type-checker hints that cross-mixin `self.*` access is valid."""

    if TYPE_CHECKING:
        # Instance attributes — all set in Orchestrator.__init__. Declared
        # Any so a mixin can both read and assign them without the checker
        # inferring a spurious narrow type.
        _CALENDAR_TOOLS: Any
        _active_user_room: Any
        _active_user_room_ts: Any
        _audio_io_active: Any
        _bg_tasks: Any
        _calendar_alerted: Any
        _claude_client: Any
        _current_state: Any
        _followup_depth: Any
        _followup_tasks: Any
        _gather: Any
        _last_pets_per_room: Any
        _last_wake_audio: Any
        _pending_live_enroll: Any
        _pending_speaker_enrollment: Any
        _pending_wakes: Any
        _room_speech_until: Any
        _scene_state: Any
        _scene_tasks: Any
        _wake_lock: Any
        _wake_window_task: Any
        activity_history: Any
        alarm_dispatcher: Any
        anomaly_detector: Any
        appliance_tracker: Any
        audio_classifier: Any
        audio_focus: Any
        belief_resolver: Any
        bus: Any
        calendar: Any
        cameras: Any
        computer: Any
        config: Any
        curiosity: Any
        dashboard: Any
        db: Any
        door_map: Any
        event_log: Any
        face_recognizer: Any
        identity: Any
        integrations: Any
        interaction_monitor: Any
        interruptibility: Any
        light_detector: Any
        llm: Any
        memory: Any
        mess_detector: Any
        mic_manager: Any
        model_registry: Any
        mqtt: Any
        nodes: Any
        notification_dispatcher: Any
        notifications: Any
        object_detector: Any
        observation_builder: Any
        pc_monitor: Any
        persona: Any
        posture: Any
        prompts: Any
        reminder_scheduler: Any
        reminders_store: Any
        room_baselines: Any
        room_settings: Any
        scene_analyzer: Any
        selfedit: Any
        sessions: Any
        sleep_tracker: Any
        speaker_id: Any
        speaker_manager: Any
        state_fusion: Any
        stt: Any
        tts: Any
        wake: Any
        wake_sources: Any
        webhooks: Any
        world_model: Any
        world_query_tools: Any
        world_store: Any
        wyze_cam_controls: Any

        # Fallback for anything not enumerated above — chiefly methods
        # defined on sibling mixins. Returns Any; no runtime __getattr__.
        def __getattr__(self, name: str) -> Any: ...
