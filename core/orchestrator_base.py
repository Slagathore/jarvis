"""
JARVIS — Ambient Home AI
========================
Mission: OrchestratorMixin — shared base for the Orchestrator concern-mixins
         (audit roadmap D6 decomposition).

         core/orchestrator.py was a 5,879-line god-object, now split into
         concern-mixins (ToolsMixin, InitMixin, ConversationMixin, LoopsMixin)
         that Orchestrator inherits. Every method still runs `self.*` against
         the one concrete Orchestrator instance, so runtime is identical.

         A type checker, looking at a mixin in isolation, cannot see that the
         mixins are only ever combined into Orchestrator. Two hints fix that:

           1. Every Orchestrator.__init__ instance attribute is declared on
              this base as `Any`, in the PLAIN class body. They are
              annotation-only (no value) so they create no runtime attribute
              — `hasattr(OrchestratorMixin, "config")` stays False. They must
              NOT be nested inside `if TYPE_CHECKING:`: a checker treats a
              conditional class-body annotation as not-authoritative, so a
              sibling `is not None` check would still mark the attribute
              Optional. Plain class-body annotations are authoritative.
           2. `__getattr__` (TYPE_CHECKING-only) covers everything else —
              chiefly methods defined on sibling mixins. No runtime
              __getattr__ exists, so a genuine typo still raises.

         This list is generated from Orchestrator.__init__; add an attribute
         there, add it here too (or regenerate).

Modules: core/orchestrator_base.py
Classes: OrchestratorMixin
"""

from typing import TYPE_CHECKING, Any


class OrchestratorMixin:
    """Base for the Orchestrator concern-mixins. Carries no behavior — only
    type-checker hints that cross-mixin `self.*` access is valid."""

    # Instance attributes — all set in Orchestrator.__init__. Annotation-only
    # (no `=`), so no runtime attribute is created; declared Any so a mixin
    # can both read and assign them without the checker inferring a spurious
    # narrow / Optional type.
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
    anomaly_scorer: Any
    appliance_tracker: Any
    audio_classifier: Any
    audio_focus: Any
    belief_resolver: Any
    bus: Any
    calendar: Any
    cameras: Any
    clap_classifier: Any
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
    mcp_gateway: Any
    pattern_miner: Any
    pc_monitor: Any
    persona: Any
    personality: Any
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
    sound_event_classifier: Any
    speaker_id: Any
    speaker_manager: Any
    state_fusion: Any
    stt: Any
    triage_gate: Any
    tts: Any
    wake: Any
    wake_sources: Any
    webhooks: Any
    world_model: Any
    world_query_tools: Any
    world_store: Any
    wyze_cam_controls: Any

    if TYPE_CHECKING:
        # Fallback for anything not enumerated above — chiefly methods
        # defined on sibling mixins. Returns Any; no runtime __getattr__.
        def __getattr__(self, name: str) -> Any: ...
