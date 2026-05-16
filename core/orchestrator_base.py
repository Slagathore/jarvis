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

         The catch a type checker hits: a method living in LoopsMixin that
         calls `self._speak(...)` (defined in LoopsMixin) and reads
         `self.config` (set in Orchestrator.__init__) looks like it's
         accessing unknown attributes — the checker can't see that the
         mixins are only ever combined into Orchestrator.

         This base fixes that with a single TYPE_CHECKING-only `__getattr__`
         stub: it tells the checker "any attribute on this class resolves to
         Any". Runtime is untouched — there is no real __getattr__, so a
         genuine typo still raises AttributeError as normal.

Modules: core/orchestrator_base.py
Classes: OrchestratorMixin
"""

from typing import TYPE_CHECKING, Any


class OrchestratorMixin:
    """Base for the Orchestrator concern-mixins. Carries no behavior — only
    the type-checker hint that cross-mixin `self.*` access is valid."""

    if TYPE_CHECKING:
        # Tells the type checker that any self.<attr> / self.<method>()
        # resolves (to Any). Mixin methods reference dozens of attributes
        # set in Orchestrator.__init__ and methods defined on sibling
        # mixins; enumerating them all would be noise. This is
        # TYPE_CHECKING-only — no runtime __getattr__ is installed.
        def __getattr__(self, name: str) -> Any: ...
