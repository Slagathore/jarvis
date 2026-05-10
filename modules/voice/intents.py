"""
JARVIS — Ambient Home AI
========================
Mission: Cross-cutting voice-intent regex parsers that don't belong to a single
         feature module. Currently: Do Not Disturb (DND) on/off + duration,
         and house-layout door-teaching commands. Each parser returns None on
         no-match so the caller can fall through to the LLM.

Modules: modules/voice/intents.py
Functions:
    parse_dnd(text) -> Optional[float]
        Returns duration in minutes for an "enable DND" command (0 means
        clear DND). None if `text` is not a DND command at all.
    parse_layout_command(text) -> Optional[dict]
        Returns a structured layout-teach intent ({"action": "...", ...})
        or None if the text isn't a layout command.
"""

import re
from typing import Optional

# "do not disturb for N units" / "shut up for N units" / "leave me alone for N units"
# "be quiet for N units" / "stop talking for N units"
# "DND for N units"
_RE_DND_ON_DURATION = re.compile(
    r"^(?:can you\s+|could you\s+|please\s+)*"
    r"(?:do not disturb|don'?t disturb|dnd|shut up|leave me alone|"
    r"be quiet|stop talking|silence yourself|don'?t bother me|quiet mode)"
    r"(?:\s+for\s+"
    r"(?P<n>\d+|a|an|one|two|three|four|five|ten|fifteen|twenty|thirty|forty[ -]?five|sixty|ninety)"
    r"\s+(?P<unit>seconds?|secs?|minutes?|mins?|hours?|hrs?|days?))?"
    r"\s*\.?\s*$",
    re.IGNORECASE,
)

# "you can talk again" / "DND off" / "stop DND" / "resume" / "talk to me again"
_RE_DND_OFF = re.compile(
    r"^(?:can you\s+|could you\s+|please\s+)*"
    r"(?:dnd off|stop dnd|cancel dnd|cancel do not disturb|"
    r"you can talk(?: again)?|talk to me(?: again)?|resume(?:\s+talking)?|"
    r"break the silence|come back)"
    r"\s*\.?\s*$",
    re.IGNORECASE,
)

_NUMBER_WORDS: dict[str, int] = {
    "a": 1, "an": 1,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30,
    "forty-five": 45, "forty five": 45, "fortyfive": 45,
    "sixty": 60, "ninety": 90,
}

_UNIT_MINUTES: dict[str, float] = {
    "second": 1 / 60, "seconds": 1 / 60, "sec": 1 / 60, "secs": 1 / 60,
    "minute": 1.0,    "minutes": 1.0,    "min": 1.0,    "mins": 1.0,
    "hour": 60.0,     "hours": 60.0,     "hr": 60.0,    "hrs": 60.0,
    "day": 1440.0,    "days": 1440.0,
}

# Default DND length when user just says "shut up" with no duration
_DEFAULT_DND_MINUTES = 30.0


def parse_dnd(text: str) -> Optional[float]:
    """
    Parse a DND voice intent.

    Returns:
        > 0.0  — Activate DND for this many minutes.
        0.0    — Clear DND (user told Jarvis to start talking again).
        None   — Not a DND intent; caller should fall through to LLM.
    """
    t = text.strip()
    if not t:
        return None

    if _RE_DND_OFF.match(t):
        return 0.0

    m = _RE_DND_ON_DURATION.match(t)
    if not m:
        return None

    n_raw = m.group("n")
    unit = m.group("unit")
    if n_raw is None or unit is None:
        # "shut up" with no duration → use default
        return _DEFAULT_DND_MINUTES

    n_lower = n_raw.lower().strip()
    n = int(n_lower) if n_lower.isdigit() else _NUMBER_WORDS.get(n_lower)
    if n is None:
        return _DEFAULT_DND_MINUTES
    minutes = n * _UNIT_MINUTES.get(unit.lower(), 1.0)
    return max(0.0, minutes)


# ── House-layout door teaching ────────────────────────────────────────────────
#
# The user teaches Jarvis the floor plan by walking to a door, optionally
# pointing at it, and naming the room it leads to. We only have to parse the
# language here — capturing the wrist landmark + persisting the entry happens
# in the orchestrator.
#
# Phrasings the regex tries to handle naturally (case-insensitive):
#   - "this door goes to the kitchen"
#   - "this is the door to the kitchen"
#   - "this door leads to the kitchen"
#   - "the door I'm pointing at goes to the kitchen"
#   - "I'm pointing at the door to the bedroom"
#   - "the kitchen door is right here"
#   - "save this as the door to the bathroom"
#   - "list the doors" / "what doors do you know"
#   - "forget the kitchen door" / "forget the door to the kitchen"
#   - "forget all doors here" / "clear the doors in this room"

_RE_LAYOUT_ADD = re.compile(
    r"^(?:and\s+|so\s+|hey\s+jarvis[,\s]+|jarvis[,\s]+)?"
    r"(?:"
    r"(?:save|remember|record|store|note|add)\s+(?:this\s+)?(?:as\s+)?(?:the\s+)?(?:door|exit|doorway)(?:\s+I'?m\s+pointing\s+at)?"
    r"|(?:this\s+(?:is\s+the\s+|is\s+a\s+)?(?:door|exit|doorway))"
    r"|(?:the\s+(?:door|exit|doorway)\s+(?:I'?m\s+pointing\s+at|right\s+(?:here|there)|here|there))"
    r"|(?:I'?m\s+pointing\s+at\s+(?:the\s+)?(?:door|exit|doorway))"
    r"|(?:the\s+(?P<short_room>[\w\s]+?)\s+(?:door|exit|doorway)\s+is\s+(?:right\s+)?(?:here|there))"
    r")"
    r"(?:\s+(?:goes|leads|opens|is\s+the\s+one)\s+(?:in)?to)?"
    r"(?:\s+to)?"
    r"(?:\s+the\s+|\s+)?"
    r"(?P<room>[\w\s]+?)"
    r"\s*[\.\?!]?\s*$",
    re.IGNORECASE,
)

_RE_LAYOUT_LIST = re.compile(
    r"^(?:hey\s+jarvis[,\s]+|jarvis[,\s]+)?"
    r"(?:(?:list|show|tell\s+me|what\s+are)\s+(?:the\s+|all\s+(?:the\s+)?)?(?:doors|exits|doorways)"
    r"|what\s+doors\s+(?:do\s+you\s+know|are\s+(?:there\s+)?(?:in\s+(?:here|this\s+room))?)"
    r"|do\s+you\s+know\s+(?:any\s+|the\s+)?(?:doors|exits))"
    r"(?:\s+(?:in|for)\s+(?:here|this\s+room))?"
    r"\s*[\.\?!]?\s*$",
    re.IGNORECASE,
)

_RE_LAYOUT_CLEAR_ALL = re.compile(
    r"^(?:hey\s+jarvis[,\s]+|jarvis[,\s]+)?"
    r"(?:forget|clear|remove|delete)\s+(?:all\s+(?:the\s+)?|every\s+|the\s+)?(?:doors|exits|doorways)"
    r"(?:\s+(?:in\s+(?:here|this\s+room)|here))?"
    r"\s*[\.\?!]?\s*$",
    re.IGNORECASE,
)

_RE_LAYOUT_CLEAR_ONE = re.compile(
    r"^(?:hey\s+jarvis[,\s]+|jarvis[,\s]+)?"
    r"(?:forget|remove|delete)\s+(?:the\s+)?(?:door\s+(?:to\s+(?:the\s+)?)?|"
    r"(?P<short_room>[\w\s]+?)\s+door)"
    r"(?P<room>[\w\s]+?)?"
    r"\s*[\.\?!]?\s*$",
    re.IGNORECASE,
)

# Light filler stripped from the end of room names: "the kitchen please" → "kitchen"
_ROOM_NAME_TRAILING_FILLER = re.compile(
    r"\s+(?:please|thanks|thank you|now|right now|alright|ok|okay)\s*$",
    re.IGNORECASE,
)


def _normalize_room_phrase(raw: Optional[str]) -> Optional[str]:
    """Trim the phrase the user spoke into a room-name candidate. Returns
    None for empty / clearly non-room phrases ("here", "there"). Caller is
    responsible for resolving against the actual config.yaml room list —
    we don't know room ids at parse time.
    """
    if not raw:
        return None
    s = _ROOM_NAME_TRAILING_FILLER.sub("", raw).strip(" .?!,")
    s = re.sub(r"^the\s+", "", s, flags=re.IGNORECASE).strip()
    if not s:
        return None
    if s.lower() in {"here", "there", "this room", "this", "that"}:
        return None
    return s


def parse_layout_command(text: str) -> Optional[dict]:
    """
    Parse a house-layout teaching command.

    Returns one of:
      {"action": "add",       "room_phrase": "<raw phrase>"}
      {"action": "list"}
      {"action": "clear_all"}
      {"action": "clear_one", "room_phrase": "<raw phrase>"}
      None — caller should fall through (DND, reminder, LLM).

    The orchestrator resolves room_phrase against the live config
    (so a typo / unknown room can be acknowledged + skipped instead
    of silently miscategorized).
    """
    if not text:
        return None
    t = text.strip()
    if not t:
        return None

    if _RE_LAYOUT_LIST.match(t):
        return {"action": "list"}

    if _RE_LAYOUT_CLEAR_ALL.match(t):
        return {"action": "clear_all"}

    m = _RE_LAYOUT_CLEAR_ONE.match(t)
    if m:
        # The phrase can come from either capture group — "the kitchen door"
        # vs "the door to the kitchen".
        phrase = _normalize_room_phrase(
            m.group("room") or m.group("short_room")
        )
        if phrase:
            return {"action": "clear_one", "room_phrase": phrase}

    m = _RE_LAYOUT_ADD.match(t)
    if m:
        phrase = _normalize_room_phrase(
            m.group("room") or m.group("short_room")
        )
        # "this is a door" / "the door is here" with no destination is too
        # ambiguous to record. Bail and let the LLM handle it.
        if phrase:
            return {"action": "add", "room_phrase": phrase}

    return None
