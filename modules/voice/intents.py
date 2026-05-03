"""
JARVIS — Ambient Home AI
========================
Mission: Cross-cutting voice-intent regex parsers that don't belong to a single
         feature module. Currently: Do Not Disturb (DND) on/off + duration.
         Each parser returns None on no-match so the caller can fall through
         to the LLM.

Modules: modules/voice/intents.py
Functions:
    parse_dnd(text) -> Optional[float]
        Returns duration in minutes for an "enable DND" command (0 means
        clear DND). None if `text` is not a DND command at all.
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
