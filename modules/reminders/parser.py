"""
JARVIS — Ambient Home AI
========================
Mission: Parse natural-language reminder requests like "remind me to take out
         the trash in 30 minutes" or "remind me to call mom at 5 pm" into
         (task, datetime) pairs. Returns None on no match so the caller can
         fall through to the regular LLM pipeline.

         Lightweight regex — no NLP dependency. Handles the common phrasings
         only. Anything more exotic falls through and the LLM can suggest the
         user rephrase.

Modules: modules/reminders/parser.py
Functions:
    parse_reminder(text) -> Optional[tuple[str, datetime]]
"""

import re
from datetime import datetime, timedelta
from typing import Optional

# "remind me to <task> in <N> <unit>"
# "set a reminder to <task> in <N> <unit>"
# "set a reminder for <task> in <N> <unit>"
# Optional leading "can you" / "please" softeners.
_RE_RELATIVE = re.compile(
    r"^(?:can you\s+|could you\s+|please\s+)*"
    r"(?:remind me to\s+|set (?:a |the )?reminder (?:to\s+|for\s+))"
    r"(?P<task>.+?)\s+in\s+"
    r"(?P<n>\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty|thirty|forty[ -]?five|sixty|ninety)"
    r"\s+(?P<unit>seconds?|secs?|minutes?|mins?|hours?|hrs?|days?)"
    r"\s*\.?\s*$",
    re.IGNORECASE,
)

# "remind me to <task> at <H>[:MM] [am|pm]"
_RE_AT = re.compile(
    r"^(?:can you\s+|could you\s+|please\s+)*"
    r"(?:remind me to\s+|set (?:a |the )?reminder (?:to\s+|for\s+))"
    r"(?P<task>.+?)\s+at\s+"
    r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?"
    r"\s*(?P<ampm>am|pm|a\.m\.|p\.m\.)?"
    r"\s*\.?\s*$",
    re.IGNORECASE,
)

_NUMBER_WORDS: dict[str, int] = {
    "a": 1, "an": 1,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "fifteen": 15, "twenty": 20, "thirty": 30,
    "forty-five": 45, "forty five": 45, "fortyfive": 45,
    "sixty": 60, "ninety": 90,
}

_UNIT_SECONDS: dict[str, int] = {
    "second": 1, "seconds": 1, "sec": 1, "secs": 1,
    "minute": 60, "minutes": 60, "min": 60, "mins": 60,
    "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600,
    "day": 86400, "days": 86400,
}


def parse_reminder(text: str) -> Optional[tuple[str, datetime]]:
    """
    Try to parse a reminder intent from a natural-language utterance.

    Returns (task, due_time) on match, None on no match.
    Accepts both digit and word numbers ("5" or "five"), optional politeness
    prefixes, optional trailing punctuation. Time-of-day forms roll forward
    to tomorrow if the requested time is already past today.
    """
    text = text.strip()
    if not text:
        return None

    m = _RE_RELATIVE.match(text)
    if m:
        return _from_relative(m)

    m = _RE_AT.match(text)
    if m:
        return _from_at(m)

    return None


def _from_relative(m: re.Match) -> Optional[tuple[str, datetime]]:
    task = m.group("task").strip()
    n_raw = m.group("n").lower().strip()
    n = int(n_raw) if n_raw.isdigit() else _NUMBER_WORDS.get(n_raw)
    if n is None:
        return None
    unit = m.group("unit").lower()
    secs = _UNIT_SECONDS.get(unit) or _UNIT_SECONDS.get(unit + "s")
    if not secs:
        return None
    return task, datetime.now() + timedelta(seconds=n * secs)


def _from_at(m: re.Match) -> Optional[tuple[str, datetime]]:
    task = m.group("task").strip()
    hour = int(m.group("hour"))
    minute = int(m.group("minute") or 0)
    ampm = (m.group("ampm") or "").lower().replace(".", "").replace(" ", "")
    if ampm == "pm" and hour < 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return None
    now = datetime.now()
    due = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    # If the time has already passed today, assume tomorrow.
    if due <= now:
        due += timedelta(days=1)
    return task, due
