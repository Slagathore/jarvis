"""
JARVIS — Ambient Home AI
========================
Mission: Lightweight regex parsers for natural-language calendar requests.
         Each parser returns None on no-match so the caller can fall through
         to the LLM. Anything more exotic than the patterns here just becomes
         a normal LLM conversation.

Modules: modules/agenda/intents.py
Functions:
    parse_calendar_query(text) -> Optional[str]
        Returns "today" | "tomorrow" | "upcoming" if `text` is a calendar query.
    parse_calendar_add(text)   -> Optional[tuple[str, datetime, datetime]]
        Returns (title, start, end) on a recognised "schedule X" request.
"""

import re
from datetime import datetime, time, timedelta
from typing import Optional

# ── Query patterns ──────────────────────────────────────────────────────────
_RE_QUERY_TODAY = re.compile(
    r"^(?:can you |could you |please )*"
    r"(?:what'?s |what is )?"
    r"(?:on |coming up on |scheduled for |in |on the )?"
    r"(?:my )?(?:calendar |agenda |schedule )?(?:for )?today\??$",
    re.IGNORECASE,
)
_RE_QUERY_TOMORROW = re.compile(
    r"^(?:can you |could you |please )*"
    r"(?:what'?s |what is )?"
    r"(?:on |coming up on |scheduled for |in |on the )?"
    r"(?:my )?(?:calendar |agenda |schedule )?(?:for )?tomorrow\??$",
    re.IGNORECASE,
)
_RE_QUERY_UPCOMING = re.compile(
    r"^(?:can you |could you |please )*"
    r"(?:what'?s |what is )?"
    r"(?:coming up|next on (?:my )?(?:calendar|agenda|schedule)|on (?:my )?(?:calendar|agenda|schedule))\??$",
    re.IGNORECASE,
)
_RE_QUERY_HAVE_ANYTHING = re.compile(
    r"^(?:can you |could you |please )*"
    r"(?:do i |have i got |is there )"
    r"(?:have )?(?:anything |any (?:meetings|events|appointments|calls)?)"
    r"(?: (?:today|coming up|on my calendar|scheduled))?\??$",
    re.IGNORECASE,
)


def parse_calendar_query(text: str) -> Optional[str]:
    """
    Return "today" | "tomorrow" | "upcoming" if the text is a calendar query.
    None otherwise.
    """
    t = text.strip().rstrip(".!?")
    if _RE_QUERY_TODAY.match(t):
        return "today"
    if _RE_QUERY_TOMORROW.match(t):
        return "tomorrow"
    if _RE_QUERY_UPCOMING.match(t) or _RE_QUERY_HAVE_ANYTHING.match(t):
        return "upcoming"
    return None


# ── Add patterns ────────────────────────────────────────────────────────────
# "schedule a meeting with X tomorrow at 3 pm"
# "add an event called X tomorrow at 3 pm"
# "put X on my calendar today at 4"
# "create a meeting for X at 5 pm tomorrow"
_RE_ADD = re.compile(
    r"^(?:can you |could you |please )*"
    r"(?:schedule|add|create|put|set up)\s+"
    r"(?:a |an |the )?"
    r"(?:meeting|event|appointment|call|reminder)?\s*"
    r"(?:(?:with|called|named|for|about|to)\s+)?"
    r"(?P<title>.+?)\s+"
    r"(?:on\s+)?"
    r"(?P<day>today|tomorrow)\s+"
    r"at\s+"
    r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?"
    r"\s*(?P<ampm>am|pm|a\.m\.|p\.m\.)?"
    r"\s*\.?\s*$",
    re.IGNORECASE,
)
# Same as above but day-then-event-and-time at the end (alternative ordering):
# "schedule X at 3 pm today"
_RE_ADD_REVERSED = re.compile(
    r"^(?:can you |could you |please )*"
    r"(?:schedule|add|create|put|set up)\s+"
    r"(?:a |an |the )?"
    r"(?:meeting|event|appointment|call|reminder)?\s*"
    r"(?:(?:with|called|named|for|about|to)\s+)?"
    r"(?P<title>.+?)\s+"
    r"at\s+"
    r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?"
    r"\s*(?P<ampm>am|pm|a\.m\.|p\.m\.)?"
    r"\s+(?P<day>today|tomorrow)"
    r"\s*\.?\s*$",
    re.IGNORECASE,
)


def parse_calendar_add(text: str) -> Optional[tuple[str, datetime, datetime]]:
    """
    Return (title, start, end) for a "schedule X today/tomorrow at HH[:MM] [am|pm]"
    request. End defaults to start + 1 hour. None if no match.
    """
    t = text.strip()
    m = _RE_ADD.match(t) or _RE_ADD_REVERSED.match(t)
    if not m:
        return None

    title = m.group("title").strip()
    if not title:
        return None

    hour = int(m.group("hour"))
    minute = int(m.group("minute") or 0)
    ampm = (m.group("ampm") or "").lower().replace(".", "").replace(" ", "")
    if ampm == "pm" and hour < 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return None

    today = datetime.now()
    base = today.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if m.group("day").lower() == "tomorrow":
        base += timedelta(days=1)
    elif base <= today:
        # "today at 3 pm" but it's already past 3 pm — assume tomorrow.
        base += timedelta(days=1)

    start = base
    end = base + timedelta(hours=1)
    return title, start, end
