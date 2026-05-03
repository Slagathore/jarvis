"""Calendar integration (Google Calendar via OAuth)."""

from modules.agenda.google_calendar import GoogleCalendar
from modules.agenda.intents import parse_calendar_add, parse_calendar_query

__all__ = ["GoogleCalendar", "parse_calendar_add", "parse_calendar_query"]
