"""
JARVIS — Ambient Home AI
========================
Mission: Read + write Cole's Google Calendar via the official Google Calendar API.
         OAuth 2.0 desktop flow on first run (opens browser once); refresh token
         is cached for all subsequent runs. All API calls run in a thread so the
         async event loop stays responsive.

Modules: modules/agenda/google_calendar.py
Classes: GoogleCalendar
Functions:
    GoogleCalendar.__init__(config)            — Load config and paths
    GoogleCalendar.authenticate()              — Async OAuth bootstrap
    GoogleCalendar.upcoming_events(hours)      — Events in the next N hours
    GoogleCalendar.add_event(title, start, ...) — Insert event
    GoogleCalendar.update_event(event_id, ...) — Patch event
    GoogleCalendar.delete_event(event_id)      — Remove event
    GoogleCalendar.is_authenticated            — Property: bool

Variables:
    GoogleCalendar._creds_path  — Path to OAuth client credentials.json
    GoogleCalendar._token_path  — Path to cached refresh token
    GoogleCalendar._calendar_id — Which calendar to operate on (default "primary")
    GoogleCalendar._timezone    — IANA timezone string for new events
    GoogleCalendar._service     — Built googleapiclient.discovery service
"""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger

# Google API readwrite scope — gives us list/create/update/delete on the user's calendars.
SCOPES = ["https://www.googleapis.com/auth/calendar"]


class GoogleCalendar:
    """
    Async wrapper around the Google Calendar API for Jarvis.

    Config (config["calendar"]):
        credentials_path: Path to OAuth client JSON downloaded from Google Cloud Console.
                          Default: "data/google_credentials.json"
        token_path:       Path where the refresh token gets cached after first auth.
                          Default: "data/google_token.json"
        calendar_id:      Which calendar to operate on. "primary" = the user's main calendar.
        timezone:         IANA timezone for new events (e.g. "America/Chicago").
    """

    def __init__(self, config: dict) -> None:
        cfg = config.get("calendar", {}) if isinstance(config.get("calendar"), dict) else {}
        self._creds_path = Path(cfg.get("credentials_path", "data/google_credentials.json"))
        self._token_path = Path(cfg.get("token_path", "data/google_token.json"))
        self._calendar_id: str = cfg.get("calendar_id", "primary")
        self._timezone: str = cfg.get("timezone", "America/Chicago")
        self._service: Optional[Any] = None
        self._authenticated: bool = False

    @property
    def is_authenticated(self) -> bool:
        return self._authenticated

    async def authenticate(self) -> bool:
        """
        Load cached token if present, else run the desktop OAuth flow.
        First-time auth opens a browser to Google's consent screen — Cole must
        click "Allow." Subsequent runs are silent (token auto-refreshes).

        Returns True on success, False if credentials file missing or auth fails.
        """
        if not self._creds_path.exists():
            logger.warning(
                f"[Calendar] No credentials at {self._creds_path} — calendar disabled. "
                "See README for OAuth setup."
            )
            return False

        try:
            self._service = await asyncio.to_thread(self._build_service_blocking)
            self._authenticated = True
            logger.info(f"[Calendar] Authenticated for calendar '{self._calendar_id}'")
            return True
        except Exception as e:
            logger.error(f"[Calendar] Authentication failed: {e}")
            return False

    def _build_service_blocking(self) -> Any:
        """
        Synchronous OAuth + service construction. Run in a thread by authenticate().
        Tries cached token first; falls back to a local-server browser flow on miss
        or when the token is unrefreshable.
        """
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build

        # Typed as Any because the loaded cred can be either oauth2.Credentials
        # or external_account_authorized_user.Credentials depending on file
        # provenance, and the InstalledAppFlow returns the same union — both
        # quack-compatibly support .valid / .refresh / .to_json.
        creds: Any = None

        if self._token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(self._token_path), SCOPES)
            except Exception as e:
                logger.warning(f"[Calendar] Could not load cached token ({e}) — re-authenticating")
                creds = None

        if creds is None or not creds.valid:
            if creds is not None and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    logger.warning(f"[Calendar] Token refresh failed ({e}) — re-authenticating")
                    creds = None

            if creds is None or not creds.valid:
                logger.info(
                    "[Calendar] Launching browser for OAuth — click 'Allow' in the page that opens"
                )
                flow = InstalledAppFlow.from_client_secrets_file(str(self._creds_path), SCOPES)
                # port=0 picks a random free port; Google's redirect uri must be
                # set to "http://localhost" in the OAuth client (Desktop app type
                # does this automatically).
                creds = flow.run_local_server(port=0, open_browser=True)

            if creds is None:
                raise RuntimeError("OAuth flow returned no credentials")
            self._token_path.parent.mkdir(parents=True, exist_ok=True)
            self._token_path.write_text(creds.to_json(), encoding="utf-8")
            logger.info(f"[Calendar] Token cached at {self._token_path}")

        # cache_discovery=False suppresses a noisy file-cache warning on Windows
        return build("calendar", "v3", credentials=creds, cache_discovery=False)

    async def upcoming_events(self, hours: float = 24, max_results: int = 25) -> list[dict[str, Any]]:
        """
        Return events starting between now and `hours` from now, soonest first.
        Each dict has: id, title, start, end, location, description, all_day.
        """
        service = self._service
        if service is None:
            return []
        now = datetime.now(timezone.utc)
        time_min = now.isoformat()
        time_max = (now + timedelta(hours=hours)).isoformat()

        def _list() -> list[dict[str, Any]]:
            resp = service.events().list(
                calendarId=self._calendar_id,
                timeMin=time_min,
                timeMax=time_max,
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
            ).execute()
            return resp.get("items", [])

        try:
            raw = await asyncio.to_thread(_list)
        except Exception as e:
            logger.warning(f"[Calendar] List events failed: {e}")
            return []

        return [self._normalize_event(e) for e in raw]

    async def add_event(
        self,
        title: str,
        start: datetime,
        end: Optional[datetime] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Insert an event. `start` must be timezone-aware or assumed in the
        configured local timezone. If `end` is omitted the event is 1h long.
        """
        service = self._service
        if service is None:
            return None
        if end is None:
            end = start + timedelta(hours=1)
        body = {
            "summary": title,
            "start": {"dateTime": start.isoformat(), "timeZone": self._timezone},
            "end":   {"dateTime": end.isoformat(),   "timeZone": self._timezone},
        }
        if description:
            body["description"] = description
        if location:
            body["location"] = location

        def _insert() -> dict[str, Any]:
            return service.events().insert(
                calendarId=self._calendar_id, body=body
            ).execute()

        try:
            created = await asyncio.to_thread(_insert)
            logger.info(f"[Calendar] Created event {created.get('id')}: {title!r}")
            return self._normalize_event(created)
        except Exception as e:
            logger.error(f"[Calendar] Create event failed: {e}")
            return None

    async def update_event(
        self,
        event_id: str,
        title: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Patch an existing event. Only fields you pass get updated."""
        service = self._service
        if service is None:
            return None
        body: dict[str, Any] = {}
        if title is not None:
            body["summary"] = title
        if start is not None:
            body["start"] = {"dateTime": start.isoformat(), "timeZone": self._timezone}
        if end is not None:
            body["end"] = {"dateTime": end.isoformat(), "timeZone": self._timezone}
        if description is not None:
            body["description"] = description
        if location is not None:
            body["location"] = location
        if not body:
            return None

        def _patch() -> dict[str, Any]:
            return service.events().patch(
                calendarId=self._calendar_id, eventId=event_id, body=body
            ).execute()

        try:
            updated = await asyncio.to_thread(_patch)
            logger.info(f"[Calendar] Updated event {event_id}")
            return self._normalize_event(updated)
        except Exception as e:
            logger.error(f"[Calendar] Update event failed: {e}")
            return None

    async def delete_event(self, event_id: str) -> bool:
        """Remove an event by id."""
        service = self._service
        if service is None:
            return False

        def _delete() -> None:
            service.events().delete(
                calendarId=self._calendar_id, eventId=event_id
            ).execute()

        try:
            await asyncio.to_thread(_delete)
            logger.info(f"[Calendar] Deleted event {event_id}")
            return True
        except Exception as e:
            logger.error(f"[Calendar] Delete event failed: {e}")
            return False

    @staticmethod
    def _normalize_event(raw: dict[str, Any]) -> dict[str, Any]:
        """Flatten Google's nested event shape into a dashboard-friendly dict."""
        start_obj = raw.get("start", {}) or {}
        end_obj = raw.get("end", {}) or {}
        all_day = "date" in start_obj and "dateTime" not in start_obj
        return {
            "id":          raw.get("id"),
            "title":       raw.get("summary", "(no title)"),
            "start":       start_obj.get("dateTime") or start_obj.get("date"),
            "end":         end_obj.get("dateTime") or end_obj.get("date"),
            "location":    raw.get("location"),
            "description": raw.get("description"),
            "all_day":     all_day,
            "html_link":   raw.get("htmlLink"),
        }
