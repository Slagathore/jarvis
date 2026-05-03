"""
JARVIS — Ambient Home AI
========================
Mission: Monitor the Windows desktop to detect what Cole is actively doing on
         his PC. Reads the active foreground window title and process name using
         psutil + win32gui, then maps them to a Jarvis activity label via the
         process_activity_map and window_title_activity_map in config.yaml.

         This is the highest-confidence activity signal because it's exact —
         there's no ambiguity about what process is running.

Modules: modules/activity/pc_monitor.py
Classes: PCMonitor
Functions:
    PCMonitor.__init__(config)       — Initialize with config maps
    PCMonitor.get_signal()           — Blocking: get current activity signal
    PCMonitor.get_signal_async()     — Async wrapper via asyncio.to_thread
    PCMonitor._get_active_window()   — Get (process_name, window_title) via win32gui
    PCMonitor._classify_activity()   — Map process/title to activity label

Variables:
    PCMonitor._process_map         — {exe_name_lower: activity} from config
    PCMonitor._window_map          — {title_keyword_lower: activity} from config
    PCMonitor._default_activity    — "idle" if nothing matches

Signal dict format:
    {
        "activity":     "gaming",
        "process_name": "cs2.exe",
        "window_title": "Counter-Strike 2",
        "confidence":   0.9,
        "context":      {"window_title": "Counter-Strike 2", "process_name": "cs2.exe"}
    }

#todo: Add idle detection via win32api GetLastInputInfo (keyboard/mouse last input time)
#todo: Add multi-monitor awareness — which display is active?
#todo: Add virtual desktop tracking (Windows 10+ desktops)
#todo: Detect video call state via window title keywords ("Meeting", "Call") for accuracy
"""

import asyncio
from typing import Optional

from loguru import logger

# Guard against non-Windows environments gracefully
psutil = None
win32gui = None
win32process = None

try:
    import psutil
    import win32api
    import win32gui
    import win32process
    _WIN32_AVAILABLE = True
except ImportError:
    win32api = None  # type: ignore[assignment]
    _WIN32_AVAILABLE = False
    logger.warning("[PCMonitor] win32gui/psutil not available — PC monitor disabled")


class PCMonitor:
    """
    Windows-only activity detector that reads the active foreground window.

    On non-Windows systems (or if pywin32 is not installed), all calls
    return a low-confidence "unknown" signal so the rest of the system
    degrades gracefully.
    """

    def __init__(self, config: dict) -> None:
        # Build lookup maps with normalized lowercase keys
        raw_process_map = config.get("process_activity_map", {})
        self._process_map: dict[str, str] = {
            k.lower(): v for k, v in raw_process_map.items()
        }

        raw_window_map = config.get("window_title_activity_map", {})
        self._window_map: dict[str, str] = {
            k.lower(): v for k, v in raw_window_map.items()
        }

        self._default_activity: str = "idle"

        # OS-level idle threshold: above this many seconds with no keyboard or
        # mouse input the activity is overridden to "away" regardless of what
        # window is focused. A locked or unattended PC running Spotify
        # shouldn't read as "browsing_general" forever.
        ctx_cfg = config.get("context", {}) if isinstance(config.get("context"), dict) else {}
        self._away_idle_seconds: int = int(ctx_cfg.get("os_idle_away_seconds", 600))

    def get_signal(self) -> dict:
        """
        Blocking call to read the current active window.
        Returns a signal dict for StateFusion.
        """
        if not _WIN32_AVAILABLE:
            return self._idle_signal()

        try:
            process_name, window_title = self._get_active_window()
        except Exception as e:
            # Don't go to "unknown" on a transient win32 hiccup — state_fusion
            # filters out "unknown" signals which leaves no data and the dashboard
            # gauge stuck. "idle" is a valid fallback that still produces a
            # reasonable interruptibility score.
            logger.debug(f"[PCMonitor] Error reading active window: {e}")
            return self._idle_signal()

        # OS idle override — if the keyboard and mouse have been quiet for a
        # while, Cole is away regardless of which window is "focused". A
        # background Spotify or browser tab shouldn't keep activity at
        # "browsing_general" overnight.
        idle_seconds = self._os_idle_seconds()
        if idle_seconds is not None and idle_seconds >= self._away_idle_seconds:
            return {
                "activity":     "away",
                "process_name": process_name,
                "window_title": window_title,
                "confidence":   0.95,
                "context": {
                    "process_name": process_name,
                    "window_title": window_title,
                    "os_idle_seconds": int(idle_seconds),
                },
            }

        activity, confidence = self._classify_activity(process_name, window_title)

        signal = {
            "activity":     activity,
            "process_name": process_name,
            "window_title": window_title,
            "confidence":   confidence,
            "context": {
                "process_name": process_name,
                "window_title": window_title,
            },
        }
        logger.debug(
            f"[PCMonitor] '{process_name}' / '{window_title}' → {activity} ({confidence:.2f})"
        )
        return signal

    async def get_signal_async(self) -> dict:
        """Async wrapper — runs the blocking win32 call in a thread pool."""
        return await asyncio.to_thread(self.get_signal)

    def _get_active_window(self) -> tuple[str, str]:
        """
        Use win32gui to get the current foreground window handle,
        then look up its process name via psutil.

        Returns:
            Tuple of (process_name, window_title), both lowercased.
        """
        if (
            not _WIN32_AVAILABLE
            or psutil is None
            or win32gui is None
            or win32process is None
        ):
            raise RuntimeError("win32/psutil dependencies are unavailable")

        hwnd = win32gui.GetForegroundWindow()
        # hwnd == 0 means no window has focus (lock screen, transition, etc.)
        # GetWindowThreadProcessId(0) raises pywintypes.error — bail early so
        # the caller still gets a signal instead of falling to "unknown".
        if not hwnd:
            return "", ""

        try:
            window_title = win32gui.GetWindowText(hwnd) or ""
        except Exception:
            window_title = ""

        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
        except Exception:
            pid = 0

        process_name = ""
        if pid:
            try:
                proc = psutil.Process(pid)
                process_name = proc.name() or ""
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                process_name = ""
            except Exception:
                process_name = ""

        return process_name.lower(), window_title.lower()

    def _classify_activity(
        self,
        process_name: str,
        window_title: str,
    ) -> tuple[str, float]:
        """
        Map process name and window title to an activity label.

        Priority:
          1. Window title keyword match (most specific)
          2. Process name exact match
          3. Default activity ("idle")

        Returns:
            Tuple of (activity_label, confidence).
        """
        # Window title keyword match — highest confidence
        for keyword, activity in self._window_map.items():
            if keyword in window_title:
                return activity, 0.95

        # Process name exact match
        if process_name and process_name in self._process_map:
            return self._process_map[process_name], 0.90

        # Partial process name match (for processes listed without .exe in map)
        for exe_pattern, activity in self._process_map.items():
            if exe_pattern in process_name:
                return activity, 0.75

        return self._default_activity, 0.4

    @staticmethod
    def _os_idle_seconds() -> Optional[float]:
        """
        Return seconds since the last keyboard or mouse input, or None if
        unavailable (non-Windows, pywin32 missing, or call failed).
        Uses GetLastInputInfo + GetTickCount.
        """
        if not _WIN32_AVAILABLE or win32api is None:
            return None
        try:
            last_input = win32api.GetLastInputInfo()
            now = win32api.GetTickCount()
            elapsed_ms = now - last_input
            if elapsed_ms < 0:
                # Tick count wraps every ~49 days; just report 0 if we hit it
                return 0.0
            return elapsed_ms / 1000.0
        except Exception:
            return None

    @staticmethod
    def _idle_signal() -> dict:
        """
        Fallback signal when the monitor can't read the active window.
        Reports "idle" (not "unknown") because state_fusion filters out
        "unknown" signals — and we'd rather show "idle" than have the
        whole pipeline collapse to no signals at all.
        """
        return {
            "activity":     "idle",
            "process_name": "",
            "window_title": "",
            "confidence":   0.3,
            "context":      {},
        }
