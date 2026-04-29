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
    import win32gui
    import win32process
    _WIN32_AVAILABLE = True
except ImportError:
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

    def get_signal(self) -> dict:
        """
        Blocking call to read the current active window.
        Returns a signal dict for StateFusion.
        """
        if not _WIN32_AVAILABLE:
            return self._unknown_signal()

        try:
            process_name, window_title = self._get_active_window()
        except Exception as e:
            logger.debug(f"[PCMonitor] Error reading active window: {e}")
            return self._unknown_signal()

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
        window_title = win32gui.GetWindowText(hwnd) or ""

        # Get the PID of the window's owning process
        _, pid = win32process.GetWindowThreadProcessId(hwnd)

        try:
            proc = psutil.Process(pid)
            process_name = proc.name() or ""
        except (psutil.NoSuchProcess, psutil.AccessDenied):
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
    def _unknown_signal() -> dict:
        """Return a no-data signal for when the monitor cannot operate."""
        return {
            "activity":     "unknown",
            "process_name": "",
            "window_title": "",
            "confidence":   0.0,
            "context":      {},
        }
