"""
Jarvis notifications package.

- `NotificationManager` (legacy): in-dashboard bell + persistent toasts.
   Lives in `manager.py`. Re-exported here so existing
   `from modules.notifications import NotificationManager` keeps working.

- `Alert`, `NotificationDispatcher`, channel classes (new in v4): unified
   phone-alert pipeline for the §29 alarm subsystem (Fire / Cat-Escape /
   Door-Open). Three channels — ntfy (recommended primary), Telegram,
   Home Assistant — fan out in parallel; per-alarm-type routing.
"""
from modules.notifications.manager import NotificationManager
from modules.notifications.dispatcher import (
    Alert,
    AlertPriority,
    NotificationDispatcher,
)
from modules.notifications.channels import (
    HAChannel,
    NotificationChannel,
    NtfyChannel,
    TelegramChannel,
)

__all__ = [
    "NotificationManager",
    "Alert",
    "AlertPriority",
    "NotificationDispatcher",
    "NotificationChannel",
    "NtfyChannel",
    "TelegramChannel",
    "HAChannel",
]
