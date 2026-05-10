"""Alarm subsystem — multi-alarm framework with state machine + audio owner +
phone-alert dispatch.

Spec: new 2.md §29.
"""
from modules.safety.alarms.state import AlarmState, AlarmType
from modules.safety.alarms.alarm import Alarm
from modules.safety.alarms.audio import AlarmAudio, NullAlarmAudio
from modules.safety.alarms.dispatcher import AlarmDispatcher
from modules.safety.alarms.cat_escape import CatEscapeAlarm
from modules.safety.alarms.door_open import DoorOpenAlarm
from modules.safety.alarms.fire import FireAlarm
from modules.safety.alarms.clown import ClownAlarm, parse_cooldown_phrase
from modules.safety.alarms.klaxon import KlaxonLibrary
from modules.safety.alarms.store import AlarmStore, NullAlarmStore

__all__ = [
    "AlarmState",
    "AlarmType",
    "Alarm",
    "AlarmAudio",
    "NullAlarmAudio",
    "AlarmDispatcher",
    "CatEscapeAlarm",
    "DoorOpenAlarm",
    "FireAlarm",
    "ClownAlarm",
    "parse_cooldown_phrase",
    "KlaxonLibrary",
    "AlarmStore",
    "NullAlarmStore",
]
