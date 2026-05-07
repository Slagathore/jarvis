"""
JARVIS — Ambient Home AI
========================
Mission: SSH-driven control of Wyze V2 cameras for the settings that don't
         live in wz_mini.conf — the Wyze-app-equivalent toggles like night
         vision mode, IR LED enable, status indicator LED. These live in
         /configs/.parameters on the cam (Wyze's own config file, not
         wz_mini's) and are sparse — only keys that have been set at least
         once are present.

         wz_mini's web tool (cam.cgi) does in-place sed replacement, which
         silently no-ops when a key is missing. This helper handles both
         cases: append the key under [SETTING] when missing, sed-replace
         when present.

         Application semantics: changes write the value to disk
         immediately. Whether they take effect live depends on Wyze's
         iCamera daemon — some keys (bitrate, FPS) are picked up by the
         daemon's polling, others (nightVision, IR LEDs) require a cam
         reboot. The dashboard surfaces a "Reboot cam" button for the
         latter category.

         Why SSH instead of cam.cgi HTTP: the CGI returns a non-standard
         HTTP status line ("HTTP/1.1 200" with no reason phrase) that
         httpx rejects, AND the CGI can't add missing keys. SSH sidesteps
         both — we already have the key authentication wired through
         WyzeSshSpeakerSink, and we can do the conditional append cleanly
         inside one shell command.

Modules: modules/vision/wyze_cam_control.py
Classes: WyzeCamControl

#todo: Add a "settings have been changed and the cam needs a reboot to
       apply" indicator on the dashboard, populated by tracking which
       keys are reboot-required. Today every change is best-effort.
#todo: Detect when iCamera goes silent on the network for >10s after a
       set — likely it's restarting on its own (some keys do this) and
       the dashboard should show "applying…" instead of pretending it's done.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Optional

from loguru import logger

# /configs/.parameters is a tiny INI-style file with a single [SETTING]
# section. Keys we know about, with their valid value sets and a human
# label for the dashboard. Adding a new key is one entry here.
WYZE_PARAMS: dict[str, dict[str, Any]] = {
    "nightVision": {
        "label": "Night Vision",
        "valid": (1, 2, 3),
        "labels": {1: "On", 2: "Off", 3: "Auto"},
        # iCamera reads /configs/.parameters at boot, not live — verified
        # 2026-05-06: writing nightVision=2 while running persists to disk
        # but the IR cut filter doesn't actually flip until reboot.
        "reboot_required": True,
    },
    "night_cut_thr": {
        "label": "Auto threshold",
        "valid": (1, 2),
        "labels": {1: "Dusk", 2: "Dark"},
        "reboot_required": True,
    },
    "night_led_ex": {
        "label": "IR LEDs (near range)",
        "valid": (1, 2),
        "labels": {1: "On", 2: "Off"},
        "reboot_required": True,
    },
    "NIGHT_LED_flag": {
        "label": "IR LEDs (far range)",
        "valid": (1, 2),
        "labels": {1: "On", 2: "Off"},
        "reboot_required": True,
    },
    "indicator": {
        "label": "Front status LED",
        "valid": (1, 2),
        "labels": {1: "On", 2: "Off"},
        # Confirmed live-apply: setting indicator=2 visibly turned the
        # front status LED off within ~1s. Wyze handles LED state via a
        # different (faster) path than the IR / night-vision settings.
        "reboot_required": False,
    },
}

_PARAMS_PATH = "/configs/.parameters"
_SECTION_HEADER = "[SETTING]"


class WyzeCamControl:
    """One instance per Wyze cam room. Holds the SSH connection details
    inherited from the room's wyze_ssh_aplay speaker config (same host,
    same key) so we don't ask Cole to configure SSH twice.
    """

    def __init__(
        self,
        room: str,
        host: str,
        ssh_user: str = "root",
        ssh_password: Optional[str] = None,
        ssh_key_path: Optional[str] = None,
        connect_timeout_s: float = 5.0,
    ) -> None:
        self._room = room
        self._host = host
        self._user = ssh_user
        self._password = ssh_password if ssh_password else None
        self._key_path = ssh_key_path if ssh_key_path else None
        self._connect_timeout = connect_timeout_s
        self._client: Optional[Any] = None
        self._lock = asyncio.Lock()

    @property
    def room(self) -> str:
        return self._room

    @property
    def host(self) -> str:
        return self._host

    # ── SSH session (lazy, reused) ───────────────────────────────────────────

    async def _connect(self) -> Optional[Any]:
        """Open or reuse the SSH client. Returns None on failure (caller
        falls back gracefully — toggling night vision shouldn't crash the
        whole dashboard if a single cam is offline).
        """
        async with self._lock:
            if self._client is not None:
                try:
                    transport = self._client.get_transport()
                    if transport is not None and transport.is_active():
                        return self._client
                except Exception:
                    pass
                self._safe_close()

            try:
                import paramiko
            except ImportError:
                logger.error(f"[WyzeCam:{self._room}] paramiko not installed")
                return None

            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                await asyncio.to_thread(
                    client.connect,
                    hostname=self._host,
                    username=self._user,
                    password=self._password,
                    key_filename=self._key_path,
                    timeout=self._connect_timeout,
                    auth_timeout=self._connect_timeout,
                    banner_timeout=self._connect_timeout,
                    allow_agent=False,
                    look_for_keys=False,
                )
            except Exception as e:
                logger.warning(
                    f"[WyzeCam:{self._room}] SSH connect to {self._host} failed: {e}"
                )
                return None
            self._client = client
            return client

    def _safe_close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    async def close(self) -> None:
        async with self._lock:
            self._safe_close()

    # ── Read all known params ────────────────────────────────────────────────

    async def get_all(self) -> dict[str, Optional[int]]:
        """Read every known WYZE_PARAMS key from /configs/.parameters.
        Missing keys come back as None (= "default"). The dashboard
        renders None as "auto/default" instead of selecting a specific
        radio button.
        """
        client = await self._connect()
        if client is None:
            return {k: None for k in WYZE_PARAMS}
        try:
            _stdin, stdout, _stderr = await asyncio.to_thread(
                client.exec_command, f"cat {_PARAMS_PATH} 2>/dev/null", 8.0
            )
            raw = await asyncio.to_thread(stdout.read)
            text = raw.decode("utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"[WyzeCam:{self._room}] read params failed: {e}")
            self._safe_close()
            return {k: None for k in WYZE_PARAMS}

        out: dict[str, Optional[int]] = {}
        for key in WYZE_PARAMS:
            m = re.search(rf"^{re.escape(key)}=(\d+)\s*$", text, flags=re.MULTILINE)
            if m:
                try:
                    out[key] = int(m.group(1))
                except ValueError:
                    out[key] = None
            else:
                out[key] = None
        return out

    # ── Set one param ────────────────────────────────────────────────────────

    async def set_param(self, key: str, value: int) -> bool:
        """Replace `key=value` in the .parameters file, or append it under
        [SETTING] if missing. Returns False on SSH failure or if the
        key/value pair fails validation against WYZE_PARAMS.

        The shell command is intentionally one-shot:
            grep -q ...     → does the key already exist?
            ↪ yes           → sed in place
            ↪ no            → append after the [SETTING] header
        Doing this server-side as a single command avoids a round-trip
        race with another writer (the Wyze app, in theory).
        """
        spec = WYZE_PARAMS.get(key)
        if spec is None:
            logger.warning(f"[WyzeCam:{self._room}] unknown param '{key}'")
            return False
        try:
            iv = int(value)
        except (TypeError, ValueError):
            logger.warning(f"[WyzeCam:{self._room}] non-int value for '{key}': {value!r}")
            return False
        if iv not in spec["valid"]:
            logger.warning(
                f"[WyzeCam:{self._room}] invalid value {iv} for '{key}'; "
                f"expected one of {spec['valid']}"
            )
            return False

        client = await self._connect()
        if client is None:
            return False
        # The shell escaping here is safe — both `key` and `iv` come from a
        # constrained allowlist (WYZE_PARAMS keys + integer-validated
        # values). No user-supplied strings reach the shell.
        cmd = (
            f"if grep -q '^{key}=' {_PARAMS_PATH}; then "
            f"  sed -i 's/^{key}=.*/{key}={iv}/' {_PARAMS_PATH}; "
            f"else "
            f"  if grep -q '^\\[SETTING\\]' {_PARAMS_PATH}; then "
            f"    sed -i '/^\\[SETTING\\]/a {key}={iv}' {_PARAMS_PATH}; "
            f"  else "
            f"    printf '[SETTING]\\n{key}={iv}\\n' >> {_PARAMS_PATH}; "
            f"  fi; "
            f"fi; "
            f"echo OK"
        )
        try:
            _stdin, stdout, stderr = await asyncio.to_thread(
                client.exec_command, cmd, 10.0
            )
            rc = await asyncio.to_thread(stdout.channel.recv_exit_status)
            if rc != 0:
                err = (await asyncio.to_thread(stderr.read)).decode("utf-8", errors="replace")
                logger.warning(
                    f"[WyzeCam:{self._room}] set_param('{key}', {iv}) exit {rc}: {err.strip()}"
                )
                return False
            logger.info(f"[WyzeCam:{self._room}] set {key}={iv}")
            return True
        except Exception as e:
            logger.warning(f"[WyzeCam:{self._room}] set_param failed: {e}")
            self._safe_close()
            return False

    # ── Reboot the cam ───────────────────────────────────────────────────────

    async def reboot(self) -> bool:
        """Trigger a clean cam reboot. Comes back online ~25-40s later
        with the new /configs/.parameters values applied.

        Backgrounds the reboot command so the SSH session can exit
        cleanly before the kernel pulls the rug out — without the &,
        we get a connection-reset on the way down and the call
        appears to fail.
        """
        client = await self._connect()
        if client is None:
            return False
        try:
            await asyncio.to_thread(
                client.exec_command, "(/sbin/reboot &) >/dev/null 2>&1; exit 0", 6.0
            )
            logger.info(f"[WyzeCam:{self._room}] reboot triggered")
            # Drop the client — the cam is going away; the next call will
            # reconnect. Don't wait for the pipe to close (it won't).
            self._safe_close()
            return True
        except Exception as e:
            logger.warning(f"[WyzeCam:{self._room}] reboot failed: {e}")
            self._safe_close()
            return False
