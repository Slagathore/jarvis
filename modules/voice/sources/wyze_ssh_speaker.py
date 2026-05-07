"""
JARVIS — Ambient Home AI
========================
Mission: Push TTS audio to a Wyze V2 cam's built-in speaker via SSH.
         The cam doesn't ship aplay or any standard ALSA tooling — wz_mini
         provides /opt/wz_mini/bin/audioplay_t20 (Ingenic T20 SoC) which
         talks to the proprietary IMP audio API via libimp.so. The path
         is: SFTP a WAV onto /tmp/, then exec the audioplay binary with
         LD_LIBRARY_PATH set so it finds the IMP shim.

         The hardware speaker is fixed at **8000 Hz / 16-bit / mono** —
         the cam will refuse anything else (or play it at the wrong speed,
         depending on firmware mood). We resample on the host side before
         shipping so the cam never sees off-rate audio.

         Persistent SSH connection — opening a fresh session per utterance
         adds ~500ms of TCP+auth overhead. Reuse one paramiko Transport
         per cam, reconnect lazily on failure.

         Half-duplex: while audioplay runs the cam's mic input is muted.
         The orchestrator already coordinates "stop talking when wake fires"
         at a higher layer, so we don't need barge-in detection here.

         Speaker quality is bad (~1cm cone, originally for delivery-driver
         barking). Cole knows; the toggle pattern means swapping in
         usb_device_spk or esp32_i2s_spk later is a one-line config change.

Modules: modules/voice/sources/wyze_ssh_speaker.py
Classes: WyzeSshSpeakerSink

#todo: Detect Wyze SoC variant (T20 vs T31) at first connect and pick the
       matching audioplay binary instead of hardcoding _t20. The Pan/V3
       cams use T31; future-proofs us for those.
#todo: Use a higher-quality resampler (scipy.signal.resample_poly) once
       scipy is on the dependency list. For TTS into a 1cm cone the
       linear interp is already inaudibly bad relative to the speaker
       itself, so this is purely a "nice to have."
"""

from __future__ import annotations

import asyncio
import io
import wave
from typing import Any, Optional

import numpy as np
from loguru import logger

from core.exceptions import AudioError
from modules.voice.sources.base import SpeakerSink

# wz_mini's audio binary. Hardcoded path because PATH on the cam doesn't
# include /opt/wz_mini/bin by default for non-interactive ssh exec sessions.
_AUDIOPLAY_BIN = "/opt/wz_mini/bin/audioplay_t20"
_LD_LIBRARY_PATH = "/opt/wz_mini/lib"

# The cam's audio output hardware is fixed at this sample rate. Confirmed
# by audioplay_t20's "Audio Out GetPubAttr samplerate:8000" output. Sending
# anything else either plays at the wrong speed or gets silently rejected.
_CAM_SPEAKER_RATE_HZ = 8000


class WyzeSshSpeakerSink(SpeakerSink):
    """SSH → SFTP → audioplay_t20 pipeline. Each play() blocks until the
    binary exits on the cam, so the caller's await resolves only after the
    audio actually finished playing — important for the orchestrator's
    TTS-then-listen cadence.
    """

    def __init__(
        self,
        room: str,
        host: str,
        ssh_user: str = "root",
        ssh_password: Optional[str] = None,
        ssh_key_path: Optional[str] = None,
        remote_play_path: str = "/tmp/jarvis_play.wav",
        volume: int = 60,
        sample_rate_hz: int = _CAM_SPEAKER_RATE_HZ,
        connect_timeout_s: float = 5.0,
    ) -> None:
        self._room = room
        self._host = host
        self._user = ssh_user
        # Empty string from .env is functionally equivalent to None — paramiko
        # treats both as "don't try password auth".
        self._password = ssh_password if ssh_password else None
        self._key_path = ssh_key_path if ssh_key_path else None
        self._remote_path = remote_play_path
        # audioplay_t20 takes 0-100. Default 60 is "audible across a small
        # room" without clipping; the speaker distorts noticeably above 80.
        self._volume = max(0, min(100, int(volume)))
        # The configured sample rate is what we resample TO. Forcing the
        # cam's native 8kHz means TTS output (typically 22050) gets
        # downsampled here once instead of by the cam at playback time.
        self._cam_rate = int(sample_rate_hz) if sample_rate_hz else _CAM_SPEAKER_RATE_HZ
        self._connect_timeout = connect_timeout_s
        # Lazy: paramiko import is slow and not every install needs it.
        self._client: Optional[Any] = None
        self._client_lock = asyncio.Lock()

    @property
    def room(self) -> str:
        return self._room

    # ── SSH connection management ────────────────────────────────────────────

    async def _ensure_connected(self) -> bool:
        """Open or refresh the SSH session. Returns False if connect fails so
        the caller can fall back to a different sink.
        """
        async with self._client_lock:
            if self._client is not None:
                # Probe transport — paramiko keeps the object around even
                # after the underlying socket dies on cam reboot.
                try:
                    transport = self._client.get_transport()
                    if transport is not None and transport.is_active():
                        return True
                except Exception:
                    pass
                # Stale — close and recreate
                self._safe_close()

            try:
                import paramiko
            except ImportError:
                logger.error(
                    f"[WyzeSpeaker:{self._room}] paramiko not installed — pip install paramiko"
                )
                return False

            client = paramiko.SSHClient()
            # AutoAddPolicy is acceptable for trusted-LAN cams; the alternative
            # would be a shared known_hosts file managed across all Jarvis cams.
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
                    f"[WyzeSpeaker:{self._room}] SSH connect to {self._host} failed: {e}"
                )
                return False

            self._client = client
            logger.info(
                f"[WyzeSpeaker:{self._room}] SSH connected to {self._user}@{self._host}"
            )
            return True

    def _safe_close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    # ── Playback ─────────────────────────────────────────────────────────────

    async def play(self, pcm: bytes, sample_rate: int) -> None:
        """Push PCM to the cam and wait for audioplay to finish. Raises
        AudioError on auth failure, transport drop, or non-zero binary exit
        — the SpeakerManager catches these and converts them into a False
        return so the smoke test (and any future fallback chain) can tell
        playback didn't actually happen.
        """
        if not pcm:
            return
        if not await self._ensure_connected():
            raise AudioError(
                f"[WyzeSpeaker:{self._room}] SSH unavailable to {self._host} — "
                "check WYZE_SSH_KEY_PATH in .env or that the cam's "
                "wz_mini/etc/ssh/authorized_keys has our public key"
            )

        # Resample to the cam's native rate, then wrap in WAV. Doing the
        # resample on the host saves the cam from having to do it (and the
        # cam often gets it wrong on off-rate input).
        resampled = self._resample_int16(pcm, sample_rate, self._cam_rate)
        wav_bytes = self._wrap_in_wav(resampled, self._cam_rate)

        # Two SSH operations: SFTP upload, then exec audioplay. Both blocking,
        # both safe to run inside a single to_thread call so we don't burn
        # two thread context-switches per utterance.
        try:
            await asyncio.to_thread(self._upload_and_play, wav_bytes)
        except AudioError:
            # Mark the connection stale; next play() will reconnect.
            self._safe_close()
            raise
        except Exception as e:
            self._safe_close()
            raise AudioError(
                f"[WyzeSpeaker:{self._room}] playback failed: {e}"
            ) from e

    def _upload_and_play(self, wav_bytes: bytes) -> None:
        """Sync helper that runs inside a thread. Uses paramiko's SFTP to
        push the WAV, then exec_command to run audioplay and waits for exit.
        """
        client = self._client
        if client is None:
            return
        sftp = client.open_sftp()
        try:
            with sftp.file(self._remote_path, "wb") as remote:
                remote.write(wav_bytes)
        finally:
            try:
                sftp.close()
            except Exception:
                pass

        # LD_LIBRARY_PATH must be set inline because the binary needs
        # libimp.so to find the proprietary Ingenic IMP audio API. The
        # cam's default PATH for ssh exec sessions doesn't include
        # /opt/wz_mini, so we use the absolute path. Volume is the second
        # positional arg (0-100). audioplay returns 0 on success.
        cmd = (
            f"LD_LIBRARY_PATH={_LD_LIBRARY_PATH} "
            f"{_AUDIOPLAY_BIN} {self._remote_path} {self._volume}"
        )
        _stdin, stdout, stderr = client.exec_command(cmd, timeout=30.0)
        rc = stdout.channel.recv_exit_status()
        if rc != 0:
            err = stderr.read().decode("utf-8", errors="replace")[:300]
            raise AudioError(
                f"[WyzeSpeaker:{self._room}] audioplay exit {rc}: {err.strip()}"
            )

    @staticmethod
    def _resample_int16(pcm_in: bytes, in_rate: int, out_rate: int) -> bytes:
        """Linear-interp resample of int16 PCM. Quality is fine for a 1cm
        cone — anything fancier (sinc, polyphase) would be inaudibly better
        through this hardware. Returns empty bytes if input is empty.
        """
        if not pcm_in:
            return b""
        if in_rate == out_rate:
            return pcm_in
        arr = np.frombuffer(pcm_in, dtype=np.int16)
        if arr.size == 0:
            return b""
        out_len = max(1, int(round(arr.size * out_rate / in_rate)))
        # endpoint=False + matching x ranges keeps the output the right
        # length and avoids a one-sample overshoot at the tail.
        x_in = np.linspace(0.0, 1.0, num=arr.size, endpoint=False, dtype=np.float64)
        x_out = np.linspace(0.0, 1.0, num=out_len, endpoint=False, dtype=np.float64)
        out = np.interp(x_out, x_in, arr.astype(np.float64))
        return np.clip(out, -32768, 32767).astype(np.int16).tobytes()

    @staticmethod
    def _wrap_in_wav(pcm: bytes, sample_rate: int) -> bytes:
        """Wrap raw int16 PCM in a minimal WAV header. audioplay_t20 reads
        the WAV header to set its decode rate, so the header's framerate
        must match the actual PCM data — we already resampled to match.
        """
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)  # int16 = 2 bytes
            w.setframerate(sample_rate)
            w.writeframes(pcm)
        return buf.getvalue()

    async def close(self) -> None:
        async with self._client_lock:
            self._safe_close()
        logger.debug(f"[WyzeSpeaker:{self._room}] Closed")
