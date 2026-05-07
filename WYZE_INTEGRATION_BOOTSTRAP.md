# JARVIS — WYZE V2 INTEGRATION BOOTSTRAP

> **Read this entire document before writing a single line of code.**
> **Owner:** Cole
> **Status:** Ready to build — handed off from a Cowork planning session.
> **Companion doc:** `JARVIS_BOOTSTRAP.md` (still authoritative for project-wide style rules).

---

## 0. WHAT THIS DOCUMENT IS

This is a **focused addendum** to `JARVIS_BOOTSTRAP.md`. The original bootstrap
treated Wyze cams as pure video. The user has since decided to wire the Wyze V2's
as **full A/V terminals** (video + mic + speaker) for bedroom, kitchen, and
living room — replacing the ESP32-CAM nodes in those rooms. ESP32 nodes survive
as cheap fallback cameras for blind spots.

This doc captures every architectural decision, every concrete spec, and every
implementation pattern needed to build the integration end-to-end without
re-deriving anything.

When you finish a section, immediately move to the next. Build in the order
defined below. Order matters — later files import earlier ones.

---

## 1. THE DECISION (and the WHY)

### What we're building
Each room declares **three independent I/O channels** in `config.yaml`:
`video`, `mic`, `speaker`. Each channel has a `type:` field that selects which
driver class handles it. To upgrade a room (e.g. swap the awful Wyze speaker for
a real USB DAC later), you change ONE `type:` value in YAML — no Python edits.

This is the **driver/source dispatch pattern**. Manager classes (`CameraManager`,
`MicManager`, `SpeakerManager`) are factories that instantiate the right driver
subclass based on config `type`. Each driver implements a small abstract
interface. The rest of Jarvis never knows or cares whether a room's mic is a USB
device, a Wyze RTSP stream, or an ESP32 over MQTT.

### Why this pattern (vs. alternatives)
| Approach | Pros | Cons |
|---|---|---|
| **Dispatch on `type:` (this plan)** | One source of truth (YAML); upgrades are config edits; new drivers are isolated files; trivially testable per-driver. | Slightly more boilerplate (one ABC, one factory). |
| Single `camera_source` string with auto-detection (the JARVIS_BOOTSTRAP draft) | Less code | Audio has no equivalent "URL prefix" — you can't auto-detect mic vs. speaker types. Mixing visual auto-detect with audio explicit-type would be inconsistent. |
| Hardcoded if/else in each manager | Quick to write | Untestable; impossible to add types without editing the manager; violates "config over code" rule. |

### Why full-Wyze-A/V despite the speaker quality
The user knows the Wyze speaker is bad. They asked for it anyway, with the
explicit caveat "make it easy to swap later." That's exactly what the toggle
pattern delivers. Build the `wyze_ssh_aplay` driver to spec, but write it
assuming it'll be replaced inside 6 months — keep it isolated, keep its
config knobs in YAML, leave a `# todo:` flag pointing to the upgrade path.

### Speaker reality check (already communicated to user)
- Wyze V2 speaker is ~1cm cone, originally for "yelling at delivery people"
- Volume tops out at ~"loud whisper"; won't fill a kitchen
- Half-duplex (mic mutes during playback)
- Playback path is SSH → SCP → `aplay` (1-3s latency)
- `wz_mini_hacks` audio-playback breaks on some firmware revs

User accepted these tradeoffs in exchange for fewer devices per room TODAY,
and the toggle pattern means the upgrade is a one-line change LATER.

---

## 2. CONCRETE SPECIFICS

### Wyze RTSP URL pattern (CRITICAL — this differs from the original bootstrap)

```
rtsp://${WYZE_RTSP_USER}:${WYZE_RTSP_PASSWORD}@<IP>:8554/video6_unicast
```

**Note the differences from JARVIS_BOOTSTRAP.md:**
- Port is **8554**, not 554. (8554 is the wz_mini_hacks HD endpoint.)
- Path is **`/video6_unicast`**, not `/unicast`. (Newer wz_mini_hacks builds.)

If a future cam uses a different wz_mini_hacks build, the path may change to
`/unicast` or `/live`. Treat the URL as fully-configurable per-room — never
construct it in code.

### Cameras and their state
| Room | Status | IP | Notes |
|---|---|---|---|
| Bedroom (or whichever room cam #1 ends up in) | **Flashed & verified** | `192.168.1.134` | RTSP confirmed working |
| Kitchen | Pending flash | TBD | Placeholder `192.168.1.135` in config — UPDATE on flash |
| Living Room | Pending flash | TBD | Placeholder `192.168.1.136` in config — UPDATE on flash |
| Office | N/A — keeps USB webcam | n/a | Don't touch, lowest-latency setup |
| Laundry Room | Existing config row, decision pending | n/a | Keep stub or delete? Ask user. |

### Credentials (LIVE, do not commit to git)
RTSP user/pass: `cole` / `admin`
SSH user: `root` (default for wz_mini_hacks)
SSH pass: per-camera, set in each `wz_mini.conf`

**Security note**: `admin` is a 5-character dictionary word. Acceptable on a
trusted home LAN for now; flag a TODO to rotate to a 16+ char random password
once the system is daily-use stable.

These go in `.env` (gitignored), referenced from `config.yaml` via `${VAR}`
interpolation handled by the typed config loader.

---

## 3. CURRENT REPO STATE (snapshot at handoff)

### Already done in the Cowork session (verify before re-doing)
- `requirements.txt` — added `av>=12.3.0`, `paramiko>=3.4.0`, `pydantic>=2.8.0`.
  Verify these are present; if so, skip re-adding.
- `.env.example` — added `WYZE_RTSP_USER`, `WYZE_RTSP_PASSWORD`, `WYZE_SSH_USER`,
  `WYZE_SSH_PASSWORD`, `WYZE_SSH_KEY_PATH`. Verify.

### NOT done — start here
- `config.yaml` — currently uses the OLD flat schema (`camera_source`, `has_node`,
  `node_ip`, `speaker_sink`). **Migrate** to the new toggled schema (see §4).
- `core/config.py` — does not exist; build it.
- `modules/vision/camera_manager.py` — does not exist.
- `modules/vision/sources/` — directory does not exist.
- `modules/voice/mic_manager.py`, `speaker_manager.py` — do not exist.
- `modules/voice/sources/` — directory does not exist.
- `dashboard/server.py` — does not exist.
- `scripts/test_wyze.py` — does not exist.

### Existing code style (FOLLOW EXACTLY — see `core/event_bus.py` for reference)
- Top-of-file docstring: Mission / Architecture / Modules / Classes / Variables / `#todo:`
- Section dividers like `# ── Public API ─────────────────────────────────────`
- `loguru.logger` with `[ModuleName]` prefix, never `print()`
- `async def` everywhere there's I/O; `asyncio.to_thread()` wraps blocking calls
- Type hints on every function signature (no exceptions)
- Custom exceptions from `core/exceptions.py` — add new ones if needed

---

## 4. CONFIG SCHEMA — THE TOGGLE TABLE

### Migration: old → new
The existing `rooms:` block uses flat fields:
```yaml
- id: bedroom
  camera_source: null
  has_node: false
  speaker_sink: local
```

Replace with the **three-channel** structure. Other Jarvis modules currently
(probably) read `camera_source` / `has_node` / `speaker_sink` directly. **Grep
the codebase** for those keys and update each call site to use
`room.video.type`, `room.video.url`, `room.speaker.type`, etc.

### Available types (the toggle dropdown)

**video.type:**
- `wyze_rtsp` — Wyze V2 + wz_mini_hacks. Fields: `url`, `transport` (tcp|udp).
- `esp32_http` — ESP32-CAM MJPEG over HTTP. Fields: `url`.
- `usb_index` — USB webcam on the PC. Fields: `device_index` (int).
- `none` — Vision disabled in this room. `grab_frame()` returns None.

**mic.type:**
- `wyze_rtsp_audio` — Demux audio from a Wyze RTSP stream via PyAV. Fields: `url`,
  `transport`, `sample_rate_hz`, `channels`. Reuses the same URL as video.
- `esp32_i2s` — ESP32 INMP441 mic relayed over MQTT. Fields: `mqtt_topic`.
- `usb_device` — sounddevice/PortAudio. Fields: `device_name` (substring match)
  or `device_index`, `sample_rate_hz`, `channels`.
- `none` — No mic in this room.

**speaker.type:**
- `wyze_ssh_aplay` — SSH → SCP → `aplay` to a Wyze cam. Fields: `host`, `ssh_user`,
  `ssh_password`, `ssh_key_path`, `remote_play_path`, `aplay_device`,
  `sample_rate_hz`.
- `esp32_i2s` — ESP32 MAX98357A speaker over MQTT. Fields: `mqtt_topic`.
- `usb_device` — sounddevice playback on a PC output. Fields: `device_name`,
  `sample_rate_hz`, `channels`.
- `none` — Silent room.

### Final config.yaml `rooms:` block (target state)

```yaml
rooms:
  - id: "office"
    display_name: "Office"
    video:
      type: "usb_index"
      device_index: 0
    mic:
      type: "usb_device"
      device_name: "default"
      sample_rate_hz: 16000
      channels: 1
    speaker:
      type: "usb_device"
      device_name: "default"
      sample_rate_hz: 22050
      channels: 1

  - id: "bedroom"
    display_name: "Bedroom"
    video:
      type: "wyze_rtsp"
      url: "rtsp://${WYZE_RTSP_USER}:${WYZE_RTSP_PASSWORD}@192.168.1.134:8554/video6_unicast"
      transport: "tcp"
    mic:
      type: "wyze_rtsp_audio"
      url: "rtsp://${WYZE_RTSP_USER}:${WYZE_RTSP_PASSWORD}@192.168.1.134:8554/video6_unicast"
      transport: "tcp"
      sample_rate_hz: 16000
      channels: 1
    speaker:
      type: "wyze_ssh_aplay"
      host: "192.168.1.134"
      ssh_user: "${WYZE_SSH_USER}"
      ssh_password: "${WYZE_SSH_PASSWORD}"
      ssh_key_path: "${WYZE_SSH_KEY_PATH}"
      remote_play_path: "/tmp/jarvis_play.wav"
      aplay_device: "plughw:0,0"
      sample_rate_hz: 16000

  - id: "kitchen"
    display_name: "Kitchen"
    video:
      type: "wyze_rtsp"
      url: "rtsp://${WYZE_RTSP_USER}:${WYZE_RTSP_PASSWORD}@192.168.1.135:8554/video6_unicast"  # ← UPDATE IP after flash
      transport: "tcp"
    mic:
      type: "wyze_rtsp_audio"
      url: "rtsp://${WYZE_RTSP_USER}:${WYZE_RTSP_PASSWORD}@192.168.1.135:8554/video6_unicast"
      transport: "tcp"
      sample_rate_hz: 16000
      channels: 1
    speaker:
      type: "wyze_ssh_aplay"
      host: "192.168.1.135"  # ← UPDATE IP after flash
      ssh_user: "${WYZE_SSH_USER}"
      ssh_password: "${WYZE_SSH_PASSWORD}"
      ssh_key_path: "${WYZE_SSH_KEY_PATH}"
      remote_play_path: "/tmp/jarvis_play.wav"
      aplay_device: "plughw:0,0"
      sample_rate_hz: 16000

  - id: "living_room"
    display_name: "Living Room"
    video:
      type: "wyze_rtsp"
      url: "rtsp://${WYZE_RTSP_USER}:${WYZE_RTSP_PASSWORD}@192.168.1.136:8554/video6_unicast"  # ← UPDATE IP after flash
      transport: "tcp"
    mic:
      type: "wyze_rtsp_audio"
      url: "rtsp://${WYZE_RTSP_USER}:${WYZE_RTSP_PASSWORD}@192.168.1.136:8554/video6_unicast"
      transport: "tcp"
      sample_rate_hz: 16000
      channels: 1
    speaker:
      type: "wyze_ssh_aplay"
      host: "192.168.1.136"  # ← UPDATE IP after flash
      ssh_user: "${WYZE_SSH_USER}"
      ssh_password: "${WYZE_SSH_PASSWORD}"
      ssh_key_path: "${WYZE_SSH_KEY_PATH}"
      remote_play_path: "/tmp/jarvis_play.wav"
      aplay_device: "plughw:0,0"
      sample_rate_hz: 16000
```

**Driver-level defaults** also belong in `config.yaml` (NOT hardcoded in Python):

```yaml
drivers:
  wyze_rtsp_video:
    buffer_size: 1               # cv2 buffer depth — 1 = freshest frame
    drain_stale_frames: 2        # cap.grab() N times before cap.read()
    reconnect_delay_s: 3.0
  wyze_rtsp_audio:
    rtsp_transport: "tcp"
    audio_chunk_ms: 30
    max_consecutive_errors: 5
  wyze_ssh_speaker:
    connect_timeout_s: 5.0
    sftp_chunk_size: 32768
    aplay_volume: 100
  esp32_http_video:
    request_timeout_s: 5.0
    reconnect_delay_s: 2.0
```

---

## 5. FILE PLAN

Build in this order. Every file gets a top-of-file docstring matching the
existing style (see `core/event_bus.py`).

### 5.1 `core/config.py` — typed config loader

**Purpose**: Load `config.yaml`, expand `${ENV_VAR}` references from `.env`,
validate via Pydantic, raise `ConfigError` on bad input.

**Key shape**:
```python
from pydantic import BaseModel, Field
from typing import Literal, Optional

class WyzeRtspVideoCfg(BaseModel):
    type: Literal["wyze_rtsp"]
    url: str
    transport: Literal["tcp", "udp"] = "tcp"

class Esp32HttpVideoCfg(BaseModel):
    type: Literal["esp32_http"]
    url: str

class UsbIndexVideoCfg(BaseModel):
    type: Literal["usb_index"]
    device_index: int

class NoneVideoCfg(BaseModel):
    type: Literal["none"]

VideoSourceCfg = WyzeRtspVideoCfg | Esp32HttpVideoCfg | UsbIndexVideoCfg | NoneVideoCfg

# Pydantic discriminator on `type` — picks the right model automatically.
class RoomConfig(BaseModel):
    id: str
    display_name: str
    video: VideoSourceCfg = Field(discriminator="type")
    mic: MicSourceCfg = Field(discriminator="type")
    speaker: SpeakerSinkCfg = Field(discriminator="type")
```

**Env-var interpolation**: scan all string values after YAML load; replace
`${VAR}` with `os.environ["VAR"]`; raise `ConfigError` if a referenced var is
unset (so a missing password fails fast instead of generating broken URLs).

**Don't reuse the existing pattern of bare-dict access** (`cfg["rooms"][0]["camera_source"]`).
Pydantic models give you autocomplete, type-checking, and a single point of
validation. Worth the migration effort.

### 5.2 `modules/vision/sources/base.py` — VideoSource ABC

```python
from abc import ABC, abstractmethod
import numpy as np

class VideoSource(ABC):
    """One driver per room. Lifecycle: open() → grab_frame()*N → close()."""

    @abstractmethod
    async def grab_frame(self) -> Optional[np.ndarray]:
        """Return the latest frame as BGR uint8 ndarray, or None on failure."""

    @abstractmethod
    async def close(self) -> None:
        """Release resources. Idempotent — safe to call twice."""
```

### 5.3 `modules/vision/sources/wyze_rtsp.py`

The three RTSP-specific touches that turn Wyze from "wonky" to "just works":

```python
import cv2

cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)   # 1) Force FFmpeg backend
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)           # 2) Always serve freshest frame

# 3) In grab_frame(), drain stale buffered frames before the real read:
for _ in range(drain_count):
    cap.grab()
ret, frame = cap.read()
```

Wrap blocking `cap.read()` in `asyncio.to_thread()`. On read failure: `cap.release()`
and set `self._cap = None` so the next `grab_frame()` reopens. Sleep
`reconnect_delay_s` before the reopen attempt.

### 5.4 `modules/vision/sources/esp32_http.py`, `usb_webcam.py`, `null_source.py`

Three more concrete VideoSource subclasses. Keep each file under 80 lines; one
responsibility per file.

### 5.5 `modules/vision/camera_manager.py` — the factory

```python
def _make_video_source(cfg: VideoSourceCfg) -> VideoSource:
    match cfg.type:
        case "wyze_rtsp":  return WyzeRtspVideo(cfg)
        case "esp32_http": return Esp32HttpVideo(cfg)
        case "usb_index":  return UsbWebcamVideo(cfg)
        case "none":       return NullVideoSource()

class CameraManager:
    def __init__(self, rooms: list[RoomConfig]):
        self._sources = {r.id: _make_video_source(r.video) for r in rooms}

    async def grab_frame(self, room_id: str) -> Optional[np.ndarray]:
        src = self._sources.get(room_id)
        return await src.grab_frame() if src else None

    async def shutdown(self):
        for src in self._sources.values():
            await src.close()
```

### 5.6 `modules/voice/sources/base.py` — MicSource + SpeakerSink ABCs

```python
class MicSource(ABC):
    @abstractmethod
    async def start(self, callback: Callable[[bytes], Awaitable[None]]) -> None:
        """Begin streaming. Calls callback with PCM int16 bytes per chunk."""
    @abstractmethod
    async def stop(self) -> None: ...

class SpeakerSink(ABC):
    @abstractmethod
    async def play(self, pcm: bytes, sample_rate: int) -> None:
        """Play raw PCM int16. Must complete before returning (caller awaits)."""
```

### 5.7 `modules/voice/sources/wyze_rtsp_mic.py` — RTSP audio demux via PyAV

This is the trickiest driver. PyAV demuxes both video and audio from one RTSP
connection; we want only audio. Wyze's wz_mini_hacks emits G.711 µ-law (PCMU)
or AAC depending on build — PyAV decodes both transparently to PCM int16.

```python
import av
import asyncio
import numpy as np

class WyzeRtspMic(MicSource):
    def __init__(self, cfg: WyzeRtspAudioCfg):
        self._cfg = cfg
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self, callback):
        self._running = True
        self._task = asyncio.create_task(self._loop(callback))

    async def _loop(self, callback):
        # av.open is blocking — run in a thread
        container = await asyncio.to_thread(
            av.open, self._cfg.url,
            options={"rtsp_transport": self._cfg.transport, "stimeout": "5000000"},
        )
        try:
            audio_stream = next(s for s in container.streams if s.type == "audio")
            resampler = av.AudioResampler(
                format="s16", layout="mono", rate=self._cfg.sample_rate_hz
            )
            for packet in container.demux(audio_stream):
                if not self._running:
                    break
                for frame in packet.decode():
                    out = resampler.resample(frame)
                    for f in (out if isinstance(out, list) else [out]):
                        pcm: bytes = f.to_ndarray().astype(np.int16).tobytes()
                        await callback(pcm)
        finally:
            await asyncio.to_thread(container.close)

    async def stop(self):
        self._running = False
        if self._task:
            await self._task
```

**Important**: PyAV `container.demux()` is a blocking generator. Either run the
whole loop in a thread (simpler) or use `asyncio.to_thread()` per-iteration
(higher overhead). Thread-the-whole-loop is fine for our use case — the
callback is async and can await the event bus, but the demux stays sync inside
the thread.

### 5.8 `modules/voice/sources/wyze_ssh_speaker.py` — SCP + aplay over SSH

```python
import paramiko
import asyncio
import wave
import io

class WyzeSshSpeaker(SpeakerSink):
    def __init__(self, cfg: WyzeSshSpeakerCfg):
        self._cfg = cfg
        self._client: Optional[paramiko.SSHClient] = None

    async def _connect(self):
        if self._client and self._client.get_transport() and self._client.get_transport().is_active():
            return
        self._client = paramiko.SSHClient()
        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        # Run blocking connect in a thread
        await asyncio.to_thread(
            self._client.connect,
            hostname=self._cfg.host,
            username=self._cfg.ssh_user,
            password=self._cfg.ssh_password or None,
            key_filename=self._cfg.ssh_key_path or None,
            timeout=self._cfg.connect_timeout_s,
        )

    async def play(self, pcm: bytes, sample_rate: int):
        await self._connect()

        # Wrap raw PCM in a WAV header so aplay can autodetect format
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)              # int16 = 2 bytes
            w.setframerate(sample_rate)
            w.writeframes(pcm)

        # SFTP upload then exec aplay — both blocking, both threaded
        def _upload_and_play():
            sftp = self._client.open_sftp()
            with sftp.file(self._cfg.remote_play_path, "wb") as remote:
                remote.write(wav_buf.getvalue())
            sftp.close()
            cmd = f"aplay -D {self._cfg.aplay_device} {self._cfg.remote_play_path}"
            stdin, stdout, stderr = self._client.exec_command(cmd)
            stdout.channel.recv_exit_status()  # Wait for completion

        await asyncio.to_thread(_upload_and_play)
```

Persistent SSH connection — opening a new SSH session per utterance adds ~500ms
of TCP+auth overhead. Reuse one connection per cam, reconnect on failure.

### 5.9 `modules/voice/sources/usb_mic.py`, `usb_speaker.py`, `null_audio.py`

USB drivers wrap `sounddevice.InputStream` / `OutputStream`. Resolve `device_name`
via substring match against `sd.query_devices()` output.

### 5.10 `modules/voice/mic_manager.py` and `speaker_manager.py`

Same factory pattern as `CameraManager`. One `MicManager.start_capture(room_id, cb)`
and `SpeakerManager.play(room_id, pcm, rate)`.

### 5.11 `dashboard/server.py` — `/stream/{room_id}` MJPEG bridge

The dashboard browser can't render RTSP directly — only `<img src=mjpeg>` works
universally. Re-broadcast every camera as MJPEG via FastAPI:

```python
@app.get("/stream/{room_id}")
async def stream(room_id: str):
    async def gen():
        while True:
            frame = await camera_manager.grab_frame(room_id)
            if frame is None:
                await asyncio.sleep(0.5); continue
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok: continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + jpeg.tobytes() + b"\r\n")
            await asyncio.sleep(1.0 / cfg.system.dashboard.stream_fps_cap)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")
```

Browser sees uniform MJPEG regardless of upstream source type. **This is the
unification point** — it's why the toggle pattern works at the dashboard layer.

### 5.12 `scripts/test_wyze.py` — smoke test

Standalone CLI. Takes one arg (`--room bedroom`). Steps:
1. Load config, instantiate managers for that room only.
2. Grab a video frame → save to `data/test_<room>_frame.jpg`.
3. Start mic for 3 seconds, save callback bytes → `data/test_<room>_audio.wav`.
4. Generate a 440Hz tone, send to speaker, verify exit code.
5. Print PASS/FAIL per stage; exit nonzero on any FAIL.

This is the gate — bedroom must pass before flashing the other two cams.

---

## 6. ROOM DIFFERENTIATION (the user's other question)

The user asked how to differentiate the three Wyze cams in code. **It's already
solved by the config schema** — explain this when you respond:

- Each room has a unique `id` (`bedroom` / `kitchen` / `living_room`)
- Each cam has a unique IP via DHCP reservation, embedded in its `url`
- All managers are keyed by `room_id`: `camera_manager.grab_frame("bedroom")`,
  `speaker_manager.play("kitchen", pcm, 16000)`, etc.
- All EventBus events carry `{"room": "bedroom", ...}` payloads — see the topic
  list in `core/event_bus.py`

Differentiation is implicit in the architecture. No additional code needed.

---

## 7. STYLE RULES (carried from JARVIS_BOOTSTRAP.md — DO NOT VIOLATE)

- **Never use placeholder code.** Every function must be real and working.
- **Comment the WHY, not the WHAT.** Reader can read Python.
- **Config over code.** Any tunable value goes in `config.yaml`.
- **Async everywhere.** All I/O non-blocking. `asyncio.to_thread()` for blocking calls.
- **Loguru, not print.** Zero `print()` in production code.
- **One responsibility per file.** Splitting beats stuffing.
- **Test scripts for every phase.** Each phase ends with a passing test.
- **Type hints on every function signature.** No exceptions.

When you finish a file, immediately move to the next. Do not stop to summarize.

---

## 8. OPEN QUESTIONS (raise these to user before/during work)

1. **Laundry Room** — existing config has it as an ESP32-CAM node (`192.168.1.101`).
   Keep it as `esp32_http` video + `esp32_i2s` mic/speaker, or remove? User
   only listed bedroom/kitchen/LR/office in the latest convo.
2. **First cam's actual room** — user said "the first wyze cam is at 192.168.1.134"
   but didn't confirm which room. Bootstrap assumes bedroom; verify.
3. **MQTT broker for ESP32 channels** — not yet running. Build the `esp32_*`
   driver stubs but mark with a `# todo:` that they need MQTT bring-up first.
4. **Existing `camera_source`/`has_node` callers** — grep before deleting fields:
   ```
   grep -rn "camera_source\|has_node\|speaker_sink\|node_ip" --include="*.py" .
   ```
   Update each call site to the new schema during migration.

---

## 9. VERIFICATION PLAN

```bash
# 1. Install new deps
pip install -r requirements.txt

# 2. Verify config loads cleanly
python -c "from core.config import load_config; print(load_config())"

# 3. Smoke-test bedroom (the only verified cam)
python scripts/test_wyze.py --room bedroom
# Expected: PASS PASS PASS, three artifacts in data/

# 4. Start dashboard, eyeball the live feed
python -m dashboard.server
# Browse to http://localhost:8000/stream/bedroom — should see live video
```

Once bedroom passes, flash kitchen + LR cams, update IPs in `config.yaml`,
re-run `test_wyze.py` for each.

---

## 10. WHAT TO DELIVER

In a single coherent push:
- Modified files: `requirements.txt` (verify), `.env.example` (verify), `config.yaml` (migrate).
- New files (in dependency order): `core/config.py`, `modules/vision/sources/{base,wyze_rtsp,esp32_http,usb_webcam,null_source}.py`,
  `modules/vision/camera_manager.py`,
  `modules/voice/sources/{base,wyze_rtsp_mic,wyze_ssh_speaker,usb_mic,usb_speaker,null_audio,esp32_mqtt_mic,esp32_mqtt_speaker}.py`,
  `modules/voice/{mic_manager,speaker_manager}.py`,
  `dashboard/server.py`,
  `scripts/test_wyze.py`.
- Updated callers of old schema fields (whatever `grep` finds).
- A short `WYZE_SETUP.md` operator guide: how to update an IP, how to swap a
  source type, common failure modes (RTSP timeouts, SSH auth refusal, aplay
  device-not-found).

---

## 11. GROUND TRUTH FROM USER (verbatim, for reference)

> "lets wire up full wyze video mic and speaker for now, but have it so i can
> change these things easily with toggles or a drop down per room so its easy
> to fix when i upgrade."

> "the first wyze cam is `rtsp://login:password@IP_ADDRESS:8554/video6_unicast`
> where login is 'cole' password is 'admin' and the ip address of the first cam
> is 192.168.1.134"

> "the wyze cameras will be located in bedroom, kitchen, and living room.
> not sure how to differentiate them within my program either?"

User is technically savvy, learning to code. Comment heavily on architectural
WHYs, give pro/con breakdowns when there's a real choice, demonstrate good
patterns rather than just describing them. Push back constructively if a
specific implementation choice is wrong — they respect that more than agreement.
