# Jarvis — Wyze V2 Operator Guide

This is the day-to-day reference for adding, swapping, and debugging Wyze
V2 cameras as full A/V terminals (video + mic + speaker). The architectural
rationale lives in `WYZE_INTEGRATION_BOOTSTRAP.md`; this doc is the cheat
sheet you grab when something is on fire.

---

## Configure a new room

Each room declares three independent channels in `config.yaml` under `rooms:`.
Pick the driver per channel via the `type:` toggle:

| channel       | available `type:` values                                                 |
|---------------|--------------------------------------------------------------------------|
| `video`       | `wyze_rtsp` · `esp32_http` · `usb_index` · `none`                        |
| `mic`         | `wyze_rtsp_audio` · `esp32_i2s_mic` · `usb_device_mic` · `none`          |
| `speaker`     | `wyze_ssh_aplay` · `esp32_i2s_spk` · `usb_device_spk` · `none`           |

Minimal Wyze room block (uses the shared credentials in `.env`):

```yaml
- id: hallway
  display_name: Hallway
  video:
    type: wyze_rtsp
    url: rtsp://${WYZE_RTSP_USER}:${WYZE_RTSP_PASSWORD}@192.168.1.137:8554/video6_unicast
    transport: tcp
  mic:
    type: wyze_rtsp_audio
    url: rtsp://${WYZE_RTSP_USER}:${WYZE_RTSP_PASSWORD}@192.168.1.137:8554/video6_unicast
    transport: tcp
    sample_rate_hz: 16000
    channels: 1
  speaker:
    type: wyze_ssh_aplay
    host: 192.168.1.137
    ssh_user: ${WYZE_SSH_USER}
    ssh_password: ${WYZE_SSH_PASSWORD}
    ssh_key_path: ${WYZE_SSH_KEY_PATH}
    remote_play_path: /tmp/jarvis_play.wav
    aplay_device: plughw:0,0
    sample_rate_hz: 16000
  fps_active: 5
  fps_idle: 1
```

The schema is validated at boot. A typo like `type: wyze_rstp` produces a
clean error pointing at the room and field — no guessing.

---

## Swap a hardware channel (the toggle pattern in action)

To replace the awful Wyze speaker in the bedroom with a USB DAC:

```yaml
# before
speaker:
  type: wyze_ssh_aplay
  host: 192.168.1.134
  ...

# after
speaker:
  type: usb_device_spk
  device_name: "Audioengine"   # substring match against sd.query_devices()
  sample_rate_hz: 22050
  channels: 1
```

Restart Jarvis. No Python edits.

---

## Smoke test a single room

```powershell
python scripts/test_wyze.py --room living_room
```

Runs three stages and writes artifacts to `data/`:

1. Captures one video frame → `data/test_bedroom_frame.jpg`
2. Records ~3 s of mic audio → `data/test_bedroom_audio.wav`
3. Plays a 1 s 440 Hz tone through the speaker

Skip flags (`--skip-video`, `--skip-mic`, `--skip-speaker`) are useful when
debugging one channel at a time or when you don't want to disturb whoever's
in the room.

`PASS PASS PASS` is the go-signal for flashing the next cam.

---

## Known limitation — single RTSP client per Wyze cam

`wz_mini_hacks` will accept multiple RTSP clients to `/video6_unicast`, but
only one of them gets audio. The second connection sees the audio stream
in PyAV's metadata and then receives zero packets.

What this means in practice:

- The smoke test releases the camera between the video and mic stages
  (see `scripts/test_wyze.py`) so each channel is verified in isolation.
- In production, **don't run YOLO/MediaPipe vision AND STT against the
  same Wyze cam at once.** The vision side wins (it opened first); STT
  silently gets nothing.
- The right fix is sharing one PyAV container per cam between video and
  mic — a planned refactor (search `SHARED-RTSP-CONTAINER` in
  `camera_manager.py`).
- Workarounds: pull video from a different `wz_mini_hacks` substream
  path (`/video7_unicast` is VGA), or only enable video in rooms where
  STT isn't needed.

## Common failure modes

### RTSP open times out

```text
[CameraManager] RTSP open timed out for 'kitchen' (rtsp://...:8554/video6_unicast)
```

Check, in order:

- Cam powered on and on the right WiFi (`ping <ip>`)
- `wz_mini_hacks` loaded (try `ssh root@<ip>` — if that works, the firmware
  is running; if it refuses connection, the SD card may have ejected)
- The URL path. Different `wz_mini_hacks` builds use different paths —
  `/video6_unicast` is the modern HD endpoint, but older builds use
  `/unicast` on port 554. The cam's `wz_mini.conf` documents which
  endpoint that build serves.

### Wyze mic is silent (PyAV opens but no chunks arrive)

```text
[WyzeMic:living_room] No audio stream in rtsp://... — is wz_mini_hacks audio enabled?
```

Some wz_mini builds ship with audio disabled by default. Edit
`wz_mini.conf` on the cam's SD card and set `ENABLE_AUDIO="true"` (the exact
key varies by build; check the README in your wz_mini source). Reboot the
cam and re-run `test_wyze.py --room <room> --skip-video --skip-speaker`.

### SSH speaker fails with auth refused

```text
[WyzeSpeaker:living_room] SSH connect to 192.168.1.134 failed: Authentication failed.
```

- Confirm `ENABLE_DROPBEAR="true"` in the cam's `wz_mini.conf`
- If using password auth, set `WYZE_SSH_PASSWORD` in `.env` to match the
  cam's `DROPBEAR_PASSWORD`
- If using key auth, generate a keypair, push the public key into
  `/root/.ssh/authorized_keys` on the cam (via the SD card or a one-time
  scp), and point `WYZE_SSH_KEY_PATH` at the private key

### `aplay` exits non-zero

```text
[WyzeSpeaker:living_room] aplay exit 1: aplay: device_list:268: no soundcards found...
```

The `wz_mini_hacks` audio module isn't loaded for this firmware build.
Either pick a build that includes ALSA support or switch this room's
`speaker.type:` to `none` and live without TTS in that room.

### Config validation fails at boot

```text
ConfigError: config.yaml: room 'kitchen' failed validation at 'video.url':
  field required
```

Open the room block, add the missing field. The error path is
`video.url`, `mic.transport`, etc. — the dot-path tells you exactly where
to look.

### Env var unset

```text
ConfigError: Config references ${WYZE_RTSP_PASSWORD} but env var is unset.
Add it to your .env file or shell environment.
```

Copy `.env.example` to `.env` and fill in the credentials. An empty value
is fine for optional fields like `WYZE_SSH_KEY_PATH` — you only need to
delete the YAML reference if you don't want the env-var lookup at all.

---

## Updating an IP

When DHCP shuffles or you swap a cam to a new room:

1. Edit the room's `video.url` and `mic.url` (both have the IP baked in)
2. Edit the room's `speaker.host`
3. Restart Jarvis or `POST /api/system/restart` from the dashboard

A future improvement is mDNS discovery so this becomes automatic, but
DHCP reservations on the router are the practical workaround today.

---

## Where things live

| File                                          | Role                                                                    |
|-----------------------------------------------|-------------------------------------------------------------------------|
| `config.yaml` `rooms:`                        | Per-room channel toggle + driver settings                               |
| `core/config.py`                              | Pydantic schema + `${VAR}` env-var expansion + boot validation          |
| `modules/vision/camera_manager.py`            | All three video paths (USB, HTTP snapshot, Wyze RTSP)                   |
| `modules/voice/sources/wyze_rtsp_mic.py`      | PyAV-based RTSP audio demux                                             |
| `modules/voice/sources/wyze_ssh_speaker.py`   | Paramiko SSH → SCP → aplay                                              |
| `modules/voice/mic_manager.py`                | Mic factory + lifecycle                                                 |
| `modules/voice/speaker_manager.py`            | Speaker factory + lifecycle                                             |
| `scripts/test_wyze.py`                        | The smoke test                                                          |
| Dashboard `/stream/{room}`                    | Multipart MJPEG live view (browser `<img src="/stream/bedroom">`)       |
| Dashboard `/api/camera/{room}/snapshot.jpg`   | Single-frame JPEG poll endpoint                                         |
