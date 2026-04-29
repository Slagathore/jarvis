# Office Node Build Guide

This is the safest first physical module to build for this repo.

## What You Get Today

- MQTT online/offline heartbeat on `jarvis/nodes/office/status`
- MJPEG camera stream on `http://<node_ip>:8080/`
- OTA updates after the first USB flash
- Node presence in the Python dashboard and MQTT tests

## What Is Not Implemented Yet

- Node-side wake-word publish
- Microphone audio publish to `jarvis/nodes/{room}/audio/in`
- Playback of `jarvis/nodes/{room}/audio/out`

The current repo uses the room node as a camera/status endpoint first. Build that path before spending time on the full voice hardware.

## Parts

- 1x AI-Thinker ESP32-CAM
- 1x USB-to-serial adapter for the first flash
- 1x stable 5V power source rated for at least 1A
- Jumper wires
- 1x breadboard or perfboard
- 1x 10k resistor from GPIO12 to GND if you wire the microphone now
- Optional for later audio work: 1x INMP441 microphone breakout
- Optional for later audio work: 1x MAX98357A amplifier breakout
- Optional for later audio work: 1x small 4 ohm or 8 ohm speaker

## Minimum Viable Build

For the first bring-up, only the ESP32-CAM, FTDI adapter, and 5V power are required.

You can add the INMP441 and MAX98357A later. The repo does not depend on them yet for a useful first node.

## First-Flash Wiring

Use this wiring for the first USB flash:

| FTDI | ESP32-CAM |
| --- | --- |
| 5V | 5V |
| GND | GND |
| TX | U0R |
| RX | U0T |

Also connect:

- `GPIO0` to `GND` before power-up to enter flash mode
- Disconnect `GPIO0` from `GND` after flashing and reboot to run normally

## Optional Audio Wiring

These pin assignments come from [node_base.yaml](./node_base.yaml):

| Signal | ESP32-CAM | Device Pin |
| --- | --- | --- |
| I2S WS / LRC | `GPIO14` | INMP441 `WS`, MAX98357A `LRC` |
| I2S BCLK | `GPIO2` | INMP441 `SCK`, MAX98357A `BCLK` |
| Mic Data In | `GPIO12` | INMP441 `SD` |
| Speaker Data Out | `GPIO15` | MAX98357A `DIN` |
| 3V3 | `3.3V` | INMP441 `VDD` |
| VIN | `5V` | MAX98357A `VIN` |
| Ground | `GND` | Common ground to every board |

Connect the speaker to the MAX98357A output terminals.

## Boot-Strap Notes

- `GPIO12` is a boot strap pin. If you wire the microphone now, keep `GPIO12` pulled low during boot with a 10k resistor to ground.
- `GPIO0` low enters flash mode. Remove that jumper for normal runtime.
- Keep all grounds common between the ESP32-CAM, FTDI adapter, microphone, amplifier, and power supply.
- Use a real 5V supply. Brownouts on ESP32-CAM boards often look like random camera or Wi-Fi failures.

## Configure The Repo

1. Copy `hardware/esphome/secrets.yaml.example` to `hardware/esphome/secrets.yaml`.
2. Fill in:
   - `wifi_ssid`
   - `wifi_password`
   - `mqtt_broker_ip`
   - `ota_password`
   - `node_office_ip`
3. Update `config.yaml` for the office room:

```yaml
rooms:
  - id: "office"
    display_name: "Office"
    camera_source: null
    has_node: true
    node_ip: "192.168.1.101"
```

`CameraManager` now derives `http://<node_ip>:8080/` automatically when `camera_source` is `null` and `has_node` is `true`.

## Flash The Node

Install ESPHome if you do not already have it:

```powershell
python -m pip install esphome
```

Validate the config:

```powershell
python -m esphome config hardware\esphome\node_office.yaml
```

Flash over USB the first time:

```powershell
python -m esphome run hardware\esphome\node_office.yaml
```

If ESPHome does not auto-detect the serial port, pass it explicitly:

```powershell
python -m esphome run hardware\esphome\node_office.yaml --device COM5
```

After the first flash, future updates can go over OTA:

```powershell
python -m esphome run hardware\esphome\node_office.yaml --device 192.168.1.101
```

## Validate Bring-Up

After boot, verify these in order:

1. `http://192.168.1.101/` loads the ESPHome web UI
2. `http://192.168.1.101:8080/` returns the camera stream
3. `python scripts/test_mqtt.py` still passes on the PC
4. `python main.py` shows the office node online in the dashboard
5. The office room camera feed is reachable from the Python side

## Expected First Success Criteria

Call this build successful when:

- the office node stays online
- the camera stream is stable
- the dashboard sees the room
- MQTT status changes show up correctly

Do not treat node audio as done yet. That is a separate firmware pass.
