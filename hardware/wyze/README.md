# Wyze V2 SD-card layout

Files staged here go onto the SD card of each Wyze V2 cam running
`wz_mini_hacks`. After copying, eject the card, slot it back in the cam,
and reboot.

## What goes on the card

```text
SD card root/
├── wz_mini.conf                   ← copy from conf/<room>.conf
├── wpa_supplicant.conf            ← copy from this dir (shared across cams)
└── wz_mini/
    └── etc/
        └── ssh/
            └── authorized_keys    ← copy from etc/ssh/authorized_keys (shared)
```

## Per-cam steps

1. Pop the SD card out of the cam.
2. Mount it on the PC.
3. Copy `conf/<room>.conf` to `<sd>/wz_mini.conf` (overwrite if present).
4. Copy `wpa_supplicant.conf` to `<sd>/wpa_supplicant.conf`.
5. Create `<sd>/wz_mini/etc/ssh/` if missing, then copy
   `etc/ssh/authorized_keys` into it.
6. Eject, slot, reboot.

The room IDs and IP plan today:

| Room        | IP (DHCP reservation)  | Conf file              |
|-------------|------------------------|------------------------|
| living_room | 192.168.1.134          | conf/living_room.conf  |
| bedroom     | 192.168.1.135 (TBD)    | conf/bedroom.conf      |
| kitchen     | 192.168.1.136 (TBD)    | conf/kitchen.conf      |

After reboot, set DHCP reservations on the router so each cam keeps the
IP listed above (or update `config.yaml` with whatever IP the router
hands out).

## What's different from the original conf

The base wz_mini.conf you pulled was missing two lines that close the
OTA-config gap. The conf files in this folder add them:

- `WEB_SERVER_OPTIONS="cam config car jpeg multicam diag status"` turns on
  the WZ Mini Web Tools at `http://<cam>/` (config editor, status page,
  jpeg snapshots, diagnostic CGI). Without it the index renders blank
  and CGI endpoints 404.
- `ENABLE_FILESERVER="true"` exposes a file PUT endpoint so future
  config tweaks (rotating creds, bitrates, paths) can be done with curl
  instead of another SD trip.

Hostnames are baked in per file (`CUSTOM_HOSTNAME="<room>"`). Everything
else (RTSP creds, audio enable, swap, motor) matches the original.

## SSH key

The keypair lives at `~/.ssh/jarvis_wyze` (private) +
`~/.ssh/jarvis_wyze.pub` (public). The public key is staged in
`etc/ssh/authorized_keys` here.

`.env` should reference the private key:

```text
WYZE_SSH_USER=root
WYZE_SSH_PASSWORD=
WYZE_SSH_KEY_PATH=C:/Users/Cole/.ssh/jarvis_wyze
```

(Already set, see `.env`.)

## Verifying the path

`wz_mini/etc/ssh/authorized_keys` matches the SD_ROOT layout in the
wz_mini_hacks repo (`wz_mini_hacks-master/SD_ROOT/wz_mini/etc/ssh/`),
which is what your build is using. wz_mini_hacks here runs OpenSSH
(`sshd`), not Dropbear, so the path is the standard OpenSSH location.

After the first cam boots, you can confirm by SSH-ing in and running:

```text
ssh -i ~/.ssh/jarvis_wyze root@192.168.1.134 \
  "find / -name authorized_keys 2>/dev/null; ls -la /etc/ssh /root/.ssh 2>/dev/null"
```

That tells you both the file the running system actually used and where
sshd is reading from, so any future-build path drift is one command away
from being diagnosed.

## Re-using for new cams

Adding a fourth cam later? Copy `conf/living_room.conf` to
`conf/<newroom>.conf`, change the `CUSTOM_HOSTNAME` line, drop a new
room block in `config.yaml`, and follow the per-cam steps above.
