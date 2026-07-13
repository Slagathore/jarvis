# Security

Jarvis is a single user home assistant. It runs on your own machine and watches and listens across your rooms. That makes the trust model simple and also unforgiving: anything with access to the host, or to the dashboard when you expose it, can reach a lot. This document explains what runs where, what leaves the machine, and what to watch for.

## Reporting a problem

If you find a security issue, please report it privately instead of opening a public issue. Open a private security advisory on the repo (the Security tab, "Report a vulnerability") at https://github.com/Slagathore/jarvis, or reach me through my GitHub profile (Slagathore). I will confirm it, fix it, and credit you if you want.

## What runs where

Everything runs locally on the host by default. Camera frames, microphone audio, wake word detection, speech to text, face and voice recognition, object and sound classification, and the world model all execute on your machine. No account and no subscription is required to run the core system.

The one part that can reach outside the machine is the language model. See the next section.

## What leaves the machine, and when

Jarvis only sends data off the machine on paths you configure. There are three of them.

1. The language model. `config.yaml` sets `ollama.model`, `vision_model`, and `action_model`. Point these at a local Ollama model (a name with no `:cloud` or `:gapi` suffix) and inference runs on your own Ollama server at `http://localhost:11434`, so nothing leaves. Point them at a cloud model (the shipped default is `kimi-k2.7-code:cloud`) and the text, plus the camera images for the vision model, that Jarvis reasons over are sent to that hosted service. Set all three to local models for a fully local install.

2. Outbound webhooks. If you configure `webhooks.outbound`, Jarvis POSTs the matching event payloads to the URLs you list, for example a Home Assistant scene or an IFTTT trigger. Nothing is sent here unless you add a URL.

3. Anything you wire yourself through the action tools or a custom integration.

Nothing else phones home. There is no telemetry.

## Dashboard access

The dashboard on port 7070 is the widest surface. It can pull live camera snapshots and recorded face and voice clips, read and write `config.yaml`, pull models, and enable and approve computer control, which drives real mouse and keyboard input through pyautogui.

By default the dashboard binds to `127.0.0.1`, so only the host machine can reach it. If you set `system.dashboard_host` to `0.0.0.0` to reach it from your phone or another computer, off box requests must present an access token. The token is generated once on first boot and stored in `data/dashboard_token`, which is gitignored. Requests from localhost are always exempt, so your access on the host machine never changes. To reach the dashboard from another device, open it once with `?token=THE_TOKEN` on the end of the URL. The page stores the token and sends it automatically after that.

Treat that token like a password. Anyone who has it and can reach the port has full control of the dashboard.

## What an attacker with local or LAN access could reach

- Someone with a login on the host machine is fully trusted by design. They can read the config, the token, the SQLite database, and the recorded samples, and can drive everything. None of it is encrypted at rest. This is inherent to a single user desktop tool.
- Someone on your LAN can reach the dashboard only if you bound it to `0.0.0.0`, and only with the token. Without the token they get a 401. Keep the token off shared channels.
- The MQTT broker used for the optional ESP32 room nodes has no authentication or TLS yet. It only matters once you deploy room nodes, and even then you should keep the broker on a trusted network.

## Known limitations

- The dashboard token is a single shared secret stored in plaintext on the local disk. That is the right tradeoff for a home tool, but it is not per user authentication.
- Data at rest, meaning the SQLite database, face and voice samples, and snapshots, is not encrypted.
- MQTT has no auth or TLS. Do not expose the broker to an untrusted network.
- Config files under `hardware/` are templates. Put real camera credentials, WiFi keys, and SSH keys in your gitignored `.env` or local files, and rotate any credential that was ever committed to a public repo.
- Computer control is real input automation. Only enable it if you trust everything that can reach the dashboard.
