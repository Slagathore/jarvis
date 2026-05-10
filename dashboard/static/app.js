/**
 * JARVIS Dashboard — app.js
 *
 * Vanilla JS WebSocket client. Connects to ws://localhost:7070/ws and
 * updates the DOM in real time as Jarvis emits events.
 *
 * On initial connect the server sends a "full_state" message so the
 * dashboard hydrates immediately even if the page reloads mid-session.
 *
 * Event types handled:
 *   full_state   → applyFullState(state, conversation)
 *   event        → applyEvent(event)
 *
 * #todo: Add visual notification bell for urgent events
 * #todo: Add history chart for interruptibility over time (Chart.js)
 * #todo: Persist conversation log to localStorage across reloads
 * #todo: Add collapsible room detail modal on room card click
 */

"use strict";

// Keeps the latest room data so updateRooms always renders the full set
let roomsCache = {};

// ── WebSocket Connection ───────────────────────────────────────────────────

const WS_URL = `ws://${window.location.host}/ws`;
let ws = null;
let reconnectTimeout = null;

function connect() {
  setWsStatus(false);

  ws = new WebSocket(WS_URL);

  ws.addEventListener("open", () => {
    setWsStatus(true);
    clearTimeout(reconnectTimeout);
  });

  ws.addEventListener("message", (msg) => {
    try {
      const data = JSON.parse(msg.data);
      if (data.type === "full_state") {
        applyFullState(data.state, data.conversation || []);
      } else if (data.type === "event") {
        applyEvent(data.event);
      }
    } catch (e) {
      console.warn("[JARVIS] Failed to parse WS message:", e);
    }
  });

  ws.addEventListener("close", () => {
    setWsStatus(false);
    reconnectTimeout = setTimeout(connect, 3000);
  });

  ws.addEventListener("error", () => {
    ws.close();
  });
}

// ── Full State Hydration ───────────────────────────────────────────────────

function applyFullState(state, conversation) {
  if (!state) return;

  updateActivity(state);
  updateAppliances(state.appliances);
  updateHealth(state.system);
  roomsCache = state.rooms || {};
  updateRooms(roomsCache);

  if (state.last_speech) {
    updateSpeech(state.last_speech);
  }

  // Replay conversation
  const log = document.getElementById("conv-log");
  if (log) {
    log.innerHTML = "";
    if (!conversation || conversation.length === 0) {
      log.innerHTML =
        '<div class="conv-empty">Waiting for first interaction...</div>';
    } else {
      conversation.forEach((entry) =>
        appendConversation(entry, /* noScroll */ true),
      );
      log.scrollTop = log.scrollHeight;
    }
  }
}

// ── Event Router ──────────────────────────────────────────────────────────

function applyEvent(event) {
  switch (event.type) {
    case "state_update":
      updateActivity(event);
      break;
    case "speech":
      updateSpeech(event);
      appendConversation({
        role: "jarvis",
        text: event.text,
        room: event.room,
        timestamp: event.timestamp,
      });
      break;
    case "user_speech":
      appendConversation({
        role: "cole",
        text: event.text,
        room: event.room,
        timestamp: event.timestamp,
      });
      break;
    case "appliance":
      updateSingleAppliance(
        event.appliance,
        event.status,
        event.runtime_minutes,
      );
      break;
    case "node_status":
      updateNodeStatus(event.room, event.online);
      break;
    case "system_health":
      updateHealth(event.health);
      break;
    case "vision":
      updateRoomVision(event.room, event);
      break;
    case "audio_level":
      updateRoomAudio(event.room, event.db);
      break;
    case "reminder_added":
      addReminder(event);
      break;
    case "reminder_fired":
      flashReminderFired(event);
      removeReminder(event.id);
      break;
    case "reminder_deleted":
      removeReminder(event.id);
      break;
    case "calendar_added":
    case "calendar_deleted":
    case "calendar_updated":
      loadCalendar();
      break;
    case "dnd":
      updateDndStatus(event.active, event.until);
      break;
    case "face_enrolled":
    case "face_deleted":
    case "speaker_deleted":
      // Legacy WHO card events — no-op now that the v2 People card has replaced it.
      break;
    case "speaker_enrolled": {
      const hint = document.getElementById("enroll-hint");
      if (hint) {
        hint.textContent = event.ok
          ? `voice sample saved (${event.prompt_id || "wake"}) for ${event.name}`
          : `voice sample failed for ${event.name}`;
      }
      loadPersons();
      break;
    }
    case "speaker_enrollment_armed":
    case "identity_voice_armed": {
      const hint = document.getElementById("enroll-hint");
      if (hint) {
        hint.textContent = `armed ${event.prompt_id || "wake"} for ${event.name} — wake + say the sentence`;
      }
      break;
    }
    case "identity_face_enrolled":
    case "identity_person_deleted":
    case "identity_person_renamed":
    case "identity_live_enrolled":
      loadPersons();
      // Refresh open profile modal too if it's the affected person
      const openId = getOpenPersonId && getOpenPersonId();
      if (openId) loadPersonSamples(openId);
      break;
    case "identity_sample_deleted": {
      const id = getOpenPersonId && getOpenPersonId();
      if (id) loadPersonSamples(id);
      loadPersons();
      break;
    }
    case "identity_pending_added":
    case "identity_pending_resolved":
      loadPending();
      loadPersons();
      break;
    case "notification.added":
    case "notification.read":
    case "notification.deleted":
      loadNotifications();
      break;
    case "model.activated":
    case "model.deleted":
    case "model.pulled":
      loadModels();
      break;
    case "memory.added":
    case "memory.updated":
    case "memory.deleted":
      loadMemory();
      break;
    case "computer.toggled":
    case "computer.pending_added":
    case "computer.confirmed":
    case "computer.rejected":
      loadComputerStatus();
      break;
    case "selfedit.toggled":
    case "selfedit.pending_added":
      loadSelfEditStatus();
      break;
  }
}

// ── DOM Updaters ──────────────────────────────────────────────────────────

function updateActivity(state) {
  const activity = (state.activity || "unknown")
    .toUpperCase()
    .replace(/_/g, " ");
  const interruptibility = state.interruptibility ?? 0.5;
  const confidence = state.confidence ?? 0;
  const signals = state.signals || [];
  const context = state.context || {};

  setText("activity-name", activity);
  setText("activity-location", state.location || "—");

  const ctxStr = context.game || context.project || context.file || "";
  setText("activity-context", ctxStr || "—");

  // Gauge
  const pct = Math.round(interruptibility * 100);
  const fill = document.getElementById("gauge-fill");
  if (fill) {
    fill.style.width = `${pct}%`;
    if (interruptibility < 0.25) {
      fill.style.background = "linear-gradient(90deg, #ff4444, #ff6644)";
    } else if (interruptibility < 0.5) {
      fill.style.background = "linear-gradient(90deg, #ffb300, #ffcc00)";
    } else {
      fill.style.background = "linear-gradient(90deg, #00d4ff, #00ff88)";
    }
  }
  setText("gauge-value", interruptibility.toFixed(2));

  // Signal chips
  const signalsEl = document.getElementById("signals");
  if (signalsEl) {
    signalsEl.innerHTML = signals
      .map((s) => `<span class="signal-chip">${s.replace(/_/g, " ")}</span>`)
      .join("");
  }

  setText("confidence", `Confidence: ${Math.round(confidence * 100)}%`);
  pulse("activity-card");
}

function updateAppliances(appliances) {
  if (!appliances) return;
  Object.entries(appliances).forEach(([name, data]) => {
    updateSingleAppliance(name, data.status, data.runtime_minutes);
  });
}

function updateSingleAppliance(name, status, runtimeMinutes) {
  const card = document.getElementById(`appl-${name}`);
  const statusEl = document.getElementById(`appl-${name}-status`);
  const timeEl = document.getElementById(`appl-${name}-time`);
  if (!card) return;

  if (statusEl) statusEl.textContent = status || "idle";

  card.classList.remove("running", "done");
  if (status === "running") card.classList.add("running");
  if (status === "done") card.classList.add("done");

  if (timeEl) {
    timeEl.textContent =
      runtimeMinutes != null ? `${Math.round(runtimeMinutes)}m` : "—";
  }
}

function updateHealth(system) {
  if (!system) return;

  setDot("h-ollama", system.ollama?.online ? "online" : "offline");
  setText("h-ollama-detail", system.ollama?.model || "—");

  setDot("h-mqtt", system.mqtt?.online ? "online" : "offline");
  setText("h-mqtt-detail", system.mqtt?.broker || "—");

  setDot("h-whisper", system.whisper?.loaded ? "online" : "offline");
  setText("h-whisper-detail", system.whisper?.model || "—");
}

// Set of room IDs that have a camera CONFIGURED (not necessarily currently
// streaming). Populated from /api/cameras at startup so the reconnect
// button appears on dropped feeds — gating on `data.has_camera` was wrong
// because that flag only flips true when vision events arrive, which
// stops happening precisely when the feed dies. RTSP rooms can manually
// reconnect; USB / HTTP cameras get the cog (settings) only.
const configuredCameraRooms = new Set();
const reconnectableCameraRooms = new Set();

async function loadConfiguredCameras() {
  try {
    const res = await fetch("/api/cameras");
    if (!res.ok) return;
    const body = await res.json();
    const list = Array.isArray(body.cameras) ? body.cameras : [];
    configuredCameraRooms.clear();
    reconnectableCameraRooms.clear();
    for (const entry of list) {
      if (!entry || !entry.room) continue;
      configuredCameraRooms.add(entry.room);
      // Only RTSP cams have the manual force-reopen path on the backend;
      // /api/camera/{room}/reconnect 400s for USB / HTTP. Don't render
      // a button that's guaranteed to fail.
      if (entry.kind === "rtsp") {
        reconnectableCameraRooms.add(entry.room);
      }
    }
    // Re-render rooms now that we know which ones have cameras configured.
    if (typeof roomsCache === "object" && roomsCache) {
      updateRooms(roomsCache);
    }
  } catch (err) {
    console.warn("[loadConfiguredCameras] failed:", err);
  }
}
loadConfiguredCameras();

function updateRooms(rooms) {
  const grid = document.getElementById("rooms-grid");
  if (!grid) return;
  grid.innerHTML = "";

  const roomIds = Object.keys(rooms || {});
  if (roomIds.length === 0) return;

  roomIds.forEach((roomId) => {
    const data = rooms[roomId] || {};
    const card = document.createElement("div");
    card.className = "room-card";
    card.id = `room-${roomId}`;

    const lightsOn = data.lights_on;
    const lightLabel =
      lightsOn == null
        ? ""
        : `<span class="room-light ${lightsOn ? "on" : "off"}">${lightsOn ? "LIGHTS ON" : "LIGHTS OFF"}</span>`;

    // Camera presence is now driven by the configured-cameras set, not by
    // `data.has_camera` (which only flips true when vision events arrive
    // — i.e. never when the feed is broken, which is exactly when the
    // reconnect button is needed).  Falls back to the old `has_camera`
    // signal until /api/cameras has answered, so first paint isn't blank.
    const hasCam =
      configuredCameraRooms.has(roomId) || Boolean(data.has_camera);
    const feedTag = hasCam
      ? `<img class="room-feed" data-room="${roomId}" alt="${roomId} feed" />`
      : "";

    // Reconnect only meaningful for RTSP cams (the backend endpoint 400s
    // for USB / HTTP). When /api/cameras hasn't loaded yet we default to
    // showing the button on any has_camera room — better to flash a
    // reconnect that 400s than to hide it on the room that needs it.
    const showReconnect =
      reconnectableCameraRooms.has(roomId) ||
      (configuredCameraRooms.size === 0 && Boolean(data.has_camera));
    const reconnectBtn = showReconnect
      ? `<button class="room-reconnect" data-room="${roomId}" title="Force-reconnect this camera's RTSP stream">⟳</button>`
      : "";

    card.innerHTML = `
      <div class="room-card-header">
        <div class="room-name">${roomId.replace(/_/g, " ").toUpperCase()}</div>
        <div class="room-card-actions">
          ${reconnectBtn}
          <button class="room-cog" data-room="${roomId}" title="Camera + audio settings for this room">⚙</button>
        </div>
      </div>
      ${feedTag}
      <div class="room-status">${data.person_present ? "● Person detected" : "○ Empty"}</div>
      <div class="room-meta">${escapeHtml(data.description || "No camera data yet")}</div>
      ${lightLabel}
    `;
    const cogBtn = card.querySelector(".room-cog");
    if (cogBtn) {
      cogBtn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        openRoomSettingsModal(roomId);
      });
    }
    const reconnectBtnEl = card.querySelector(".room-reconnect");
    if (reconnectBtnEl) {
      reconnectBtnEl.addEventListener("click", async (ev) => {
        ev.stopPropagation();
        // Disable + spin while reconnecting; backend can take 5-30s for
        // RTSP open against a slow / dropped cam. The UI shouldn't let
        // double-clicks queue more attempts.
        reconnectBtnEl.disabled = true;
        reconnectBtnEl.classList.add("spinning");
        try {
          const res = await fetch(`/api/camera/${encodeURIComponent(roomId)}/reconnect`, {
            method: "POST",
          });
          if (!res.ok) {
            const txt = await res.text().catch(() => "");
            console.warn(`[reconnect] ${roomId} ->`, res.status, txt);
            reconnectBtnEl.classList.add("failed");
            setTimeout(() => reconnectBtnEl.classList.remove("failed"), 2000);
          } else {
            // Force a snapshot refresh so the user sees the new feed
            // immediately without waiting for the 250ms poll cycle.
            refreshRoomFeeds();
          }
        } catch (err) {
          console.warn(`[reconnect] ${roomId} threw:`, err);
          reconnectBtnEl.classList.add("failed");
          setTimeout(() => reconnectBtnEl.classList.remove("failed"), 2000);
        } finally {
          reconnectBtnEl.disabled = false;
          reconnectBtnEl.classList.remove("spinning");
        }
      });
    }
    grid.appendChild(card);
  });

  refreshRoomFeeds();
}

// ── Camera feed refresh ───────────────────────────────────────────────────
// Poll-style refresh: cheaper than MJPEG long-poll, gracefully degrades when
// the snapshot endpoint 404s (rooms without cameras hide the <img>).
function refreshRoomFeeds() {
  const imgs = document.querySelectorAll("img.room-feed");
  const stamp = Date.now();
  imgs.forEach((img) => {
    const room = img.dataset.room;
    if (!room) return;
    const card = document.getElementById(`room-${room}`);
    img.onerror = () => {
      img.classList.add("dead");
      // Mark the whole card so the ⟳ button can pulse via CSS — the
      // user needs to spot it without thinking when a feed dies.
      if (card) card.classList.add("offline");
    };
    img.onload = () => {
      img.classList.remove("dead");
      if (card) card.classList.remove("offline");
    };
    img.src = `/api/camera/${encodeURIComponent(room)}/snapshot.jpg?t=${stamp}`;
  });
}

// 250ms = ~4 fps in the dashboard. Smooth enough for "is something
// happening in this room" without thrashing the JPEG encoder. Capture-side
// rate is independent (config: rooms[].fps_active); for the office webcam
// at 30fps capture, this gives ~4fps in the browser because the snapshot
// endpoint takes a fresh frame per request and the network round trip
// dominates. Bump if you want a smoother feed.
setInterval(refreshRoomFeeds, 250);

function updateRoomVision(roomId, data) {
  // Vision events imply the room has a camera, so make sure has_camera sticks
  // through cache merges even if the initial full_state didn't carry it.
  const merged = Object.assign({}, roomsCache[roomId] || {}, data, {
    has_camera: true,
  });
  roomsCache[roomId] = merged;
  const card = document.getElementById(`room-${roomId}`);
  if (!card) {
    updateRooms(roomsCache);
    return;
  }

  // Card exists but may be missing the <img> if it was first rendered before
  // has_camera was known. Rebuild from scratch in that case.
  if (!card.querySelector("img.room-feed")) {
    updateRooms(roomsCache);
    return;
  }

  const statusEl = card.querySelector(".room-status");
  if (statusEl) {
    statusEl.textContent = data.person_present
      ? "● Person detected"
      : "○ Empty";
  }

  const metaEl = card.querySelector(".room-meta");
  if (metaEl && data.description) {
    metaEl.textContent = data.description;
  }

  if (data.lights_on != null) {
    let el = card.querySelector(".room-light");
    if (!el) {
      el = document.createElement("span");
      card.appendChild(el);
    }
    el.className = `room-light ${data.lights_on ? "on" : "off"}`;
    el.textContent = data.lights_on ? "LIGHTS ON" : "LIGHTS OFF";
  }

  card.classList.add("active");
  setTimeout(() => card.classList.remove("active"), 2000);
}

function updateRoomAudio(roomId, db) {
  if (!roomId || db == null) return;
  roomsCache[roomId] = Object.assign({}, roomsCache[roomId] || {}, {
    audio_db: db,
  });
  const card = document.getElementById(`room-${roomId}`);
  if (!card) return;
  let meter = card.querySelector(".room-meter");
  if (!meter) {
    meter = document.createElement("div");
    meter.className = "room-meter";
    meter.innerHTML = '<div class="room-meter-fill"></div><div class="room-meter-label">—</div>';
    card.appendChild(meter);
  }
  // Map -60 dBFS → 0%, 0 dBFS → 100%, clamp.
  const pct = Math.max(0, Math.min(100, Math.round((db + 60) * (100 / 60))));
  const fill = meter.querySelector(".room-meter-fill");
  const label = meter.querySelector(".room-meter-label");
  if (fill) fill.style.width = `${pct}%`;
  if (label) label.textContent = `${db.toFixed(0)} dBFS`;
}

function updateNodeStatus(roomId, online) {
  const card = document.getElementById(`room-${roomId}`);
  if (card) {
    card.classList.toggle("node-online", online);
  }
}

function updateSpeech(data) {
  setText("speech-text", `"${data.text || "—"}"`);
  setText("speech-room", data.room ? data.room.toUpperCase() : "—");
  setText("speech-time", formatTime(data.timestamp));
  pulse("speech-card");
}

function appendConversation(entry, skipScroll = false) {
  const log = document.getElementById("conv-log");
  if (!log) return;

  const empty = log.querySelector(".conv-empty");
  if (empty) empty.remove();

  const el = document.createElement("div");
  el.className = `conv-entry ${entry.role}`;
  el.innerHTML = `
    <div class="conv-speaker">${entry.role === "jarvis" ? "JARVIS" : "COLE"} · ${entry.room?.toUpperCase() || ""}</div>
    <div class="conv-text">${escapeHtml(entry.text)}</div>
    <div class="conv-time">${formatTime(entry.timestamp)}</div>
  `;
  log.appendChild(el);

  if (!skipScroll) {
    log.scrollTop = log.scrollHeight;
  }

  // Keep max 50 entries in DOM
  while (log.children.length > 50) {
    log.removeChild(log.firstChild);
  }
}

// ── Utilities ─────────────────────────────────────────────────────────────

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function setDot(id, state) {
  const el = document.getElementById(id);
  if (el) el.className = `dot ${state}`;
}

function pulse(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.style.boxShadow = "0 0 30px #00d4ff55";
  setTimeout(() => {
    el.style.boxShadow = "";
  }, 600);
}

function setWsStatus(online) {
  const el = document.getElementById("ws-status");
  if (!el) return;
  el.innerHTML = online
    ? '<span class="dot online"></span> LIVE'
    : '<span class="dot offline"></span> RECONNECTING';
}

function formatTime(isoStr) {
  if (!isoStr) return "—";
  try {
    const d = new Date(isoStr);
    return d.toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return "—";
  }
}

function escapeHtml(str) {
  return (str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// ── Clock ─────────────────────────────────────────────────────────────────

function updateClock() {
  const el = document.getElementById("clock");
  if (el) {
    el.textContent = new Date().toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  }
}

setInterval(updateClock, 1000);
updateClock();

// ── Voice Switcher ────────────────────────────────────────────────────────

function loadVoices() {
  fetch("/api/voices")
    .then((r) => r.json())
    .then(({ voices, active }) => {
      const sel = document.getElementById("voice-select");
      if (!sel) return;
      sel.innerHTML = voices
        .map((v) => `<option value="${v}"${v === active ? " selected" : ""}>${v}</option>`)
        .join("");
    })
    .catch(() => {});
}

function applyVoice() {
  const sel = document.getElementById("voice-select");
  if (!sel || !sel.value) return;
  fetch("/api/voice", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ voice: sel.value }),
  }).catch((e) => console.warn("[JARVIS] Voice switch failed:", e));
}

const voiceApplyBtn = document.getElementById("voice-apply");
if (voiceApplyBtn) voiceApplyBtn.addEventListener("click", applyVoice);
loadVoices();

// ── Text Chat ─────────────────────────────────────────────────────────────

function sendChat() {
  const input = document.getElementById("chat-input");
  if (!input) return;
  const text = input.value.trim();
  if (!text) return;

  appendConversation({
    role: "cole",
    text,
    room: "dashboard",
    timestamp: new Date().toISOString(),
  });
  input.value = "";

  fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, room: "office" }),
  }).catch((e) => console.warn("[JARVIS] Chat send failed:", e));
}

const chatInput = document.getElementById("chat-input");
const chatSend = document.getElementById("chat-send");
if (chatInput) chatInput.addEventListener("keydown", (e) => { if (e.key === "Enter") sendChat(); });
if (chatSend) chatSend.addEventListener("click", sendChat);

// ── Reminders ─────────────────────────────────────────────────────────────

let remindersCache = [];

function loadReminders() {
  fetch("/api/reminders")
    .then((r) => r.json())
    .then(({ reminders }) => {
      remindersCache = reminders || [];
      renderReminders();
    })
    .catch(() => {});
}

function renderReminders() {
  const list = document.getElementById("reminders-list");
  if (!list) return;
  list.innerHTML = "";
  if (remindersCache.length === 0) {
    list.innerHTML = '<div class="reminders-empty">No pending reminders.</div>';
    return;
  }
  remindersCache
    .slice()
    .sort((a, b) => (a.trigger_time || "").localeCompare(b.trigger_time || ""))
    .forEach((r) => list.appendChild(reminderRow(r)));
}

function reminderRow(r) {
  const row = document.createElement("div");
  row.className = "reminder-row";
  row.dataset.id = String(r.id);
  const due = r.trigger_time ? new Date(r.trigger_time) : null;
  const when = due ? formatRelative(due) : "—";
  const recurLabel = r.recurrence_seconds
    ? ` · repeats ${formatInterval(r.recurrence_seconds)}`
    : "";
  row.innerHTML = `
    <div class="reminder-info">
      <div class="reminder-msg">${escapeHtml(r.message || "")}</div>
      <div class="reminder-when">${when}${recurLabel}</div>
    </div>
    <button class="reminder-dismiss" aria-label="Dismiss">×</button>
  `;
  if (due && due <= new Date()) row.classList.add("due");
  row.querySelector(".reminder-dismiss").addEventListener("click", () => {
    fetch(`/api/reminders/${r.id}`, { method: "DELETE" }).catch(() => {});
  });
  return row;
}

function addReminder(event) {
  remindersCache = remindersCache.filter((r) => r.id !== event.id);
  remindersCache.push({
    id: event.id,
    message: event.message,
    trigger_time: event.trigger_time,
    recurrence_seconds: event.recurrence_seconds,
  });
  renderReminders();
}

function removeReminder(id) {
  remindersCache = remindersCache.filter((r) => r.id !== id);
  renderReminders();
}

function flashReminderFired(event) {
  // Pulse the speech card so a fired reminder is visually obvious — the
  // actual audio comes from Jarvis via the existing TTS pipeline.
  pulse("speech-card");
}

function formatInterval(secs) {
  if (!secs) return "";
  if (secs % 86400 === 0) {
    const d = secs / 86400;
    return d === 1 ? "daily" : d === 7 ? "weekly" : `every ${d} days`;
  }
  if (secs % 3600 === 0) {
    const h = secs / 3600;
    return h === 1 ? "hourly" : `every ${h}h`;
  }
  if (secs % 60 === 0) return `every ${secs / 60}m`;
  return `every ${secs}s`;
}

function formatRelative(when) {
  const now = new Date();
  const ms = when - now;
  const past = ms < 0;
  const abs = Math.abs(ms);
  const mins = Math.round(abs / 60000);
  if (mins < 1) return past ? "just now" : "in <1 min";
  if (mins < 60) return past ? `${mins}m ago` : `in ${mins}m`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return past ? `${hours}h ago` : `in ${hours}h`;
  return when.toLocaleString("en-US", {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false,
  });
}

function submitReminder() {
  const textEl = document.getElementById("reminder-text");
  const minEl = document.getElementById("reminder-min");
  const recurEl = document.getElementById("reminder-recur");
  if (!textEl || !minEl) return;
  const text = textEl.value.trim();
  const mins = parseInt(minEl.value, 10);
  if (!text || !mins || mins < 1) return;
  const due = new Date(Date.now() + mins * 60 * 1000);
  const recur = recurEl ? parseInt(recurEl.value, 10) : 0;
  const body = { message: text, trigger_time: due.toISOString() };
  if (recur && recur > 0) body.recurrence_seconds = recur;
  fetch("/api/reminders", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
    .then(() => {
      textEl.value = "";
    })
    .catch((e) => console.warn("[JARVIS] Reminder add failed:", e));
}

const reminderTextEl = document.getElementById("reminder-text");
const reminderAddBtn = document.getElementById("reminder-add");
if (reminderTextEl) {
  reminderTextEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") submitReminder();
  });
}
if (reminderAddBtn) reminderAddBtn.addEventListener("click", submitReminder);

// Re-render every 30s so "in 5m" labels stay current
setInterval(renderReminders, 30000);
loadReminders();

// ── Calendar ──────────────────────────────────────────────────────────────

let calendarCache = [];

function loadCalendar() {
  fetch("/api/calendar/upcoming?hours=24")
    .then((r) => r.json())
    .then(({ events, authenticated }) => {
      const list = document.getElementById("calendar-list");
      if (!list) return;
      if (!authenticated) {
        list.innerHTML = '<div class="calendar-empty">Calendar not connected.</div>';
        return;
      }
      calendarCache = events || [];
      renderCalendar();
    })
    .catch(() => {});
}

function renderCalendar() {
  const list = document.getElementById("calendar-list");
  if (!list) return;
  if (calendarCache.length === 0) {
    list.innerHTML = '<div class="calendar-empty">Nothing in the next 24h.</div>';
    return;
  }
  const now = new Date();
  list.innerHTML = "";
  calendarCache.forEach((e) => {
    const start = e.start ? new Date(e.start) : null;
    const row = document.createElement("div");
    row.className = "calendar-row";
    const minutesAway = start ? (start - now) / 60000 : Infinity;
    if (minutesAway > 0 && minutesAway < 60) row.classList.add("soon");
    const when = start ? formatCalendarWhen(start, now) : "—";
    row.innerHTML = `
      <div class="calendar-when">${when}</div>
      <div class="calendar-title">${escapeHtml(e.title || "(untitled)")}</div>
    `;
    list.appendChild(row);
  });
}

function formatCalendarWhen(when, now) {
  const sameDay = when.toDateString() === now.toDateString();
  const opts = { hour: "2-digit", minute: "2-digit", hour12: false };
  if (sameDay) return when.toLocaleTimeString("en-US", opts);
  // Tomorrow / further
  const tomorrow = new Date(now); tomorrow.setDate(tomorrow.getDate() + 1);
  if (when.toDateString() === tomorrow.toDateString()) {
    return "tmrw " + when.toLocaleTimeString("en-US", opts);
  }
  return when.toLocaleString("en-US", { month: "short", day: "numeric", ...opts });
}

setInterval(loadCalendar, 5 * 60 * 1000);  // refresh every 5 min
loadCalendar();

// ── Config editor ─────────────────────────────────────────────────────────

function loadConfig() {
  const ta = document.getElementById("config-yaml");
  const status = document.getElementById("config-status");
  if (!ta) return;
  fetch("/api/config")
    .then((r) => r.json())
    .then(({ yaml }) => {
      ta.value = yaml || "";
      if (status) {
        status.textContent = "loaded";
        status.className = "config-status";
      }
    })
    .catch((e) => {
      if (status) {
        status.textContent = "load failed";
        status.className = "config-status error";
      }
      console.warn("[JARVIS] config load failed:", e);
    });
}

function saveConfig() {
  const ta = document.getElementById("config-yaml");
  const status = document.getElementById("config-status");
  if (!ta) return;
  fetch("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ yaml: ta.value }),
  })
    .then(async (r) => {
      const body = await r.json().catch(() => ({}));
      if (!r.ok) {
        if (status) {
          status.textContent = body.detail || `save failed (${r.status})`;
          status.className = "config-status error";
        }
        return;
      }
      if (status) {
        status.textContent = body.restart_required
          ? "saved — restart required"
          : "saved";
        status.className = "config-status ok";
      }
    })
    .catch((e) => {
      if (status) {
        status.textContent = "save failed";
        status.className = "config-status error";
      }
      console.warn("[JARVIS] config save failed:", e);
    });
}

const configToggle = document.getElementById("config-toggle");
const configBody = document.getElementById("config-body");
const configLoadBtn = document.getElementById("config-load");
const configSaveBtn = document.getElementById("config-save");
if (configToggle && configBody) {
  configToggle.addEventListener("click", () => {
    const showing = configBody.style.display !== "none";
    configBody.style.display = showing ? "none" : "block";
    configToggle.textContent = showing
      ? "CONFIG (click to expand)"
      : "CONFIG (click to collapse)";
    if (!showing) loadConfig();
  });
}
if (configLoadBtn) configLoadBtn.addEventListener("click", loadConfig);
if (configSaveBtn) configSaveBtn.addEventListener("click", saveConfig);

// ── DND ───────────────────────────────────────────────────────────────────

function updateDndStatus(active, until) {
  const el = document.getElementById("dnd-status");
  if (!el) return;
  if (active && until) {
    const dt = new Date(until);
    el.textContent = `until ${dt.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false })}`;
    el.classList.add("active");
  } else {
    el.textContent = "off";
    el.classList.remove("active");
  }
}

function loadDndStatus() {
  fetch("/api/dnd")
    .then((r) => r.json())
    .then(({ active, until }) => updateDndStatus(active, until))
    .catch(() => {});
}

function setDnd(minutes) {
  fetch("/api/dnd", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ minutes }),
  }).catch((e) => console.warn("[JARVIS] DND set failed:", e));
}

const dndOnBtn = document.getElementById("dnd-on");
const dndOffBtn = document.getElementById("dnd-off");
if (dndOnBtn) {
  dndOnBtn.addEventListener("click", () => {
    const sel = document.getElementById("dnd-duration");
    const mins = sel ? parseInt(sel.value, 10) : 30;
    setDnd(mins);
  });
}
if (dndOffBtn) dndOffBtn.addEventListener("click", () => setDnd(0));
loadDndStatus();

// ── PEOPLE (Identity v2) ──────────────────────────────────────────────────

function renderPersons(persons) {
  const el = document.getElementById("persons-list");
  if (!el) return;
  if (!persons || persons.length === 0) {
    el.innerHTML = `<div class="who-empty">No people enrolled.</div>`;
    return;
  }
  el.innerHTML = "";
  persons.forEach((p) => {
    const row = document.createElement("div");
    row.className = "person-row";
    const initial = (p.name || "?").trim().charAt(0).toUpperCase();
    // Bust thumbnail cache after a new face sample lands so the row updates
    // without a hard reload. _personsCacheVersion increments on every refresh.
    const v = _personsCacheVersion;
    const portrait = p.has_thumbnail
      ? `<img class="person-portrait" src="/api/identity/persons/${p.id}/thumbnail.jpg?v=${v}" alt="${escapeHtml(p.name)}" />`
      : `<div class="person-portrait person-portrait-fallback">${escapeHtml(initial)}</div>`;
    row.innerHTML = `
      ${portrait}
      <span class="who-name">${escapeHtml(p.name)}</span>
      <span class="person-counts">
        <span title="Face samples">F${p.face_sample_count || 0}</span>
        <span title="Voice samples">V${p.voice_sample_count || 0}</span>
      </span>
      <button class="who-delete" data-id="${p.id}" title="Delete person + all samples">×</button>
    `;
    // Click portrait or name → open profile modal. Delete button stops propagation.
    const open = () => openPersonProfile(p);
    row.querySelector(".person-portrait").addEventListener("click", open);
    row.querySelector(".who-name").addEventListener("click", open);
    row.querySelector(".who-name").style.cursor = "pointer";
    row.querySelector(".who-delete").addEventListener("click", (ev) => {
      ev.stopPropagation();
      if (!confirm(`Delete '${p.name}' and all their samples?`)) return;
      fetch(`/api/identity/persons/${p.id}`, { method: "DELETE" }).catch(() => {});
    });
    el.appendChild(row);
  });
}

// Cached persons list — drives the pending-review assign dropdown so it can
// list everyone already in the system without an extra fetch per row.
let _personsCache = [];
// Bumped on every refresh so thumbnail URLs include a cache-busting query
// param — otherwise browsers serve the stale 404/old image after enroll.
let _personsCacheVersion = 0;

function loadPersons() {
  return fetch("/api/identity/persons")
    .then((r) => r.json())
    .then(({ persons }) => {
      _personsCache = persons || [];
      _personsCacheVersion += 1;
      renderPersons(_personsCache);
      // Re-render pending too, since its dropdown reads from _personsCache.
      const pl = document.getElementById("pending-list");
      if (pl && pl.dataset.lastItems) {
        try {
          renderPending(JSON.parse(pl.dataset.lastItems));
        } catch (_) {}
      }
    })
    .catch(() => {});
}

function _enrollHint(text, isError = false) {
  const hint = document.getElementById("enroll-hint");
  if (!hint) return;
  hint.textContent = text;
  hint.style.color = isError ? "var(--accent-warn, #ff7a59)" : "";
}

function _enrollGetName() {
  const nameEl = document.getElementById("person-name");
  if (!nameEl) return null;
  const name = nameEl.value.trim();
  if (!name) {
    _enrollHint("enter a name first", true);
    return null;
  }
  return name;
}

function _enrollGetRoom() {
  const sel = document.getElementById("person-room");
  return sel ? sel.value : "office";
}

function snapPose(pose) {
  const name = _enrollGetName();
  if (!name) return;
  const room = _enrollGetRoom();
  _enrollHint(`capturing ${pose}…`);
  fetch("/api/identity/face/enroll", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, pose, room }),
  })
    .then(async (r) => {
      const body = await r.json().catch(() => ({}));
      if (r.ok) {
        _enrollHint(`captured ${pose} for ${name}`);
        loadPersons();
      } else {
        _enrollHint(body.detail || `${pose} failed`, true);
      }
    })
    .catch(() => _enrollHint(`${pose} failed`, true));
}

function armVoice(promptId) {
  const name = _enrollGetName();
  if (!name) return;
  fetch("/api/identity/voice/arm", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, prompt_id: promptId }),
  })
    .then((r) => {
      if (r.ok) _enrollHint(`armed ${promptId} for ${name} — say wake word, then the sentence`);
    })
    .catch(() => _enrollHint("arm failed", true));
}

document.querySelectorAll(".pose-btn").forEach((btn) => {
  btn.addEventListener("click", () => snapPose(btn.dataset.pose));
});
document.querySelectorAll(".voice-arm-btn").forEach((btn) => {
  btn.addEventListener("click", () => armVoice(btn.dataset.promptId));
});

// Populate the room selector from /api/cameras (or fall back to the rooms in state)
function populatePersonRoomSelect() {
  const sel = document.getElementById("person-room");
  if (!sel) return;
  fetch("/api/state")
    .then((r) => r.json())
    .then((state) => {
      const rooms = Object.keys(state.rooms || {});
      if (rooms.length === 0) return;
      sel.innerHTML = "";
      rooms.forEach((rid) => {
        const opt = document.createElement("option");
        opt.value = rid;
        opt.textContent = rid;
        if (rid === "office") opt.selected = true;
        sel.appendChild(opt);
      });
    })
    .catch(() => {});
}

// ── PERSON PROFILE MODAL ──────────────────────────────────────────────────

function openPersonProfile(person) {
  let modal = document.getElementById("person-modal");
  if (!modal) {
    modal = document.createElement("div");
    modal.id = "person-modal";
    modal.className = "person-modal";
    modal.innerHTML = `
      <div class="person-modal-backdrop"></div>
      <div class="person-modal-body">
        <div class="person-modal-header">
          <img class="person-modal-portrait" id="person-modal-portrait" alt="portrait" />
          <div class="person-modal-title">
            <input type="text" id="person-modal-name" class="person-modal-name" />
            <div class="person-modal-meta" id="person-modal-meta"></div>
          </div>
          <button class="person-modal-close" id="person-modal-close">×</button>
        </div>
        <div class="person-modal-section">
          <div class="enroll-row-label">FACE SAMPLES</div>
          <div class="sample-grid" id="person-faces"></div>
        </div>
        <div class="person-modal-section">
          <div class="enroll-row-label">VOICE SAMPLES</div>
          <div class="sample-list" id="person-voices"></div>
        </div>
      </div>
    `;
    document.body.appendChild(modal);
    modal.querySelector(".person-modal-backdrop").addEventListener("click", closePersonProfile);
    document.getElementById("person-modal-close").addEventListener("click", closePersonProfile);
  }
  modal.classList.add("open");
  modal.dataset.personId = String(person.id);

  const portrait = document.getElementById("person-modal-portrait");
  if (person.has_thumbnail) {
    portrait.src = `/api/identity/persons/${person.id}/thumbnail.jpg?v=${_personsCacheVersion}`;
    portrait.style.display = "";
  } else {
    portrait.style.display = "none";
  }

  const nameInput = document.getElementById("person-modal-name");
  nameInput.value = person.name;
  nameInput.onblur = () => {
    const newName = nameInput.value.trim();
    if (!newName || newName === person.name) return;
    fetch(`/api/identity/persons/${person.id}/rename`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: newName }),
    }).catch(() => {});
  };

  document.getElementById("person-modal-meta").textContent =
    `Created ${person.created_at || "—"} · ${person.face_sample_count || 0} face / ${person.voice_sample_count || 0} voice samples`;

  loadPersonSamples(person.id);
}

function closePersonProfile() {
  const modal = document.getElementById("person-modal");
  if (modal) modal.classList.remove("open");
}

function loadPersonSamples(personId) {
  fetch(`/api/identity/persons/${personId}/samples`)
    .then((r) => r.json())
    .then(({ face, voice }) => {
      renderFaceSamples(face || []);
      renderVoiceSamples(voice || []);
    })
    .catch(() => {});
}

function renderFaceSamples(samples) {
  const el = document.getElementById("person-faces");
  if (!el) return;
  if (!samples.length) {
    el.innerHTML = `<div class="who-empty">No face samples.</div>`;
    return;
  }
  el.innerHTML = "";
  samples.forEach((s) => {
    const cell = document.createElement("div");
    cell.className = "sample-cell";
    cell.innerHTML = `
      ${s.has_image
        ? `<img class="sample-thumb" src="/api/identity/face_samples/${s.id}/image.jpg" alt="sample" />`
        : `<div class="sample-thumb sample-thumb-empty">no image</div>`}
      <div class="sample-meta">
        <div>${escapeHtml(s.pose || "candid")}</div>
        <div class="sample-source">${escapeHtml(s.source)}</div>
      </div>
      <button class="sample-delete" data-id="${s.id}" title="Delete this sample">×</button>
    `;
    cell.querySelector(".sample-delete").addEventListener("click", () => {
      if (!confirm("Delete this face sample?")) return;
      fetch(`/api/identity/face_samples/${s.id}`, { method: "DELETE" })
        .then(() => loadPersonSamples(getOpenPersonId()))
        .catch(() => {});
    });
    el.appendChild(cell);
  });
}

function renderVoiceSamples(samples) {
  const el = document.getElementById("person-voices");
  if (!el) return;
  if (!samples.length) {
    el.innerHTML = `<div class="who-empty">No voice samples.</div>`;
    return;
  }
  el.innerHTML = "";
  samples.forEach((s) => {
    const row = document.createElement("div");
    row.className = "sample-row";
    row.innerHTML = `
      <div class="sample-meta">
        <div>${escapeHtml(s.prompt_id || "candid")}</div>
        <div class="sample-source">${escapeHtml(s.source)} · ${s.captured_at}</div>
      </div>
      <button class="sample-delete" data-id="${s.id}" title="Delete this sample">×</button>
    `;
    row.querySelector(".sample-delete").addEventListener("click", () => {
      if (!confirm("Delete this voice sample?")) return;
      fetch(`/api/identity/voice_samples/${s.id}`, { method: "DELETE" })
        .then(() => loadPersonSamples(getOpenPersonId()))
        .catch(() => {});
    });
    el.appendChild(row);
  });
}

function getOpenPersonId() {
  const modal = document.getElementById("person-modal");
  return modal && modal.dataset.personId ? parseInt(modal.dataset.personId, 10) : null;
}

// ── PENDING REVIEW ────────────────────────────────────────────────────────

const NEW_PERSON_SENTINEL = "__new__";

function _personOptions(selectedName) {
  const sel = (selectedName || "").toLowerCase();
  let html = "";
  _personsCache.forEach((p) => {
    const isSelected = p.name && p.name.toLowerCase() === sel ? "selected" : "";
    html += `<option value="${escapeHtml(p.name)}" ${isSelected}>${escapeHtml(p.name)}</option>`;
  });
  html += `<option value="${NEW_PERSON_SENTINEL}">+ new person…</option>`;
  return html;
}

function renderPending(items) {
  const el = document.getElementById("pending-list");
  if (!el) return;
  // Cache so loadPersons() can re-render after the persons list refreshes
  // without re-fetching pending. Lets the dropdown stay in sync hot.
  el.dataset.lastItems = JSON.stringify(items || []);
  if (!items || items.length === 0) {
    el.innerHTML = `<div class="who-empty">No pending review items.</div>`;
    return;
  }
  el.innerHTML = "";
  items.forEach((p) => {
    const row = document.createElement("div");
    row.className = "pending-row";
    const isCluster = p.kind && p.kind.startsWith("pending_cluster_");
    const modality = p.kind && p.kind.includes("voice") ? "voice" : "face";
    const hint = isCluster
      ? `Unknown ${modality} cluster #${p.cluster_id || "?"} — best match ${(p.similarity || 0).toFixed(2)}`
      : `Drift on ${escapeHtml(p.person_name || "unknown")} — sim ${(p.similarity || 0).toFixed(2)} (anchored via ${p.anchored_via || "?"})`;
    let preview = "";
    if (p.has_image) {
      preview = `<img class="pending-thumb" src="/api/identity/pending/${p.id}/image.jpg" alt="capture" />`;
    } else if (p.has_audio) {
      preview = `<audio controls class="pending-audio" src="/api/identity/pending/${p.id}/audio.wav"></audio>`;
    }
    // Pre-select the existing person if this is a drift case, so a one-click
    // assign actually reuses the right person row.
    const optionsHtml = _personOptions(p.person_name);
    row.innerHTML = `
      ${preview}
      <div class="pending-meta">
        <div class="pending-hint">${hint}</div>
        <div class="pending-actions">
          ${
            !isCluster
              ? `<button class="dev-btn pending-confirm" data-id="${p.id}">YES, IT'S ${escapeHtml(p.person_name || "")}</button>`
              : ""
          }
          <select class="dev-select pending-select">${optionsHtml}</select>
          <input type="text" class="reminder-input pending-name" placeholder="New person name" hidden />
          <button class="dev-btn pending-assign" data-id="${p.id}">${isCluster ? "ASSIGN" : "REASSIGN"}</button>
          <button class="dev-btn pending-reject" data-id="${p.id}">REJECT</button>
        </div>
      </div>
    `;
    const select = row.querySelector(".pending-select");
    const newInput = row.querySelector(".pending-name");
    select.addEventListener("change", () => {
      if (select.value === NEW_PERSON_SENTINEL) {
        newInput.hidden = false;
        newInput.focus();
      } else {
        newInput.hidden = true;
        newInput.value = "";
      }
    });
    const confirmBtn = row.querySelector(".pending-confirm");
    if (confirmBtn) {
      confirmBtn.addEventListener("click", () => resolvePending(p.id, "confirm"));
    }
    row.querySelector(".pending-assign").addEventListener("click", () => {
      let target = select.value;
      if (target === NEW_PERSON_SENTINEL) {
        target = newInput.value.trim();
        if (!target) {
          newInput.focus();
          return;
        }
        // Warn if the user is creating a new person with a name that already
        // exists case-insensitively — likely they meant to pick the existing
        // entry from the dropdown.
        const collision = _personsCache.find(
          (pp) => pp.name && pp.name.toLowerCase() === target.toLowerCase()
        );
        if (collision) {
          if (
            !confirm(
              `'${target}' will reuse the existing person '${collision.name}'.\n` +
              `If this is genuinely a different person with the same name, ` +
              `pick a distinct label first (e.g. '${target} S').\n\nProceed?`
            )
          ) {
            return;
          }
          target = collision.name; // preserve original casing
        }
      }
      if (!target) return;
      resolvePending(p.id, "assign", target);
    });
    row.querySelector(".pending-reject").addEventListener("click", () => resolvePending(p.id, "reject"));
    el.appendChild(row);
  });
}

function loadPending() {
  fetch("/api/identity/pending")
    .then((r) => r.json())
    .then(({ pending }) => renderPending(pending || []))
    .catch(() => {});
}

function resolvePending(id, action, targetName) {
  const body = { action };
  if (targetName) body.target_name = targetName;
  fetch(`/api/identity/pending/${id}/resolve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
    .then(() => {
      loadPending();
      loadPersons();
    })
    .catch(() => {});
}

populatePersonRoomSelect();
loadPersons();
loadPending();

// ── NOTIFICATIONS (header bell) ───────────────────────────────────────────

function loadNotifications() {
  fetch("/api/notifications")
    .then((r) => r.json())
    .then(({ items, unread }) => {
      renderNotifications(items || []);
      updateBellBadge(unread || 0);
    })
    .catch(() => {});
}

function renderNotifications(items) {
  const el = document.getElementById("bell-list");
  if (!el) return;
  if (!items.length) {
    el.innerHTML = `<div class="who-empty">No notifications.</div>`;
    return;
  }
  el.innerHTML = "";
  items.forEach((n) => {
    const row = document.createElement("div");
    row.className = `bell-item bell-${n.severity || "info"}${n.read ? " bell-read" : ""}`;
    row.innerHTML = `
      <div class="bell-item-body">
        <div class="bell-item-title">${escapeHtml(n.title)}</div>
        ${n.message ? `<div class="bell-item-msg">${escapeHtml(n.message)}</div>` : ""}
        <div class="bell-item-meta">${escapeHtml(n.created_at || "")}</div>
      </div>
      <button class="bell-item-x" data-id="${n.id}" title="Dismiss">×</button>
    `;
    row.querySelector(".bell-item-body").addEventListener("click", () => {
      navigateToNotification(n);
      fetch(`/api/notifications/${n.id}/read`, { method: "POST" }).catch(() => {});
    });
    row.querySelector(".bell-item-x").addEventListener("click", (ev) => {
      ev.stopPropagation();
      fetch(`/api/notifications/${n.id}`, { method: "DELETE" }).catch(() => {});
    });
    el.appendChild(row);
  });
}

function updateBellBadge(unread) {
  const badge = document.getElementById("bell-badge");
  if (!badge) return;
  if (unread > 0) {
    badge.textContent = String(unread);
    badge.hidden = false;
  } else {
    badge.hidden = true;
  }
}

function navigateToNotification(n) {
  // Close the dropdown first
  const dd = document.getElementById("bell-dropdown");
  if (dd) dd.hidden = true;
  if (n.action === "open_pending") {
    const card = document.getElementById("pending-card");
    if (card) card.scrollIntoView({ behavior: "smooth", block: "center" });
  } else if (n.action === "open_person" && n.target_id) {
    const person = _personsCache.find((p) => p.id === n.target_id);
    if (person) openPersonProfile(person);
  }
}

const bellBtn = document.getElementById("bell-btn");
const bellDropdown = document.getElementById("bell-dropdown");
const bellMarkAll = document.getElementById("bell-mark-all");
if (bellBtn && bellDropdown) {
  bellBtn.addEventListener("click", (ev) => {
    ev.stopPropagation();
    bellDropdown.hidden = !bellDropdown.hidden;
    if (!bellDropdown.hidden) loadNotifications();
  });
  document.addEventListener("click", (ev) => {
    if (bellDropdown.hidden) return;
    if (!document.getElementById("bell-wrap").contains(ev.target)) {
      bellDropdown.hidden = true;
    }
  });
}
if (bellMarkAll) {
  bellMarkAll.addEventListener("click", () => {
    fetch("/api/notifications/read_all", { method: "POST" }).catch(() => {});
  });
}
loadNotifications();

// ── SELF-EDIT (kill switch + pending edits + revert) ─────────────────────

function loadSelfEditStatus() {
  fetch("/api/selfedit/status")
    .then((r) => r.json())
    .then((s) => {
      const toggle = document.getElementById("selfedit-toggle");
      const status = document.getElementById("selfedit-status");
      if (toggle) toggle.checked = !!s.enabled;
      if (status) {
        status.textContent = s.available
          ? (s.enabled ? "ENABLED — Jarvis can edit its own code" : "disabled (read-only)")
          : "unavailable";
        status.className = "control-status" + (s.enabled ? " on" : "");
      }
    })
    .catch(() => {});
}

document.getElementById("selfedit-toggle")?.addEventListener("change", (e) => {
  fetch("/api/selfedit/enable", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled: e.target.checked }),
  }).then(loadSelfEditStatus).catch(() => {});
});
loadSelfEditStatus();

// ── SYSTEM (restart / shutdown) ───────────────────────────────────────────
document.getElementById("system-restart")?.addEventListener("click", () => {
  if (!confirm("Restart Jarvis?\n\nIf you started via the supervisor wrapper it'll come back up automatically (plain relaunch — no heartbeat watch, no git changes).\nIf you started via plain `python main.py`, it'll just exit and stay off.")) return;
  fetch("/api/system/restart", { method: "POST" }).catch(() => {});
});
document.getElementById("system-shutdown")?.addEventListener("click", () => {
  if (!confirm("Shutdown Jarvis?\n\nThis exits the process cleanly. Even with the supervisor, Jarvis WILL stay off until you start it again manually.")) return;
  fetch("/api/system/shutdown", { method: "POST" }).catch(() => {});
});

// ── COMPUTER CONTROL (kill switch + pending action queue) ────────────────

function loadComputerStatus() {
  fetch("/api/computer/status")
    .then((r) => r.json())
    .then((s) => {
      const toggle = document.getElementById("computer-toggle");
      const status = document.getElementById("computer-status");
      if (toggle) toggle.checked = !!s.enabled;
      if (status) {
        status.textContent = s.available
          ? (s.enabled ? "ENABLED — Jarvis can drive mouse + keyboard" : "disabled")
          : "unavailable";
        status.className = "control-status" + (s.enabled ? " on" : "");
      }
      renderComputerPending(s.pending || []);
    })
    .catch(() => {});
}

function renderComputerPending(pending) {
  const el = document.getElementById("computer-pending");
  if (!el) return;
  if (!pending.length) { el.innerHTML = ""; return; }
  el.innerHTML = `<div class="control-pending-title">Pending approval</div>` +
    pending.map((p) => `
      <div class="control-pending-row">
        <div class="control-pending-info">
          <div class="control-pending-action">${escapeHtml(p.action_type)}</div>
          <div class="control-pending-args">${escapeHtml(JSON.stringify(p.args))}</div>
          <div class="control-pending-reason">${escapeHtml(p.reason)}</div>
        </div>
        <button class="dev-btn pending-approve" data-id="${p.id}">APPROVE</button>
        <button class="dev-btn pending-reject" data-id="${p.id}">REJECT</button>
      </div>
    `).join("");
  el.querySelectorAll(".pending-approve").forEach((btn) => {
    btn.addEventListener("click", () => {
      fetch(`/api/computer/pending/${btn.dataset.id}/approve`, { method: "POST" })
        .then(loadComputerStatus).catch(() => {});
    });
  });
  el.querySelectorAll(".pending-reject").forEach((btn) => {
    btn.addEventListener("click", () => {
      fetch(`/api/computer/pending/${btn.dataset.id}/reject`, { method: "POST" })
        .then(loadComputerStatus).catch(() => {});
    });
  });
}

document.getElementById("computer-toggle")?.addEventListener("change", (e) => {
  fetch("/api/computer/enable", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled: e.target.checked }),
  }).then(loadComputerStatus).catch(() => {});
});
loadComputerStatus();

// ── MEMORY v2 ────────────────────────────────────────────────────────────

let _memoryDebounce = null;

function loadMemory() {
  const kindEl = document.getElementById("memory-kind-filter");
  const searchEl = document.getElementById("memory-search");
  const kind = kindEl ? kindEl.value : "";
  const search = searchEl ? searchEl.value.trim() : "";
  const url = search
    ? `/api/memory/search?q=${encodeURIComponent(search)}&k=20`
    : `/api/memory?limit=80${kind ? `&kind=${encodeURIComponent(kind)}` : ""}`;
  fetch(url)
    .then((r) => r.json())
    .then(({ items }) => renderMemory(items || []))
    .catch(() => {});
}

function renderMemory(items) {
  const el = document.getElementById("memory-list");
  if (!el) return;
  if (!items.length) {
    el.innerHTML = `<div class="who-empty">No memories.</div>`;
    return;
  }
  el.innerHTML = "";
  items.forEach((m) => {
    const row = document.createElement("div");
    row.className = `memory-row mem-${m.kind || "fact"}`;
    row.dataset.id = String(m.id);
    const subj = m.subject ? `<span class="mem-subject">[${escapeHtml(m.subject)}]</span> ` : "";
    const imp = (m.importance || 0).toFixed(2);
    const ts = (m.created_at || "").slice(0, 16);
    row.innerHTML = `
      <div class="mem-line1">
        <span class="mem-kind">${escapeHtml(m.kind || "fact")}</span>
        <span class="mem-importance">${imp}</span>
        <button class="mem-edit" data-id="${m.id}" title="Edit">✎</button>
        <button class="mem-del" data-id="${m.id}" title="Delete">×</button>
      </div>
      <div class="mem-content">${subj}${escapeHtml(m.content)}</div>
      <div class="mem-meta">${ts}${typeof m.score === "number" ? ` · score ${m.score.toFixed(2)}` : ""}</div>
    `;
    row.querySelector(".mem-del").addEventListener("click", () => {
      if (!confirm("Delete this memory?")) return;
      fetch(`/api/memory/${m.id}`, { method: "DELETE" }).catch(() => {});
    });
    row.querySelector(".mem-edit").addEventListener("click", () => beginMemoryEdit(row, m));
    el.appendChild(row);
  });
}

function beginMemoryEdit(row, m) {
  // Replace the static row with an inline edit form. Saving fires the
  // existing POST /api/memory/{id} endpoint; cancelling restores the row.
  const original = row.innerHTML;
  row.classList.add("mem-editing");
  row.innerHTML = `
    <div class="mem-line1">
      <select class="dev-select mem-edit-kind">
        ${["fact","preference","event","instruction","thought","question"].map(k =>
          `<option value="${k}" ${k === (m.kind || "fact") ? "selected" : ""}>${k}</option>`
        ).join("")}
      </select>
      <input type="number" step="0.05" min="0" max="1" class="mem-edit-importance" value="${(m.importance || 0).toFixed(2)}" />
    </div>
    <input type="text" class="reminder-input mem-edit-subject" placeholder="subject (optional)" value="${escapeHtml(m.subject || "")}" />
    <textarea class="reminder-input mem-edit-content" rows="3">${escapeHtml(m.content || "")}</textarea>
    <div class="mem-edit-actions">
      <button class="dev-btn mem-edit-save">SAVE</button>
      <button class="dev-btn mem-edit-cancel">CANCEL</button>
    </div>
  `;
  const cancel = () => { row.innerHTML = original; row.classList.remove("mem-editing"); _attachMemoryRowHandlers(row, m); };
  row.querySelector(".mem-edit-cancel").addEventListener("click", cancel);
  row.querySelector(".mem-edit-save").addEventListener("click", () => {
    const body = {
      kind:       row.querySelector(".mem-edit-kind").value,
      importance: parseFloat(row.querySelector(".mem-edit-importance").value),
      subject:    row.querySelector(".mem-edit-subject").value.trim() || null,
      content:    row.querySelector(".mem-edit-content").value.trim(),
    };
    if (!body.content) return;
    fetch(`/api/memory/${m.id}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(() => {
      // memory.updated WS event will fire loadMemory; if it doesn't (offline
      // race), still refresh after a short delay as a fallback.
      setTimeout(loadMemory, 250);
    }).catch(() => {});
  });
}

function _attachMemoryRowHandlers(row, m) {
  // Re-attach handlers after a cancelled edit (we did innerHTML replace).
  row.querySelector(".mem-del")?.addEventListener("click", () => {
    if (!confirm("Delete this memory?")) return;
    fetch(`/api/memory/${m.id}`, { method: "DELETE" }).catch(() => {});
  });
  row.querySelector(".mem-edit")?.addEventListener("click", () => beginMemoryEdit(row, m));
}

document.getElementById("memory-kind-filter")?.addEventListener("change", loadMemory);
document.getElementById("memory-search")?.addEventListener("input", () => {
  if (_memoryDebounce) clearTimeout(_memoryDebounce);
  _memoryDebounce = setTimeout(loadMemory, 250);
});

loadMemory();

// ── LLM MODEL SELECTOR ────────────────────────────────────────────────────

let _modelsCache = { installed: [], catalog: [] };

function loadModels() {
  fetch("/api/models")
    .then((r) => r.json())
    .then((data) => {
      _modelsCache = data;
      renderModels();
    })
    .catch(() => {});
}

function _capsBadges(caps) {
  if (!caps) return "";
  const items = [];
  if (caps.is_api_direct) {
    items.push(`<span class="cap-badge cap-api">API</span>`);
  } else if (caps.is_cloud) {
    items.push(`<span class="cap-badge cap-cloud">CLOUD</span>`);
  }
  if (caps.tool_use) items.push(`<span class="cap-badge cap-tools">TOOLS</span>`);
  if (caps.vision) items.push(`<span class="cap-badge cap-vision">VISION</span>`);
  if (caps.thinking_mode) items.push(`<span class="cap-badge cap-think">CoT</span>`);
  if (caps.context_window) {
    const k = caps.context_window >= 1000000
      ? `${Math.round(caps.context_window / 1000000)}M`
      : `${Math.round(caps.context_window / 1000)}k`;
    items.push(`<span class="cap-badge cap-ctx">${k} ctx</span>`);
  }
  return items.join(" ");
}

function renderModels() {
  const installed = _modelsCache.installed || [];
  const catalog = _modelsCache.catalog || [];
  const chatSel = document.getElementById("model-chat-select");
  const visSel = document.getElementById("model-vision-select");
  const actSel = document.getElementById("model-action-select");
  const datalist = document.getElementById("model-catalog");
  const list = document.getElementById("model-installed-list");
  const activeCaps = document.getElementById("model-active-caps");
  if (!chatSel || !visSel || !list) return;

  if (installed.length === 0) {
    chatSel.innerHTML = `<option value="">no models installed</option>`;
    visSel.innerHTML = `<option value="">no models installed</option>`;
    if (actSel) actSel.innerHTML = `<option value="">no models installed</option>`;
    list.innerHTML = `<div class="who-empty">No models installed. Pull one above.</div>`;
  } else {
    const opts = (active_field) => installed.map((m) =>
      `<option value="${escapeHtml(m.name)}" ${m[active_field] ? "selected" : ""}>${escapeHtml(m.name)}</option>`
    ).join("");
    chatSel.innerHTML = opts("active_chat");
    visSel.innerHTML = opts("active_vision");
    if (actSel) actSel.innerHTML = `<option value="">— disabled —</option>` + opts("active_action");

    list.innerHTML = "";
    installed.forEach((m) => {
      const row = document.createElement("div");
      row.className = "model-row-installed";
      const sizeMb = m.size_bytes ? `${(m.size_bytes / 1e9).toFixed(1)} GB` : "";
      // Active-role badges. Per Cole's spec, the tune gear shows ONLY for
      // models currently active in some role, and sits between the name
      // and the role labels.
      const isActive = m.active_chat || m.active_vision || m.active_action;
      const roleBadges = [
        m.active_chat   ? `<span class="model-role-badge role-chat">CHAT</span>`     : "",
        m.active_vision ? `<span class="model-role-badge role-vision">VISION</span>` : "",
        m.active_action ? `<span class="model-role-badge role-action">ACTION</span>` : "",
      ].join("");
      const tuneBtn = isActive
        ? `<button class="model-tune" data-name="${escapeHtml(m.name)}" title="Tune sampling parameters + thinking mode">⚙</button>`
        : "";
      row.innerHTML = `
        <div class="model-row-line1">
          <span class="model-name">${escapeHtml(m.name)}</span>
          ${tuneBtn}
          <span class="model-roles">${roleBadges}</span>
          <span class="model-size">${sizeMb}</span>
          <button class="model-del" data-name="${escapeHtml(m.name)}" title="Remove">×</button>
        </div>
        <div class="model-row-caps">${_capsBadges(m.capabilities)}</div>
        <div class="model-row-notes">${escapeHtml((m.capabilities && m.capabilities.notes) || "")}</div>
      `;
      row.querySelector(".model-del").addEventListener("click", (ev) => {
        ev.stopPropagation();
        if (m.active_chat) { alert("Can't delete the active chat model — switch first."); return; }
        if (!confirm(`Delete '${m.name}' from disk?`)) return;
        fetch(`/api/models/${encodeURIComponent(m.name)}`, { method: "DELETE" }).catch(() => {});
      });
      const tuneEl = row.querySelector(".model-tune");
      if (tuneEl) {
        tuneEl.addEventListener("click", (ev) => {
          ev.stopPropagation();
          openModelTuneModal(m.name);
        });
      }
      list.appendChild(row);
    });
  }

  const activeChat = installed.find((m) => m.active_chat);
  if (activeChat && activeCaps) {
    activeCaps.innerHTML = _capsBadges(activeChat.capabilities);
  } else if (activeCaps) {
    activeCaps.innerHTML = "";
  }

  if (datalist) {
    datalist.innerHTML = catalog.map((c) =>
      `<option value="${escapeHtml(c.name)}">${escapeHtml(c.notes || "")}</option>`
    ).join("");
  }

  // Reflect the active selection in the inline cogs — has to happen here
  // (after the selects are populated) AND on every change handler call.
  if (typeof _syncInlineTuneButtons === "function") _syncInlineTuneButtons();
}

function setActiveModel(kind, name) {
  if (!name) return;
  fetch("/api/models/active", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, kind }),
  }).catch(() => {});
}

const chatSelEl = document.getElementById("model-chat-select");
const visSelEl = document.getElementById("model-vision-select");
const actSelEl = document.getElementById("model-action-select");
const pullBtnEl = document.getElementById("model-pull-btn");
if (chatSelEl) chatSelEl.addEventListener("change", (e) => { setActiveModel("chat", e.target.value); _syncInlineTuneButtons(); });
if (visSelEl) visSelEl.addEventListener("change", (e) => { setActiveModel("vision", e.target.value); _syncInlineTuneButtons(); });
if (actSelEl) actSelEl.addEventListener("change", (e) => { setActiveModel("action", e.target.value); _syncInlineTuneButtons(); });

// Inline tune cogs sit next to the CHAT/VISION/ACTION dropdowns. Show
// only when the dropdown has a non-empty selection — there's no model
// to tune when ACTION is "— disabled —" or a select is still loading.
// Each cog reuses the same openModelTuneModal() the per-row gear in the
// installed-list uses, so behavior is identical.
function _syncInlineTuneButtons() {
  const map = [
    ["model-chat-select",   "model-chat-tune"],
    ["model-vision-select", "model-vision-tune"],
    ["model-action-select", "model-action-tune"],
  ];
  map.forEach(([selId, btnId]) => {
    const sel = document.getElementById(selId);
    const btn = document.getElementById(btnId);
    if (!sel || !btn) return;
    const hasModel = !!(sel.value && sel.value.trim());
    btn.hidden = !hasModel;
    if (hasModel) btn.dataset.modelName = sel.value;
  });
}
["model-chat-tune", "model-vision-tune", "model-action-tune"].forEach((btnId) => {
  const btn = document.getElementById(btnId);
  if (!btn) return;
  btn.addEventListener("click", (ev) => {
    ev.stopPropagation();
    const name = btn.dataset.modelName;
    if (name) openModelTuneModal(name);
  });
});
if (pullBtnEl) pullBtnEl.addEventListener("click", () => {
  const inp = document.getElementById("model-pull-name");
  const name = (inp && inp.value || "").trim();
  if (!name) return;
  pullModel(name);
});

function pullModel(name) {
  const wrap = document.getElementById("model-progress");
  const text = document.getElementById("model-progress-text");
  const fill = document.getElementById("model-progress-fill");
  if (wrap) wrap.hidden = false;
  if (text) text.textContent = `pulling ${name}…`;
  if (fill) fill.style.width = "0%";

  fetch("/api/models/pull", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  })
    .then(async (resp) => {
      if (!resp.body) return;
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop();
        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          try {
            const evt = JSON.parse(line.slice(5).trim());
            if (text) text.textContent = `${name}: ${evt.status || "..."}`;
            if (evt.completed && evt.total && fill) {
              fill.style.width = `${Math.floor((evt.completed / evt.total) * 100)}%`;
            }
            if (evt.status === "done" || evt.status === "success") {
              if (text) text.textContent = `${name}: complete`;
              if (fill) fill.style.width = "100%";
              setTimeout(() => { if (wrap) wrap.hidden = true; }, 1500);
              loadModels();
            }
            if (evt.status === "error") {
              if (text) text.textContent = `${name}: ${evt.error || "failed"}`;
            }
          } catch (_) {}
        }
      }
    })
    .catch((e) => {
      if (text) text.textContent = `${name}: ${e}`;
    });
}

loadModels();
// Periodic refresh — picks up models pulled via the Ollama CLI (or by
// other means outside the dashboard) without needing a page reload.
// 30s is frequent enough to feel "live" without spamming the daemon.
setInterval(loadModels, 30000);

// ── MODEL TUNE MODAL (sampling params + thinking) ─────────────────────────

let _modelPresetsCache = [];

function _loadPresetsOnce() {
  if (_modelPresetsCache.length) return Promise.resolve(_modelPresetsCache);
  return fetch("/api/models/presets")
    .then((r) => r.json())
    .then(({ presets }) => { _modelPresetsCache = presets || []; return _modelPresetsCache; })
    .catch(() => []);
}

async function openModelTuneModal(name) {
  await _loadPresetsOnce();
  const settingsResp = await fetch(`/api/models/${encodeURIComponent(name)}/settings`).then((r) => r.json()).catch(() => ({ settings: null }));
  const s = settingsResp.settings || {};

  let modal = document.getElementById("model-tune-modal");
  if (!modal) {
    modal = document.createElement("div");
    modal.id = "model-tune-modal";
    modal.className = "person-modal";
    modal.innerHTML = `
      <div class="person-modal-backdrop"></div>
      <div class="person-modal-body">
        <div class="person-modal-header">
          <div class="person-modal-title">
            <div class="model-tune-name" id="model-tune-name"></div>
            <div class="person-modal-meta">Sampling parameters + thinking mode. Empty fields = use the model's defaults.</div>
          </div>
          <button class="person-modal-close" id="model-tune-close">×</button>
        </div>

        <div class="model-tune-row">
          <label>Preset</label>
          <select class="dev-select" id="model-tune-preset">
            <option value="">— custom —</option>
          </select>
        </div>

        <div class="model-tune-grid">
          <label>temperature <input type="number" step="0.05" id="t-temperature" /></label>
          <label>top_p       <input type="number" step="0.05" id="t-top_p" /></label>
          <label>top_k       <input type="number" step="1"    id="t-top_k" /></label>
          <label>min_p       <input type="number" step="0.05" id="t-min_p" /></label>
          <label>presence_penalty   <input type="number" step="0.1" id="t-presence_penalty" /></label>
          <label>repetition_penalty <input type="number" step="0.05" id="t-repetition_penalty" /></label>
        </div>

        <div class="model-tune-row">
          <label class="model-tune-think">
            Thinking
            <select id="t-thinking_enabled" class="dev-select" title="Use default = let the model decide (no think kwarg sent). On/Off = force the value, useful when a model's default trips an API constraint (e.g. Gemini 3 thinking requires thought_signature handling Ollama-cloud doesn't preserve).">
              <option value="">Use model default (recommended)</option>
              <option value="true">On — force thinking</option>
              <option value="false">Off — force no thinking</option>
            </select>
          </label>
        </div>

        <div class="model-tune-actions">
          <button class="dev-btn" id="model-tune-save">SAVE</button>
          <button class="dev-btn" id="model-tune-reset" title="Drop overrides; use modelfile defaults">RESET</button>
          <button class="dev-btn" id="model-tune-cancel">CANCEL</button>
        </div>
      </div>
    `;
    document.body.appendChild(modal);
    modal.querySelector(".person-modal-backdrop").addEventListener("click", () => modal.classList.remove("open"));
    document.getElementById("model-tune-close").addEventListener("click", () => modal.classList.remove("open"));
    document.getElementById("model-tune-cancel").addEventListener("click", () => modal.classList.remove("open"));

    const presetSel = document.getElementById("model-tune-preset");
    _modelPresetsCache.forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = p.label || p.id;
      presetSel.appendChild(opt);
    });
    presetSel.addEventListener("change", () => {
      const sel = _modelPresetsCache.find((p) => p.id === presetSel.value);
      if (!sel) return;
      ["temperature","top_p","top_k","min_p","presence_penalty","repetition_penalty"].forEach((k) => {
        const el = document.getElementById(`t-${k}`);
        if (el && sel[k] !== undefined) el.value = sel[k];
      });
      const t = document.getElementById("t-thinking_enabled");
      if (t) {
        // Presets explicitly set thinking on or off — they're not for
        // "model default" cases (those wouldn't need a preset). So
        // missing thinking_enabled in the preset → leave the dropdown
        // alone; explicit boolean → set it.
        if (sel.thinking_enabled === true) t.value = "true";
        else if (sel.thinking_enabled === false) t.value = "false";
      }
    });
  }

  document.getElementById("model-tune-name").textContent = name;
  modal.dataset.modelName = name;
  ["temperature","top_p","top_k","min_p","presence_penalty","repetition_penalty"].forEach((k) => {
    const el = document.getElementById(`t-${k}`);
    if (el) el.value = (s[k] !== null && s[k] !== undefined) ? s[k] : "";
  });
  const t = document.getElementById("t-thinking_enabled");
  if (t) {
    // Tri-state hydration. None/undefined → "Use default" (no override
    // sent — model decides). True/False → force the value. The previous
    // version coerced None to True, which silently wrote thinking=on to
    // every model the user opened the modal for. That tripped Gemini 3's
    // thought_signature requirement on Ollama-cloud and broke chat.
    if (s.thinking_enabled === true) t.value = "true";
    else if (s.thinking_enabled === false) t.value = "false";
    else t.value = "";
  }
  const presetSel = document.getElementById("model-tune-preset");
  if (presetSel) presetSel.value = s.preset || "";

  const saveBtn = document.getElementById("model-tune-save");
  const resetBtn = document.getElementById("model-tune-reset");
  saveBtn.onclick = () => {
    const body = { preset: document.getElementById("model-tune-preset").value || "custom" };
    ["temperature","top_p","top_k","min_p","presence_penalty","repetition_penalty"].forEach((k) => {
      const v = document.getElementById(`t-${k}`).value;
      body[k] = (v === "" ? null : (k === "top_k" ? parseInt(v, 10) : parseFloat(v)));
    });
    // Tri-state save: "" → null (no override), "true" → true, "false" → false.
    // Backend's _build_options_and_think skips the kwarg entirely when null,
    // so the model uses its own default behavior.
    {
      const tv = document.getElementById("t-thinking_enabled").value;
      body.thinking_enabled = (tv === "true" ? true : (tv === "false" ? false : null));
    }
    fetch(`/api/models/${encodeURIComponent(name)}/settings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(() => modal.classList.remove("open"));
  };
  resetBtn.onclick = () => {
    if (!confirm(`Drop sampling overrides for '${name}'? It'll fall back to the model's modelfile defaults.`)) return;
    fetch(`/api/models/${encodeURIComponent(name)}/settings`, { method: "DELETE" })
      .then(() => modal.classList.remove("open"));
  };

  modal.classList.add("open");
}

// ── PER-FEED ROOM SETTINGS MODAL (rotate / flip / volume / TTS / talkback) ──
//
// Single modal reused across all rooms. Mirrors the model-tune modal's
// layout (same .person-modal scaffold) so the dashboard feels cohesive.
// State is pulled live from /api/room/{room}/settings on open and pushed
// back via /api/room/{room}/settings on every slider change (debounced
// for smoothness without spamming the backend).

let _roomSettingsDebounce = 0;

async function openRoomSettingsModal(room) {
  // Pull current settings + speaker/mic capability info in parallel.
  const [settingsResp, micStatus] = await Promise.all([
    fetch(`/api/room/${encodeURIComponent(room)}/settings`).then((r) => r.json()).catch(() => ({ settings: {} })),
    fetch(`/api/mic/${encodeURIComponent(room)}/status`).then((r) => r.json()).catch(() => ({})),
  ]);
  const s = settingsResp.settings || {};
  const hasSpeaker = !!micStatus.has_speaker;
  const speakerType = String(micStatus.speaker_type || "none");
  // Volume slider only meaningful for Wyze (audioplay_t20 takes 0-100).
  // USB / ESP / null sinks ignore the override; hide the slider in those
  // cases to avoid suggesting it does something it doesn't.
  const volumeSupported = (speakerType === "wyze_ssh_aplay");

  let modal = document.getElementById("room-settings-modal");
  if (!modal) {
    modal = document.createElement("div");
    modal.id = "room-settings-modal";
    modal.className = "person-modal";
    modal.innerHTML = `
      <div class="person-modal-backdrop"></div>
      <div class="person-modal-body">
        <div class="person-modal-header">
          <div class="person-modal-title">
            <div class="model-tune-name" id="room-settings-name"></div>
            <div class="person-modal-meta" id="room-settings-meta"></div>
          </div>
          <button class="person-modal-close" id="room-settings-close">×</button>
        </div>

        <div class="room-settings-section">
          <div class="room-settings-section-title">CAMERA</div>
          <div class="room-settings-row">
            <label>Rotation</label>
            <select id="rs-rotation">
              <option value="0">0°</option>
              <option value="90">90° CW</option>
              <option value="180">180°</option>
              <option value="270">270° (90° CCW)</option>
            </select>
          </div>
          <div class="room-settings-row">
            <label><input type="checkbox" id="rs-flip_h" /> Flip horizontally (mirror)</label>
            <label><input type="checkbox" id="rs-flip_v" /> Flip vertically</label>
          </div>
          <div class="room-settings-row">
            <label>Brightness <input type="range" id="rs-brightness" min="0.5" max="1.5" step="0.05" /></label>
            <span class="room-settings-val" id="rs-brightness-val"></span>
          </div>
          <div class="room-settings-row">
            <label>Contrast <input type="range" id="rs-contrast" min="0.5" max="1.5" step="0.05" /></label>
            <span class="room-settings-val" id="rs-contrast-val"></span>
          </div>
          <div class="room-settings-row">
            <button class="dev-btn" id="rs-snapshot">DOWNLOAD SNAPSHOT</button>
          </div>
        </div>

        <div class="room-settings-section" id="rs-speaker-section">
          <div class="room-settings-section-title">SPEAKER</div>
          <div class="room-settings-row" id="rs-volume-row">
            <label>Volume <input type="range" id="rs-volume" min="0" max="100" step="1" /></label>
            <span class="room-settings-val" id="rs-volume-val"></span>
          </div>
          <div class="room-settings-row">
            <label><input type="checkbox" id="rs-muted" /> Muted (drop all TTS in this room)</label>
          </div>
          <div class="room-settings-row">
            <input type="text" id="rs-speak-text" class="reminder-input" placeholder="Type something to speak in this room…" />
            <button class="dev-btn" id="rs-speak-btn">SPEAK</button>
          </div>
          <div class="room-settings-row">
            <button class="dev-btn" id="rs-talkback-btn" title="Hold to push your mic into this room's speaker">🎤 TALKBACK (HOLD)</button>
            <span class="room-settings-val" id="rs-talkback-status"></span>
          </div>
        </div>

        <div class="room-settings-section" id="rs-wyze-section" hidden>
          <div class="room-settings-section-title">WYZE CAMERA</div>
          <div id="rs-wyze-fields"></div>
          <div class="room-settings-row">
            <button class="dev-btn" id="rs-wyze-reboot" title="Reboot the cam — needed for some settings to take effect (~30s downtime)">REBOOT CAM</button>
            <span class="room-settings-val" id="rs-wyze-status"></span>
          </div>
        </div>

        <div class="room-settings-section">
          <div class="room-settings-actions">
            <button class="dev-btn" id="rs-reset" title="Wipe all tweaks; revert to config.yaml defaults">RESET ALL</button>
          </div>
        </div>
      </div>
    `;
    document.body.appendChild(modal);
    modal.querySelector(".person-modal-backdrop").addEventListener("click", () => modal.classList.remove("open"));
    document.getElementById("room-settings-close").addEventListener("click", () => modal.classList.remove("open"));
  }

  // Header label
  document.getElementById("room-settings-name").textContent =
    `${room.replace(/_/g, " ").toUpperCase()} — settings`;
  document.getElementById("room-settings-meta").textContent =
    `Speaker: ${speakerType}` + (volumeSupported ? "" : "  (volume slider disabled — driver doesn't take a volume arg)");

  // Hide the speaker section entirely if there's no speaker
  const spkSection = document.getElementById("rs-speaker-section");
  if (spkSection) spkSection.hidden = !hasSpeaker;
  const volRow = document.getElementById("rs-volume-row");
  if (volRow) volRow.hidden = !volumeSupported;

  modal.dataset.room = room;

  // Hydrate form from current settings
  document.getElementById("rs-rotation").value = String(s.rotation ?? 0);
  document.getElementById("rs-flip_h").checked = !!s.flip_h;
  document.getElementById("rs-flip_v").checked = !!s.flip_v;
  const setSlider = (id, v, def) => {
    const el = document.getElementById(id);
    const valEl = document.getElementById(`${id}-val`);
    el.value = String(v ?? def);
    if (valEl) valEl.textContent = el.value;
  };
  setSlider("rs-brightness", s.brightness, 1.0);
  setSlider("rs-contrast", s.contrast, 1.0);
  setSlider("rs-volume", s.volume, 60);
  document.getElementById("rs-muted").checked = !!s.muted;

  // Wire change handlers (idempotent — overwriting onchange replaces prior closures)
  const debouncedSave = (patch) => {
    clearTimeout(_roomSettingsDebounce);
    _roomSettingsDebounce = setTimeout(() => {
      fetch(`/api/room/${encodeURIComponent(room)}/settings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      }).catch(() => {});
    }, 150);
  };
  document.getElementById("rs-rotation").onchange = (e) => debouncedSave({ rotation: parseInt(e.target.value, 10) });
  document.getElementById("rs-flip_h").onchange = (e) => debouncedSave({ flip_h: e.target.checked });
  document.getElementById("rs-flip_v").onchange = (e) => debouncedSave({ flip_v: e.target.checked });
  ["rs-brightness", "rs-contrast", "rs-volume"].forEach((id) => {
    const el = document.getElementById(id);
    el.oninput = () => {
      const valEl = document.getElementById(`${id}-val`);
      if (valEl) valEl.textContent = el.value;
      const key = id.slice(3); // "rs-foo" → "foo"
      const isInt = (key === "volume");
      debouncedSave({ [key]: isInt ? parseInt(el.value, 10) : parseFloat(el.value) });
    };
  });
  document.getElementById("rs-muted").onchange = (e) => debouncedSave({ muted: e.target.checked });

  // Snapshot download — open the snapshot URL with a timestamp so the
  // browser doesn't serve a cached frame; the download attribute hints
  // a filename to the save dialog.
  document.getElementById("rs-snapshot").onclick = () => {
    const a = document.createElement("a");
    a.href = `/api/camera/${encodeURIComponent(room)}/snapshot.jpg?t=${Date.now()}`;
    a.download = `${room}_snapshot_${Date.now()}.jpg`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  // TTS speak
  document.getElementById("rs-speak-btn").onclick = () => {
    const text = (document.getElementById("rs-speak-text").value || "").trim();
    if (!text) return;
    fetch(`/api/room/${encodeURIComponent(room)}/speak`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    }).catch(() => {});
    document.getElementById("rs-speak-text").value = "";
  };

  // Talkback — hold-to-talk pattern. getUserMedia + AudioContext capture
  // → resample to 16kHz int16 → POST raw bytes to /api/room/{room}/play_pcm.
  // Hold semantic is friendlier than press-toggle for a one-off "tell the
  // dog to get off the couch" use case.
  _wireTalkback(room);

  // Reset all tweaks
  document.getElementById("rs-reset").onclick = () => {
    if (!confirm(`Wipe all runtime tweaks for '${room}'?`)) return;
    fetch(`/api/room/${encodeURIComponent(room)}/settings`, { method: "DELETE" })
      .then(() => modal.classList.remove("open"));
  };

  // WYZE CAMERA section — only renders for rooms with a wyze cam wired
  // up (orchestrator's wyze_cam_controls dict). Hidden otherwise.
  await _renderWyzeSection(room);

  modal.classList.add("open");
}

// Renders the WYZE CAMERA section (night vision, IR LEDs, status LED, reboot).
// Pulls live values from /api/wyze/{room}/cam — if 404, the section stays hidden
// (room isn't a Wyze cam, so the controls would be meaningless).
async function _renderWyzeSection(room) {
  const section = document.getElementById("rs-wyze-section");
  const fieldsEl = document.getElementById("rs-wyze-fields");
  const statusEl = document.getElementById("rs-wyze-status");
  if (!section || !fieldsEl) return;

  let resp;
  try {
    resp = await fetch(`/api/wyze/${encodeURIComponent(room)}/cam`);
  } catch (_) {
    section.hidden = true;
    return;
  }
  if (resp.status === 404) {
    section.hidden = true;  // not a Wyze cam
    return;
  }
  if (!resp.ok) {
    section.hidden = false;
    fieldsEl.innerHTML = `<div class="room-settings-row">cam unreachable (HTTP ${resp.status})</div>`;
    return;
  }
  const data = await resp.json();
  const values = data.values || {};
  const spec = data.spec || {};

  section.hidden = false;
  // Build a row per known param. Each row has a label + a select. The
  // <option value=""> represents "default" (key not present in
  // /configs/.parameters); selecting it doesn't write anything (no
  // "remove key from file" API yet).
  const rowsHtml = Object.keys(spec).map((key) => {
    const s = spec[key];
    const cur = values[key];
    const opts = [`<option value="">default</option>`].concat(
      s.options.map((o) =>
        `<option value="${o.value}" ${cur == o.value ? "selected" : ""}>${escapeHtml(o.label)}</option>`
      )
    ).join("");
    const rebootHint = s.reboot_required ? `<span class="room-settings-val" title="needs reboot to apply">⟳</span>` : "";
    return `
      <div class="room-settings-row">
        <label>${escapeHtml(s.label)} ${rebootHint}</label>
        <select data-wyze-key="${escapeHtml(key)}">${opts}</select>
      </div>`;
  }).join("");
  fieldsEl.innerHTML = rowsHtml;

  // Wire each select to POST on change. A single key per change so we
  // can show per-key success/failure cleanly in the status line.
  fieldsEl.querySelectorAll("select[data-wyze-key]").forEach((sel) => {
    sel.onchange = async (ev) => {
      const key = sel.dataset.wyzeKey;
      const v = ev.target.value;
      if (v === "") return;  // "default" — no-op; we don't remove keys
      statusEl.textContent = `setting ${key}…`;
      const r = await fetch(`/api/wyze/${encodeURIComponent(room)}/cam`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ [key]: parseInt(v, 10) }),
      });
      const j = await r.json().catch(() => ({}));
      const ok = j.results && j.results[key];
      statusEl.textContent = ok
        ? `✓ ${key}=${v}` + (spec[key].reboot_required ? " (reboot to apply)" : "")
        : `✗ ${key} failed`;
    };
  });

  // REBOOT CAM
  const rebootBtn = document.getElementById("rs-wyze-reboot");
  if (rebootBtn) {
    rebootBtn.onclick = async () => {
      if (!confirm(`Reboot the ${room} cam? Stream will drop for ~30s.`)) return;
      statusEl.textContent = "rebooting…";
      const r = await fetch(`/api/wyze/${encodeURIComponent(room)}/reboot`, { method: "POST" });
      statusEl.textContent = r.ok ? "✓ rebooting (cam back in ~30s)" : `✗ reboot failed (${r.status})`;
    };
  }
}

// Talkback recorder. Captures via getUserMedia + an AudioWorklet (or
// ScriptProcessor fallback for older browsers — we use the simpler
// MediaRecorder + decode-on-server path because Wyze speakers truncate
// hard at 8kHz anyway, so quality isn't the bottleneck).
function _wireTalkback(room) {
  const btn = document.getElementById("rs-talkback-btn");
  const status = document.getElementById("rs-talkback-status");
  if (!btn) return;
  let mediaStream = null;
  let audioCtx = null;
  let processor = null;
  let source = null;
  const targetRate = 16000;
  const chunks = [];

  const start = async () => {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
      status.textContent = "mic permission denied";
      return;
    }
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    source = audioCtx.createMediaStreamSource(mediaStream);
    // ScriptProcessor is deprecated but still ubiquitously supported.
    // Buffer 4096 = ~85ms at 48kHz capture; small enough for low latency,
    // big enough to avoid GC pressure.
    processor = audioCtx.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (ev) => {
      const inBuf = ev.inputBuffer.getChannelData(0);
      // Downsample by linear decimation. Browser sample rates are
      // typically 44.1k or 48k; targetRate is 16k. Quality is fine for a
      // 1cm Wyze cone; better resampling adds code for inaudible gain.
      const ratio = audioCtx.sampleRate / targetRate;
      const outLen = Math.floor(inBuf.length / ratio);
      const out = new Int16Array(outLen);
      for (let i = 0; i < outLen; i++) {
        const v = inBuf[Math.floor(i * ratio)] || 0;
        out[i] = Math.max(-32768, Math.min(32767, v * 32767));
      }
      chunks.push(out);
    };
    source.connect(processor);
    processor.connect(audioCtx.destination);
    status.textContent = "● recording";
  };

  const stop = async () => {
    if (!mediaStream) return;
    try { processor && processor.disconnect(); } catch {}
    try { source && source.disconnect(); } catch {}
    try { audioCtx && audioCtx.close(); } catch {}
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
    audioCtx = null;
    processor = null;
    source = null;

    // Concatenate captured int16 chunks into one buffer + POST as
    // application/octet-stream. The endpoint takes raw PCM bytes; a
    // ?rate=16000 query tells the speaker driver the input rate.
    let total = 0;
    for (const c of chunks) total += c.length;
    if (total === 0) {
      status.textContent = "(no audio captured)";
      return;
    }
    const merged = new Int16Array(total);
    let off = 0;
    for (const c of chunks) { merged.set(c, off); off += c.length; }
    chunks.length = 0;
    status.textContent = "uploading…";
    try {
      const r = await fetch(`/api/room/${encodeURIComponent(room)}/play_pcm?rate=${targetRate}`, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: merged.buffer,
      });
      status.textContent = r.ok ? "✓ played" : `failed (${r.status})`;
    } catch (e) {
      status.textContent = "upload failed";
    }
  };

  // Pointer events cover mouse + touch + pen uniformly.
  btn.onpointerdown = (ev) => { ev.preventDefault(); start(); };
  btn.onpointerup = (ev) => { ev.preventDefault(); stop(); };
  btn.onpointerleave = () => { if (mediaStream) stop(); };
  btn.onpointercancel = () => { if (mediaStream) stop(); };
}

// ── PERSONA SYSTEM (dropdown + command box + status badge) ────────────────
//
// Hidden personas (uwu) intentionally do NOT appear in the dropdown — the
// /api/personas endpoint filters them out server-side. The only way to
// activate one is to type the name into the command input. That's the
// designed safety property; don't add an "advanced" toggle that lists them.

async function loadPersonas() {
  const sel = document.getElementById("persona-select");
  const badge = document.getElementById("persona-badge");
  const status = document.getElementById("persona-status");
  if (!sel) return;
  let data;
  try {
    const r = await fetch("/api/personas");
    if (!r.ok) {
      sel.innerHTML = `<option>persona system not configured</option>`;
      sel.disabled = true;
      return;
    }
    data = await r.json();
  } catch (e) {
    return;
  }
  const personas = data.personas || [];
  const active = data.active || "";
  sel.disabled = false;
  sel.innerHTML = personas.map((p) => {
    const lock = p.requires_privacy ? " 🔒" : "";
    return `<option value="${escapeHtml(p.name)}" ${p.name === active ? "selected" : ""}>${escapeHtml(p.display)}${lock}</option>`;
  }).join("");
  // If the active persona is hidden (e.g. uwu), it WON'T appear in the
  // dropdown — synthesize a transient option so the user sees the truth.
  // Marked italic so it's visually distinct from configured options.
  if (active && !personas.some((p) => p.name === active)) {
    const opt = document.createElement("option");
    opt.value = active;
    opt.textContent = `${active} (hidden)`;
    opt.selected = true;
    opt.style.fontStyle = "italic";
    sel.appendChild(opt);
  }
  if (badge) {
    const lockIcon = data.locked ? " 🔒" : "";
    badge.textContent = active ? `${active}${lockIcon}` : "";
  }
  if (status) status.textContent = "";
  // Refresh resume offer state alongside
  await refreshPersonaCurrent();
}

async function refreshPersonaCurrent() {
  try {
    const r = await fetch("/api/persona/current");
    if (!r.ok) return;
    const data = await r.json();
    const resumeEl = document.getElementById("persona-resume");
    const resumeText = document.getElementById("persona-resume-text");
    if (data.pending_resume && resumeEl) {
      resumeEl.hidden = false;
      if (resumeText) {
        resumeText.textContent = `Resume '${data.pending_resume}'?`;
      }
    } else if (resumeEl) {
      resumeEl.hidden = true;
    }
  } catch {}
}

const personaSel = document.getElementById("persona-select");
if (personaSel) {
  personaSel.addEventListener("change", async (e) => {
    const name = e.target.value;
    if (!name) return;
    const r = await fetch("/api/persona/set", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    const status = document.getElementById("persona-status");
    if (!r.ok) {
      const j = await r.json().catch(() => ({}));
      if (status) status.textContent = `✗ ${j.detail || r.status}`;
    } else {
      if (status) status.textContent = `✓ ${name}`;
    }
    await loadPersonas();
  });
}

const personaCmdBtn = document.getElementById("persona-cmd-btn");
const personaCmd = document.getElementById("persona-cmd");
async function runPersonaCommand() {
  if (!personaCmd) return;
  const text = (personaCmd.value || "").trim();
  if (!text) return;
  const status = document.getElementById("persona-status");
  const r = await fetch("/api/persona/command", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!r.ok) {
    const j = await r.json().catch(() => ({}));
    if (status) status.textContent = `✗ ${j.detail || r.status}`;
  } else {
    const j = await r.json();
    if (status) status.textContent = `✓ active: ${j.active}${j.locked ? " 🔒" : ""}`;
    personaCmd.value = "";  // clear so the typed name doesn't linger on screen
  }
  await loadPersonas();
}
if (personaCmdBtn) personaCmdBtn.addEventListener("click", runPersonaCommand);
if (personaCmd) {
  personaCmd.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); runPersonaCommand(); }
  });
}

const personaResumeYes = document.getElementById("persona-resume-yes");
const personaResumeNo = document.getElementById("persona-resume-no");
if (personaResumeYes) {
  personaResumeYes.addEventListener("click", async () => {
    await fetch("/api/persona/command", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: "resume" }),
    });
    await loadPersonas();
  });
}
if (personaResumeNo) {
  personaResumeNo.addEventListener("click", async () => {
    // Clearing pending_resume is a side effect of any non-resume action.
    // Simplest "decline" = revert to default explicitly.
    await fetch("/api/persona/command", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: "revert" }),
    });
    await loadPersonas();
  });
}

// Re-fetch personas on connection + every 10s so other sessions
// (auto-revert, dashboard tabs) stay in sync.
loadPersonas();
setInterval(loadPersonas, 10000);

// ── Pets (World Model §22) ────────────────────────────────────────────────
// Render every resident cat + dog with current state, likely-room
// (with unmonitored_home fallback), and a per-pet care summary
// (last food / litterbox / water / leash event timestamps).
//
// Care-summary kinds we render. Order matters — left-to-right in the UI.
// Cats only see litterbox + food; dogs only see food + water + leash.
// Anything else the system fires (future kinds) is ignored here.
const CARE_KIND_LABELS = {
  litterbox_visit: { label: "litter", icon: "▣" },
  food_dish_visit: { label: "food", icon: "◆" },
  dog_food_visit: { label: "food", icon: "◆" },
  dog_water_visit: { label: "water", icon: "≈" },
  leash_interaction: { label: "leash", icon: "⌒" },
};
const CARE_KINDS_BY_SPECIES = {
  cat: ["litterbox_visit", "food_dish_visit"],
  dog: ["dog_food_visit", "dog_water_visit", "leash_interaction"],
};

function formatRelativeTs(ts) {
  if (!ts) return "—";
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return "—";
  const ageS = (Date.now() - d.getTime()) / 1000;
  if (ageS < 60) return `${Math.round(ageS)}s ago`;
  if (ageS < 3600) return `${Math.round(ageS / 60)}m ago`;
  if (ageS < 86400) return `${Math.round(ageS / 3600)}h ago`;
  return `${Math.round(ageS / 86400)}d ago`;
}

function renderPetCard(pet) {
  const div = document.createElement("div");
  div.className = `pet-row pet-${pet.species || "unknown"}`;

  // Where the pet "is". When we haven't seen them recently and the
  // pet has an unmonitored_home, render with a hedge — matches the
  // way where_is_pet's likely_room_inferred flag wants to be phrased.
  let location = pet.last_seen_room || pet.likely_room || "?";
  let hedge = "";
  if (pet.likely_room_inferred && pet.likely_room) {
    location = pet.likely_room;
    hedge = ' <span class="pet-hedge">(probably)</span>';
  }
  if (pet.last_seen_landmark) {
    location += ` · ${pet.last_seen_landmark}`;
  }

  // Care chips — per-species subset only. Each chip shows "last seen N
  // ago" or "—" if we have no event of that kind in the window.
  const kinds = CARE_KINDS_BY_SPECIES[pet.species] || [];
  const chips = kinds
    .map((k) => {
      const meta = CARE_KIND_LABELS[k];
      if (!meta) return "";
      const slot = (pet.care || {})[k];
      const lastTs = slot ? slot.last_ts : null;
      const cls = lastTs ? "pet-care-chip on" : "pet-care-chip off";
      return `
        <span class="${cls}" title="last ${meta.label}: ${
          lastTs ? new Date(lastTs).toLocaleString() : "no record in 24h"
        }">
          <span class="pet-care-icon">${meta.icon}</span>
          <span class="pet-care-label">${meta.label}</span>
          <span class="pet-care-ago">${formatRelativeTs(lastTs)}</span>
        </span>`;
    })
    .join("");

  const speciesGlyph = pet.species === "dog" ? "𓃡" : "𓃠";
  const stateClass = `pet-state-${(pet.state || "unknown").replace(/_/g, "-")}`;

  div.innerHTML = `
    <div class="pet-row-head">
      <span class="pet-glyph">${speciesGlyph}</span>
      <span class="pet-name">${escapeHtml(pet.name || "?")}</span>
      <span class="pet-state ${stateClass}">${escapeHtml(pet.state || "?")}</span>
    </div>
    <div class="pet-row-where">
      <span class="pet-where-label">in</span>
      <span class="pet-where-room">${escapeHtml(location)}</span>${hedge}
    </div>
    <div class="pet-care">${chips}</div>
  `;
  return div;
}

async function loadPets() {
  try {
    const res = await fetch("/api/world_model/pets");
    if (!res.ok) return;
    const body = await res.json();
    const list = document.getElementById("pets-list");
    if (!list) return;
    if (!body.available) {
      list.innerHTML =
        '<div class="who-empty">World model unavailable.</div>';
      return;
    }
    const pets = Array.isArray(body.pets) ? body.pets : [];
    if (pets.length === 0) {
      list.innerHTML =
        '<div class="who-empty">No resident pets configured.</div>';
      return;
    }
    list.innerHTML = "";
    // Group by species; cats first then dogs, alphabetical inside.
    const order = (a, b) => {
      const sa = a.species || "z", sb = b.species || "z";
      if (sa !== sb) return sa.localeCompare(sb);
      return (a.name || "").localeCompare(b.name || "");
    };
    pets.sort(order).forEach((p) => list.appendChild(renderPetCard(p)));
  } catch (err) {
    console.warn("[loadPets] failed:", err);
  }
}
loadPets();
// 30s cadence — care-summary chips don't need real-time updates
// (food/litterbox events fire on the order of hours), state changes
// flow in via the WebSocket world.entity_event handler below.
setInterval(loadPets, 30000);

// ── World Events tail (live tail of world_entity_events) ──────────────────

const EVENT_TYPE_GLYPH = {
  first_seen: "+",
  reappeared: "↺",
  moved_to: "→",
  moved_within_room: "·",
  lost_visibility: "?",
  departed: "✕",
  entered_unmonitored: "▢",
  interacted_with: "◇",
  posture_changed: "⌇",
  stationary_long: "⏚",
  camera_degraded: "!",
  camera_restored: "✓",
  name_linked: "@",
};

function renderEventRow(ev) {
  const glyph = EVENT_TYPE_GLYPH[ev.event_type] || "·";
  const meta = ev.metadata || {};
  // §22.9 events ride as `interacted_with` with metadata.interaction_kind;
  // surface that directly so the user sees "litterbox_visit" not just
  // "interacted_with".
  let label = ev.event_type || "?";
  if (meta.interaction_kind) {
    label = meta.interaction_kind;
  }
  const room = ev.room ? ` <span class="event-room">${escapeHtml(ev.room)}</span>` : "";
  const lm = meta.landmark ? ` <span class="event-landmark">@${escapeHtml(meta.landmark)}</span>` : "";
  const name = ev.entity_name || `?_${ev.entity_type || "ent"}`;
  const ago = formatRelativeTs(ev.ts);
  const div = document.createElement("div");
  div.className = `event-row event-type-${ev.event_type || "unknown"}`;
  div.innerHTML = `
    <span class="event-glyph">${glyph}</span>
    <span class="event-name">${escapeHtml(name)}</span>
    <span class="event-label">${escapeHtml(label)}</span>${room}${lm}
    <span class="event-ago">${ago}</span>
  `;
  return div;
}

async function loadWorldEvents() {
  try {
    const res = await fetch("/api/world_model/events?limit=20");
    if (!res.ok) return;
    const body = await res.json();
    const list = document.getElementById("world-events-list");
    if (!list) return;
    if (!body.available) {
      list.innerHTML =
        '<div class="who-empty">World model unavailable.</div>';
      return;
    }
    const events = Array.isArray(body.events) ? body.events : [];
    if (events.length === 0) {
      list.innerHTML =
        '<div class="who-empty">No events yet (last 12h).</div>';
      return;
    }
    list.innerHTML = "";
    events.forEach((ev) => list.appendChild(renderEventRow(ev)));
  } catch (err) {
    console.warn("[loadWorldEvents] failed:", err);
  }
}
loadWorldEvents();
// 5s cadence so landmark dwell events appear quickly during dev. Cheap:
// the endpoint reads the indexed event log + decodes JSON, no I/O fanout.
setInterval(loadWorldEvents, 5000);

// ── Interactions timeline (§24.6) ─────────────────────────────────────────
// Pre-filtered tail of INTERACTED_WITH / PICKED_UP / PLACED_DOWN / HANDED_OFF
// events. The verb-template + thumbnail makes each row read like a sentence
// ("Cole picked up wallet · office · 3m ago") rather than the raw event log.

const INTERACTION_TEMPLATES = {
  picked_up: (e, m) =>
    `${escapeHtml(m.person_name || "?")} picked up ` +
    `${escapeHtml(m.object_name || e.entity_name || "object")}`,
  placed_down: (e, m) =>
    `${escapeHtml(m.person_name || "?")} placed ` +
    `${escapeHtml(m.object_name || e.entity_name || "object")} ` +
    `down`,
  handed_off: (e, m) =>
    `${escapeHtml(m.from_person_name || "?")} handed ` +
    `${escapeHtml(m.object_name || e.entity_name || "object")} ` +
    `to ${escapeHtml(m.to_person_name || "?")}`,
  interacted_with: (e, m) =>
    `${escapeHtml(e.entity_name || "?")} touched ` +
    `${escapeHtml(m.object_name || "an object")}`,
};

const INTERACTION_GLYPH = {
  picked_up: "↑",
  placed_down: "↓",
  handed_off: "⇄",
  interacted_with: "·",
};

function renderInteractionRow(ev) {
  const meta = ev.metadata || {};
  const tpl = INTERACTION_TEMPLATES[ev.event_type] ||
    ((e) => escapeHtml(e.event_type || "interaction"));
  const sentence = tpl(ev, meta);
  const room = ev.room
    ? `<span class="event-room">${escapeHtml(ev.room)}</span>`
    : "";
  const thumb = ev.thumbnail_url
    ? `<img class="interaction-thumb" src="${ev.thumbnail_url}"
            alt="snapshot" loading="lazy"
            onerror="this.style.display='none';" />`
    : '<div class="interaction-thumb empty"></div>';
  const glyph = INTERACTION_GLYPH[ev.event_type] || "·";
  const ago = formatRelativeTs(ev.ts);
  const div = document.createElement("div");
  div.className = `interaction-row interaction-${ev.event_type || "unknown"}`;
  div.innerHTML = `
    ${thumb}
    <div class="interaction-body">
      <div class="interaction-line">
        <span class="interaction-glyph">${glyph}</span>
        <span class="interaction-text">${sentence}</span>
      </div>
      <div class="interaction-meta">${room}<span class="interaction-ago">${ago}</span></div>
    </div>
  `;
  return div;
}

async function loadInteractions() {
  try {
    const res = await fetch("/api/world_model/interactions?limit=30");
    if (!res.ok) return;
    const body = await res.json();
    const list = document.getElementById("interactions-list");
    if (!list) return;
    if (!body.available) {
      list.innerHTML =
        '<div class="who-empty">World model unavailable.</div>';
      return;
    }
    const events = Array.isArray(body.events) ? body.events : [];
    if (events.length === 0) {
      list.innerHTML =
        '<div class="who-empty">No interactions in the last 24h.</div>';
      return;
    }
    list.innerHTML = "";
    events.forEach((ev) => list.appendChild(renderInteractionRow(ev)));
  } catch (err) {
    console.warn("[loadInteractions] failed:", err);
  }
}
loadInteractions();
// 10s cadence — interactions don't fire every second the way landmark
// dwell does; cheaper than the 5s WORLD EVENTS poll.
setInterval(loadInteractions, 10000);

// ── Init ──────────────────────────────────────────────────────────────────

connect();
