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

    card.innerHTML = `
      <div class="room-name">${roomId.replace(/_/g, " ").toUpperCase()}</div>
      <div class="room-status">${data.person_present ? "● Person detected" : "○ Empty"}</div>
      <div class="room-meta">${escapeHtml(data.description || "No camera data yet")}</div>
      ${lightLabel}
    `;
    grid.appendChild(card);
  });
}

function updateRoomVision(roomId, data) {
  roomsCache[roomId] = Object.assign({}, roomsCache[roomId] || {}, data);
  const card = document.getElementById(`room-${roomId}`);
  if (!card) {
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

// ── Init ──────────────────────────────────────────────────────────────────

connect();
