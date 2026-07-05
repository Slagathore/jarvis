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
let activeTab = "home";

// ── Visibility-aware polling ──────────────────────────────────────────────
// All recurring dashboard polls go through safeInterval() so they:
//  1. Pause while the browser tab is hidden — Chrome throttles background
//     timers but the snapshot poll still triggers camera reads + JPEG
//     encoding on the server every tick. We want the work to stop, not
//     just slow down.
//  2. Skip a tick if the previous run hasn't finished (in-flight guard) —
//     a slow LLM/DB request used to stack up callbacks behind a 5s poll.
//  3. Fire once on visibility return so the UI refreshes immediately
//     instead of waiting for the next interval tick.
const _safeIntervals = [];
function safeInterval(fn, intervalMs) {
  const entry = { fn, intervalMs, timer: null, running: false };
  const tick = async () => {
    if (document.hidden || entry.running) return;
    entry.running = true;
    try {
      await fn();
    } catch (e) {
      console.warn("[safeInterval]", e);
    } finally {
      entry.running = false;
    }
  };
  entry.timer = setInterval(tick, intervalMs);
  _safeIntervals.push(entry);
  return entry;
}
function stopSafeInterval(entry) {
  if (!entry) return;
  if (entry.timer) clearInterval(entry.timer);
  const idx = _safeIntervals.indexOf(entry);
  if (idx >= 0) _safeIntervals.splice(idx, 1);
}
document.addEventListener("visibilitychange", () => {
  if (document.hidden) return;
  // Tab just became visible — run every safeInterval once so the user
  // doesn't stare at stale data while waiting for the next tick.
  _safeIntervals.forEach((e) => {
    if (e.running) return;
    try {
      const r = e.fn();
      if (r && typeof r.then === "function") r.catch(() => {});
    } catch (err) {
      console.warn("[safeInterval:visible]", err);
    }
  });
});

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
  renderWakeCalibration(Object.values(state.wake_calibration || {}));

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
      updateRoomAudio(event.room, event.db, event.peak_db);
      scheduleWakeCalibrationRefresh();
      break;
    case "wake_score":
    case "wake_calibration":
      scheduleWakeCalibrationRefresh();
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
    case "identity_pending_collapsed":
    case "identity_pending_bulk_resolved":
      refreshReviewsBadge();
      loadPersons();
      // If the user has the Reviews tab open, refresh its grid so a
      // dropped row disappears immediately.
      if (document.getElementById("tab-pane-reviews") &&
          !document.getElementById("tab-pane-reviews").hidden) {
        loadReviewsTab();
      }
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
    case "world.entity_event":
      // A new entity event fired on the bus. Refresh the panels
      // immediately rather than waiting on the next 5s/10s poll. The
      // panel functions are idempotent and rate-limited by their own
      // setInterval, so an extra refresh here is cheap.
      loadWorldEvents();
      // Interactions only fire for a small subset of event types —
      // only re-pull when one of those types lands.
      if (event.event_type === "interacted_with" ||
          event.event_type === "picked_up" ||
          event.event_type === "placed_down" ||
          event.event_type === "handed_off") {
        loadInteractions();
      }
      // Pet cards key off entity state changes — refresh so the
      // "last seen N ago" text stays current.
      if (event.entity_type === "cat" || event.entity_type === "dog") {
        if (typeof loadPets === "function") loadPets();
      }
      break;
    case "world.state_snapshot":
      // Throttled to ≤1/5s by the server; cheap UI refresh.
      if (typeof loadPets === "function") loadPets();
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

// Attention tier: flag a card as needing eyes ("attention" amber /
// "alert" red / null to clear). Pairs with .card-attention/.card-alert
// in style.css so problem cards pop out of the wall of ambient status.
function setCardTier(cardId, tier) {
  const card = document.getElementById(cardId);
  if (!card) return;
  card.classList.toggle("card-attention", tier === "attention");
  card.classList.toggle("card-alert", tier === "alert");
}

function updateHealth(system) {
  if (!system) return;

  setDot("h-ollama", system.ollama?.online ? "online" : "offline");
  setText("h-ollama-detail", system.ollama?.model || "—");

  setDot("h-mqtt", system.mqtt?.online ? "online" : "offline");
  setText("h-mqtt-detail", system.mqtt?.broker || "—");

  setDot("h-whisper", system.whisper?.loaded ? "online" : "offline");
  setText("h-whisper-detail", system.whisper?.model || "—");

  const anyDown =
    !system.ollama?.online || !system.mqtt?.online || !system.whisper?.loaded;
  setCardTier("health-card", anyDown ? "attention" : null);
}

async function loadDegradedStatus() {
  try {
    const res = await fetch("/api/degraded");
    if (!res.ok) return;
    renderDegradedStatus(await res.json());
  } catch (err) {
    console.warn("[degraded] load failed:", err);
  }
}

function renderDegradedStatus(data) {
  const summary = document.getElementById("degraded-summary");
  const list = document.getElementById("degraded-list");
  if (!summary || !list) return;
  const items = Array.isArray(data?.items) ? data.items : [];
  const overall = data?.overall === "ok" ? "ok" : "degraded";
  summary.textContent = overall === "ok" ? "All core capabilities nominal" : "Running with degraded capabilities";
  summary.className = `degraded-summary ${overall}`;
  setCardTier("degraded-card", overall === "ok" ? null : "attention");
  list.innerHTML = items.map((item) => {
    const status = item.status || "unknown";
    const detail = item.detail ? ` · ${escapeHtml(item.detail)}` : "";
    return `
      <div class="degraded-item ${status}">
        <span class="dot ${status === "loaded" ? "online" : status === "disabled" ? "" : "offline"}"></span>
        <span class="degraded-name">${escapeHtml(item.name || "unknown")}</span>
        <span class="degraded-detail">${escapeHtml(status)}${detail}</span>
      </div>`;
  }).join("");
}

let _wakeCalibrationTimer = null;
let _wakeCalibrationLoading = false;
let _wakeCalibrationRefreshSoon = null;

async function loadWakeCalibration() {
  if (_wakeCalibrationLoading) return;
  _wakeCalibrationLoading = true;
  try {
    const res = await fetch("/api/wake_calibration");
    if (!res.ok) return;
    const body = await res.json();
    renderWakeCalibration(body.rooms || []);
  } catch (err) {
    console.warn("[wake_calibration] load failed:", err);
  } finally {
    _wakeCalibrationLoading = false;
  }
}

function scheduleWakeCalibrationRefresh() {
  if (_wakeCalibrationRefreshSoon) return;
  _wakeCalibrationRefreshSoon = setTimeout(() => {
    _wakeCalibrationRefreshSoon = null;
    loadWakeCalibration();
  }, 500);
}

function renderWakeCalibration(rooms) {
  const el = document.getElementById("wake-calibration-list");
  if (!el) return;
  const list = Array.isArray(rooms) ? rooms : [];
  if (!list.length) {
    el.innerHTML = `<div class="who-empty">Waiting for mic levels…</div>`;
    return;
  }
  el.innerHTML = list.map((r) => {
    const room = r.room || "unknown";
    const rms = r.rms_db == null ? "—" : `${Number(r.rms_db).toFixed(0)} dB`;
    const peak = r.peak_db == null ? "—" : `${Number(r.peak_db).toFixed(0)} dB`;
    const score = Number(r.wake_score || 0);
    const sensitivity = Number(r.sensitivity || 0.5);
    const suggested = Number(r.suggested_sensitivity || sensitivity);
    const fp = Number(r.false_positive_count || 0);
    const pct = Math.max(0, Math.min(100, Math.round(score * 100)));
    const sensPct = Math.max(0, Math.min(100, Math.round(sensitivity * 100)));
    return `
      <div class="wake-row">
        <div class="wake-row-head">
          <span class="wake-room">${escapeHtml(room.replace(/_/g, " ").toUpperCase())}</span>
          <button class="wake-fp-btn" data-room="${escapeHtml(room)}" title="Mark the last wake in this room as false">False wake</button>
        </div>
        <div class="wake-stats">
          <span>RMS ${rms}</span>
          <span>Peak ${peak}</span>
          <span>False ${fp}</span>
          <span>Suggest ${suggested.toFixed(2)}</span>
        </div>
        <div class="wake-score-track" title="Wake score ${score.toFixed(3)} / sensitivity ${sensitivity.toFixed(2)}">
          <div class="wake-score-fill" style="width:${pct}%"></div>
          <div class="wake-score-threshold" style="left:${sensPct}%"></div>
        </div>
      </div>`;
  }).join("");
  el.querySelectorAll(".wake-fp-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const room = btn.dataset.room;
      if (!room) return;
      btn.disabled = true;
      try {
        await fetch(`/api/wake_calibration/${encodeURIComponent(room)}/false_positive`, {
          method: "POST",
        });
        loadWakeCalibration();
      } finally {
        btn.disabled = false;
      }
    });
  });
}

loadDegradedStatus();
safeInterval(loadDegradedStatus, 5000);
loadWakeCalibration();
_wakeCalibrationTimer = safeInterval(loadWakeCalibration, 3000);

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

    const tagPetBtn = hasCam
      ? `<button class="room-tag-pet" data-room="${roomId}" title="Point at a cat or dog in this room and tell me which pet it is">🐾</button>`
      : "";
    card.innerHTML = `
      <div class="room-card-header">
        <div class="room-name">${roomId.replace(/_/g, " ").toUpperCase()}</div>
        <div class="room-card-actions">
          ${tagPetBtn}
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
    const tagBtn = card.querySelector(".room-tag-pet");
    if (tagBtn) {
      tagBtn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        openLivePetTagModal(roomId);
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
            const body = await res.json().catch(() => ({}));
            if (!body.reconnected) {
              console.warn(`[reconnect] ${roomId} did not reopen`, body);
              reconnectBtnEl.classList.add("failed");
              setTimeout(() => reconnectBtnEl.classList.remove("failed"), 2000);
              return;
            }
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
  if (activeTab !== "home") return;
  const imgs = document.querySelectorAll("img.room-feed");
  const stamp = Date.now();
  imgs.forEach((img) => {
    const room = img.dataset.room;
    if (!room) return;
    // In-flight guard: if the previous snapshot for this room is still
    // loading, skip this tick rather than abandoning the request and
    // stacking another one. Stale dataset is cleared on load/error.
    if (img.dataset.loading === "1") return;
    const card = document.getElementById(`room-${room}`);
    img.onerror = () => {
      img.dataset.loading = "";
      img.classList.add("dead");
      // Mark the whole card so the ⟳ button can pulse via CSS — the
      // user needs to spot it without thinking when a feed dies.
      if (card) card.classList.add("offline");
    };
    img.onload = () => {
      img.dataset.loading = "";
      img.classList.remove("dead");
      if (card) card.classList.remove("offline");
    };
    img.dataset.loading = "1";
    img.src = `/api/camera/${encodeURIComponent(room)}/snapshot.jpg?t=${stamp}`;
    // Wire click → lightbox (once). The dashboard down-scales 1080p
    // Wyze frames to ~640px; the lightbox shows the original snapshot
    // URL at full resolution.
    if (!img.dataset.lightboxWired) {
      img.dataset.lightboxWired = "1";
      img.classList.add("room-feed-clickable");
      img.title = "Click for full-resolution snapshot";
      img.addEventListener("click", () => {
        const ts = new Date().toISOString();
        // Pull a fresh frame so the lightbox shows current state, not
        // whatever the polling loop happened to have cached.
        openImageLightbox(
          `/api/camera/${encodeURIComponent(room)}/snapshot.jpg?lb=${Date.now()}`,
          room.replace(/_/g, " "),
          ts,
        );
      });
    }
  });
}

// 2000ms = 0.5 fps. The dashboard is a "what's happening in each room"
// status board, not a video player — humans can't perceive motion
// smoothness much below ~10fps and we're nowhere near that anyway. Faster
// polling triggers cv2.imencode + frame reads across every configured room.
// safeInterval pauses on document.hidden, the active-tab guard above stops
// camera polling while the dashboard is on Settings/Logs/Perf, and the
// server-side preview cache deduplicates back-to-back requests.
safeInterval(refreshRoomFeeds, 2000);

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

function updateRoomAudio(roomId, db, peakDb) {
  if (!roomId || db == null) return;
  roomsCache[roomId] = Object.assign({}, roomsCache[roomId] || {}, {
    audio_db: db,
    audio_peak_db: peakDb,
  });
  const card = document.getElementById(`room-${roomId}`);
  if (!card) return;
  let meter = card.querySelector(".room-meter");
  if (!meter) {
    meter = document.createElement("div");
    meter.className = "room-meter";
    meter.innerHTML = `
      <div class="room-meter-fill"></div>
      <div class="room-meter-peak"></div>
      <div class="room-meter-label">—</div>`;
    card.appendChild(meter);
  }
  // Map -60 dBFS → 0%, 0 dBFS → 100%, clamp. Peak is rendered as a
  // thin marker line so we can see clipping margins for wake-word
  // debugging.
  const dbToPct = (v) => Math.max(0, Math.min(100, Math.round((v + 60) * (100 / 60))));
  const pct = dbToPct(db);
  const peakPct = peakDb != null ? dbToPct(peakDb) : null;
  const fill = meter.querySelector(".room-meter-fill");
  const peak = meter.querySelector(".room-meter-peak");
  const label = meter.querySelector(".room-meter-label");
  if (fill) {
    fill.style.width = `${pct}%`;
    // Visual cue for clipping risk: turn the fill warmer above -6dB.
    fill.classList.toggle("hot", db > -6);
  }
  if (peak && peakPct != null) {
    peak.style.left = `${peakPct}%`;
    peak.style.display = "block";
    peak.classList.toggle("clipping", peakDb > -1);
  } else if (peak) {
    peak.style.display = "none";
  }
  if (label) {
    label.textContent = peakDb != null
      ? `${db.toFixed(0)} / pk ${peakDb.toFixed(0)} dBFS`
      : `${db.toFixed(0)} dBFS`;
  }
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

safeInterval(updateClock, 1000);
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
safeInterval(renderReminders, 30000);
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

safeInterval(loadCalendar, 5 * 60 * 1000);  // refresh every 5 min
loadCalendar();

// ── Config editor ─────────────────────────────────────────────────────────

function loadConfig() {
  const ta = document.getElementById("config-yaml");
  const status = document.getElementById("config-status");
  if (!ta) return;
  fetch("/api/tunables")
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
  fetch("/api/tunables", {
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
      // The old in-home pending-list card was retired in favour of the
      // dedicated Pending Reviews tab — that tab pulls its own data
      // when loaded, and the reviews-tab-badge tracks unread count.
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

// (renderPending was retired alongside the home-tab pending-list card;
// the Pending Reviews tab is now the single surface for identity
// resolution. _personOptions is still used by the per-card dropdown in
// the Reviews tab — that's why it stays defined above.)

function _positionFaceBbox(img, overlay) {
  // bbox stored as [x1, y1, x2, y2] in source-image pixel coords.
  // Scale into the <img>'s displayed pixel coords using the
  // natural-vs-rendered ratio. Both images use object-fit:contain so
  // the ratio is uniform.
  const x1 = parseFloat(overlay.dataset.x1);
  const y1 = parseFloat(overlay.dataset.y1);
  const x2 = parseFloat(overlay.dataset.x2);
  const y2 = parseFloat(overlay.dataset.y2);
  if (!Number.isFinite(x1)) return;
  const natW = img.naturalWidth, natH = img.naturalHeight;
  if (!natW || !natH) return;
  const rect = img.getBoundingClientRect();
  // contain-fit: actual rendered image area inside the box.
  const scale = Math.min(rect.width / natW, rect.height / natH);
  const renderW = natW * scale, renderH = natH * scale;
  const offsetX = (rect.width - renderW) / 2;
  const offsetY = (rect.height - renderH) / 2;
  overlay.style.left = `${offsetX + x1 * scale}px`;
  overlay.style.top = `${offsetY + y1 * scale}px`;
  overlay.style.width = `${(x2 - x1) * scale}px`;
  overlay.style.height = `${(y2 - y1) * scale}px`;
}

// Old in-home loadPending() / resolvePending() helpers were retired
// alongside the home-tab pending-list card. The Pending Reviews tab is
// now the only place pending rows surface; its badge in the tab bar is
// refreshed via refreshReviewsBadge() on every WS event.

populatePersonRoomSelect();
loadPersons();
refreshReviewsBadge();

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
    // Switch the user to the Pending Reviews tab (which replaced the
    // old home-tab pending-list card) instead of scrolling to a card
    // that no longer exists.
    const btn = document.querySelector('.tab-btn[data-tab="reviews"]');
    if (btn) btn.click();
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
safeInterval(loadModels, 30000);

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
    a.href = `/api/camera/${encodeURIComponent(room)}/snapshot.jpg?full=1&t=${Date.now()}`;
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
safeInterval(loadPersonas, 10000);

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
  div.className = `pet-row pet-${pet.species || "unknown"} pet-row-clickable`;
  div.title = "Click for details + recent events";
  div.addEventListener("click", () => openPetLoreModal(pet));

  // Where the pet "is" — distinguish "currently visible" from
  // "last seen N minutes ago in X" so the unseen / departed states
  // actually tell Cole where the pet was last, not just "unseen".
  const state = String(pet.state || "").toLowerCase();
  const isSeen = state === "in_room_seen" || state === "present";
  const lastRoom = pet.last_seen_room || pet.likely_room || "?";
  const lastLandmark = pet.last_seen_landmark;
  const lastAgo = formatRelativeTs(pet.last_seen_ts);

  let location;
  let hedge = "";
  if (isSeen) {
    location = lastRoom;
    if (lastLandmark) location += ` · ${lastLandmark}`;
  } else if (state === "departed") {
    location = `departed · last seen in ${lastRoom}`;
    if (lastLandmark) location += ` · ${lastLandmark}`;
    if (pet.last_seen_ts) hedge = ` <span class="pet-hedge">${escapeHtml(lastAgo)}</span>`;
  } else if (state === "unmonitored_zone" && pet.unmonitored_home) {
    location = `probably in ${pet.unmonitored_home}`;
    if (pet.last_seen_ts) {
      hedge = ` <span class="pet-hedge">last seen in ${escapeHtml(lastRoom)} ${escapeHtml(lastAgo)}</span>`;
    }
  } else {
    // in_room_unseen, ambiguous, or anything else — be explicit that
    // this is a memory, not a live sighting.
    location = `last seen in ${lastRoom}`;
    if (lastLandmark) location += ` · ${lastLandmark}`;
    if (pet.last_seen_ts) hedge = ` <span class="pet-hedge">${escapeHtml(lastAgo)}</span>`;
    else hedge = ' <span class="pet-hedge">(no recent record)</span>';
  }
  if (pet.likely_room_inferred && pet.likely_room && !isSeen
      && state !== "unmonitored_zone" && state !== "departed") {
    // Cost-function guess — fold it in as an extra hint so the user
    // sees both the literal last sighting and the inferred likely room.
    if (pet.likely_room !== pet.last_seen_room) {
      hedge += ` <span class="pet-hedge">· probably ${escapeHtml(pet.likely_room)} now</span>`;
    }
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
      <span class="pet-where-prefix">${isSeen ? "in " : ""}</span><span class="pet-where-room">${escapeHtml(location)}</span>${hedge}
    </div>
    <div class="pet-care">${chips}</div>
  `;
  return div;
}

async function loadPets() {
  const list = document.getElementById("pets-list");
  if (!list) return;
  try {
    const res = await fetch("/api/world_model/pets");
    if (!res.ok) {
      list.innerHTML =
        '<div class="who-empty">Pets unavailable.</div>';
      return;
    }
    const body = await res.json();
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
    list.innerHTML =
      '<div class="who-empty">Pets unavailable.</div>';
  }
}
loadPets();
// 30s cadence — care-summary chips don't need real-time updates
// (food/litterbox events fire on the order of hours), state changes
// flow in via the WebSocket world.entity_event handler below.
safeInterval(loadPets, 30000);

// ── Pet lore card modal ───────────────────────────────────────────────────
// Click a pet row → open a modal with the full seed metadata (color,
// coat, personality, notes, distinctive features, etc.) and a tail of
// recent events for that pet. Closes on backdrop click or ESC.

function _kvRows(obj) {
  if (!obj || typeof obj !== "object") return "";
  const rows = [];
  for (const [k, v] of Object.entries(obj)) {
    if (v === null || v === undefined || v === "" ||
        (Array.isArray(v) && v.length === 0)) continue;
    let pretty;
    if (Array.isArray(v)) {
      pretty = v.map((x) => escapeHtml(String(x))).join(", ");
    } else if (typeof v === "object") {
      pretty = escapeHtml(JSON.stringify(v));
    } else {
      pretty = escapeHtml(String(v));
    }
    const label = k.replace(/_/g, " ");
    rows.push(
      `<div class="lore-row"><span class="lore-key">${escapeHtml(label)}</span>` +
      `<span class="lore-val">${pretty}</span></div>`,
    );
  }
  return rows.join("");
}

async function _fetchPetEvents(name) {
  try {
    const res = await fetch(
      `/api/world_model/pets/${encodeURIComponent(name)}/events?limit=30&hours_ago=168`,
    );
    if (!res.ok) return [];
    const body = await res.json();
    return Array.isArray(body.events) ? body.events : [];
  } catch (e) {
    console.warn("[fetchPetEvents] failed:", e);
    return [];
  }
}

async function _fetchPetThumbnails(name) {
  try {
    const res = await fetch(
      `/api/world_model/pets/${encodeURIComponent(name)}/thumbnails?limit=8`,
    );
    if (!res.ok) return [];
    const body = await res.json();
    return Array.isArray(body.thumbnails) ? body.thumbnails : [];
  } catch (e) {
    console.warn("[fetchPetThumbnails] failed:", e);
    return [];
  }
}

function _renderPetThumbnailStrip(thumbs, petName) {
  if (!thumbs || thumbs.length === 0) {
    return '<div class="who-empty">No snapshots yet. Run the cluster builder or wait for more captures.</div>';
  }
  return `<div class="lore-thumbs">` +
    thumbs.map((t) => {
      const ts = t.ts ? new Date(t.ts).toLocaleString() : "";
      const room = escapeHtml(t.room || "?");
      return `
        <div class="lore-thumb" data-event-id="${escapeHtml(t.event_id || "")}"
             title="${escapeHtml(petName)} · ${room} · ${escapeHtml(ts)}">
          <img src="${t.url}" alt="${escapeHtml(petName)}"
               loading="lazy"
               onerror="this.parentElement.classList.add('lore-thumb-broken');" />
          <div class="lore-thumb-meta">
            <span class="lore-thumb-room">${room}</span>
            <span class="lore-thumb-ago">${formatRelativeTs(t.ts)}</span>
          </div>
        </div>`;
    }).join("") +
    `</div>`;
}

async function _fetchPetSamples(name) {
  try {
    const res = await fetch(
      `/api/world_model/pets/${encodeURIComponent(name)}/samples`,
    );
    if (!res.ok) return [];
    const body = await res.json();
    return Array.isArray(body.samples) ? body.samples : [];
  } catch (e) {
    console.warn("[fetchPetSamples] failed:", e);
    return [];
  }
}

function _renderPetSamples(samples, petName) {
  if (!samples || samples.length === 0) {
    return '<div class="who-empty">No confirmed visual samples yet. '
      + 'Samples are saved when you tag this pet in the cluster labeler.</div>';
  }
  return `<div class="lore-thumbs">` +
    samples.map((s) => {
      const ts = s.created_at ? new Date(s.created_at).toLocaleString() : "";
      const room = escapeHtml(s.room || "?");
      return `
        <div class="lore-thumb" data-sample-id="${s.id}"
             style="position:relative;"
             title="${escapeHtml(petName)} · ${room} · ${escapeHtml(ts)} · ${escapeHtml(s.source || "")}">
          <button class="sample-del" data-sample-id="${s.id}"
                  title="Delete this sample"
                  style="position:absolute;top:2px;right:2px;z-index:2;
                         background:rgba(132,58,58,0.92);color:#fff;border:none;
                         border-radius:3px;cursor:pointer;font-size:11px;
                         line-height:1;padding:3px 6px;">×</button>
          <img src="${s.url}" alt="${escapeHtml(petName)}" loading="lazy"
               onerror="this.parentElement.classList.add('lore-thumb-broken');" />
          <div class="lore-thumb-meta">
            <span class="lore-thumb-room">${room}</span>
            <span class="lore-thumb-ago">${escapeHtml(s.source || "")}</span>
          </div>
        </div>`;
    }).join("") +
    `</div>`;
}

// Wire the delete buttons in the pet-samples slot. Re-renders the slot
// after a successful delete.
function _wirePetSampleDeletes(slot, petName) {
  slot.querySelectorAll(".sample-del").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const id = btn.dataset.sampleId;
      if (!id || !confirm("Delete this visual sample? This is permanent.")) return;
      btn.disabled = true;
      try {
        const r = await fetch(`/api/world_model/pet_samples/${id}`,
                              { method: "DELETE" });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const fresh = await _fetchPetSamples(petName);
        slot.innerHTML = _renderPetSamples(fresh, petName);
        _wirePetSampleDeletes(slot, petName);
      } catch (err) {
        console.warn("[petSampleDelete] failed:", err);
        btn.disabled = false;
        alert("Delete failed: " + err.message);
      }
    });
  });
}

function _renderPetEventList(events) {
  if (!events || events.length === 0) {
    return '<div class="who-empty">No events in the last week.</div>';
  }
  return events
    .map((e) => {
      const ts = e.ts ? new Date(e.ts).toLocaleString() : "?";
      const room = e.room ? ` · ${escapeHtml(e.room)}` : "";
      const lm = (e.metadata && e.metadata.landmark)
        ? ` <span class="lore-lm">${escapeHtml(e.metadata.landmark)}</span>` : "";
      const glyph = EVENT_TYPE_GLYPH[e.event_type] || "·";
      return `<div class="lore-event">
        <span class="lore-evt-glyph">${glyph}</span>
        <span class="lore-evt-type">${escapeHtml(e.event_type || "?")}</span>${lm}
        <span class="lore-evt-room">${room}</span>
        <span class="lore-evt-ts">${ts}</span>
      </div>`;
    })
    .join("");
}

function closePetLoreModal() {
  const m = document.getElementById("pet-lore-modal");
  if (m) m.remove();
  document.removeEventListener("keydown", _petLoreKeydown);
}

function _petLoreKeydown(e) {
  if (e.key === "Escape") closePetLoreModal();
}

async function openPetLoreModal(pet) {
  closePetLoreModal();
  const seed = pet.seed || {};
  const speciesGlyph = pet.species === "dog" ? "𓃡" : "𓃠";
  const overlay = document.createElement("div");
  overlay.id = "pet-lore-modal";
  overlay.className = "modal-overlay";
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) closePetLoreModal();
  });

  overlay.innerHTML = `
    <div class="modal-card">
      <div class="modal-head">
        <span class="pet-glyph">${speciesGlyph}</span>
        <span class="modal-title">${escapeHtml(pet.name || "?")}</span>
        <span class="modal-sub">${escapeHtml(pet.species || "")}</span>
        <button class="modal-close" aria-label="Close">×</button>
      </div>
      <div class="modal-body">
        <div class="modal-section">
          <div class="modal-section-label">CURRENT STATE</div>
          <div class="lore-rows">
            <div class="lore-row"><span class="lore-key">state</span>
              <span class="lore-val">${escapeHtml(pet.state || "?")}</span></div>
            <div class="lore-row"><span class="lore-key">likely room</span>
              <span class="lore-val">${escapeHtml(pet.likely_room || pet.last_seen_room || "?")}${
                pet.likely_room_inferred ? " (inferred)" : ""
              }</span></div>
            <div class="lore-row"><span class="lore-key">last seen</span>
              <span class="lore-val">${
                pet.last_seen_ts ? new Date(pet.last_seen_ts).toLocaleString() : "—"
              }</span></div>
          </div>
        </div>
        <div class="modal-section">
          <div class="modal-section-label">SNAPSHOTS</div>
          <div class="lore-thumbs-slot" id="lore-thumbs-slot">
            <div class="who-empty">Loading…</div>
          </div>
        </div>
        <div class="modal-section">
          <div class="modal-section-label">VISUAL SAMPLES</div>
          <div class="lore-thumbs-slot" id="pet-samples-slot">
            <div class="who-empty">Loading…</div>
          </div>
        </div>
        <div class="modal-section">
          <div class="modal-section-label">LORE</div>
          <div class="lore-rows">${_kvRows(seed) ||
            '<div class="who-empty">No seed metadata in config.yaml for this pet.</div>'
          }</div>
        </div>
        <div class="modal-section">
          <div class="modal-section-label">RECENT EVENTS (7 days)</div>
          <div class="lore-events" id="lore-events-list">
            <div class="who-empty">Loading…</div>
          </div>
        </div>
        <div class="modal-foot">
          <a href="/clusters" class="modal-link" target="_blank">
            Wrong pet? Open cluster labeler →
          </a>
        </div>
      </div>
    </div>`;

  overlay.querySelector(".modal-close").addEventListener("click", closePetLoreModal);
  document.body.appendChild(overlay);
  document.addEventListener("keydown", _petLoreKeydown);

  // Fire both fetches in parallel — the events list is the longer
  // wait (DB scan); thumbnails come back fast.
  const [events, thumbs, samples] = await Promise.all([
    _fetchPetEvents(pet.name),
    _fetchPetThumbnails(pet.name),
    _fetchPetSamples(pet.name),
  ]);
  const eventsSlot = overlay.querySelector("#lore-events-list");
  if (eventsSlot) eventsSlot.innerHTML = _renderPetEventList(events);
  const samplesSlot = overlay.querySelector("#pet-samples-slot");
  if (samplesSlot) {
    samplesSlot.innerHTML = _renderPetSamples(samples, pet.name);
    _wirePetSampleDeletes(samplesSlot, pet.name);
  }
  const thumbsSlot = overlay.querySelector("#lore-thumbs-slot");
  if (thumbsSlot) {
    thumbsSlot.innerHTML = _renderPetThumbnailStrip(thumbs, pet.name);
    // Click thumbnail → lightbox at full resolution.
    thumbsSlot.querySelectorAll(".lore-thumb img").forEach((img) => {
      img.style.cursor = "zoom-in";
      img.addEventListener("click", () => openImageLightbox(
        img.src,
        `${pet.name} · ${img.parentElement.parentElement.querySelector(".lore-thumb-room").textContent}`,
        img.parentElement.parentElement.dataset.eventTs || null,
      ));
    });
  }
}

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
  // Coalesced run counter ("Anna lost_visibility ×4 over 8s") — only
  // shown when the renderer collapsed multiple adjacent identical
  // events. Span stays hidden for singleton rows.
  const countBadge = (ev._collapse_count && ev._collapse_count > 1)
    ? ` <span class="event-count" title="${ev._collapse_count} consecutive identical events">×${ev._collapse_count}</span>`
    : "";
  const div = document.createElement("div");
  div.className = `event-row event-type-${ev.event_type || "unknown"}`;
  // Cat/dog events get a click target — opens the relabel modal so the
  // user can fix mis-attributions ("this 'Serval' was actually Socks").
  const isAnimal = ev.entity_type === "cat" || ev.entity_type === "dog";
  const tagBtn = isAnimal
    ? `<button class="event-tag-btn" title="Tag this as a specific pet">tag</button>`
    : "";
  div.innerHTML = `
    <span class="event-glyph">${glyph}</span>
    <span class="event-name">${escapeHtml(name)}</span>
    <span class="event-label">${escapeHtml(label)}</span>${countBadge}${room}${lm}
    <span class="event-ago">${ago}</span>${tagBtn}
  `;
  if (isAnimal) {
    div.querySelector(".event-tag-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      openPetTagModal(ev);
    });
  }
  return div;
}

// ── Pet tag modal ─────────────────────────────────────────────────────────
// Dropdown of resident pets (filtered to the event's species) +
// thumbnail of the event being relabeled, if a snapshot exists. Hits
// POST /api/world_model/events/{id}/relabel.

let _petsCacheForTagging = null;
async function _ensurePetsCacheForTagging() {
  if (_petsCacheForTagging) return _petsCacheForTagging;
  try {
    const res = await fetch("/api/world_model/pets");
    if (!res.ok) return [];
    const body = await res.json();
    _petsCacheForTagging = Array.isArray(body.pets) ? body.pets : [];
    return _petsCacheForTagging;
  } catch {
    return [];
  }
}

function closePetTagModal() {
  const m = document.getElementById("pet-tag-modal");
  if (m) m.remove();
  document.removeEventListener("keydown", _petTagKeydown);
}
function _petTagKeydown(e) {
  if (e.key === "Escape") closePetTagModal();
}

// ── Live "Tag pet in frame" modal ─────────────────────────────────────────
// Triggered by the 🐾 button on a camera tile. Fetches a fresh snapshot
// + recent cat/dog detections in that room, draws clickable bboxes on
// top of the snapshot. Click a box → pick pet → POST tag_in_frame so
// the world model relabels all overlapping recent events.

function closeLivePetTagModal() {
  const m = document.getElementById("live-pet-tag-modal");
  if (m) m.remove();
  document.removeEventListener("keydown", _livePetKeydown);
}
function _livePetKeydown(e) {
  if (e.key === "Escape") closeLivePetTagModal();
}

async function openLivePetTagModal(room) {
  closeLivePetTagModal();
  const pets = await _ensurePetsCacheForTagging();
  if (pets.length === 0) {
    alert("No resident pets configured in config.yaml.");
    return;
  }
  // YOLO runs on EVERY open. Use ONE live detector pass for cats,
  // dogs, and display-only people so all boxes come from the same
  // captured frame. Separate pet/person calls made repeated clicks look
  // inconsistent because each call grabbed a fresh camera frame.
  const stamp = Date.now();
  const snapUrl = `/api/camera/${encodeURIComponent(room)}/snapshot.jpg?lb=${stamp}`;
  let [liveBody, recentBody] = await Promise.all([
    fetch("/api/world_model/yolo_now", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ room, species: ["cat", "dog", "person"] }),
    }).then(r => r.ok ? r.json() : { detections: [] }).catch(() => ({ detections: [] })),
    fetch(`/api/world_model/recent_animal_detections?room=${encodeURIComponent(room)}&seconds=30`)
      .then(r => r.ok ? r.json() : { detections: [] }).catch(() => ({ detections: [] })),
  ]);
  const allLiveDets = Array.isArray(liveBody.detections) ? liveBody.detections : [];
  const liveDets = allLiveDets.filter((d) => d.species === "cat" || d.species === "dog");
  const personDets = allLiveDets.filter((d) => d.species === "person");
  const recentDets = Array.isArray(recentBody.detections) ? recentBody.detections : [];

  // Merge: live detections first (they're the actual current frame),
  // each enriched with the closest matching recent event's
  // entity_name (so the bbox label shows what the system currently
  // thinks it is). Fall back to recent events for cats sitting still.
  function _iou(a, b) {
    const [a1, b1, a2, b2] = a;
    const [c1, d1, c2, d2] = b;
    const ix1 = Math.max(a1, c1), iy1 = Math.max(b1, d1);
    const ix2 = Math.min(a2, c2), iy2 = Math.min(b2, d2);
    const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
    const aa = (a2 - a1) * (b2 - b1);
    const bb = (c2 - c1) * (d2 - d1);
    const u = aa + bb - inter;
    return u > 0 ? inter / u : 0;
  }
  const unique = [];
  for (const live of liveDets) {
    let bestMatch = null;
    let bestIou = 0;
    for (const r of recentDets) {
      if (!r.bbox || r.bbox.length !== 4) continue;
      if (r.entity_type !== live.species) continue;
      const i = _iou(live.bbox, r.bbox);
      if (i > bestIou) { bestIou = i; bestMatch = r; }
    }
    const hasRecentName = bestIou > 0.3 && bestMatch;
    const suggestedName = live.suggested_name || (live.pet_match && live.pet_match.accepted ? live.pet_match.pet_name : null);
    unique.push({
      bbox: live.bbox,
      entity_type: live.species,
      entity_name: hasRecentName
        ? bestMatch.entity_name
        : suggestedName
          ? suggestedName
        : `(new ${live.species})`,
      entity_id: hasRecentName ? bestMatch.entity_id : (live.suggested_entity_id || null),
      ts: hasRecentName ? bestMatch.ts : new Date().toISOString(),
      confidence: live.confidence,
      pet_match: live.pet_match || null,
      source: "yolo_live",
    });
  }
  // Drop in any recent events whose bbox doesn't have a corresponding
  // live detection — covers cats that moved enough to log but are
  // briefly out of frame on the snapshot we just fetched.
  for (const r of recentDets) {
    if (!r.bbox || r.bbox.length !== 4) continue;
    const overlaps = unique.some(u => _iou(u.bbox, r.bbox) > 0.3);
    if (overlaps) continue;
    unique.push({ ...r, source: "event_log" });
    if (unique.length >= 12) break;
  }
  // Append person detections as display-only "flag" boxes. They
  // render in a distinct color (see CSS .live-pet-box-person) so
  // Cole can see who's where while tagging pets — no assign action
  // attaches to them, identity flow lives in the Pending Reviews tab.
  for (const p of personDets) {
    if (!p.bbox || p.bbox.length !== 4) continue;
    unique.push({
      bbox: p.bbox,
      entity_type: "person",
      entity_name: "(person)",
      entity_id: null,
      ts: new Date().toISOString(),
      confidence: p.confidence,
      source: "yolo_live_person",
    });
  }

  const overlay = document.createElement("div");
  overlay.id = "live-pet-tag-modal";
  overlay.className = "modal-overlay";
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) closeLivePetTagModal();
  });

  const opts = pets.map((p) =>
    `<option value="${escapeHtml(p.name)}" data-species="${escapeHtml(p.species)}">${escapeHtml(p.name)} (${escapeHtml(p.species)})</option>`,
  ).join("");

  overlay.innerHTML = `
    <div class="modal-card live-pet-card">
      <div class="modal-head">
        <span class="modal-title">Tag a pet in ${escapeHtml(room.replace(/_/g, " "))}</span>
        <button class="modal-close" aria-label="Close">×</button>
      </div>
      <div class="modal-body">
        <div class="live-pet-hint">
          Yellow / purple boxes = pets JARVIS sees in the frame
          <i>right now</i> (YOLO live). Box label shows what the system
          currently thinks they are. Click a box, pick the correct
          pet, Save. Updates the last 30s of events whose bboxes
          overlap (IoU &gt; 0.3).
        </div>
        <div class="live-pet-canvas-wrap">
          <img id="live-pet-snapshot" src="${snapUrl}" alt="snapshot" />
          <div id="live-pet-overlays"></div>
        </div>
        <div class="live-pet-tools">
          <button class="dev-btn" id="live-pet-draw"
                  title="Drag a box over a pet YOLO missed in the full-frame pass. JARVIS crops that region, upscales it, and re-runs detection at a low threshold — it can recover small or low-contrast pets the normal pass skipped.">
            ✏ Draw region to recheck
          </button>
          <span class="live-pet-status" id="live-pet-draw-status"></span>
        </div>
        ${unique.length === 0
          ? `<div class="who-empty">No cat/dog detections in the last 30s. The frame is shown but there's nothing to relabel — try again when a pet is visible.</div>`
          : `<div class="live-pet-detections">
              <div class="live-pet-section-label">Recent detections (click one):</div>
              <div class="live-pet-det-list" id="live-pet-det-list"></div>
            </div>`}
        <div class="live-pet-form" id="live-pet-form" hidden>
          <label>
            This is actually:
            <select class="dev-select" id="live-pet-select">${opts}</select>
          </label>
          <button class="dev-btn" id="live-pet-save">Save</button>
          <button class="dev-btn live-pet-not-pet" id="live-pet-not-pet"
                  title="Not a pet at all — delete these bogus detections + mark this region as a false-positive zone for 6h.">
            Not a pet
          </button>
          <button class="dev-btn live-pet-recheck" id="live-pet-recheck"
                  title="Rerun YOLO on just this region at a lower confidence threshold. Catches small/partial pets the full-frame pass missed.">
            Recheck region
          </button>
          <button class="dev-btn live-pet-cancel" id="live-pet-cancel">Cancel</button>
          <span class="live-pet-status" id="live-pet-status"></span>
        </div>
      </div>
    </div>`;
  overlay.querySelector(".modal-close").addEventListener("click", closeLivePetTagModal);
  document.body.appendChild(overlay);
  document.addEventListener("keydown", _livePetKeydown);

  // Once the snapshot loads, render bbox overlays positioned in the
  // displayed coordinate space. Bboxes from the DB are in source-pixel
  // coords, so scale via natural-vs-rendered ratio.
  const img = overlay.querySelector("#live-pet-snapshot");
  const overlaysSlot = overlay.querySelector("#live-pet-overlays");
  const detListSlot = overlay.querySelector("#live-pet-det-list");

  let selectedDet = null;

  function renderOverlays() {
    overlaysSlot.innerHTML = "";
    if (!img.naturalWidth || !img.naturalHeight) return;
    const rect = img.getBoundingClientRect();
    const wrapRect = overlaysSlot.parentElement.getBoundingClientRect();
    const scale = Math.min(rect.width / img.naturalWidth, rect.height / img.naturalHeight);
    const renderW = img.naturalWidth * scale, renderH = img.naturalHeight * scale;
    const offsetX = (rect.width - renderW) / 2 + (rect.left - wrapRect.left);
    const offsetY = (rect.height - renderH) / 2 + (rect.top - wrapRect.top);

    unique.forEach((d, idx) => {
      const [x1, y1, x2, y2] = d.bbox;
      const box = document.createElement("div");
      box.className = `live-pet-box live-pet-box-${d.entity_type}`;
      box.dataset.idx = idx;
      box.style.left = `${offsetX + x1 * scale}px`;
      box.style.top = `${offsetY + y1 * scale}px`;
      box.style.width = `${(x2 - x1) * scale}px`;
      box.style.height = `${(y2 - y1) * scale}px`;
      box.innerHTML = `<span class="live-pet-box-label">${escapeHtml(d.entity_name || d.entity_type)}</span>`;
      box.addEventListener("click", () => _selectDet(idx));
      overlaysSlot.appendChild(box);
    });
  }

  function _selectDet(idx) {
    selectedDet = unique[idx];
    overlay.querySelectorAll(".live-pet-box").forEach((b) => {
      b.classList.toggle("active", Number(b.dataset.idx) === idx);
    });
    overlay.querySelectorAll(".live-pet-det-row").forEach((r) => {
      r.classList.toggle("active", Number(r.dataset.idx) === idx);
    });
    const form = overlay.querySelector("#live-pet-form");
    if (form) {
      form.hidden = false;
      // Group the dropdown by species but DO NOT hide the wrong-species
      // entries — YOLO sometimes misclassifies a cat as a dog (or
      // vice versa) and the user must be able to override. Sort so the
      // detected species comes first, then a separator, then the other.
      const sel = overlay.querySelector("#live-pet-select");
      const detected = selectedDet.entity_type;
      const opts = Array.from(sel.options);
      // Restore visibility in case a prior selection had hidden some.
      opts.forEach((o) => { o.hidden = false; });
      // Stable sort: detected species first, alphabetical inside each group.
      opts.sort((a, b) => {
        const sa = a.dataset.species, sb = b.dataset.species;
        if (sa === detected && sb !== detected) return -1;
        if (sb === detected && sa !== detected) return 1;
        return a.text.localeCompare(b.text);
      });
      // Re-insert in sorted order; insert a disabled visual separator
      // between species groups so it's obvious which pets match the
      // detected species and which require an override.
      sel.innerHTML = "";
      let prevSpecies = null;
      opts.forEach((o) => {
        if (prevSpecies !== null && o.dataset.species !== prevSpecies) {
          const sep = document.createElement("option");
          sep.disabled = true;
          sep.textContent = `── (override: ${o.dataset.species}) ──`;
          sel.appendChild(sep);
        }
        sel.appendChild(o);
        prevSpecies = o.dataset.species;
      });
      // Default to the first matching-species option.
      const first = opts.find((o) => o.dataset.species === detected) || opts[0];
      if (first) sel.value = first.value;
    }
  }

  if (img.complete && img.naturalWidth) renderOverlays();
  else img.addEventListener("load", renderOverlays);
  window.addEventListener("resize", renderOverlays);

  // Side list of detections too, in case the bbox is too small to click.
  if (detListSlot) {
    detListSlot.innerHTML = unique
      .map((d, i) => `
        <div class="live-pet-det-row" data-idx="${i}">
          <span class="live-pet-det-name">${escapeHtml(d.entity_name || `?_${d.entity_type}`)}</span>
          <span class="live-pet-det-species">${escapeHtml(d.entity_type)}</span>
          <span class="live-pet-det-ago">${formatRelativeTs(d.ts)}</span>
        </div>`)
      .join("");
    detListSlot.querySelectorAll(".live-pet-det-row").forEach((row) => {
      row.addEventListener("click", () => _selectDet(Number(row.dataset.idx)));
    });
  }

  // Mark a detection as resolved (visually) and reset the form so
  // the user can pick the next box without re-opening the modal —
  // there are usually 2-3 animals in frame and re-opening loses the
  // YOLO call's context.
  function _markResolved(idx, statusText) {
    if (idx == null) return;
    const boxEl = overlay.querySelector(`.live-pet-box[data-idx="${idx}"]`);
    const rowEl = overlay.querySelector(`.live-pet-det-row[data-idx="${idx}"]`);
    if (boxEl) {
      boxEl.classList.remove("active");
      boxEl.classList.add("resolved");
      const lbl = boxEl.querySelector(".live-pet-box-label");
      if (lbl && statusText) lbl.textContent = statusText;
    }
    if (rowEl) {
      rowEl.classList.remove("active");
      rowEl.classList.add("resolved");
    }
    selectedDet = null;
    overlay.querySelector("#live-pet-form").hidden = true;
  }

  function _selectedIdx() {
    if (!selectedDet) return null;
    const i = unique.indexOf(selectedDet);
    return i >= 0 ? i : null;
  }

  // Save handler.
  const saveBtn = overlay.querySelector("#live-pet-save");
  const cancelBtn = overlay.querySelector("#live-pet-cancel");
  const notPetBtn = overlay.querySelector("#live-pet-not-pet");
  const recheckBtn = overlay.querySelector("#live-pet-recheck");
  const statusEl = overlay.querySelector("#live-pet-status");
  if (saveBtn) {
    saveBtn.addEventListener("click", async () => {
      if (!selectedDet) return;
      // A "person" box here is usually a pet YOLO misclassified — that is
      // exactly the case the user needs to correct, so tagging it as a pet
      // is allowed and goes through the same tag_in_frame path. The backend
      // matches person-typed event rows too and flips them to the pet.
      const target = overlay.querySelector("#live-pet-select").value;
      if (!target) return;
      if (selectedDet.entity_type === "person") {
        statusEl.textContent = `correcting person → ${target}…`;
      }
      const idx = _selectedIdx();
      statusEl.textContent = "saving…";
      statusEl.className = "live-pet-status";
      try {
        const res = await fetch("/api/world_model/tag_in_frame", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            room,
            pet_name: target,
            bbox: selectedDet.bbox,
            seconds: 30,
          }),
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || `HTTP ${res.status}`);
        }
        const body = await res.json();
        statusEl.textContent = `relabeled ${body.relabeled} event(s) as ${body.pet_name} ✓ — pick the next box.`;
        statusEl.classList.add("ok");
        _markResolved(idx, body.pet_name);
        loadWorldEvents();
      } catch (e) {
        statusEl.textContent = `failed: ${e.message || e}`;
        statusEl.classList.add("err");
      }
    });
  }
  if (notPetBtn) {
    notPetBtn.addEventListener("click", async () => {
      if (!selectedDet) return;
      const idx = _selectedIdx();
      statusEl.textContent = "marking false-positive…";
      statusEl.className = "live-pet-status";
      try {
        const res = await fetch("/api/world_model/not_an_animal", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            room,
            bbox: selectedDet.bbox,
            seconds: 30,
          }),
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || `HTTP ${res.status}`);
        }
        const body = await res.json();
        statusEl.textContent = `cleared ${body.deleted_events} bogus event(s); region suppressed for 6h ✓`;
        statusEl.classList.add("ok");
        _markResolved(idx, "(not a pet)");
        loadWorldEvents();
      } catch (e) {
        statusEl.textContent = `failed: ${e.message || e}`;
        statusEl.classList.add("err");
      }
    });
  }
  // Re-run YOLO on one region — cropped, upscaled by YOLO's own
  // letterbox, at a low confidence threshold. The backend takes any
  // bbox, so this serves both the per-detection "Recheck region"
  // button and the freehand "Draw region" tool below.
  // `statusTarget` is whichever status span the caller wants updated.
  async function _recheckRegion(bbox, statusTarget) {
    const st = statusTarget || statusEl;
    st.textContent = "rechecking region at lower threshold…";
    st.className = st.id === "live-pet-draw-status"
      ? "live-pet-status" : "live-pet-status";
    try {
      const res = await fetch("/api/world_model/yolo_region", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ room, bbox, conf: 0.08, padding: 0.20 }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const body = await res.json();
      const found = (body.detections || []).filter(
        (d) => d.class === "cat" || d.class === "dog" || d.class === "person",
      );
      if (found.length === 0) {
        st.textContent = "no cat/dog/person found even at low threshold.";
        st.classList.add("err");
        return 0;
      }
      // Inject the new detections into `unique` and re-render overlays.
      for (const d of found) {
        unique.push({
          bbox: d.box,
          entity_type: d.class,
          entity_name: d.class === "person" ? "(person)" : `(rechecked ${d.class})`,
          entity_id: null,
          ts: new Date().toISOString(),
          confidence: d.confidence,
          source: "yolo_recheck",
        });
      }
      renderOverlays();
      if (detListSlot) {
        detListSlot.innerHTML = unique
          .map((d, i) => `
            <div class="live-pet-det-row" data-idx="${i}">
              <span class="live-pet-det-name">${escapeHtml(d.entity_name || `?_${d.entity_type}`)}</span>
              <span class="live-pet-det-species">${escapeHtml(d.entity_type)}</span>
              <span class="live-pet-det-ago">${formatRelativeTs(d.ts)}</span>
            </div>`)
          .join("");
        detListSlot.querySelectorAll(".live-pet-det-row").forEach((row) => {
          row.addEventListener("click", () => _selectDet(Number(row.dataset.idx)));
        });
      }
      st.textContent = `found ${found.length} new detection(s) ✓ — click to tag.`;
      st.classList.add("ok");
      return found.length;
    } catch (e) {
      st.textContent = `recheck failed: ${e.message || e}`;
      st.classList.add("err");
      return 0;
    }
  }

  if (recheckBtn) {
    recheckBtn.addEventListener("click", () => {
      if (!selectedDet) return;
      _recheckRegion(selectedDet.bbox, statusEl);
    });
  }

  // ── Freehand "draw a region" recheck ─────────────────────────────────────
  // YOLO's full-frame pass misses small / low-contrast / partially-hidden
  // pets. This lets the user drag a box over WHERE they know a pet is so
  // the detector looks there specifically. Works even with zero existing
  // detections (the exact case the per-detection Recheck button can't
  // handle — there's no box to select).
  const drawBtn = overlay.querySelector("#live-pet-draw");
  const drawStatus = overlay.querySelector("#live-pet-draw-status");
  let drawArmed = false;

  function _setDrawArmed(on) {
    drawArmed = on;
    if (drawBtn) drawBtn.classList.toggle("armed", on);
    overlaysSlot.style.cursor = on ? "crosshair" : "";
    // #live-pet-overlays is pointer-events:none in CSS (so the image and
    // the detection boxes handle clicks). While the draw tool is armed we
    // flip the container to pointer-events:auto so it actually receives
    // the pointerdown — otherwise the drag handler never fires.
    overlaysSlot.style.pointerEvents = on ? "auto" : "";
    if (drawStatus) {
      drawStatus.textContent = on
        ? "drag a box over the pet, then release"
        : "";
      drawStatus.className = "live-pet-status";
    }
  }

  if (drawBtn) {
    drawBtn.addEventListener("click", () => _setDrawArmed(!drawArmed));
  }

  overlaysSlot.addEventListener("pointerdown", (ev) => {
    if (!drawArmed || !img.naturalWidth) return;
    ev.preventDefault();
    const oRect = overlaysSlot.getBoundingClientRect();
    const iRect = img.getBoundingClientRect();
    const scale = Math.min(
      iRect.width / img.naturalWidth, iRect.height / img.naturalHeight);
    const renderW = img.naturalWidth * scale, renderH = img.naturalHeight * scale;
    const offX = (iRect.width - renderW) / 2 + (iRect.left - oRect.left);
    const offY = (iRect.height - renderH) / 2 + (iRect.top - oRect.top);
    const startX = ev.clientX - oRect.left;
    const startY = ev.clientY - oRect.top;

    const rectEl = document.createElement("div");
    rectEl.className = "live-pet-draw-rect";
    rectEl.style.cssText =
      "position:absolute;border:2px dashed #ffd24d;" +
      "background:rgba(255,210,77,0.12);pointer-events:none;z-index:5;";
    overlaysSlot.appendChild(rectEl);

    let curX = startX, curY = startY;
    const onMove = (mv) => {
      curX = mv.clientX - oRect.left;
      curY = mv.clientY - oRect.top;
      rectEl.style.left = `${Math.min(startX, curX)}px`;
      rectEl.style.top = `${Math.min(startY, curY)}px`;
      rectEl.style.width = `${Math.abs(curX - startX)}px`;
      rectEl.style.height = `${Math.abs(curY - startY)}px`;
    };
    const onUp = async () => {
      document.removeEventListener("pointermove", onMove);
      document.removeEventListener("pointerup", onUp);
      rectEl.remove();
      _setDrawArmed(false);
      // Overlay-local rect → source-pixel bbox (inverse of renderOverlays).
      const toSrc = (lx, ly) => [
        Math.max(0, Math.min(img.naturalWidth, Math.round((lx - offX) / scale))),
        Math.max(0, Math.min(img.naturalHeight, Math.round((ly - offY) / scale))),
      ];
      const [sx1, sy1] = toSrc(Math.min(startX, curX), Math.min(startY, curY));
      const [sx2, sy2] = toSrc(Math.max(startX, curX), Math.max(startY, curY));
      if (sx2 - sx1 < 8 || sy2 - sy1 < 8) {
        if (drawStatus) drawStatus.textContent = "region too small — try again.";
        return;
      }
      await _recheckRegion([sx1, sy1, sx2, sy2], drawStatus);
    };
    document.addEventListener("pointermove", onMove);
    document.addEventListener("pointerup", onUp);
  });
  if (cancelBtn) {
    cancelBtn.addEventListener("click", () => {
      selectedDet = null;
      overlay.querySelector("#live-pet-form").hidden = true;
      overlay.querySelectorAll(".live-pet-box, .live-pet-det-row").forEach((el) => {
        if (!el.classList.contains("resolved")) el.classList.remove("active");
      });
    });
  }
}

async function openPetTagModal(ev) {
  closePetTagModal();
  const pets = await _ensurePetsCacheForTagging();
  const species = ev.entity_type;
  const candidates = pets.filter((p) => p.species === species);
  if (candidates.length === 0) {
    openImageLightbox(
      null,
      `No resident ${species}s configured to tag this as.`,
      ev.ts,
    );
    return;
  }
  const thumbUrl = ev.id
    ? `/api/world_model/cluster/event/${encodeURIComponent(ev.id)}/image.jpg`
    : null;
  const overlay = document.createElement("div");
  overlay.id = "pet-tag-modal";
  overlay.className = "modal-overlay";
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) closePetTagModal();
  });
  const opts = candidates.map((p) =>
    `<option value="${escapeHtml(p.name)}"${
      p.name === ev.entity_name ? " selected" : ""
    }>${escapeHtml(p.name)}</option>`,
  ).join("");
  overlay.innerHTML = `
    <div class="modal-card pet-tag-card">
      <div class="modal-head">
        <span class="modal-title">Tag ${escapeHtml(species)} event</span>
        <button class="modal-close" aria-label="Close">×</button>
      </div>
      <div class="modal-body">
        <div class="pet-tag-current">
          Currently labeled <b>${escapeHtml(ev.entity_name || "?")}</b>
          in <b>${escapeHtml(ev.room || "?")}</b>
          (${escapeHtml(ev.event_type || "?")}).
        </div>
        ${thumbUrl
          ? `<img class="pet-tag-thumb" src="${thumbUrl}"
                  onerror="this.outerHTML='<div class=\\'pet-tag-noimg\\'>No snapshot stored for this event.</div>'"
                  alt="snapshot" />`
          : `<div class="pet-tag-noimg">No snapshot for this event.</div>`
        }
        <label class="pet-tag-label">
          This is actually:
          <select class="dev-select" id="pet-tag-select">${opts}</select>
        </label>
        <div class="pet-tag-actions">
          <button class="dev-btn" id="pet-tag-save">Save</button>
          <span class="pet-tag-status" id="pet-tag-status"></span>
        </div>
      </div>
    </div>`;
  overlay.querySelector(".modal-close").addEventListener("click", closePetTagModal);
  document.body.appendChild(overlay);
  document.addEventListener("keydown", _petTagKeydown);

  overlay.querySelector("#pet-tag-save").addEventListener("click", async () => {
    const sel = overlay.querySelector("#pet-tag-select");
    const status = overlay.querySelector("#pet-tag-status");
    const target = sel.value;
    if (!target) return;
    status.textContent = "saving…";
    try {
      const res = await fetch(
        `/api/world_model/events/${encodeURIComponent(ev.id)}/relabel`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pet_name: target }),
        },
      );
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      status.textContent = "saved ✓";
      status.classList.add("ok");
      // Refresh the events panel so the relabel is visible.
      setTimeout(() => { closePetTagModal(); loadWorldEvents(); }, 600);
    } catch (e) {
      status.textContent = `failed: ${e.message || e}`;
      status.classList.add("err");
    }
  });
}

function _coalesceEvents(events, maxGapSeconds = 30) {
  // Server returns DESC by ts. Walk from newest to oldest; if the
  // next-older event has the same entity_id + event_type + room and
  // happened within `maxGapSeconds`, fold its count into the current
  // entry instead of emitting a new row. Lets the UI show
  // "Anna lost_visibility ×7" instead of seven near-identical rows.
  const out = [];
  let cur = null;
  for (const ev of events) {
    const sameKey = (
      cur &&
      cur.entity_id === ev.entity_id &&
      cur.event_type === ev.event_type &&
      cur.room === ev.room
    );
    const dt = cur
      ? Math.abs(new Date(cur.ts).getTime() - new Date(ev.ts).getTime()) / 1000
      : Infinity;
    if (sameKey && dt <= maxGapSeconds) {
      cur._collapse_count = (cur._collapse_count || 1) + 1;
    } else {
      cur = { ...ev, _collapse_count: 1 };
      out.push(cur);
    }
  }
  return out;
}

async function loadWorldEvents() {
  try {
    // Pull a bigger window so collapsing has signal — the panel still
    // renders ~20 rows, but they each represent a real change rather
    // than a flicker.
    const res = await fetch("/api/world_model/events?limit=120");
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
    const coalesced = _coalesceEvents(events).slice(0, 25);
    list.innerHTML = "";
    coalesced.forEach((ev) => list.appendChild(renderEventRow(ev)));
  } catch (err) {
    console.warn("[loadWorldEvents] failed:", err);
  }
}
loadWorldEvents();
// 5s cadence so landmark dwell events appear quickly during dev. Cheap:
// the endpoint reads the indexed event log + decodes JSON, no I/O fanout.
safeInterval(loadWorldEvents, 5000);

// ── Anomalies (§25 — behavioral anomaly review queue) ─────────────────────
// AnomalyScorer fires world.anomaly when a resident's world event scores
// unusual against their nightly behavioral profile. This card is the review
// queue: "not unusual" POSTs an invalidate, which feeds auto_tune's
// false-positive rate so the score threshold self-corrects over time.

function renderAnomalyRow(a) {
  const div = document.createElement("div");
  div.className = "anomaly-row" + (a.invalidated ? " anomaly-invalidated" : "");
  const score = Number(a.score || 0);
  const ev = a.event || {};
  const comps = a.components || {};
  // Components are per-signal sub-scores; show the ones that contributed,
  // strongest first, as "room_at_time 6.5 · time_of_day 4.0".
  const compStr =
    Object.entries(comps)
      .filter(([, v]) => Number(v) > 0)
      .sort((x, y) => Number(y[1]) - Number(x[1]))
      .map(([k, v]) => `${escapeHtml(k)} ${Number(v).toFixed(1)}`)
      .join(" · ") || "—";
  const where = [ev.event_type, ev.room]
    .filter(Boolean)
    .map(escapeHtml)
    .join(" · ");
  const ago = formatRelativeTs(a.ts);
  const scoreClass = score >= 8 ? " anomaly-score-high" : "";
  div.innerHTML = `
    <div class="anomaly-score${scoreClass}" title="anomaly score 0–10">${score.toFixed(1)}</div>
    <div class="anomaly-body">
      <div class="anomaly-line">
        <span class="anomaly-name">${escapeHtml(a.entity_name || "?")}</span>
        ${where ? `<span class="anomaly-where">${where}</span>` : ""}
        <span class="anomaly-ago">${escapeHtml(ago)}</span>
      </div>
      <div class="anomaly-components">${compStr}</div>
      ${
        a.invalidated
          ? `<div class="anomaly-fp">marked not unusual${
              a.invalidated_reason
                ? ": " + escapeHtml(a.invalidated_reason)
                : ""
            }</div>`
          : ""
      }
    </div>
    ${
      a.invalidated
        ? ""
        : `<button class="anomaly-dismiss dev-btn">not unusual</button>`
    }
  `;
  const btn = div.querySelector(".anomaly-dismiss");
  if (btn) {
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      btn.textContent = "…";
      try {
        const res = await fetch(
          `/api/world_model/anomalies/${encodeURIComponent(a.id)}/invalidate`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ reason: "dashboard review" }),
          },
        );
        if (res.ok) {
          loadAnomalies();
        } else {
          btn.disabled = false;
          btn.textContent = "not unusual";
        }
      } catch (e) {
        console.warn("[anomaly invalidate] failed:", e);
        btn.disabled = false;
        btn.textContent = "not unusual";
      }
    });
  }
  return div;
}

async function loadAnomalies() {
  try {
    const res = await fetch("/api/world_model/anomalies?limit=40");
    if (!res.ok) return;
    const body = await res.json();
    const list = document.getElementById("anomalies-list");
    if (!list) return;
    const thresholdEl = document.getElementById("anomaly-threshold");
    if (!body.available) {
      list.innerHTML =
        '<div class="who-empty">Anomaly scoring unavailable.</div>';
      if (thresholdEl) thresholdEl.textContent = "";
      setCardTier("anomalies-card", null);
      return;
    }
    if (thresholdEl) {
      thresholdEl.textContent =
        body.threshold != null ? `· fires above ${body.threshold}` : "";
    }
    const anomalies = Array.isArray(body.anomalies) ? body.anomalies : [];
    setCardTier("anomalies-card", anomalies.length > 0 ? "attention" : null);
    if (anomalies.length === 0) {
      list.innerHTML =
        '<div class="who-empty">No anomalies — behavior looks normal.</div>';
      return;
    }
    list.innerHTML = "";
    anomalies.forEach((a) => list.appendChild(renderAnomalyRow(a)));
  } catch (err) {
    console.warn("[loadAnomalies] failed:", err);
  }
}
loadAnomalies();
// 15s cadence — anomalies are rare (per-entity 10-min cooldown), so a
// slow poll keeps the review queue fresh without busy-work.
safeInterval(loadAnomalies, 15000);

// ── Review tab: unknown objects ───────────────────────────────────────────
// Things Jarvis saw recurring but can't name. The fix for the old voice-only
// answer window — name or dismiss them here, any time, with the photo + the
// "keeps showing up here" evidence in front of you.

function renderObjReviewCard(item) {
  const card = document.createElement("div");
  card.className = "objreview-card";
  const key = item.key;
  const cls = item.yolo_class || "object";
  const room = item.room || "?";
  const count = item.count || 0;
  const loc = item.location || {};
  const stab = loc.stability != null ? Math.round(loc.stability * 100) : null;

  const img = document.createElement("img");
  img.className = "objreview-crop";
  img.src = "/api/object_vocab/review/crop.jpg?key=" + encodeURIComponent(key);
  img.alt = cls;
  img.addEventListener("error", () => {
    const ph = document.createElement("div");
    ph.className = "objreview-nocrop";
    ph.textContent = "no photo";
    if (img.parentNode) img.replaceWith(ph);
  });

  const meta = document.createElement("div");
  meta.className = "objreview-meta";
  const clsEl = document.createElement("div");
  clsEl.className = "objreview-cls";
  clsEl.textContent = cls;
  const lineEl = document.createElement("div");
  lineEl.className = "objreview-line";
  lineEl.textContent = room + " · seen " + count + "x";
  meta.append(clsEl, lineEl);
  if (stab != null) {
    const stabEl = document.createElement("div");
    stabEl.className = "objreview-line objreview-dim";
    stabEl.textContent =
      stab >= 70 ? "parked in one spot · " + stab + "% stable"
        : stab >= 40 ? "roughly one area · " + stab + "% stable"
          : "moving around · " + stab + "% stable";
    meta.append(stabEl);
  }

  const nameInput = document.createElement("input");
  nameInput.type = "text";
  nameInput.className = "reminder-input objreview-name";
  nameInput.placeholder = "what is it?";

  const answerBtn = document.createElement("button");
  answerBtn.className = "dev-btn";
  answerBtn.textContent = "Name it";
  const dismissBtn = document.createElement("button");
  dismissBtn.className = "dev-btn";
  dismissBtn.textContent = "Dismiss";

  answerBtn.addEventListener("click", async () => {
    const name = nameInput.value.trim();
    if (!name) { nameInput.focus(); return; }
    answerBtn.disabled = true; dismissBtn.disabled = true;
    try {
      const res = await fetch("/api/object_vocab/review/answer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key: key, name: name }),
      });
      if (res.ok) { loadObjectVocabReview(); }
      else { answerBtn.disabled = false; dismissBtn.disabled = false; }
    } catch (e) {
      console.warn("[objreview answer] failed:", e);
      answerBtn.disabled = false; dismissBtn.disabled = false;
    }
  });
  dismissBtn.addEventListener("click", async () => {
    answerBtn.disabled = true; dismissBtn.disabled = true;
    try {
      const res = await fetch("/api/object_vocab/review/dismiss", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key: key }),
      });
      if (res.ok) { loadObjectVocabReview(); }
      else { answerBtn.disabled = false; dismissBtn.disabled = false; }
    } catch (e) {
      console.warn("[objreview dismiss] failed:", e);
      answerBtn.disabled = false; dismissBtn.disabled = false;
    }
  });
  nameInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") answerBtn.click();
  });

  const actions = document.createElement("div");
  actions.className = "objreview-actions";
  actions.append(nameInput, answerBtn, dismissBtn);
  card.append(img, meta, actions);
  return card;
}

async function loadObjectVocabReview() {
  const grid = document.getElementById("objreview-grid");
  if (!grid) return;
  try {
    const res = await fetch("/api/object_vocab/review");
    if (!res.ok) return;
    const body = await res.json();
    if (!body.available) {
      grid.innerHTML =
        '<div class="who-empty">Object learning is disabled.</div>';
      return;
    }
    const items = Array.isArray(body.items) ? body.items : [];
    if (items.length === 0) {
      grid.innerHTML =
        '<div class="who-empty">No unknown objects right now.</div>';
      return;
    }
    grid.innerHTML = "";
    items.forEach((it) => grid.appendChild(renderObjReviewCard(it)));
  } catch (err) {
    console.warn("[loadObjectVocabReview] failed:", err);
  }
}

// ── Review tab: unknown sounds ────────────────────────────────────────────
// Sounds the cascade heard but couldn't identify (not wake / event / speech).
// Each has a saved clip — play it back, name it, or dismiss the one-offs.

function renderSoundCard(item) {
  const card = document.createElement("div");
  card.className = "objreview-card";
  const id = item.id;
  const room = item.room || "?";
  const dur = item.duration_s != null ? item.duration_s : 0;

  const audio = document.createElement("audio");
  audio.className = "soundreview-audio";
  audio.controls = true;
  audio.preload = "none";
  audio.src = "/api/sound_vocab/review/clip.wav?id=" + encodeURIComponent(id);

  const meta = document.createElement("div");
  meta.className = "objreview-meta";
  const title = document.createElement("div");
  title.className = "objreview-cls";
  title.textContent = item.guess ? "sound · " + item.guess
    : "unidentified sound";
  const line = document.createElement("div");
  line.className = "objreview-line";
  const when = item.ts ? new Date(item.ts * 1000).toLocaleString() : "";
  line.textContent = room + " · " + dur.toFixed(1) + "s"
    + (when ? " · " + when : "");
  meta.append(title, line);

  const nameInput = document.createElement("input");
  nameInput.type = "text";
  nameInput.className = "reminder-input objreview-name";
  nameInput.placeholder = "what was it?";

  const answerBtn = document.createElement("button");
  answerBtn.className = "dev-btn";
  answerBtn.textContent = "Name it";
  const dismissBtn = document.createElement("button");
  dismissBtn.className = "dev-btn";
  dismissBtn.textContent = "Dismiss";

  answerBtn.addEventListener("click", async () => {
    const name = nameInput.value.trim();
    if (!name) { nameInput.focus(); return; }
    answerBtn.disabled = true; dismissBtn.disabled = true;
    try {
      const res = await fetch("/api/sound_vocab/review/answer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: id, name: name }),
      });
      if (res.ok) { loadSoundVocabReview(); }
      else { answerBtn.disabled = false; dismissBtn.disabled = false; }
    } catch (e) {
      console.warn("[soundreview answer] failed:", e);
      answerBtn.disabled = false; dismissBtn.disabled = false;
    }
  });
  dismissBtn.addEventListener("click", async () => {
    answerBtn.disabled = true; dismissBtn.disabled = true;
    try {
      const res = await fetch("/api/sound_vocab/review/dismiss", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: id }),
      });
      if (res.ok) { loadSoundVocabReview(); }
      else { answerBtn.disabled = false; dismissBtn.disabled = false; }
    } catch (e) {
      console.warn("[soundreview dismiss] failed:", e);
      answerBtn.disabled = false; dismissBtn.disabled = false;
    }
  });
  nameInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") answerBtn.click();
  });

  const actions = document.createElement("div");
  actions.className = "objreview-actions";
  actions.append(nameInput, answerBtn, dismissBtn);
  card.append(audio, meta, actions);
  return card;
}

async function loadSoundVocabReview() {
  const grid = document.getElementById("soundreview-grid");
  if (!grid) return;
  try {
    const res = await fetch("/api/sound_vocab/review");
    if (!res.ok) return;
    const body = await res.json();
    if (!body.available) {
      grid.innerHTML =
        '<div class="who-empty">Sound learning is disabled.</div>';
      return;
    }
    const items = Array.isArray(body.items) ? body.items : [];
    if (items.length === 0) {
      grid.innerHTML =
        '<div class="who-empty">No unidentified sounds right now.</div>';
      return;
    }
    grid.innerHTML = "";
    items.forEach((it) => grid.appendChild(renderSoundCard(it)));
  } catch (err) {
    console.warn("[loadSoundVocabReview] failed:", err);
  }
}

// ── Routine (§25 — PatternMiner behavioral heatmap) ───────────────────────
// A resident's learned room-by-hour week — the baseline the AnomalyScorer
// judges against. Cells where a recent anomaly fired are ringed so you can
// see WHY each scored unusual (it landed in a cold cell). PatternMiner
// builds profiles in UTC; the grid is shifted to local time so it reads
// against your actual clock. Resident persons only.

const ROUTINE_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const ROUTINE_ROOM_COLORS = {
  office: "#5b9bd5",
  bedroom: "#9b6dd5",
  kitchen: "#e8a445",
  living_room: "#6ce6a6",
  laundry_room: "#d56d9b",
};

let _routineData = null; // last /pattern_profile response
let _routineAnomalies = []; // last /anomalies response (heatmap overlay)

function _routineRoomColor(room) {
  if (ROUTINE_ROOM_COLORS[room]) return ROUTINE_ROOM_COLORS[room];
  // Stable hue for any room outside the fixed palette.
  let h = 0;
  for (let i = 0; i < room.length; i++) {
    h = (h * 31 + room.charCodeAt(i)) % 360;
  }
  return `hsl(${h}, 55%, 62%)`;
}

function _routineDominant(dist) {
  // {room: probability} → [room, prob] of the max, or [null, 0].
  let best = null;
  let bestP = 0;
  for (const [room, p] of Object.entries(dist || {})) {
    if (Number(p) > bestP) {
      best = room;
      bestP = Number(p);
    }
  }
  return [best, bestP];
}

function _routinePeakHourLocal(byWeekday, offsetH) {
  // byWeekday = {weekday: {hour(UTC): count}} → modal hour, shifted local.
  const totals = {};
  for (const perHour of Object.values(byWeekday || {})) {
    for (const [h, c] of Object.entries(perHour || {})) {
      totals[h] = (totals[h] || 0) + Number(c);
    }
  }
  let peak = null;
  let peakC = 0;
  for (const [h, c] of Object.entries(totals)) {
    if (c > peakC) {
      peak = Number(h);
      peakC = c;
    }
  }
  if (peak == null) return null;
  return ((peak - offsetH) % 24 + 24) % 24;
}

function _routineFmtHour(h) {
  if (h == null) return "—";
  const ampm = h < 12 ? "AM" : "PM";
  const h12 = h % 12 === 0 ? 12 : h % 12;
  return `${h12} ${ampm}`;
}

function renderRoutine() {
  const body = document.getElementById("routine-body");
  if (!body || !_routineData) return;
  const sel = document.getElementById("routine-resident");
  const residents = _routineData.residents || [];
  if (residents.length === 0) {
    body.innerHTML = '<div class="who-empty">No resident profiles yet.</div>';
    return;
  }
  const chosen =
    residents.find((r) => r.id === (sel && sel.value)) || residents[0];
  const profile = chosen.profile;
  if (!profile || !profile.n_events) {
    body.innerHTML =
      `<div class="who-empty">No routine learned for ` +
      `${escapeHtml(chosen.name)} yet — PatternMiner builds it nightly ` +
      `from the world-event log.</div>`;
    return;
  }
  const rbwh = profile.room_by_weekday_hour || {};
  // UTC→local: getTimezoneOffset() is minutes WEST of UTC (positive in the
  // Americas) and already accounts for the current DST state.
  const offsetH = new Date().getTimezoneOffset() / 60;

  // Anomaly overlay: local weekday-hour cells where a recent anomaly fired
  // for THIS resident (Date getters below are already local time).
  const hot = new Set();
  for (const a of _routineAnomalies) {
    if ((a.entity_name || "") !== chosen.name) continue;
    const ts = a.event && a.event.ts ? a.event.ts : a.ts;
    const d = ts ? new Date(ts) : null;
    if (d && !isNaN(d.getTime())) {
      hot.add(`${(d.getDay() + 6) % 7}:${d.getHours()}`);
    }
  }

  // ── Heatmap grid (local time) ──
  let grid = '<div class="routine-hours"><span></span>';
  for (let h = 0; h < 24; h++) {
    grid += `<span>${h % 6 === 0 ? h : ""}</span>`;
  }
  grid += "</div>";
  const roomsSeen = new Set();
  for (let lwd = 0; lwd < 7; lwd++) {
    grid += `<div class="routine-row">` +
      `<span class="routine-day">${ROUTINE_DAYS[lwd]}</span>`;
    for (let lh = 0; lh < 24; lh++) {
      // Local (lwd,lh) → the UTC bucket PatternMiner stored it under.
      const utcTotal = ((lwd * 24 + lh + offsetH) % 168 + 168) % 168;
      const uwd = Math.floor(utcTotal / 24);
      const uh = utcTotal % 24;
      const byHour = rbwh[uwd] || rbwh[String(uwd)] || {};
      const dist = byHour[uh] || byHour[String(uh)] || {};
      const [room, prob] = _routineDominant(dist);
      let style = "";
      let title = `${ROUTINE_DAYS[lwd]} ${_routineFmtHour(lh)}`;
      if (room) {
        roomsSeen.add(room);
        style =
          `background:${_routineRoomColor(room)};` +
          `opacity:${(0.2 + 0.8 * prob).toFixed(2)}`;
        title += ` — ${room} ${Math.round(prob * 100)}%`;
      } else {
        title += " — no data";
      }
      const hotCls = hot.has(`${lwd}:${lh}`) ? " routine-cell-anomaly" : "";
      grid +=
        `<span class="routine-cell${hotCls}" style="${style}" ` +
        `title="${escapeHtml(title)}"></span>`;
    }
    grid += "</div>";
  }

  // ── Legend ──
  let legend = '<div class="routine-legend">';
  for (const room of [...roomsSeen].sort()) {
    legend +=
      `<span class="routine-legend-item">` +
      `<span class="routine-legend-swatch" ` +
      `style="background:${_routineRoomColor(room)}"></span>` +
      `${escapeHtml(room)}</span>`;
  }
  if (hot.size > 0) {
    legend +=
      '<span class="routine-legend-item">' +
      '<span class="routine-legend-swatch routine-cell-anomaly"></span>' +
      "anomaly</span>";
  }
  legend += "</div>";

  // ── Digest ──
  const arr = _routinePeakHourLocal(profile.arrival_by_weekday, offsetH);
  const dep = _routinePeakHourLocal(profile.departure_by_weekday, offsetH);
  const seqs = (profile.morning_routine || {}).most_common_sequences || [];
  const morning = seqs.length ? seqs[0].sequence.join(" → ") : "—";
  const cop = Object.entries(profile.co_presence || {}).sort(
    (a, b) => b[1] - a[1],
  )[0];
  const copStr = cop
    ? `${escapeHtml(cop[0])} (${Math.round(cop[1] * 100)}%)`
    : "—";
  const updated = chosen.profile_updated
    ? formatRelativeTs(chosen.profile_updated)
    : "—";
  const digest =
    '<div class="routine-digest">' +
    `<div><span class="routine-k">Arrives</span> ${_routineFmtHour(arr)}` +
    ` &nbsp;·&nbsp; <span class="routine-k">Leaves</span> ` +
    `${_routineFmtHour(dep)}</div>` +
    `<div><span class="routine-k">Morning</span> ${escapeHtml(morning)}</div>` +
    `<div><span class="routine-k">Usually with</span> ${copStr}</div>` +
    `<div class="routine-meta">${profile.n_events} events / 30 days ` +
    `· rebuilt ${escapeHtml(updated)} · local time</div>` +
    "</div>";

  body.innerHTML =
    `<div class="routine-heatmap">${grid}</div>` + legend + digest;
}

async function loadRoutine() {
  try {
    const [pRes, aRes] = await Promise.all([
      fetch("/api/world_model/pattern_profile"),
      fetch("/api/world_model/anomalies?limit=60"),
    ]);
    if (!pRes.ok) return;
    _routineData = await pRes.json();
    if (aRes.ok) {
      const ab = await aRes.json();
      _routineAnomalies = Array.isArray(ab.anomalies) ? ab.anomalies : [];
    }
    const sel = document.getElementById("routine-resident");
    const body = document.getElementById("routine-body");
    if (!sel || !body) return;
    if (!_routineData.available) {
      body.innerHTML =
        '<div class="who-empty">World model unavailable.</div>';
      return;
    }
    const residents = _routineData.residents || [];
    // Repopulate the selector; keep the current pick if still present,
    // else default to the richest profile (most events — i.e. you).
    const prev = sel.value;
    sel.innerHTML = residents
      .map(
        (r) =>
          `<option value="${escapeHtml(r.id)}">${escapeHtml(r.name)}</option>`,
      )
      .join("");
    if (residents.some((r) => r.id === prev)) {
      sel.value = prev;
    } else {
      let best = residents[0];
      for (const r of residents) {
        const n = (r.profile && r.profile.n_events) || 0;
        const bn = (best && best.profile && best.profile.n_events) || 0;
        if (n > bn) best = r;
      }
      if (best) sel.value = best.id;
    }
    renderRoutine();
  } catch (err) {
    console.warn("[loadRoutine] failed:", err);
  }
}
const _routineSel = document.getElementById("routine-resident");
if (_routineSel) _routineSel.addEventListener("change", renderRoutine);
loadRoutine();
// Profiles rebuild nightly + anomalies are rare — a slow refresh is plenty.
safeInterval(loadRoutine, 300000);

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
  // Best-effort URL: prefer the server-provided thumbnail_url; else fall
  // back to the cluster-event endpoint which 404s gracefully if no
  // snapshot exists. Always make the whole row clickable so the user
  // gets a "no snapshot stored" lightbox instead of a dead row.
  const url = ev.thumbnail_url
    || (ev.id
        ? `/api/world_model/cluster/event/${encodeURIComponent(ev.id)}/image.jpg`
        : null);
  const hasThumb = !!ev.thumbnail_url;
  const thumb = hasThumb
    ? `<img class="interaction-thumb interaction-thumb-clickable"
            src="${ev.thumbnail_url}"
            alt="snapshot" loading="lazy"
            onerror="this.style.display='none';" />`
    : '<div class="interaction-thumb empty">(no thumb)</div>';
  const glyph = INTERACTION_GLYPH[ev.event_type] || "·";
  const ago = formatRelativeTs(ev.ts);
  const div = document.createElement("div");
  div.className = `interaction-row interaction-row-clickable interaction-${ev.event_type || "unknown"}`;
  div.title = "Click for full-size snapshot";
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
  div.addEventListener("click", () => {
    if (url) {
      openImageLightbox(url, sentence, ev.ts);
    } else {
      openImageLightbox(null, sentence, ev.ts);
    }
  });
  return div;
}

// Simple lightbox for full-size snapshot viewing — used by the
// interactions thumbnail click handler. Reuses the modal-overlay CSS.

function _imgLightboxKeydown(e) {
  if (e.key === "Escape") closeImageLightbox();
}

function closeImageLightbox() {
  const m = document.getElementById("img-lightbox");
  if (m) m.remove();
  document.removeEventListener("keydown", _imgLightboxKeydown);
}

function openImageLightbox(url, caption, ts, faceBbox = null) {
  closeImageLightbox();
  const overlay = document.createElement("div");
  overlay.id = "img-lightbox";
  overlay.className = "modal-overlay img-lightbox-overlay";
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) closeImageLightbox();
  });
  const tsStr = ts ? new Date(ts).toLocaleString() : "";
  // The image may not exist (event predates snapshot, or no snapshot
  // was saved for this event). Render it anyway and rely on onerror
  // to swap in a "no image" placeholder so the lightbox still opens.
  const bboxAttrs = faceBbox && faceBbox.length === 4
    ? ` data-x1="${faceBbox[0]}" data-y1="${faceBbox[1]}" data-x2="${faceBbox[2]}" data-y2="${faceBbox[3]}"`
    : "";
  const bboxHtml = faceBbox && faceBbox.length === 4
    ? `<div class="face-bbox-overlay face-bbox-lightbox"${bboxAttrs}></div>`
    : "";
  const imgHtml = url
    ? `<div class="lightbox-img-wrap">
         <img class="lightbox-img" src="${url}" alt="snapshot"
              onerror="this.outerHTML='<div class=\\'lightbox-noimg\\'>No snapshot stored for this event.</div>'" />
         ${bboxHtml}
       </div>`
    : `<div class="lightbox-noimg">No snapshot stored for this event.</div>`;
  overlay.innerHTML = `
    <div class="lightbox-card">
      <button class="modal-close lightbox-close" aria-label="Close">×</button>
      ${imgHtml}
      <div class="lightbox-caption">
        <span>${escapeHtml(caption || "")}</span>
        <span class="lightbox-ts">${escapeHtml(tsStr)}</span>
      </div>
    </div>`;
  overlay.querySelector(".lightbox-close").addEventListener(
    "click", closeImageLightbox
  );
  document.body.appendChild(overlay);
  document.addEventListener("keydown", _imgLightboxKeydown);
  // Position the bbox after the image lays out.
  if (faceBbox) {
    const img = overlay.querySelector(".lightbox-img");
    const ov = overlay.querySelector(".face-bbox-lightbox");
    if (img && ov) {
      const reposition = () => _positionFaceBbox(img, ov);
      if (img.complete && img.naturalWidth > 0) reposition();
      else img.addEventListener("load", reposition);
      window.addEventListener("resize", reposition);
    }
  }
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
safeInterval(loadInteractions, 10000);

// ── Clown alarm (§29.8 v4.1) ──────────────────────────────────────────────

function _clownToast(msg, isError = false) {
  // Reuse pet panel's toast if present, else fall back to console.
  const t = document.getElementById("toast");
  if (t) {
    t.textContent = msg;
    t.classList.toggle("error", isError);
    t.classList.add("show");
    setTimeout(() => t.classList.remove("show"), 2400);
  } else {
    (isError ? console.warn : console.log)("[clown]", msg);
  }
}

async function loadClownStatus() {
  try {
    const res = await fetch("/api/clown/status");
    if (!res.ok) return;
    const body = await res.json();
    const statusEl = document.getElementById("clown-status");
    if (!statusEl) return;
    if (!body.available) {
      statusEl.innerHTML =
        '<div class="who-empty">Clown alarm not wired.</div>';
      return;
    }
    const cooldown = body.cooldown_remaining_seconds || 0;
    const cooldownLabel =
      cooldown > 60 * 60 * 24 * 365 ? "INDEFINITE"
      : cooldown > 60 ? `${Math.round(cooldown / 60)}m`
      : cooldown > 0 ? `${Math.round(cooldown)}s`
      : "—";
    statusEl.innerHTML = `
      <div class="clown-state-line">
        <span class="clown-state-label">STATE</span>
        <span class="clown-state-value">${escapeHtml(body.state)}</span>
        <span class="clown-state-label">COOLDOWN</span>
        <span class="clown-state-value">${escapeHtml(cooldownLabel)}</span>
      </div>
      <div class="clown-pool-line">
        Pool: ${body.pool_size} entries (${body.pool_improv_slots} improv)
        ${body.cooldown_reason ? `· ${escapeHtml(body.cooldown_reason)}` : ""}
      </div>
    `;
    const list = document.getElementById("clown-improv-list");
    if (!list) return;
    const events = body.recent_improv_events || [];
    if (events.length === 0) {
      list.innerHTML =
        '<div class="who-empty">No recent generations.</div>';
      return;
    }
    list.innerHTML = "";
    events.slice().reverse().forEach((ev) => {
      const row = document.createElement("div");
      row.className = `clown-improv-row outcome-${ev.outcome}`;
      const supplementBadge =
        ev.cross_style_supplement_count > 0
          ? `<span class="clown-improv-badge">+${ev.cross_style_supplement_count} cross</span>`
          : "";
      row.innerHTML = `
        <div class="clown-improv-head">
          <span class="clown-improv-style">${escapeHtml(ev.style_seed || "?")}</span>
          <span class="clown-improv-outcome">${escapeHtml(ev.outcome || "?")}</span>
          <span class="clown-improv-examples">${ev.examples_used_count} ex ${supplementBadge}</span>
          <span class="clown-improv-ago">${formatRelativeTs(ev.ts)}</span>
        </div>
        ${ev.final_text
          ? `<div class="clown-improv-text">${escapeHtml(ev.final_text)}</div>`
          : ""}
        ${ev.error
          ? `<div class="clown-improv-error">${escapeHtml(ev.error)}</div>`
          : ""}
      `;
      list.appendChild(row);
    });
  } catch (err) {
    console.warn("[loadClownStatus] failed:", err);
  }
}
loadClownStatus();
safeInterval(loadClownStatus, 8000);

(function wireClownControls() {
  const testBtn = document.getElementById("clown-test");
  if (testBtn) {
    testBtn.addEventListener("click", async () => {
      testBtn.disabled = true;
      try {
        const res = await fetch("/api/clown/test_fire", { method: "POST" });
        if (!res.ok) throw new Error(await res.text());
        _clownToast("Test fire dispatched.");
        setTimeout(loadClownStatus, 500);
      } catch (err) {
        _clownToast(`Test fire failed: ${err.message || err}`, true);
      } finally {
        testBtn.disabled = false;
      }
    });
  }
  const cooldownBtn = document.getElementById("clown-cooldown");
  if (cooldownBtn) {
    cooldownBtn.addEventListener("click", async () => {
      const phraseEl = document.getElementById("clown-cooldown-phrase");
      const phrase = (phraseEl && phraseEl.value.trim()) || "for an hour";
      cooldownBtn.disabled = true;
      try {
        const res = await fetch("/api/clown/cooldown", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ phrase }),
        });
        if (!res.ok) throw new Error(await res.text());
        const body = await res.json();
        _clownToast(
          body.indefinite
            ? "Suppressed indefinitely."
            : `Cooldown set: ${body.reason || `${body.seconds}s`}.`
        );
        loadClownStatus();
      } catch (err) {
        _clownToast(`Cooldown failed: ${err.message || err}`, true);
      } finally {
        cooldownBtn.disabled = false;
      }
    });
  }
  const reenableBtn = document.getElementById("clown-reenable");
  if (reenableBtn) {
    reenableBtn.addEventListener("click", async () => {
      reenableBtn.disabled = true;
      try {
        const res = await fetch("/api/clown/reenable", { method: "POST" });
        if (!res.ok) throw new Error(await res.text());
        _clownToast("Re-enabled.");
        loadClownStatus();
      } catch (err) {
        _clownToast(`Re-enable failed: ${err.message || err}`, true);
      } finally {
        reenableBtn.disabled = false;
      }
    });
  }
  const reloadBtn = document.getElementById("clown-reload");
  if (reloadBtn) {
    reloadBtn.addEventListener("click", async () => {
      reloadBtn.disabled = true;
      try {
        const res = await fetch("/api/clown/reload_pool", { method: "POST" });
        if (!res.ok) throw new Error(await res.text());
        const body = await res.json();
        _clownToast(`Reloaded ${body.loaded} pool entries.`);
        loadClownStatus();
      } catch (err) {
        _clownToast(`Reload failed: ${err.message || err}`, true);
      } finally {
        reloadBtn.disabled = false;
      }
    });
  }
})();

// ── Tabs: Home / Settings / Logs ──────────────────────────────────────────

(function setupTabs() {
  const bar = document.getElementById("tab-bar");
  if (!bar) return;
  const buttons = bar.querySelectorAll(".tab-btn");
  const panes = {
    home: document.getElementById("tab-pane-home"),
    reviews: document.getElementById("tab-pane-reviews"),
    perf: document.getElementById("tab-pane-perf"),
    settings: document.getElementById("tab-pane-settings"),
    logs: document.getElementById("tab-pane-logs"),
  };
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab;
      activeTab = target || "home";
      buttons.forEach((b) => b.classList.toggle("active", b === btn));
      Object.entries(panes).forEach(([k, el]) => {
        if (!el) return;
        el.hidden = k !== target;
      });
      if (target === "settings") loadSettings();
      if (target === "reviews") {
        loadReviewsTab();
        loadObjectVocabReview();
        loadSoundVocabReview();
      }
      if (target === "logs") connectLogStream();
      if (target !== "logs") disconnectLogStream();
      if (target === "perf") startPerfRefresh();
      else stopPerfRefresh();
      if (target === "home") refreshRoomFeeds();
    });
  });
  // Keep the tab badge in sync independent of which tab is open.
  refreshReviewsBadge();
  safeInterval(refreshReviewsBadge, 15000);
})();

// ── Pending Reviews tab ───────────────────────────────────────────────────

let _reviewsItems = [];
let _reviewsSelected = new Set();

async function refreshReviewsBadge() {
  try {
    const res = await fetch("/api/identity/pending");
    if (!res.ok) return;
    const body = await res.json();
    const n = (body.pending || []).length;
    const badge = document.getElementById("reviews-tab-badge");
    if (badge) {
      badge.textContent = String(n);
      badge.hidden = n === 0;
    }
  } catch {}
}

async function _loadBankStats() {
  try {
    const res = await fetch("/api/identity/bank_stats");
    if (!res.ok) return [];
    const body = await res.json();
    return Array.isArray(body.persons) ? body.persons : [];
  } catch {
    return [];
  }
}

let _reviewsPersons = [];

async function loadReviewsTab() {
  const grid = document.getElementById("reviews-grid");
  const statsEl = document.getElementById("reviews-stats");
  const targetSel = document.getElementById("reviews-bulk-target");
  if (!grid || !targetSel) return;

  grid.innerHTML = '<div class="who-empty">Loading…</div>';

  const [items, persons] = await Promise.all([
    fetch("/api/identity/pending").then((r) => r.json()).then((b) => b.pending || []).catch(() => []),
    _loadBankStats(),
  ]);
  _reviewsItems = items;
  _reviewsSelected.clear();
  _reviewsPersons = persons;
  // Keep _personsCache in sync so per-card dropdowns built via
  // _personOptions() include current bank too — _personsCache is the
  // canonical source for that helper and may not have been hydrated
  // if the user lands on this tab first.
  if (persons.length && _personsCache.length === 0) {
    _personsCache = persons;
    _personsCacheVersion += 1;
  }

  // Populate bulk-target dropdown with enrolled persons. Add the
  // "+ new person…" sentinel as the last option so bulk-assign can
  // create-and-attach in one click.
  const bulkOpts = persons
    .map((p) => `<option value="${escapeHtml(p.name)}">${escapeHtml(p.name)} (${p.face_samples} face samples)</option>`)
    .join("");
  targetSel.innerHTML = bulkOpts +
    `<option value="${NEW_PERSON_SENTINEL}">+ new person…</option>`;

  // Show / hide the new-name input next to the bulk dropdown.
  const bulkNewName = document.getElementById("reviews-bulk-new-name");
  if (bulkNewName) {
    bulkNewName.hidden = true;
    bulkNewName.value = "";
    targetSel.removeEventListener("change", _onBulkTargetChange);
    targetSel.addEventListener("change", _onBulkTargetChange);
  }

  // Per-person bank stats summary.
  if (statsEl) {
    statsEl.innerHTML = persons.length
      ? "Bank: " + persons
          .map((p) => `${escapeHtml(p.name)}=${p.face_samples}`)
          .join(", ")
      : "";
  }

  if (items.length === 0) {
    grid.innerHTML = '<div class="who-empty">No pending reviews.</div>';
    _updateReviewsCount();
    return;
  }
  grid.innerHTML = "";
  items.forEach((p) => grid.appendChild(_renderReviewCard(p)));
  _updateReviewsCount();
}

function _onBulkTargetChange() {
  const sel = document.getElementById("reviews-bulk-target");
  const input = document.getElementById("reviews-bulk-new-name");
  if (!sel || !input) return;
  if (sel.value === NEW_PERSON_SENTINEL) {
    input.hidden = false;
    input.focus();
  } else {
    input.hidden = true;
    input.value = "";
  }
}

function _renderReviewCard(p) {
  const div = document.createElement("div");
  div.className = "review-card";
  div.dataset.id = p.id;
  const isCluster = p.kind && p.kind.startsWith("pending_cluster_");
  const modality = p.kind && p.kind.includes("voice") ? "voice" : "face";
  const sim = (p.similarity || 0).toFixed(2);
  const suggested = p.suggested_person_name
    ? `<span class="review-suggested">looks like <b>${escapeHtml(p.suggested_person_name)}</b> (${(p.suggested_similarity || 0).toFixed(2)})</span>`
    : "";
  const bboxJson = p.face_bbox ? JSON.stringify(p.face_bbox) : "";
  const hasImg = p.has_image;
  const preselectName = p.suggested_person_name || p.person_name || null;
  const personOpts = _personOptions(preselectName);
  div.innerHTML = `
    <div class="review-head">
      <label class="review-check">
        <input type="checkbox" class="review-cb" />
      </label>
      <span class="review-id">#${p.id}</span>
      ${suggested}
      <span class="review-sim">sim ${sim}</span>
    </div>
    <div class="review-thumb-wrap" data-bbox='${escapeHtml(bboxJson)}'>
      ${hasImg
        ? `<img class="review-thumb" src="/api/identity/pending/${p.id}/image.jpg" alt="capture" />`
        : `<div class="review-thumb empty">(${modality})</div>`}
      ${p.face_bbox
        ? `<div class="face-bbox-overlay review-bbox"
              data-x1="${p.face_bbox[0]}" data-y1="${p.face_bbox[1]}"
              data-x2="${p.face_bbox[2]}" data-y2="${p.face_bbox[3]}"></div>`
        : ""}
    </div>
    <div class="review-assign">
      <select class="dev-select review-person-sel">${personOpts}</select>
      <input type="text" class="reminder-input review-new-name"
             placeholder="New person name" hidden />
      <button class="dev-btn review-quick" data-action="assign-this">Assign</button>
    </div>
    <div class="review-foot">
      ${p.suggested_person_name
        ? `<button class="dev-btn review-quick" data-action="suggested">Assign suggested</button>`
        : ""}
      <button class="dev-btn review-quick" data-action="reject">Reject</button>
    </div>
  `;
  const cb = div.querySelector(".review-cb");
  const sel = div.querySelector(".review-person-sel");
  const newNameInput = div.querySelector(".review-new-name");

  // Full-card click → toggle the checkbox. Excluded: the thumbnail
  // (opens lightbox), buttons (have their own actions), and the
  // per-card dropdown / new-name input (so clicking to focus them
  // doesn't accidentally select). The checkbox label itself still
  // works as a label, so a click on it does the toggle directly.
  const _toggleSelection = () => {
    cb.checked = !cb.checked;
    cb.dispatchEvent(new Event("change"));
  };
  div.addEventListener("click", (e) => {
    if (e.target.closest("button")) return;
    if (e.target.closest(".review-thumb")) return;
    if (e.target.closest(".review-person-sel")) return;
    if (e.target.closest(".review-new-name")) return;
    if (e.target.closest(".review-check")) return; // label handles it
    _toggleSelection();
  });

  cb.addEventListener("change", () => {
    if (cb.checked) _reviewsSelected.add(p.id);
    else _reviewsSelected.delete(p.id);
    _updateReviewsCount();
    div.classList.toggle("review-selected", cb.checked);
  });

  // Per-card dropdown: switching to "+ new person…" reveals an input.
  sel.addEventListener("change", () => {
    if (sel.value === NEW_PERSON_SENTINEL) {
      newNameInput.hidden = false;
      newNameInput.focus();
    } else {
      newNameInput.hidden = true;
      newNameInput.value = "";
    }
  });

  // Bbox overlay positioning.
  const img = div.querySelector(".review-thumb");
  const overlay = div.querySelector(".review-bbox");
  if (img && img.tagName === "IMG" && overlay) {
    const pos = () => _positionFaceBbox(img, overlay);
    if (img.complete && img.naturalWidth > 0) pos();
    else img.addEventListener("load", pos);
    window.addEventListener("resize", pos);
  }
  // Clicking the thumb opens the lightbox (with bbox).
  if (img && img.tagName === "IMG") {
    img.style.cursor = "zoom-in";
    img.addEventListener("click", (e) => {
      e.stopPropagation();
      openImageLightbox(
        `/api/identity/pending/${p.id}/image.jpg`,
        p.suggested_person_name || (isCluster ? `Cluster #${p.cluster_id}` : `Drift on ${p.person_name}`),
        p.captured_at,
        p.face_bbox || null,
      );
    });
  }
  // Per-card action buttons.
  div.querySelectorAll(".review-quick").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const action = btn.dataset.action;
      try {
        if (action === "suggested") {
          if (!p.suggested_person_name) return;
          await fetch(`/api/identity/pending/${p.id}/resolve`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              action: "assign",
              target_name: p.suggested_person_name,
            }),
          });
        } else if (action === "assign-this") {
          let target = sel.value;
          if (target === NEW_PERSON_SENTINEL) {
            target = (newNameInput.value || "").trim();
            if (!target) {
              newNameInput.focus();
              return;
            }
            const collision = _personsCache.find(
              (pp) => pp.name && pp.name.toLowerCase() === target.toLowerCase(),
            );
            if (collision) {
              if (!confirm(
                `'${target}' will reuse the existing person '${collision.name}'.\n` +
                `If this is genuinely a different person with the same name, ` +
                `pick a distinct label first (e.g. '${target} S').\n\nProceed?`,
              )) return;
              target = collision.name;
            }
          }
          if (!target) return;
          await fetch(`/api/identity/pending/${p.id}/resolve`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "assign", target_name: target }),
          });
          loadPersons();
        } else if (action === "reject") {
          await fetch(`/api/identity/pending/${p.id}/resolve`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "reject" }),
          });
        }
        div.classList.add("review-resolving");
        setTimeout(() => {
          div.remove();
          refreshReviewsBadge();
        }, 200);
      } catch (err) {
        console.warn("[reviews] quick action failed:", err);
      }
    });
  });
  return div;
}

function _updateReviewsCount() {
  const c = document.getElementById("reviews-count");
  if (c) c.textContent = `${_reviewsSelected.size} selected`;
}

(function wireReviewsControls() {
  const selectAll = document.getElementById("reviews-select-all");
  const assignBtn = document.getElementById("reviews-bulk-assign");
  const rejectBtn = document.getElementById("reviews-bulk-reject");
  const targetSel = document.getElementById("reviews-bulk-target");
  const status = document.getElementById("reviews-status");
  if (!selectAll || !assignBtn || !rejectBtn) return;

  selectAll.addEventListener("change", () => {
    const cbs = document.querySelectorAll("#reviews-grid .review-cb");
    cbs.forEach((cb) => {
      cb.checked = selectAll.checked;
      cb.dispatchEvent(new Event("change"));
    });
  });

  const setStatus = (msg, cls = "") => {
    if (!status) return;
    status.textContent = msg;
    status.className = `reviews-status ${cls}`;
  };

  const doBulk = async (action) => {
    const ids = Array.from(_reviewsSelected);
    if (ids.length === 0) {
      setStatus("Nothing selected.", "err");
      return;
    }
    let target = action === "assign" ? targetSel.value : null;
    if (action === "assign") {
      if (target === NEW_PERSON_SENTINEL) {
        const newName = document.getElementById("reviews-bulk-new-name");
        const typed = newName ? newName.value.trim() : "";
        if (!typed) {
          if (newName) newName.focus();
          setStatus("Type a name for the new person.", "err");
          return;
        }
        const collision = _personsCache.find(
          (pp) => pp.name && pp.name.toLowerCase() === typed.toLowerCase(),
        );
        if (collision) {
          if (!confirm(
            `'${typed}' will reuse the existing person '${collision.name}'.\n` +
            `Proceed (re-uses) or cancel and pick a distinct label?`,
          )) return;
          target = collision.name;
        } else {
          target = typed;
        }
      }
      if (!target) {
        setStatus("Pick a person for the bulk target.", "err");
        return;
      }
    }
    setStatus(`Processing ${ids.length}…`);
    assignBtn.disabled = rejectBtn.disabled = true;
    try {
      const res = await fetch("/api/identity/pending/bulk", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ids,
          action,
          target_name: target,
        }),
      });
      const body = await res.json();
      setStatus(
        `Done: ${body.ok} ok, ${body.skipped_quality} skipped (quality), ${body.failed} failed.`,
        body.failed ? "err" : "ok",
      );
      // Reload the tab so resolved rows disappear and stats refresh.
      await loadReviewsTab();
      refreshReviewsBadge();
      // Bulk-creating a new person? Re-fetch the persons cache so
      // any future per-card dropdowns include the new name.
      if (action === "assign") loadPersons();
    } catch (e) {
      setStatus(`Failed: ${e.message || e}`, "err");
    } finally {
      assignBtn.disabled = rejectBtn.disabled = false;
    }
  };

  assignBtn.addEventListener("click", () => doBulk("assign"));
  rejectBtn.addEventListener("click", () => doBulk("reject"));

  const collapseBtn = document.getElementById("reviews-collapse");
  const rejectAllBtn = document.getElementById("reviews-reject-all");
  if (collapseBtn) {
    collapseBtn.addEventListener("click", async () => {
      collapseBtn.disabled = true;
      setStatus("Collapsing duplicates…");
      try {
        const res = await fetch("/api/identity/pending/collapse", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ modality: "face", min_sim: 0.35 }),
        });
        const body = await res.json();
        setStatus(
          `Collapsed: kept ${body.kept} representatives, rejected ${body.rejected} duplicates (scanned ${body.scanned}).`,
          "ok",
        );
        await loadReviewsTab();
        refreshReviewsBadge();
      } catch (e) {
        setStatus(`Collapse failed: ${e.message || e}`, "err");
      } finally {
        collapseBtn.disabled = false;
      }
    });
  }
  const pruneBtn = document.getElementById("reviews-prune-bank");
  if (pruneBtn) {
    pruneBtn.addEventListener("click", async () => {
      if (!confirm(
        "Drop near-duplicate face samples (≥0.97 cosine) from every " +
        "person's bank?\n\nKeeps the oldest representative of each " +
        "near-duplicate pair. Safe to run repeatedly — the cap (60 " +
        "face / 40 voice) limits growth, this just trims the over-" +
        "represented poses that pull centroids the wrong direction.",
      )) return;
      pruneBtn.disabled = true;
      setStatus("Pruning…");
      try {
        const res = await fetch("/api/identity/bank_prune", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ modality: "both" }),
        });
        const body = await res.json();
        const total = (body.results || [])
          .reduce((acc, r) => acc + (r.total_dropped || 0), 0);
        setStatus(`Pruned ${total} redundant sample(s).`, total ? "ok" : "");
        await loadReviewsTab();
      } catch (e) {
        setStatus(`Prune failed: ${e.message || e}`, "err");
      } finally {
        pruneBtn.disabled = false;
      }
    });
  }
  if (rejectAllBtn) {
    rejectAllBtn.addEventListener("click", async () => {
      if (!confirm(
        "Reject EVERY unresolved pending row?\n\n" +
        "This is the nuclear option for runaway queues. " +
        "It doesn't enroll anyone — just clears the backlog.\n\n" +
        "Continue?",
      )) return;
      rejectAllBtn.disabled = true;
      setStatus("Rejecting all…");
      try {
        const res = await fetch("/api/identity/pending/reject_all", {
          method: "POST",
        });
        const body = await res.json();
        setStatus(`Rejected ${body.rejected} unresolved rows.`, "ok");
        await loadReviewsTab();
        refreshReviewsBadge();
      } catch (e) {
        setStatus(`Reject all failed: ${e.message || e}`, "err");
      } finally {
        rejectAllBtn.disabled = false;
      }
    });
  }
})();

// ── Perf tab ──────────────────────────────────────────────────────────────

let _perfTimer = null;

async function loadPerfTab() {
  try {
    const res = await fetch("/api/perf");
    if (!res.ok) return;
    const body = await res.json();
    renderPerf(body);
  } catch (e) {
    console.warn("[perf] load failed:", e);
  }
}

function renderPerf(state) {
  const grid = document.getElementById("perf-grid");
  if (!grid) return;
  const uptimeEl = document.getElementById("perf-uptime");
  const uptimeS = Math.max(1, state.uptime_s || 1);
  if (uptimeEl) {
    const s = Math.round(uptimeS);
    const m = Math.floor(s / 60);
    const sec = s % 60;
    uptimeEl.textContent = `uptime: ${m}m ${sec}s`;
  }
  const timings = state.timings || {};
  const counters = state.counters || {};
  const modelCalls = Array.isArray(state.model_calls) ? state.model_calls : [];
  const entries = Object.entries(timings);
  const counterEntries = Object.entries(counters);
  if (entries.length === 0 && counterEntries.length === 0 && modelCalls.length === 0) {
    grid.innerHTML = '<div class="who-empty">No timing samples yet. Wait a few seconds for the hot paths to fire.</div>';
    return;
  }
  // Sort by avg_ms descending — heaviest at top so the lag culprit is
  // visible without scrolling.
  entries.sort((a, b) => (b[1].avg_ms || 0) - (a[1].avg_ms || 0));
  // The "lag budget" reference: for a 30fps loop you have ~33ms; for
  // 5fps you have 200ms. Color avg by where it sits in the 0-100ms range.
  const _color = (ms) => {
    if (ms < 10) return "perf-green";
    if (ms < 30) return "perf-amber";
    if (ms < 60) return "perf-orange";
    return "perf-red";
  };
  const timingHtml = entries.map(([name, t]) => `
    <div class="perf-card">
      <div class="perf-name">${escapeHtml(name)}</div>
      <div class="perf-row">
        <span class="perf-label">avg</span>
        <span class="perf-val ${_color(t.avg_ms)}">${t.avg_ms.toFixed(1)} ms</span>
      </div>
      <div class="perf-row">
        <span class="perf-label">p50</span>
        <span class="perf-val">${t.p50_ms.toFixed(1)} ms</span>
      </div>
      <div class="perf-row">
        <span class="perf-label">p95</span>
        <span class="perf-val ${_color(t.p95_ms)}">${t.p95_ms.toFixed(1)} ms</span>
      </div>
      <div class="perf-row">
        <span class="perf-label">max</span>
        <span class="perf-val">${t.max_ms.toFixed(1)} ms</span>
      </div>
      <div class="perf-row">
        <span class="perf-label">last</span>
        <span class="perf-val">${t.last_ms.toFixed(1)} ms</span>
      </div>
      <div class="perf-row">
        <span class="perf-label">n</span>
        <span class="perf-val">${t.n}</span>
      </div>
    </div>
  `).join("");
  // Counters render as a row of per-sec rates. Skip when empty.
  const counterHtml = counterEntries.length
    ? `<div class="perf-card perf-counters">
        <div class="perf-name">counters (per/sec total)</div>
        ${counterEntries.map(([n, v]) => `
          <div class="perf-row">
            <span class="perf-label">${escapeHtml(n)}</span>
            <span class="perf-val">${(v / uptimeS).toFixed(1)}/s · ${v}</span>
          </div>`).join("")}
      </div>`
    : "";
  const modelHtml = modelCalls.length
    ? `<div class="perf-card perf-counters perf-model-calls">
        <div class="perf-name">model cost / latency (today)</div>
        ${modelCalls.map((m) => `
          <div class="perf-model-row">
            <div class="perf-model-name">${escapeHtml(m.provider)} · ${escapeHtml(m.model)}</div>
            <div class="perf-row">
              <span class="perf-label">calls</span>
              <span class="perf-val">${m.calls} (${m.cloud_calls || 0} cloud)</span>
            </div>
            <div class="perf-row">
              <span class="perf-label">avg</span>
              <span class="perf-val ${_color(m.avg_latency_ms || 0)}">${Number(m.avg_latency_ms || 0).toFixed(1)} ms</span>
            </div>
            <div class="perf-row">
              <span class="perf-label">timeouts</span>
              <span class="perf-val">${Math.round(Number(m.timeout_rate || 0) * 100)}%</span>
            </div>
            <div class="perf-row">
              <span class="perf-label">tool iters</span>
              <span class="perf-val">${Number(m.avg_tool_iterations || 0).toFixed(1)}</span>
            </div>
          </div>`).join("")}
      </div>`
    : "";
  grid.innerHTML = modelHtml + timingHtml + counterHtml;
}

function startPerfRefresh() {
  stopPerfRefresh();
  loadPerfTab();
  const auto = document.getElementById("perf-auto-refresh");
  if (auto && auto.checked) {
    _perfTimer = safeInterval(loadPerfTab, 2000);
  }
}

function stopPerfRefresh() {
  if (_perfTimer) {
    stopSafeInterval(_perfTimer);
    _perfTimer = null;
  }
}

(function wirePerfControls() {
  const auto = document.getElementById("perf-auto-refresh");
  const refresh = document.getElementById("perf-refresh-now");
  if (auto) {
    auto.addEventListener("change", () => {
      if (auto.checked) startPerfRefresh();
      else stopPerfRefresh();
    });
  }
  if (refresh) refresh.addEventListener("click", () => loadPerfTab());
})();

// ── Settings tab ──────────────────────────────────────────────────────────

const _SETTINGS_WM_LABELS = {
  visibility_grace_seconds: "Visibility grace floor (seconds)",
  visibility_window_seconds: "Visibility smoothing window (seconds)",
  visibility_min_samples: "Visibility min samples in window",
  visibility_seen_fraction_floor: "Visibility seen-fraction floor (0-1)",
  person_continuity_seconds: "Person continuity (seconds)",
  movement_jitter_threshold: "Movement jitter threshold (0-1)",
  posture_debounce_frames: "Posture debounce (frames)",
  interaction_debounce_frames: "Interaction debounce (frames)",
  landmark_dwell_frames: "Landmark dwell (frames)",
  T_handoff_seconds: "Hand-off window (seconds)",
  stationary_long_minutes: "Stationary threshold (minutes)",
};

let _settingsState = null;

async function loadSettings() {
  const status = document.getElementById("settings-status");
  if (status) { status.textContent = "loading…"; status.className = "settings-status"; }
  try {
    const res = await fetch("/api/tunables");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    _settingsState = await res.json();
    renderSettings(_settingsState);
    if (status) { status.textContent = ""; }
  } catch (e) {
    if (status) { status.textContent = `failed: ${e.message || e}`; status.className = "settings-status err"; }
  }
}

function renderSettings(state) {
  const wm = state.world_model || {};
  const snap = state.snapshots || {};
  const logs = (state.logs && state.logs.console_debug_blacklist) || [];

  const wmEl = document.getElementById("settings-wm");
  if (wmEl) {
    wmEl.innerHTML = Object.entries(_SETTINGS_WM_LABELS)
      .map(([k, label]) => {
        const v = wm[k];
        let step = "1";
        if (k === "movement_jitter_threshold" || k === "visibility_seen_fraction_floor") {
          step = "0.05";
        } else if (k.endsWith("_seconds") || k.endsWith("_minutes")) {
          step = "0.5";
        }
        return `<label>
          <span>${escapeHtml(label)}</span>
          <input type="number" step="${step}" data-key="${k}" value="${v !== undefined ? v : ""}" />
        </label>`;
      })
      .join("");
  }

  const snapEl = document.getElementById("settings-snap");
  if (snapEl) {
    snapEl.innerHTML = Object.entries(snap)
      .map(([k, v]) => `<label>
          <span>${escapeHtml(k)}</span>
          <input type="number" step="1" data-key="${k}" value="${v}" />
        </label>`)
      .join("");
  }

  const logsEl = document.getElementById("settings-logs");
  if (logsEl) {
    const rows = logs.map((mod, i) => `
      <div class="log-mod-row">
        <input type="text" value="${escapeHtml(mod)}" data-idx="${i}" />
        <button class="dev-btn log-mod-remove" data-idx="${i}">Remove</button>
      </div>
    `).join("");
    const addRow = `
      <div class="log-mod-row">
        <input type="text" id="log-mod-add" placeholder="modules.something.module" />
        <button class="dev-btn" id="log-mod-add-btn">Add</button>
      </div>`;
    logsEl.innerHTML = rows + addRow;
    logsEl.querySelectorAll(".log-mod-remove").forEach((b) => {
      b.addEventListener("click", () => {
        const i = Number(b.dataset.idx);
        const updated = logs.filter((_, j) => j !== i);
        applyLogBlacklist(updated);
      });
    });
    const addBtn = document.getElementById("log-mod-add-btn");
    const addInput = document.getElementById("log-mod-add");
    if (addBtn && addInput) {
      addBtn.addEventListener("click", () => {
        const v = addInput.value.trim();
        if (!v) return;
        applyLogBlacklist([...logs, v]);
      });
    }
  }
}

function _collectSettingsForm() {
  const wm = {};
  document.querySelectorAll("#settings-wm input[data-key]").forEach((el) => {
    const k = el.dataset.key;
    const v = el.value.trim();
    if (v === "") return;
    wm[k] = Number(v);
  });
  const snap = {};
  document.querySelectorAll("#settings-snap input[data-key]").forEach((el) => {
    const k = el.dataset.key;
    const v = el.value.trim();
    if (v === "") return;
    snap[k] = Number(v);
  });
  return { world_model: wm, snapshots: snap };
}

async function _patchConfig(body) {
  const status = document.getElementById("settings-status");
  try {
    const res = await fetch("/api/tunables", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const out = await res.json();
    _settingsState = out.state || _settingsState;
    renderSettings(_settingsState);
    if (status) {
      if (out.errors && out.errors.length) {
        status.textContent = `applied with ${out.errors.length} error(s)`;
        status.className = "settings-status err";
        console.warn("[config] errors:", out.errors);
      } else {
        status.textContent = "applied ✓";
        status.className = "settings-status ok";
      }
    }
  } catch (e) {
    if (status) { status.textContent = `failed: ${e.message || e}`; status.className = "settings-status err"; }
  }
}

async function applyLogBlacklist(list) {
  await _patchConfig({ logs: { console_debug_blacklist: list } });
}

(function wireSettingsButtons() {
  const save = document.getElementById("settings-save");
  const reload = document.getElementById("settings-reload");
  if (save) save.addEventListener("click", () => _patchConfig(_collectSettingsForm()));
  if (reload) reload.addEventListener("click", () => loadSettings());
})();

// ── Logs tab — WebSocket stream ──────────────────────────────────────────

let _logSocket = null;
let _logBuffer = [];
let _logPaused = false;
const LOG_MAX_LINES = 2000;

function connectLogStream() {
  if (_logSocket && _logSocket.readyState <= 1) return;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  _logSocket = new WebSocket(`${proto}://${location.host}/ws/logs`);
  _logSocket.addEventListener("open", () => {
    _sendLogFilter();
  });
  _logSocket.addEventListener("message", (ev) => {
    if (_logPaused) return;
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type !== "log" || !msg.record) return;
    appendLogLine(msg.record);
  });
  _logSocket.addEventListener("close", () => { _logSocket = null; });
}

function disconnectLogStream() {
  if (_logSocket) {
    try { _logSocket.close(); } catch {}
    _logSocket = null;
  }
}

function _sendLogFilter() {
  if (!_logSocket || _logSocket.readyState !== 1) return;
  const minLevel = (document.getElementById("logs-min-level") || {}).value || "INFO";
  const includeRaw = (document.getElementById("logs-include") || {}).value || "";
  const include = includeRaw.split(",").map((s) => s.trim()).filter(Boolean);
  _logSocket.send(JSON.stringify({ min_level: minLevel, include }));
}

function appendLogLine(rec) {
  const pre = document.getElementById("logs-stream");
  if (!pre) return;
  const ts = (rec.ts || "").slice(11, 19);
  const levelCls = `log-${(rec.level || "info").toLowerCase()}`;
  const line = `${ts} [<span class="${levelCls}">${escapeHtml(rec.level)}</span>] ` +
               `<span class="log-name">${escapeHtml(rec.name || "")}:${rec.line || 0}</span> ${escapeHtml(rec.message || "")}\n`;
  _logBuffer.push(line);
  if (_logBuffer.length > LOG_MAX_LINES) {
    _logBuffer = _logBuffer.slice(-LOG_MAX_LINES);
  }
  // Append the new line at the end without rebuilding the whole pane.
  const wasAtBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 8;
  pre.insertAdjacentHTML("beforeend", line);
  // Trim the head if we're over the limit.
  if (pre.childNodes.length > LOG_MAX_LINES * 2) {
    pre.innerHTML = _logBuffer.join("");
  }
  if (wasAtBottom) pre.scrollTop = pre.scrollHeight;
}

(function wireLogButtons() {
  const apply = document.getElementById("logs-apply");
  const clear = document.getElementById("logs-clear");
  const pause = document.getElementById("logs-pause");
  if (apply) apply.addEventListener("click", () => _sendLogFilter());
  if (clear) clear.addEventListener("click", () => {
    _logBuffer = [];
    const pre = document.getElementById("logs-stream");
    if (pre) pre.innerHTML = "";
  });
  if (pause) pause.addEventListener("change", () => {
    _logPaused = pause.checked;
  });
})();

// ── Init ──────────────────────────────────────────────────────────────────

connect();
