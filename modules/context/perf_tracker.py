"""
JARVIS — Ambient Home AI
========================
Mission: Lightweight performance tracker. Holds rolling timing samples
         for hot-path operations (YOLO inference, observation building,
         scene description, etc.) and per-event-bus throughput counts so
         the dashboard Perf tab can answer "which thing is making it
         laggy?" without guesswork.

         Designed for ~zero overhead in the hot path: every call sites
         appends one float to a deque, no locking (the GIL is sufficient
         for `deque.append`), and reads aggregate on demand inside the
         dashboard endpoint.

Modules: modules/context/perf_tracker.py
Classes: PerfTracker, TimingContext (context manager helper)
"""
from __future__ import annotations

import time
from datetime import datetime
from collections import deque
from contextlib import contextmanager
from threading import Lock
from typing import Iterator, Optional


class PerfTracker:
    """Process-singleton holding per-metric rolling deques. Capped at
    `window` samples per metric (default 300) so memory stays bounded
    even for 30fps timings. Aggregation walks the deque once per
    request — O(N) but N is small."""

    _instance: Optional["PerfTracker"] = None

    def __init__(self, window: int = 300) -> None:
        self._window = window
        self._timings: dict[str, deque] = {}
        self._counters: dict[str, int] = {}
        self._model_calls: dict[tuple[str, str, str], dict] = {}
        self._lock = Lock()
        self._started_at = time.monotonic()

    @classmethod
    def instance(cls) -> "PerfTracker":
        if cls._instance is None:
            cls._instance = PerfTracker()
        return cls._instance

    def record_ms(self, name: str, ms: float) -> None:
        """Append one timing sample (in milliseconds) to the named
        rolling window. Cheap enough for per-frame use."""
        d = self._timings.get(name)
        if d is None:
            with self._lock:
                d = self._timings.get(name)
                if d is None:
                    d = deque(maxlen=self._window)
                    self._timings[name] = d
        d.append(float(ms))

    def increment(self, name: str, n: int = 1) -> None:
        """Bump a counter — used for events/sec style metrics."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + n

    def record_model_call(
        self,
        provider: str,
        model: str,
        *,
        latency_ms: float,
        ok: bool,
        timeout: bool = False,
        cloud: bool = False,
        tool_iterations: int = 0,
    ) -> None:
        """Record one LLM/provider call for cost/latency visibility.

        The tracker deliberately stores daily aggregates rather than raw
        request rows. It answers operational questions ("is Gemini slow
        today?", "are timeouts spiking?", "how many cloud calls did we make?")
        without growing memory over a long-running assistant process.
        """
        day = datetime.now().date().isoformat()
        key = (day, provider or "unknown", model or "unknown")
        with self._lock:
            row = self._model_calls.get(key)
            if row is None:
                row = {
                    "day": day,
                    "provider": provider or "unknown",
                    "model": model or "unknown",
                    "calls": 0,
                    "ok": 0,
                    "errors": 0,
                    "timeouts": 0,
                    "cloud_calls": 0,
                    "latency_total_ms": 0.0,
                    "latency_max_ms": 0.0,
                    "tool_iterations_total": 0,
                    "tool_loops": 0,
                }
                self._model_calls[key] = row
            row["calls"] += 1
            row["ok" if ok else "errors"] += 1
            row["timeouts"] += 1 if timeout else 0
            row["cloud_calls"] += 1 if cloud else 0
            row["latency_total_ms"] += max(0.0, float(latency_ms))
            row["latency_max_ms"] = max(
                float(row["latency_max_ms"]), max(0.0, float(latency_ms))
            )
            if tool_iterations:
                row["tool_iterations_total"] += max(0, int(tool_iterations))
                row["tool_loops"] += 1

    @contextmanager
    def timeit(self, name: str) -> Iterator[None]:
        """Context manager: `with tracker.timeit('yolo'): ...`"""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record_ms(name, (time.perf_counter() - t0) * 1000.0)

    def snapshot(self) -> dict:
        """Aggregate current state. Returns:
          {
            "uptime_s": float,
            "timings": {
              name: {
                "n": int, "avg_ms": float, "p50_ms": float,
                "p95_ms": float, "max_ms": float, "last_ms": float,
                "per_s": float,
              }, ...
            },
            "counters": {name: int},
          }
        """
        out_timings: dict[str, dict] = {}
        uptime = max(0.001, time.monotonic() - self._started_at)
        for name, d in list(self._timings.items()):
            samples = list(d)
            if not samples:
                continue
            samples_sorted = sorted(samples)
            n = len(samples)
            avg = sum(samples) / n
            p50 = samples_sorted[n // 2]
            p95 = samples_sorted[min(n - 1, int(n * 0.95))]
            mx = samples_sorted[-1]
            last = samples[-1]
            # rough per-second rate — the deque doesn't carry timestamps,
            # so we infer from window vs uptime (close enough for orders
            # of magnitude on the dashboard).
            per_s = min(n / uptime, n)
            out_timings[name] = {
                "n": n,
                "avg_ms": avg,
                "p50_ms": p50,
                "p95_ms": p95,
                "max_ms": mx,
                "last_ms": last,
                "per_s": per_s,
            }
        model_calls: list[dict] = []
        with self._lock:
            model_rows = list(self._model_calls.values())
        for row in model_rows:
            calls = max(1, int(row["calls"]))
            tool_loops = max(1, int(row.get("tool_loops", 0)))
            out = dict(row)
            out["avg_latency_ms"] = float(row["latency_total_ms"]) / calls
            out["timeout_rate"] = float(row["timeouts"]) / calls
            out["avg_tool_iterations"] = (
                float(row["tool_iterations_total"]) / tool_loops
                if row.get("tool_loops") else 0.0
            )
            model_calls.append(out)

        return {
            "uptime_s": uptime,
            "timings": out_timings,
            "counters": dict(self._counters),
            "model_calls": sorted(
                model_calls,
                key=lambda r: (r.get("day", ""), r.get("provider", ""), r.get("model", "")),
                reverse=True,
            ),
        }


def perf() -> PerfTracker:
    """Module-level shortcut."""
    return PerfTracker.instance()
