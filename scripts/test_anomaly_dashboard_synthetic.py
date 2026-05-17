"""
JARVIS — §25 Anomaly Dashboard
==============================
Synthetic test for the §25 anomaly review-queue endpoints added to
dashboard/server.py:

  GET  /api/world_model/anomalies
  POST /api/world_model/anomalies/{id}/invalidate

A real DashboardServer is built and driven with FastAPI's TestClient; the
orchestrator is a stub carrying a stub AnomalyScorer, so no DB / world
model is needed. The stub's recent_anomalies() returns rows shaped like
the world_anomalies table — `components` / `event` as JSON TEXT — so the
endpoint's server-side JSON decode is exercised.

Covers:
  • no orchestrator / no scorer  -> available:false, empty queue
  • populated queue              -> decoded components+event, threshold
  • invalidate a known anomaly   -> scorer.invalidate called, ok:true
  • invalidate an unknown id     -> 404
  • invalidate with no scorer    -> 503

Run: python scripts/test_anomaly_dashboard_synthetic.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from dashboard.server import DashboardServer


# ── Stubs ───────────────────────────────────────────────────────────────────


class StubAnomalyScorer:
    """Mimics AnomalyScorer's dashboard-facing surface."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.threshold = 6.0
        self.invalidate_calls: list[tuple[str, str]] = []

    async def recent_anomalies(self, limit: int = 50) -> list[dict]:
        # Mirror the real query: newest-first, limit-capped, dict rows.
        return [dict(r) for r in self._rows[:limit]]

    async def invalidate(self, anomaly_id: str, reason: str = "") -> bool:
        self.invalidate_calls.append((anomaly_id, reason))
        return any(r["id"] == anomaly_id for r in self._rows)


class StubOrchestrator:
    def __init__(self, scorer) -> None:
        self.anomaly_scorer = scorer


def _row(anomaly_id: str, score: float, invalidated: int = 0) -> dict:
    """A world_anomalies row — components/event are JSON TEXT, as stored."""
    return {
        "id": anomaly_id,
        "ts": "2026-05-17T03:14:00+00:00",
        "entity_id": "ent-cole",
        "entity_name": "Cole",
        "score": score,
        "components": json.dumps({"room_at_time": 6.5, "time_of_day": 4.0}),
        "event": json.dumps({"event_type": "reappeared", "room": "kitchen"}),
        "invalidated": invalidated,
        "invalidated_reason": "" if not invalidated else "false alarm",
        "invalidated_ts": None,
    }


def _client(scorer) -> TestClient:
    server = DashboardServer(host="127.0.0.1", port=0)
    if scorer is not None:
        server._orchestrator = StubOrchestrator(scorer)
    return TestClient(server.app)


# ── Tests ───────────────────────────────────────────────────────────────────


def test_no_scorer_reports_unavailable() -> None:
    client = _client(scorer=None)
    res = client.get("/api/world_model/anomalies")
    assert res.status_code == 200, res.status_code
    body = res.json()
    assert body["available"] is False, body
    assert body["anomalies"] == [], body
    print("PASS: no anomaly subsystem -> available:false, empty queue")


def test_populated_queue_decodes_json_columns() -> None:
    scorer = StubAnomalyScorer([_row("a1", 8.2), _row("a2", 6.1, invalidated=1)])
    client = _client(scorer)
    res = client.get("/api/world_model/anomalies?limit=40")
    assert res.status_code == 200, res.status_code
    body = res.json()
    assert body["available"] is True, body
    assert body["threshold"] == 6.0, body
    items = body["anomalies"]
    assert len(items) == 2, items
    first = items[0]
    # components / event must arrive as objects, not raw JSON strings.
    assert isinstance(first["components"], dict), first["components"]
    assert first["components"]["room_at_time"] == 6.5, first
    assert isinstance(first["event"], dict), first["event"]
    assert first["event"]["event_type"] == "reappeared", first
    # invalidated coerced to a real bool for the frontend.
    assert first["invalidated"] is False, first
    assert items[1]["invalidated"] is True, items[1]
    print("PASS: populated queue decodes components/event JSON + threshold")


def test_invalidate_known_anomaly() -> None:
    scorer = StubAnomalyScorer([_row("a1", 8.2)])
    client = _client(scorer)
    res = client.post(
        "/api/world_model/anomalies/a1/invalidate",
        json={"reason": "dashboard review"},
    )
    assert res.status_code == 200, (res.status_code, res.text)
    assert res.json()["ok"] is True, res.json()
    assert scorer.invalidate_calls == [("a1", "dashboard review")], (
        scorer.invalidate_calls
    )
    print("PASS: invalidate a known anomaly -> scorer.invalidate called")


def test_invalidate_unknown_anomaly_404() -> None:
    scorer = StubAnomalyScorer([_row("a1", 8.2)])
    client = _client(scorer)
    res = client.post(
        "/api/world_model/anomalies/does-not-exist/invalidate", json={}
    )
    assert res.status_code == 404, (res.status_code, res.text)
    print("PASS: invalidate an unknown id -> 404")


def test_invalidate_without_scorer_503() -> None:
    client = _client(scorer=None)
    res = client.post("/api/world_model/anomalies/a1/invalidate", json={})
    assert res.status_code == 503, (res.status_code, res.text)
    print("PASS: invalidate with no anomaly subsystem -> 503")


def main() -> None:
    test_no_scorer_reports_unavailable()
    test_populated_queue_decodes_json_columns()
    test_invalidate_known_anomaly()
    test_invalidate_unknown_anomaly_404()
    test_invalidate_without_scorer_503()
    print("\nAll §25 anomaly dashboard tests passed.")


if __name__ == "__main__":
    main()
