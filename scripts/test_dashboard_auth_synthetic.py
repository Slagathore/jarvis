"""
JARVIS — Dashboard access-token auth
====================================
Synthetic test for the token guard added in dashboard/auth.py +
dashboard/server.py (H1). Drives the REAL FastAPI app and the REAL
TokenAuthMiddleware through hand-built ASGI scopes so the peer address is
fully controllable — TestClient hardcodes its client host, which would hide
exactly the localhost-vs-LAN distinction this control turns on.

Proves the control's contract:
  • localhost request with NO token           -> passes (Cole's own access)
  • off-box (LAN) request with NO token        -> 401
  • off-box request WITH the token             -> passes (header / query / cookie)
  • off-box request with the WRONG token       -> 401
  • /api/health is exempt off-box              -> passes (liveness probe)
  • WebSocket off-box with no token            -> handshake refused (close 1008)
  • WebSocket off-box with the token           -> reaches the app
  • no token configured (control disabled)     -> fails open, never blocks boot

Run: python scripts/test_dashboard_auth_synthetic.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.auth import (  # noqa: E402
    ENV_VAR,
    TokenAuthMiddleware,
    is_local_host,
    load_or_create_token,
    tokens_match,
)
from dashboard.server import DashboardServer  # noqa: E402

TOKEN = "smoke-token-abcdef0123456789"
LAN = "192.168.1.50"
LOCAL = "127.0.0.1"


def _server(auth_token=TOKEN, require_auth=True):
    # auth_token is pinned so no token file is ever read or written here.
    return DashboardServer(
        host=LOCAL, port=0, auth_token=auth_token, require_auth=require_auth
    )


async def _http(app, path, client_host, headers=None, query=b""):
    """Invoke the full ASGI app for one GET; return (status, body_bytes)."""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": query if isinstance(query, bytes) else query.encode(),
        "root_path": "",
        "headers": [
            (k.lower().encode(), v.encode()) for k, v in (headers or {}).items()
        ],
        "client": (client_host, 51234),
        "server": ("127.0.0.1", 7070),
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    out: list = []

    async def send(message):
        out.append(message)

    await app(scope, receive, send)
    start = next(m for m in out if m["type"] == "http.response.start")
    body = b"".join(
        m.get("body", b"") for m in out if m["type"] == "http.response.body"
    )
    return start["status"], body


def _get(app, path, client_host, headers=None, query=b""):
    return asyncio.run(_http(app, path, client_host, headers, query))


# ── HTTP: the core control contract ─────────────────────────────────────────


def test_localhost_passes_without_token() -> None:
    status, body = _get(_server().app, "/api/state", LOCAL)
    assert status == 200, status
    assert b'"state"' in body, body
    print("PASS: localhost request with no token -> 200 (Cole's access intact)")


def test_offbox_without_token_is_401() -> None:
    status, _ = _get(_server().app, "/api/state", LAN)
    assert status == 401, status
    print("PASS: off-box request without token -> 401")


def test_offbox_with_header_token_passes() -> None:
    status, body = _get(
        _server().app, "/api/state", LAN, headers={"X-Dashboard-Token": TOKEN}
    )
    assert status == 200, status
    assert b'"state"' in body, body
    print("PASS: off-box request with header token -> 200")


def test_offbox_with_query_token_passes() -> None:
    status, _ = _get(_server().app, "/api/state", LAN, query="token=" + TOKEN)
    assert status == 200, status
    print("PASS: off-box request with ?token= -> 200 (WebSocket/img path)")


def test_offbox_with_cookie_token_passes() -> None:
    status, _ = _get(
        _server().app,
        "/api/state",
        LAN,
        headers={"Cookie": "jarvis_dashboard_token=" + TOKEN},
    )
    assert status == 200, status
    print("PASS: off-box request with cookie token -> 200")


def test_offbox_with_wrong_token_is_401() -> None:
    status, _ = _get(
        _server().app, "/api/state", LAN, headers={"X-Dashboard-Token": "nope"}
    )
    assert status == 401, status
    print("PASS: off-box request with wrong token -> 401")


def test_health_is_exempt_offbox() -> None:
    status, _ = _get(_server().app, "/api/health", LAN)
    assert status == 200, status
    print("PASS: /api/health is exempt off-box -> 200 (liveness probe)")


def test_control_disabled_fails_open() -> None:
    # Bound wide with no token: warns at startup, but must not hard-block.
    app = _server(auth_token=None, require_auth=False).app
    status, _ = _get(app, "/api/state", LAN)
    assert status == 200, status
    print("PASS: no token configured -> fails open, never blocks Cole's boot")


# ── WebSocket: same guard covers the WS handshake ───────────────────────────


async def _ws_reaches_app(token, client_host, query=b""):
    """Return True if the guard let a WS handshake through to the inner app."""
    reached = {"v": False}

    async def inner(scope, receive, send):
        reached["v"] = True
        await send({"type": "websocket.accept"})

    mw = TokenAuthMiddleware(inner, token=token, exempt_paths={"/api/health"})
    scope = {
        "type": "websocket",
        "path": "/ws",
        "raw_path": b"/ws",
        "query_string": query if isinstance(query, bytes) else query.encode(),
        "headers": [],
        "client": (client_host, 51234),
    }
    sent: list = []

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        sent.append(message)

    await mw(scope, receive, send)
    closed = any(m.get("type") == "websocket.close" for m in sent)
    return reached["v"], closed


def test_ws_offbox_without_token_refused() -> None:
    reached, closed = asyncio.run(_ws_reaches_app(TOKEN, LAN))
    assert not reached, "guard must not reach the app"
    assert closed, "guard must close the handshake"
    print("PASS: off-box WebSocket with no token -> handshake refused (close)")


def test_ws_offbox_with_token_reaches_app() -> None:
    reached, closed = asyncio.run(
        _ws_reaches_app(TOKEN, LAN, query="token=" + TOKEN)
    )
    assert reached, "valid token must reach the app"
    assert not closed, "valid token must not be closed"
    print("PASS: off-box WebSocket with ?token= -> reaches the app")


def test_ws_localhost_reaches_app_without_token() -> None:
    reached, closed = asyncio.run(_ws_reaches_app(TOKEN, LOCAL))
    assert reached and not closed
    print("PASS: localhost WebSocket with no token -> reaches the app")


# ── Unit: helpers the middleware is built on ────────────────────────────────


def test_is_local_host_covers_loopback() -> None:
    assert is_local_host("127.0.0.1")
    assert is_local_host("::1")
    assert is_local_host("127.5.5.5")  # 127.0.0.0/8 is all loopback
    assert is_local_host("::ffff:127.0.0.1")
    assert not is_local_host("192.168.1.50")
    assert not is_local_host("0.0.0.0")
    assert not is_local_host(None)
    print("PASS: is_local_host -> loopback only")


def test_tokens_match_is_exact() -> None:
    assert tokens_match("abc", "abc")
    assert not tokens_match("abc", "abd")
    assert not tokens_match("", "abc")
    assert not tokens_match("abc", None)
    print("PASS: tokens_match -> exact, empty/None rejected")


def test_load_or_create_token_env_and_file() -> None:
    prev = os.environ.get(ENV_VAR)
    try:
        os.environ[ENV_VAR] = "env-pinned-token"
        assert load_or_create_token("unused") == "env-pinned-token"
        os.environ.pop(ENV_VAR, None)
        with tempfile.TemporaryDirectory() as d:
            first = load_or_create_token(d)
            assert first and len(first) >= 20, first
            # Persisted: a second call returns the same value.
            assert load_or_create_token(d) == first
            assert (Path(d) / "dashboard_token").read_text().strip() == first
        print("PASS: load_or_create_token -> env wins, else generate + persist")
    finally:
        if prev is None:
            os.environ.pop(ENV_VAR, None)
        else:
            os.environ[ENV_VAR] = prev


def main() -> None:
    test_localhost_passes_without_token()
    test_offbox_without_token_is_401()
    test_offbox_with_header_token_passes()
    test_offbox_with_query_token_passes()
    test_offbox_with_cookie_token_passes()
    test_offbox_with_wrong_token_is_401()
    test_health_is_exempt_offbox()
    test_control_disabled_fails_open()
    test_ws_offbox_without_token_refused()
    test_ws_offbox_with_token_reaches_app()
    test_ws_localhost_reaches_app_without_token()
    test_is_local_host_covers_loopback()
    test_tokens_match_is_exact()
    test_load_or_create_token_env_and_file()
    print("\nAll dashboard auth tests passed.")


if __name__ == "__main__":
    main()
