"""
JARVIS — Dashboard access token
===============================
Mission: Gate the dashboard's HTTP + WebSocket surface with a single shared
         token WITHOUT adding friction for the local user.

         The dashboard exposes cameras, mics, recorded face/voice clips,
         config read+write, model pulls, and computer control. On a trusted
         LAN that is acceptable; the moment the bind address is anything but
         loopback, an unauthenticated actor on the network can reach all of
         it. This module adds a token that is required ONLY for genuine
         off-box requests:

           - Requests from localhost / 127.0.0.1 / ::1 are exempt entirely,
             so the machine's own user needs nothing new.
           - The token is auto-generated once and persisted to a gitignored
             local file, so it never has to be typed or managed.
           - Off-box browsers present it via the X-Dashboard-Token header, a
             `token` query parameter (WebSockets and <img> tags cannot set
             headers, so the frontend also drops it into a same-origin
             cookie), or the cookie directly.

Modules: dashboard/auth.py
Functions:
    load_or_create_token(data_dir)   — env var -> file -> generate+persist
    is_local_host(host)              — loopback check (v4 /8, v6, mapped v4)
    extract_token(scope)             — pull a token out of an ASGI request
    tokens_match(provided, expected) — constant-time compare
Classes:
    TokenAuthMiddleware              — pure-ASGI guard over http + websocket
"""

from __future__ import annotations

import hmac
import os
import secrets
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import parse_qs, unquote

from loguru import logger

TOKEN_HEADER = b"x-dashboard-token"
TOKEN_QUERY = "token"
TOKEN_COOKIE = "jarvis_dashboard_token"
ENV_VAR = "JARVIS_DASHBOARD_TOKEN"
TOKEN_FILENAME = "dashboard_token"

# 127.0.0.0/8 is entirely loopback; ::1 is v6 loopback; ::ffff:127.x is a
# v4-mapped loopback address some stacks report.
_LOOPBACK_EXACT = frozenset({"127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1"})


def is_local_host(host: Optional[str]) -> bool:
    """True when the peer address is loopback (Cole's own machine)."""
    if not host:
        return False
    h = host.strip().lower()
    if h in _LOOPBACK_EXACT:
        return True
    if h.startswith("127.") or h.startswith("::ffff:127."):
        return True
    return False


def load_or_create_token(data_dir: "str | os.PathLike[str]" = "data") -> str:
    """Return the dashboard token, creating and persisting one if needed.

    Resolution order:
      1. The ``JARVIS_DASHBOARD_TOKEN`` env var (lets Cole pin a value or a
         deployment inject one).
      2. A previously persisted ``<data_dir>/dashboard_token`` file.
      3. A freshly generated token, written to that file (best effort, 0600).

    Never raises — if the file cannot be written the token still works for
    the current process; the caller logs a warning so off-box access can be
    restored by setting the env var.
    """
    env_val = os.environ.get(ENV_VAR, "").strip()
    if env_val:
        return env_val

    path = Path(data_dir) / TOKEN_FILENAME
    try:
        if path.exists():
            existing = path.read_text(encoding="utf-8").strip()
            if existing:
                return existing
    except OSError as e:
        logger.warning(f"[Dashboard] Could not read auth token {path}: {e}")

    token = secrets.token_urlsafe(32)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(token, encoding="utf-8")
        try:
            os.chmod(path, 0o600)  # honored on POSIX, harmless on Windows
        except OSError:
            pass
        logger.info(
            f"[Dashboard] Generated a new access token at {path} "
            "(gitignored; used only for off-box access, localhost is exempt)"
        )
    except OSError as e:
        logger.warning(
            f"[Dashboard] Could not persist auth token to {path} ({e}); "
            f"set {ENV_VAR} to a fixed value if you need off-box access"
        )
    return token


def _header(headers: Iterable[tuple[bytes, bytes]], name: bytes) -> Optional[str]:
    for k, v in headers:
        if k.lower() == name:
            try:
                return v.decode("latin-1")
            except Exception:
                return None
    return None


def extract_token(scope: dict) -> Optional[str]:
    """Pull a token from an ASGI scope: header, bearer, query, or cookie."""
    headers = scope.get("headers") or []

    tok = _header(headers, TOKEN_HEADER)
    if tok:
        return tok.strip()

    auth = _header(headers, b"authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[7:].strip()

    qs = scope.get("query_string") or b""
    if qs:
        vals = parse_qs(qs.decode("latin-1")).get(TOKEN_QUERY)
        if vals and vals[0]:
            return vals[0].strip()

    cookie = _header(headers, b"cookie")
    if cookie:
        for part in cookie.split(";"):
            name, sep, value = part.strip().partition("=")
            if sep and name == TOKEN_COOKIE:
                return unquote(value).strip()

    return None


def tokens_match(provided: Optional[str], expected: Optional[str]) -> bool:
    """Constant-time token comparison."""
    if not provided or not expected:
        return False
    return hmac.compare_digest(provided, expected)


class TokenAuthMiddleware:
    """Pure-ASGI guard covering BOTH http and websocket scopes.

    First matching rule wins and lets the request through:
      1. Non http/websocket scope (lifespan)           -> pass
      2. No token configured (control disabled)        -> pass (fail open;
         a loud warning was logged at startup so Cole is never hard-blocked
         from booting his own machine)
      3. Peer is loopback (local trust)                -> pass
      4. Path is exempt (health probe)                 -> pass
      5. A valid token is presented                    -> pass
      6. Otherwise                                     -> 401 (http) /
         close 1008 (websocket)
    """

    def __init__(
        self,
        app,
        token: Optional[str],
        exempt_paths: Iterable[str] = (),
    ) -> None:
        self.app = app
        self.token = token or None
        self.exempt_paths = frozenset(exempt_paths)

    async def __call__(self, scope, receive, send):
        if scope.get("type") not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        if self.token is None:
            await self.app(scope, receive, send)
            return

        client = scope.get("client") or ()
        host = client[0] if client else None
        if is_local_host(host):
            await self.app(scope, receive, send)
            return

        if scope.get("path", "") in self.exempt_paths:
            await self.app(scope, receive, send)
            return

        if tokens_match(extract_token(scope), self.token):
            await self.app(scope, receive, send)
            return

        await self._reject(scope, receive, send)

    async def _reject(self, scope, receive, send):
        if scope["type"] == "websocket":
            # Drain the connect message, then decline the handshake. Per ASGI,
            # sending websocket.close before accept fails the handshake (the
            # client sees an HTTP 403).
            try:
                await receive()
            except Exception:
                pass
            await send({"type": "websocket.close", "code": 1008})
            return
        body = b'{"detail":"Dashboard token required or invalid"}'
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"www-authenticate", b"Bearer"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
