# security.py
# Purpose: security headers on every response and per-client rate limits on the scoring endpoints.
# Author: Colile Sibanda

from __future__ import annotations

import math
import time
from collections import OrderedDict, deque
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from phishlens.config.schema import ServingConfig

WINDOW_SECONDS = 60.0
MAX_TRACKED_CLIENTS = 10_000  # bounds memory if many addresses send one request each
SCORING_PATHS = ("/predict", "/explain")
GLOBAL_KEY = "*"

BASE_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Strict-Transport-Security": "max-age=31536000",
}
# The browser UI loads only its own files: no third-party script, style or font.
UI_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
)
# JSON responses never need to load anything.
API_CSP = "default-src 'none'; frame-ancestors 'none'"


class SlidingWindowLimiter:
    """Count requests per key over the last `WINDOW_SECONDS` seconds.

    Pure apart from the injected clock, so it is testable without sleeping.
    The least recently seen keys are dropped beyond `MAX_TRACKED_CLIENTS`.

    Inputs: clock : returns the current time in seconds (default monotonic).
    Output: a limiter with `allow(key, limit) -> (allowed, retry_after)`.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()

    def allow(self, key: str, limit: int) -> tuple[bool, int]:
        """Record one request for `key` if it is within `limit` per window.

        Inputs: key : client address or GLOBAL_KEY; limit : requests per window.
        Output: (allowed, seconds until the oldest counted request expires).
        """
        now = self._clock()
        hits = self._hits.setdefault(key, deque())
        self._hits.move_to_end(key)
        while hits and now - hits[0] >= WINDOW_SECONDS:
            hits.popleft()
        if len(hits) >= limit:
            return False, max(1, math.ceil(WINDOW_SECONDS - (now - hits[0])))
        hits.append(now)
        while len(self._hits) > MAX_TRACKED_CLIENTS:
            self._hits.popitem(last=False)
        return True, 0


def client_address(request: Request, trusted_proxy_hops: int) -> str:
    """Identify the client, trusting only as many proxies as configured.

    Behind N trusted proxies the real client is the Nth address from the right
    of X-Forwarded-For; anything further left was written by the client and can
    be forged, so it is never used.

    Inputs: request : the incoming request; trusted_proxy_hops : 0 for direct.
    Output: the client address as a string.
    """
    direct = request.client.host if request.client else "unknown"
    if trusted_proxy_hops <= 0:
        return direct
    forwarded = [part.strip() for part in request.headers.get("x-forwarded-for", "").split(",") if part.strip()]
    return forwarded[-trusted_proxy_hops] if len(forwarded) >= trusted_proxy_hops else direct


def install_security(app: FastAPI, settings: ServingConfig, limiter: SlidingWindowLimiter | None = None) -> None:
    """Add the rate-limit and security-header middleware to the app.

    Inputs: app : the FastAPI app; settings : the serving config; limiter :
    optional, for tests that control the clock.
    Output: none; mutates the app.
    """
    limiter = limiter if limiter is not None else SlidingWindowLimiter()

    @app.middleware("http")
    async def _guard(request: Request, call_next):
        if request.method == "POST" and request.url.path in SCORING_PATHS:
            refusal = _check_limits(request, settings, limiter)
            if refusal is not None:
                return _with_headers(request, refusal)
        return _with_headers(request, await call_next(request))


def _check_limits(request: Request, settings: ServingConfig, limiter: SlidingWindowLimiter):
    """Apply the per-client and global limits to one scoring request.

    Inputs: request; settings : the serving config; limiter.
    Output: a 429 response if a limit is exceeded, else None.
    """
    client = client_address(request, settings.trusted_proxy_hops)
    checks = [(client, settings.rate_limit_per_minute, "requests from this client")]
    if request.url.path == "/explain":
        checks.append((f"{client}|explain", settings.explain_rate_limit_per_minute, "explanations from this client"))
        checks.append((GLOBAL_KEY, settings.explain_global_limit_per_minute, "explanations across all users"))
    for key, limit, what in checks:
        allowed, retry_after = limiter.allow(key, limit)
        if not allowed:
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(retry_after)},
                content={"detail": f"Too many {what}: the limit is {limit} per minute. Try again in {retry_after} s."},
            )
    return None


def _with_headers(request: Request, response):
    """Attach the security headers; the UI gets a page CSP, JSON a deny-all one.

    The interactive docs at /docs load their script from a CDN, so they keep
    the base headers only.

    Inputs: request; response.
    Output: the same response, with headers set.
    """
    for name, value in BASE_HEADERS.items():
        response.headers.setdefault(name, value)
    path = request.url.path
    if path.startswith("/ui"):
        response.headers["Content-Security-Policy"] = UI_CSP
    elif not path.startswith(("/docs", "/redoc")):
        response.headers["Content-Security-Policy"] = API_CSP
    return response
