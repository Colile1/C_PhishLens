# test_security.py
# Purpose: unit tests for the rate limiter, client identification and security headers.
# Author: Colile Sibanda

from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from phishlens.config.loader import load_config
from phishlens.serving.app import create_app
from phishlens.serving.security import SlidingWindowLimiter, client_address


class _Clock:
    """A clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _Request:
    """The two request attributes client_address reads."""

    def __init__(self, host: str, forwarded: str | None = None) -> None:
        self.client = type("Client", (), {"host": host})()
        self.headers = {"x-forwarded-for": forwarded} if forwarded else {}


def test_limiter_refuses_over_the_limit_and_recovers_after_the_window() -> None:
    clock = _Clock()
    limiter = SlidingWindowLimiter(clock)
    assert [limiter.allow("a", 2)[0] for _ in range(3)] == [True, True, False]
    assert limiter.allow("b", 2)[0] is True  # limits are per key
    allowed, retry_after = limiter.allow("a", 2)
    assert not allowed and 1 <= retry_after <= 60
    clock.now += 60
    assert limiter.allow("a", 2)[0] is True


def test_client_address_ignores_forged_forwarded_entries() -> None:
    forged = _Request("10.0.0.1", "6.6.6.6, 203.0.113.7")
    assert client_address(forged, trusted_proxy_hops=0) == "10.0.0.1"
    assert client_address(forged, trusted_proxy_hops=1) == "203.0.113.7"
    assert client_address(_Request("10.0.0.1"), trusted_proxy_hops=1) == "10.0.0.1"


@pytest.fixture(scope="module")
def strict_client():
    """An app whose /explain allows two requests per client per minute."""
    config = load_config()
    config = replace(config, serving=replace(config.serving, explain_rate_limit_per_minute=2))
    with TestClient(create_app(config)) as client:
        yield client


def test_explain_is_rate_limited_with_retry_after(strict_client: TestClient) -> None:
    codes = [strict_client.post("/explain", json={"text": "verify your account"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    refused = strict_client.post("/explain", json={"text": "verify your account"})
    assert refused.status_code == 429
    assert int(refused.headers["retry-after"]) >= 1
    assert "per minute" in refused.json()["detail"]
    assert strict_client.post("/predict", json={"text": "hello"}).status_code == 200


def test_security_headers_and_ui_csp(strict_client: TestClient) -> None:
    health = strict_client.get("/health")
    assert health.headers["x-content-type-options"] == "nosniff"
    assert health.headers["x-frame-options"] == "DENY"
    assert health.headers["content-security-policy"].startswith("default-src 'none'")
    ui = strict_client.get("/ui/")
    assert "script-src 'self'" in ui.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in ui.headers["content-security-policy"]
    assert "content-security-policy" not in strict_client.get("/docs").headers
