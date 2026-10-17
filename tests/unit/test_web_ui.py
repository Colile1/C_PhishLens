# test_web_ui.py
# Purpose: the browser front end is served, complete, and adds no endpoint.
# Author: Colile Sibanda

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from phishlens.config.loader import load_config
from phishlens.serving.app import WEB_DIR, create_app

PAGES = ("index.html", "model.html", "guide.html")
ASSET_PATTERN = re.compile(r'(?:src|href)="(static/[^"]+)"')


@pytest.fixture(scope="module")
def client():
    """Build a TestClient over the real app, without running the lifespan.

    The pages are static, so the model does not need to load for them.

    Inputs: none.
    Output: a TestClient wrapping the app.
    """
    return TestClient(create_app(load_config()))


def test_ui_root_serves_the_analyse_page(client: TestClient) -> None:
    response = client.get("/ui/")
    assert response.status_code == 200
    assert "Check an email" in response.text


def test_site_root_redirects_to_the_ui(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/ui/"


@pytest.mark.parametrize("page", PAGES)
def test_every_asset_a_page_references_is_served(client: TestClient, page: str) -> None:
    html = client.get(f"/ui/{page}")
    assert html.status_code == 200
    assets = ASSET_PATTERN.findall(html.text)
    assert assets, f"{page} references no static assets"
    for asset in assets:
        assert client.get(f"/ui/{asset}").status_code == 200, f"{page}: {asset} missing"


def test_ui_adds_no_api_endpoint(client: TestClient) -> None:
    paths = set(client.get("/openapi.json").json()["paths"])
    assert paths == {"/health", "/metadata", "/predict", "/explain"}


def test_ui_source_is_ascii_only() -> None:
    for path in WEB_DIR.rglob("*"):
        if path.is_file():
            data = path.read_bytes()
            assert data.isascii(), f"{path.name} contains non-ASCII characters"
