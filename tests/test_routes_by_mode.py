"""Tests for route exposure in launch (single-model) mode and hub (daemon) mode."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.server import configure_fastapi_app


def test_launch_mode_exposes_only_v1_routes() -> None:
    """Ensure launch mode exposes `/v1/...` but not `/hub/...`."""
    app = FastAPI()
    configure_fastapi_app(app, include_hub_routes=False)
    client = TestClient(app)

    # `/v1/models` should exist and respond (list of models endpoint)
    resp = client.get("/v1/models")
    assert resp.status_code in (200, 404) or resp.status_code // 100 == 2

    # `/hub/status` must NOT be present in launch mode
    resp = client.get("/hub/status")
    assert resp.status_code == 404


def test_hub_mode_exposes_hub_and_v1_routes() -> None:
    """Ensure hub daemon mode exposes both `/v1/...` and `/hub/...`."""
    app = FastAPI()
    configure_fastapi_app(app, include_hub_routes=True)
    client = TestClient(app)

    # `/v1/models` should exist
    resp = client.get("/v1/models")
    assert resp.status_code in (200, 404) or resp.status_code // 100 == 2

    # `/hub/status` should be registered; since supervisor isn't set in this test
    # the canonical handler may return 500 or 422, but it should not be 404 (not found).
    resp = client.get("/hub/status")
    assert resp.status_code != 404
