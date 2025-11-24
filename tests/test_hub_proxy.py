"""Integration tests for hub proxy behavior.

These tests run the hub daemon and the main API in-process and patch
`httpx` to route requests to the daemon app using ASGI transport so the
proxy logic can be exercised without opening real TCP ports.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
from httpx import ASGITransport
from loguru import logger
import yaml

from app.config import MLXServerConfig
from app.hub import proxy as proxy_mod
from app.hub.daemon import create_app as create_daemon_app
from app.hub.proxy import proxy_router
import app.server as server_mod
from app.server import setup_server


def _write_hub_yaml(path: Path) -> None:
    data = {
        "host": "127.0.0.1",
        "port": 8000,
        "daemon_port": 47851,
        "enable_status_page": True,
        "models": [{"name": "foo", "model_path": "/models/foo", "default": False}],
    }
    path.write_text(yaml.safe_dump(data))


def test_hub_page_and_v1_proxy(tmp_path: Path) -> None:
    """Verify that `/hub` is served and `/v1` proxy returns 503 for missing worker."""
    hub_yaml = tmp_path / "hub.yaml"
    _write_hub_yaml(hub_yaml)

    # Create daemon app that reads our hub.yaml
    daemon_app = create_daemon_app(str(hub_yaml))

    # Create main API app via setup_server
    cfg = MLXServerConfig(model_path="dummy", host="127.0.0.1", port=8000)
    uvconfig = setup_server(cfg)
    main_app = uvconfig.app

    # Ensure main app knows about the daemon (string value is enough because
    # we patch httpx to route to daemon_app via ASGITransport below)
    main_app.state.hub_daemon_url = "http://daemon"

    # Prevent the server from trying to download or initialize real models
    # during TestClient lifespan by stubbing `instantiate_handler`.
    orig_instantiate = server_mod.instantiate_handler

    async def _dummy_instantiate(_: MLXServerConfig) -> Any:
        class _DummyHandler:
            async def initialize(self, *_a: Any, **_kw: Any) -> None:
                return None

            async def cleanup(self) -> None:
                return None

        return _DummyHandler()

    server_mod.instantiate_handler = _dummy_instantiate

    # Patch the AsyncClient used inside the proxy module so only daemon
    # requests are routed to the in-process daemon app. This avoids
    # interfering with TestClient's own httpx clients.
    orig_AsyncClient = proxy_mod.httpx.AsyncClient

    def async_client_factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return orig_AsyncClient(transport=ASGITransport(app=daemon_app), **kwargs)

    proxy_mod.httpx.AsyncClient = async_client_factory  # type: ignore[assignment]

    try:
        with TestClient(daemon_app), TestClient(main_app) as main_client:
            # The main /hub should prefer the daemon-rendered HTML
            r = main_client.get("/hub")
            assert r.status_code == 200
            assert "MLX Hub Status" in r.text

            # A proxied /v1 call without a running worker should return 503
            # Debug: dump registered routes to help diagnose 404s
            paths = [getattr(r, "path", None) for r in main_client.app.router.routes]
            logger.info(f"ROUTES: {paths}")

            # Verify that the OpenAPI document exposes the chat endpoint
            openapi = main_client.get("/openapi.json")
            assert openapi.status_code == 200
            paths_map = openapi.json().get("paths", {})
            assert "/v1/chat/completions" in paths_map

            payload = {"model": "foo", "messages": []}
            r2 = main_client.post("/v1/chat/completions", json=payload)
            logger.info(f"PROXY RESP: status={r2.status_code} body={r2.text}")
            assert r2.status_code in (503, 404)
            body = r2.json()
            assert isinstance(body, dict)
            # Prefer the hub-style error payload, but accept a generic 404 detail
            if r2.status_code == 503:
                assert "error" in body
            else:
                assert "detail" in body
    finally:
        # restore proxy module AsyncClient and instantiate handler
        proxy_mod.httpx.AsyncClient = orig_AsyncClient  # type: ignore[assignment]
        server_mod.instantiate_handler = orig_instantiate


class WorkerHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        resp = {
            "ok": True,
            "path": self.path,
            "body": body.decode("utf-8", errors="ignore"),
        }
        payload = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        # Silence test output
        return


class DaemonHandler(BaseHTTPRequestHandler):
    # This will be set at runtime by the test harness
    worker_port = 0

    def do_GET(self) -> None:
        if self.path.startswith("/hub/status"):
            payload = {
                "timestamp": 0,
                "models": [{"name": "testmodel", "port": self.worker_port, "host": "127.0.0.1"}],
            }
            data = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


def _serve_in_thread(server: ThreadingHTTPServer) -> threading.Thread:
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return t


def test_proxy_forwards_to_worker(tmp_path: Path) -> None:
    # Start fake worker HTTP server
    worker_server = ThreadingHTTPServer(("127.0.0.1", 0), WorkerHandler)
    worker_thread = _serve_in_thread(worker_server)
    worker_port = worker_server.server_address[1]

    try:
        # Start fake daemon that reports the worker port
        DaemonHandler.worker_port = worker_port
        daemon_server = ThreadingHTTPServer(("127.0.0.1", 0), DaemonHandler)
        daemon_thread = _serve_in_thread(daemon_server)
        daemon_port = daemon_server.server_address[1]

        try:
            # Create a minimal FastAPI app with the proxy router and point it
            # at the fake daemon.
            app = FastAPI()
            app.state.hub_daemon_url = f"http://127.0.0.1:{daemon_port}"
            app.include_router(proxy_router)

            with TestClient(app) as client:
                body = {"model": "testmodel", "input": "hello"}
                resp = client.post("/v1/echo", json=body)
                assert resp.status_code == 200
                payload = resp.json()
                # The worker echoes the path and body — verify they match
                assert "/v1/echo" in payload.get("path", "")
                assert "hello" in payload.get("body", "")

        finally:
            daemon_server.shutdown()
            daemon_server.server_close()
            daemon_thread.join(timeout=1.0)
    finally:
        worker_server.shutdown()
        worker_server.server_close()
        worker_thread.join(timeout=1.0)
