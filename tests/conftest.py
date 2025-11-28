"""Shared test fixtures and helpers for the test suite.

This module provides small factory fixtures used by multiple test modules
to avoid duplication of common test helpers.
"""

from __future__ import annotations

from collections.abc import Callable
from http import HTTPStatus
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException
import pytest

from app.config import MLXServerConfig
from app.hub.config import MLXHubConfig
from app.server import LazyHandlerManager


@pytest.fixture
def make_hub_config(tmp_path: Path) -> Callable[..., MLXHubConfig]:
    """Flexible factory for creating `MLXHubConfig` instances.

    Returns a callable that accepts keyword overrides for fields like
    `models`, `host`, and `port` so tests can quickly construct configs.
    """

    def _factory(**overrides: object) -> MLXHubConfig:
        cfg = MLXHubConfig(
            host=overrides.get("host", "127.0.0.1"),
            port=overrides.get("port", 8123),
            model_starting_port=overrides.get("model_starting_port", 8000),
            log_level=overrides.get("log_level", "INFO"),
            log_path=overrides.get("log_path", tmp_path / "logs"),
            enable_status_page=overrides.get("enable_status_page", True),
            source_path=overrides.get("source_path", tmp_path / "hub.yaml"),
        )
        models = overrides.get(
            "models",
            [MLXServerConfig(model_path="/models/foo", name="foo", model_type="lm")],
        )
        cfg.models = models
        return cfg

    return _factory


@pytest.fixture
def hub_config_with_defaults(
    make_hub_config: Callable[..., MLXHubConfig], make_model_mock: Callable[..., MagicMock]
) -> MLXHubConfig:
    """Create a hub config with default and regular models for lifecycle tests.

    Uses the shared `make_model_mock` factory to construct model-like objects so
    tests remain consistent and avoid duplicated mock construction logic.
    """
    default_model = make_model_mock(
        "default_model",
        "/models/default",
        "lm",
        is_default=True,
        group=None,
        auto_unload_minutes=120,
    )

    regular_model = make_model_mock(
        "regular_model",
        "/models/regular",
        "lm",
        is_default=False,
        group=None,
        auto_unload_minutes=120,
    )

    return make_hub_config(models=[default_model, regular_model])


@pytest.fixture
def mock_handler_manager() -> MagicMock:
    """Mocked `LazyHandlerManager`-like object for tests."""
    manager = MagicMock(spec=LazyHandlerManager)
    manager.is_vram_loaded.return_value = False
    manager.ensure_loaded = AsyncMock()
    manager.unload = AsyncMock(return_value=True)
    return manager


@pytest.fixture
def hub_config_file(tmp_path: Path) -> Path:
    """Write a minimal `hub.yaml` used by CLI tests and return its path."""
    config = tmp_path / "hub.yaml"
    log_dir = tmp_path / "logs"
    config.write_text(
        f"""
log_path: {log_dir}
models:
  - name: alpha
    model_path: /models/a
    model_type: lm
""".strip(),
    )
    return config


@pytest.fixture
def make_config(tmp_path: Path) -> Callable[[], MLXHubConfig]:
    """Factory fixture that returns a function creating a hub config.

    The returned function takes no arguments and closes over the test's
    temporary path so callers do not need to accept `tmp_path` themselves.
    """

    def _make_config() -> MLXHubConfig:
        return MLXHubConfig(
            log_path=tmp_path / "logs",
            models=[MLXServerConfig(model_path="/models/foo", name="foo", model_type="lm")],
        )

    return _make_config


@pytest.fixture
def live_snapshot() -> Callable[[int], dict]:
    """Factory fixture returning a function that generates a live snapshot payload.

    Returns a callable `fn(pid: int=4321) -> dict` to mirror the previous
    test-local helper used across tests.
    """

    def _live_snapshot(pid: int = 4321) -> dict:
        return {
            "models": [
                {
                    "name": "foo",
                    "state": "running",
                    "pid": pid,
                    "log_path": "/tmp/foo.log",
                    "started_at": 1,
                },
            ],
        }

    return _live_snapshot


@pytest.fixture
def write_hub_yaml(tmp_path: Path) -> Callable[[str, str], Path]:
    """Helper to write hub YAML content to a file in `tmp_path`.

    Usage: `path = write_hub_yaml(yaml_text, filename="hub.yaml")`
    """

    def _write(content: str, filename: str = "hub.yaml") -> Path:
        p = tmp_path / filename
        p.write_text(content.strip())
        return p

    return _write


@pytest.fixture
def make_model_mock() -> Callable[..., MagicMock]:
    """Factory fixture that creates configurable MagicMock model objects.

    Returns a callable with signature
    `(name: str = "mock_model", model_path: str = "/models/mock", model_type: str = "lm", *,
    is_default: bool = False, group: str | None = None, auto_unload_minutes: int | None = None) -> MagicMock`.
    """

    def _factory(
        name: str = "mock_model",
        model_path: str = "/models/mock",
        model_type: str = "lm",
        *,
        is_default: bool = False,
        group: str | None = None,
        auto_unload_minutes: int | None = None,
    ) -> MagicMock:
        m = MagicMock()
        m.name = name
        m.model_path = model_path
        m.model_type = model_type
        m.is_default_model = is_default
        m.group = group
        m.auto_unload_minutes = auto_unload_minutes
        return m

    return _factory


class _StubServiceState:
    """Stub state for testing hub service interactions."""

    def __init__(self) -> None:
        self.available = True
        self.reload_calls = 0
        self.reload_result: dict[str, list[str]] = {"started": [], "stopped": [], "unchanged": []}
        self.status_payload: dict[str, Any] = {"models": [], "timestamp": 1}
        self.start_calls: list[str] = []
        self.stop_calls: list[str] = []
        self.shutdown_called = False
        self.controller_stop_calls = 0


class _StubController:
    """Stub controller for testing hub controller interactions."""

    def __init__(self) -> None:
        self.loaded: list[str] = []
        self.unloaded: list[str] = []
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.reload_count = 0

    async def load_model(self, name: str) -> None:
        self.loaded.append(name)
        if name == "denied":
            raise HTTPException(status_code=HTTPStatus.TOO_MANY_REQUESTS, detail="group busy")

    async def unload_model(self, name: str) -> None:
        self.unloaded.append(name)
        if name == "missing":
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="not loaded")

    async def start_model(self, name: str) -> None:
        self.started.append(name)
        if name == "saturated":
            raise HTTPException(status_code=HTTPStatus.TOO_MANY_REQUESTS, detail="group full")

    async def stop_model(self, name: str) -> None:
        self.stopped.append(name)

    async def reload_config(self) -> dict[str, Any]:
        self.reload_count += 1
        return {"started": [], "stopped": ["old_model"], "unchanged": ["alpha", "beta"]}

    def get_status(self) -> dict[str, Any]:
        return {
            "timestamp": 1234567890.0,
            "models": [
                {
                    "name": "alpha",
                    "state": "running",
                    "pid": 12345,
                    "port": 8124,
                    "started_at": 1234567800.0,
                    "exit_code": None,
                    "memory_loaded": True,
                    "group": None,
                    "is_default": True,
                    "model_path": "test/path",
                    "auto_unload_minutes": 120,
                },
                {
                    "name": "beta",
                    "state": "stopped",
                    "pid": None,
                    "port": 8125,
                    "started_at": None,
                    "exit_code": None,
                    "memory_loaded": False,
                    "group": None,
                    "is_default": False,
                    "model_path": "test/path2",
                    "auto_unload_minutes": None,
                },
            ],
        }


@pytest.fixture
def stub_service_state() -> _StubServiceState:
    """Return a fresh stub service state instance for tests."""
    return _StubServiceState()


@pytest.fixture
def stub_controller() -> _StubController:
    """Return a fresh stub controller instance for tests."""
    return _StubController()


class _StubServiceClient:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.reload_calls = 0
        self.shutdown_called = False
        self.is_available_calls = 0

    def is_available(self) -> bool:
        self.is_available_calls += 1
        return True

    def start_model(self, name: str) -> None:
        self.started.append(name)

    def stop_model(self, name: str) -> None:
        self.stopped.append(name)

    def reload(self) -> dict[str, list[str]]:
        self.reload_calls += 1
        return {"started": [], "stopped": [], "unchanged": []}

    def shutdown(self) -> None:
        self.shutdown_called = True


@pytest.fixture
def stub_service_client() -> _StubServiceClient:
    """Provide a fresh stub service client for CLI tests."""
    return _StubServiceClient()
