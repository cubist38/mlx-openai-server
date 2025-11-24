"""CLI regression tests for hub action subcommands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
from click.testing import CliRunner
import pytest

from app.cli import _render_watch_table, cli
from app.hub.config import MLXHubConfig


@pytest.fixture
def hub_config_file(tmp_path: Path) -> Path:
    """Write a minimal hub.yaml used for CLI tests."""

    config = tmp_path / "hub.yaml"
    log_dir = tmp_path / "logs"
    config.write_text(
        f"""
log_path: {log_dir}
models:
  - name: alpha
    model_path: /models/a
    model_type: lm
""".strip()
    )
    return config


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


def test_hub_reload_cli_reloads_service(
    monkeypatch: pytest.MonkeyPatch, hub_config_file: Path
) -> None:
    """`hub reload` should trigger a service reload via HubServiceClient."""

    stub = _StubServiceClient()

    def _call_stub(
        _config: MLXHubConfig,
        method: str,
        path: str,
        *,
        json: dict[str, object] | None = None,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        if method == "POST" and path == "/hub/reload":
            return stub.reload()
        if method == "GET" and path == "/health":
            return {"status": "ok"}
        raise RuntimeError(f"unexpected call {method} {path}")

    monkeypatch.setattr("app.cli._call_daemon_api", _call_stub)

    runner = CliRunner()
    result = runner.invoke(cli, ["hub", "--config", str(hub_config_file), "reload"])

    assert result.exit_code == 0
    assert stub.reload_calls == 1


def test_hub_stop_cli_requests_shutdown(
    monkeypatch: pytest.MonkeyPatch, hub_config_file: Path
) -> None:
    """`hub stop` should reload config then shut down the service."""

    stub = _StubServiceClient()

    def _call_stub(
        _config: MLXHubConfig,
        method: str,
        path: str,
        *,
        json: dict[str, object] | None = None,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        # emulate availability check and reload/shutdown behavior
        if method == "GET" and path == "/health":
            return {"status": "ok"}
        if method == "POST" and path == "/hub/reload":
            return stub.reload()
        if method == "POST" and path == "/hub/shutdown":
            stub.shutdown()
            return {"message": "shutdown"}
        if method == "POST" and path.startswith("/hub/models/") and path.endswith("/start"):
            name = path.split("/")[-2]
            stub.start_model(name)
            return {"message": "started"}
        if method == "POST" and path.startswith("/hub/models/") and path.endswith("/stop"):
            name = path.split("/")[-2]
            stub.stop_model(name)
            return {"message": "stopped"}
        raise RuntimeError(f"unexpected call {method} {path}")

    monkeypatch.setattr("app.cli._call_daemon_api", _call_stub)
    # Prior tests used an internal build hook; the CLI now uses the daemon API.

    runner = CliRunner()
    result = runner.invoke(cli, ["hub", "--config", str(hub_config_file), "stop"])

    assert result.exit_code == 0
    assert stub.reload_calls == 1, (
        f"reloads={stub.reload_calls} availability_checks={stub.is_available_calls}"
    )
    assert stub.shutdown_called is True


def test_hub_start_model_cli_uses_service_client(
    monkeypatch: pytest.MonkeyPatch, hub_config_file: Path
) -> None:
    """`hub start-model` should instruct the HubServiceClient to start models."""

    stub = _StubServiceClient()

    def _call_stub(
        _config: MLXHubConfig,
        method: str,
        path: str,
        *,
        json: dict[str, object] | None = None,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        if method == "POST" and path.startswith("/hub/models/") and path.endswith("/start"):
            name = path.split("/")[-2]
            stub.start_model(name)
            return {"message": "started"}
        if method == "POST" and path == "/hub/reload":
            return stub.reload()
        if method == "GET" and path == "/health":
            return {"status": "ok"}
        raise RuntimeError(f"unexpected call {method} {path}")

    monkeypatch.setattr("app.cli._call_daemon_api", _call_stub)

    runner = CliRunner()
    result = runner.invoke(cli, ["hub", "--config", str(hub_config_file), "start-model", "alpha"])

    assert result.exit_code == 0
    assert stub.started == ["alpha"]
    assert stub.reload_calls == 1


def test_hub_stop_model_cli_uses_service_client(
    monkeypatch: pytest.MonkeyPatch, hub_config_file: Path
) -> None:
    """`hub stop-model` should request stop_model for the provided names."""

    stub = _StubServiceClient()

    def _call_stub(
        _config: MLXHubConfig,
        method: str,
        path: str,
        *,
        json: dict[str, object] | None = None,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        if method == "POST" and path.startswith("/hub/models/") and path.endswith("/stop"):
            name = path.split("/")[-2]
            stub.stop_model(name)
            return {"message": "stopped"}
        if method == "POST" and path == "/hub/reload":
            return stub.reload()
        if method == "GET" and path == "/health":
            return {"status": "ok"}
        raise RuntimeError(f"unexpected call {method} {path}")

    monkeypatch.setattr("app.cli._call_daemon_api", _call_stub)

    runner = CliRunner()
    result = runner.invoke(cli, ["hub", "--config", str(hub_config_file), "stop-model", "alpha"])

    assert result.exit_code == 0
    assert stub.stopped == ["alpha"]
    assert stub.reload_calls == 1


def test_hub_start_model_cli_requires_model_names(hub_config_file: Path) -> None:
    """The CLI should fail fast if no model names are provided."""

    runner = CliRunner()
    result = runner.invoke(cli, ["hub", "--config", str(hub_config_file), "start-model"])

    assert result.exit_code != 0
    assert "Missing argument" in result.output
    assert "MODEL_NAMES" in result.output


def test_render_watch_table_formats_columns() -> None:
    """The watch table helper should render sorted, columnized rows."""

    models = [
        {
            "name": "beta",
            "state": "failed",
            "pid": 2222,
            "group": "vlm",
            "started_at": 1950.0,
            "exit_code": 1,
            "log_path": "/tmp/logs/beta.log",
        },
        {
            "name": "alpha",
            "state": "running",
            "pid": 1111,
            "group": "lm",
            "started_at": 1900.0,
            "exit_code": None,
            "log_path": "/tmp/logs/alpha.log",
        },
    ]

    table = _render_watch_table(models, now=2000.0)

    assert "NAME" in table.splitlines()[0]
    assert "alpha" in table
    assert "1m40s" in table  # uptime derived from now - started_at
    assert "beta.log" in table
    assert "EXIT" in table


def test_render_watch_table_handles_empty_payload() -> None:
    """The watch table helper should gracefully render empty snapshots."""

    assert _render_watch_table([], now=0) == "  (no managed processes)"


def test_hub_load_model_cli_calls_controller(
    monkeypatch: pytest.MonkeyPatch, hub_config_file: Path
) -> None:
    """`hub load-model` should delegate to the controller helper."""

    captured: list[tuple[tuple[str, ...], str]] = []

    def _fake_run_actions(_config: object, names: tuple[str, ...], action: str) -> None:
        captured.append((names, action))

    monkeypatch.setattr("app.cli._run_memory_actions", _fake_run_actions)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "hub",
            "--config",
            str(hub_config_file),
            "load-model",
            "alpha",
            "beta",
        ],
    )

    assert result.exit_code == 0
    assert captured == [(("alpha", "beta"), "load-model")]


def test_hub_unload_model_cli_surfaces_errors(
    monkeypatch: pytest.MonkeyPatch, hub_config_file: Path
) -> None:
    """`hub unload-model` should propagate helper failures as CLI errors."""

    def _fake_run_actions(_config: object, _names: tuple[str, ...], _action: str) -> None:
        raise click.ClickException("boom: test")

    monkeypatch.setattr("app.cli._run_memory_actions", _fake_run_actions)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["hub", "--config", str(hub_config_file), "unload-model", "alpha"],
    )

    assert result.exit_code != 0
    assert "boom: test" in result.output
