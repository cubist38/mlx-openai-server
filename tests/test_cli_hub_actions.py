"""CLI regression tests for hub action subcommands."""

from __future__ import annotations

from pathlib import Path

import click
from click.testing import CliRunner
import pytest

from app.cli import _render_watch_table, cli


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
    monkeypatch.setattr("app.cli._require_service_client", lambda _cfg: stub)

    runner = CliRunner()
    result = runner.invoke(cli, ["hub", "--config", str(hub_config_file), "reload"])

    assert result.exit_code == 0
    assert stub.reload_calls == 1


def test_hub_stop_cli_requests_shutdown(
    monkeypatch: pytest.MonkeyPatch, hub_config_file: Path
) -> None:
    """`hub stop` should reload config then shut down the service."""

    stub = _StubServiceClient()
    monkeypatch.setattr("app.cli._require_service_client", lambda _cfg: stub)
    build_calls = {"count": 0}

    def _fake_build(_cfg: object) -> _StubServiceClient:
        build_calls["count"] += 1
        return stub

    monkeypatch.setattr("app.cli._build_service_client", _fake_build)

    runner = CliRunner()
    result = runner.invoke(cli, ["hub", "--config", str(hub_config_file), "stop"])

    assert result.exit_code == 0
    assert build_calls["count"] == 1
    assert stub.reload_calls == 1, (
        f"reloads={stub.reload_calls} availability_checks={stub.is_available_calls}"
    )
    assert stub.shutdown_called is True


def test_hub_start_model_cli_uses_service_client(
    monkeypatch: pytest.MonkeyPatch, hub_config_file: Path
) -> None:
    """`hub start-model` should instruct the HubServiceClient to start models."""

    stub = _StubServiceClient()
    monkeypatch.setattr("app.cli._require_service_client", lambda _cfg: stub)

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
    monkeypatch.setattr("app.cli._require_service_client", lambda _cfg: stub)

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


def test_hub_memory_load_cli_calls_controller(
    monkeypatch: pytest.MonkeyPatch, hub_config_file: Path
) -> None:
    """`hub load-model` should delegate to the controller helper."""

    captured: list[tuple[tuple[str, ...], str, str]] = []

    def _fake_run_actions(
        _config: object, names: tuple[str, ...], action: str, *, reason: str
    ) -> None:
        captured.append((names, action, reason))

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
            "--reason",
            "dashboard",
        ],
    )

    assert result.exit_code == 0
    assert captured == [(("alpha", "beta"), "load-model", "dashboard")]


def test_hub_memory_unload_cli_surfaces_errors(
    monkeypatch: pytest.MonkeyPatch, hub_config_file: Path
) -> None:
    """`hub unload-model` should propagate helper failures as CLI errors."""

    def _fake_run_actions(
        _config: object, _names: tuple[str, ...], _action: str, *, reason: str
    ) -> None:
        raise click.ClickException(f"boom: {reason}")

    monkeypatch.setattr("app.cli._run_memory_actions", _fake_run_actions)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["hub", "--config", str(hub_config_file), "unload-model", "alpha", "--reason", "test"],
    )

    assert result.exit_code != 0
    assert "boom: test" in result.output
