"""CLI regression tests for hub action subcommands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
from click.testing import CliRunner
import pytest

from app.cli import _render_watch_table, cli
from app.hub.config import MLXHubConfig

# `hub_config_file` is provided by `tests/conftest.py`


# `_StubServiceClient` is provided by `tests/conftest.py` as the
# `stub_service_client` fixture.


def test_hub_reload_cli_reloads_service(
    monkeypatch: pytest.MonkeyPatch,
    hub_config_file: Path,
    stub_service_client: object,
) -> None:
    """`hub reload` should trigger a service reload via HubServiceClient."""
    stub = stub_service_client

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
    monkeypatch: pytest.MonkeyPatch,
    hub_config_file: Path,
    stub_service_client: object,
) -> None:
    """`hub stop` should reload config then shut down the service."""
    stub = stub_service_client

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
    monkeypatch: pytest.MonkeyPatch,
    hub_config_file: Path,
    stub_service_client: object,
) -> None:
    """`hub start-model` should instruct the HubServiceClient to start models."""
    stub = stub_service_client

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
    monkeypatch: pytest.MonkeyPatch,
    hub_config_file: Path,
    stub_service_client: object,
) -> None:
    """`hub stop-model` should request stop_model for the provided names."""
    stub = stub_service_client

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
            "loaded": "no",
            "auto_unload": "-",
            "type": "vlm",
            "group": "vlm",
            "default": "-",
            "model": "-",
        },
        {
            "name": "alpha",
            "state": "running",
            "loaded": "yes",
            "auto_unload": "5min",
            "type": "lm",
            "group": "lm",
            "default": "✓",
            "model": "some/model",
        },
    ]

    table = _render_watch_table(models)

    assert "NAME" in table.splitlines()[0]
    assert "alpha" in table
    assert "beta" in table
    assert "LOADED" in table
    assert "AUTO-UNLOAD" in table
    assert "TYPE" in table
    assert "GROUP" in table
    assert "DEFAULT" in table
    assert "MODEL" in table


def test_render_watch_table_handles_empty_payload() -> None:
    """The watch table helper should gracefully render empty snapshots."""
    assert _render_watch_table([]) == "  (no managed models)"


def test_hub_load_model_cli_calls_controller(
    monkeypatch: pytest.MonkeyPatch,
    hub_config_file: Path,
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
    assert captured == [(("alpha", "beta"), "load")]


def test_hub_unload_model_cli_surfaces_errors(
    monkeypatch: pytest.MonkeyPatch,
    hub_config_file: Path,
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


def test_hub_load_model_cli_extracts_user_friendly_error_messages(
    monkeypatch: pytest.MonkeyPatch,
    hub_config_file: Path,
) -> None:
    """`hub load-model` should extract user-friendly messages from structured error responses."""

    def _fake_run_actions(_config: object, _names: tuple[str, ...], _action: str) -> None:
        # Simulate the error that would come from a 409 constraint violation
        raise click.ClickException(
            "Daemon responded 409: Loading model 'llama32_3b' would violate group max_loaded constraint"
        )

    monkeypatch.setattr("app.cli._run_memory_actions", _fake_run_actions)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["hub", "--config", str(hub_config_file), "load-model", "llama32_3b"],
    )

    assert result.exit_code != 0
    # The error message should be the clean, user-friendly version
    assert "Loading model 'llama32_3b' would violate group max_loaded constraint" in result.output
    # Should not contain the verbose JSON structure
    assert "{'error':" not in result.output


def test_hub_config_option_loads_specified_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--config` option should load the specified config file, not the default."""
    # Create two different config files
    config1 = tmp_path / "config1.yaml"
    config1.write_text(
        """
models:
  - name: model-from-config1
    model_path: /path/from/config1
    model_type: lm
""".strip()
    )

    config2 = tmp_path / "config2.yaml"
    config2.write_text(
        """
models:
  - name: model-from-config2
    model_path: /path/from/config2
    model_type: lm
""".strip()
    )

    # Mock _load_hub_config_or_fail to capture the path it was called with
    loaded_paths: list[str] = []

    def mock_load_config(config_path: str | None) -> MLXHubConfig:
        loaded_paths.append(str(config_path) if config_path else "None")
        # Return a minimal config for the test
        return MLXHubConfig(
            models=[],
            host="127.0.0.1",
            port=8000,
            enable_status_page=False,
            log_path=tmp_path / "logs",
        )

    monkeypatch.setattr("app.cli._load_hub_config_or_fail", mock_load_config)

    runner = CliRunner()
    # Test with config1
    result = runner.invoke(cli, ["hub", "--config", str(config1), "status"], catch_exceptions=False)

    assert result.exit_code == 0
    assert len(loaded_paths) == 1
    assert loaded_paths[0] == str(config1)

    # Clear for next test
    loaded_paths.clear()

    # Test with config2
    result = runner.invoke(cli, ["hub", "--config", str(config2), "status"], catch_exceptions=False)

    assert result.exit_code == 0
    assert len(loaded_paths) == 1
    assert loaded_paths[0] == str(config2)

    # Clear for next test
    loaded_paths.clear()

    # Test without --config option (should use None, which means default)
    result = runner.invoke(cli, ["hub", "status"], catch_exceptions=False)

    assert result.exit_code == 0
    assert len(loaded_paths) == 1
    assert loaded_paths[0] == "None"  # None is passed when no --config option
