"""Verify persistent registration deletion without touching model files."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from click.testing import CliRunner
from fastapi.testclient import TestClient
import pytest

from app.config import MultiModelServerConfig, load_config_from_yaml
from app.core.model_registry import ModelRegistry
from app.model_management import (
    ModelManagementError,
    ModelManager,
    find_local_server,
    management_request,
    serve_model_management,
)


@pytest.fixture
def control_dir(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Keep Unix socket names below macOS's 104-byte path limit."""
    with tempfile.TemporaryDirectory(prefix="mlx-control-", dir="/tmp") as directory:
        monkeypatch.setenv("MLX_SERVER_RUNTIME_DIR", directory)
        yield Path(directory)


def _config(tmp_path: Path, port: int = 8123) -> MultiModelServerConfig:
    """Create a commented config and a checkpoint sentinel."""
    weights = tmp_path / "checkpoint"
    weights.mkdir(exist_ok=True)
    (weights / "model.safetensors").write_bytes(b"keep these model bytes")
    path = tmp_path / f"config-{port}.yaml"
    path.write_text(
        "# Keep this server comment\n"
        f"server:\n  host: 127.0.0.1\n  port: {port}\n"
        "models:\n"
        f"  - model_path: {weights}\n"
        "    served_model_name: alpha\n"
        "    aliases: [first]\n"
        "    on_demand: true\n"
        "  - model_path: org/embedding\n"
        "    served_model_name: beta\n"
        "    model_type: embeddings\n"
        "    on_demand: true\n"
        "    on_demand_idle_timeout: 2m # Keep this model comment\n"
    )
    path.chmod(0o600)
    return load_config_from_yaml(str(path))


async def _registry(config: MultiModelServerConfig) -> ModelRegistry:
    """Register test models without loading any checkpoints."""
    registry = ModelRegistry()
    for model in config.models:
        await registry.register_on_demand_model(
            model_id=model.served_model_name,
            model_cfg_dict=asdict(model),
            model_type=model.model_type,
            model_path=model.model_path,
            context_length=None,
            queue_config={},
            aliases=model.alias_names(),
        )
    return registry


@pytest.mark.asyncio
async def test_delete_alias_persists_keeps_comments_and_files(tmp_path: Path) -> None:
    """Removing an alias removes its canonical entry and preserves checkpoints."""
    config = _config(tmp_path)
    path = Path(config.source_path)
    before = path.read_bytes()
    registry = await _registry(config)
    manager = ModelManager(config, registry)

    result = await manager.dispatch({"operation": "delete", "model": "first"})

    assert result["model"] == "alpha"
    assert result["files_deleted"] is False
    assert registry.list_model_ids() == ["beta"]
    assert not registry.has_model("first")
    assert [model.served_model_name for model in load_config_from_yaml(str(path)).models] == [
        "beta"
    ]
    assert Path(result["backup"]).read_bytes() == before
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(Path(result["backup"]).stat().st_mode) == 0o600
    assert "# Keep this server comment" in path.read_text()
    assert "# Keep this model comment" in path.read_text()
    assert (tmp_path / "checkpoint/model.safetensors").read_bytes() == b"keep these model bytes"


@pytest.mark.asyncio
async def test_last_model_can_be_removed_and_config_reloaded(tmp_path: Path) -> None:
    """Deleting every model leaves a valid configuration for the next restart."""
    config = _config(tmp_path)
    registry = await _registry(config)
    manager = ModelManager(config, registry)
    await manager.dispatch({"operation": "delete", "model": "alpha"})
    await manager.dispatch({"operation": "delete", "model": "beta"})

    assert registry.list_model_ids() == []
    assert registry.get_model_status() == []
    assert load_config_from_yaml(config.source_path).models == []
    assert (tmp_path / "checkpoint/model.safetensors").exists()


@pytest.mark.asyncio
async def test_busy_model_is_not_changed(tmp_path: Path) -> None:
    """An active request prevents cleanup and persistent removal."""
    config = _config(tmp_path)
    path = Path(config.source_path)
    before = path.read_bytes()
    registry = await _registry(config)
    cleanup = AsyncMock()
    registry._handlers["alpha"] = SimpleNamespace(cleanup=cleanup)
    registry._on_demand_loaded.add("alpha")
    registry._on_demand_ref_count["alpha"] = 1

    with pytest.raises(ValueError, match="active request"):
        await ModelManager(config, registry).dispatch({"operation": "delete", "model": "alpha"})

    cleanup.assert_not_awaited()
    assert registry.has_model("alpha")
    assert path.read_bytes() == before
    assert not path.with_suffix(".yaml.bak").exists()


@pytest.mark.asyncio
async def test_idle_model_is_cleaned_up_before_removal(tmp_path: Path) -> None:
    """An idle loaded model can be deleted without leaving a worker."""
    config = _config(tmp_path)
    registry = await _registry(config)
    cleanup = AsyncMock()
    registry._handlers["alpha"] = SimpleNamespace(cleanup=cleanup)
    registry._on_demand_loaded.add("alpha")
    registry._on_demand_ref_count["alpha"] = 0

    await ModelManager(config, registry).dispatch({"operation": "delete", "model": "alpha"})

    cleanup.assert_awaited_once()
    assert registry.get_loaded_model_count() == 0
    assert not registry.has_model("alpha")


@pytest.mark.asyncio
async def test_failed_cleanup_does_not_edit_config(tmp_path: Path) -> None:
    """A worker which failed to stop stays registered and configured."""
    config = _config(tmp_path)
    path = Path(config.source_path)
    before = path.read_bytes()
    registry = await _registry(config)
    registry._handlers["alpha"] = SimpleNamespace(
        cleanup=AsyncMock(side_effect=RuntimeError("worker did not stop"))
    )
    registry._on_demand_loaded.add("alpha")

    with pytest.raises(RuntimeError, match="did not stop"):
        await ModelManager(config, registry).dispatch({"operation": "delete", "model": "alpha"})
    assert registry.has_model("alpha")
    assert path.read_bytes() == before


@pytest.mark.asyncio
async def test_changed_target_is_not_removed(tmp_path: Path) -> None:
    """A name reused for changed settings must not delete an unreviewed entry."""
    config = _config(tmp_path)
    registry = await _registry(config)
    path = Path(config.source_path)
    edited = path.read_text().replace("aliases: [first]", "aliases: [replacement]")
    path.write_text(edited)

    with pytest.raises(ModelManagementError, match="changed on disk"):
        await ModelManager(config, registry).dispatch({"operation": "delete", "model": "alpha"})
    assert registry.has_model("alpha")
    assert path.read_text() == edited


@pytest.mark.asyncio
async def test_write_failure_keeps_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed atomic replacement cannot leave only an in-memory deletion."""
    from app import model_management

    config = _config(tmp_path)
    registry = await _registry(config)
    path = Path(config.source_path)
    before = path.read_bytes()
    replace = model_management.os.replace

    def fail_config_replace(source: str | Path, destination: str | Path) -> None:
        """Fail only the final config commit, after its backup was written."""
        if Path(destination) == path:
            raise PermissionError("read-only configuration")
        return replace(source, destination)

    monkeypatch.setattr(model_management.os, "replace", fail_config_replace)
    with pytest.raises(PermissionError, match="read-only"):
        await ModelManager(config, registry).dispatch({"operation": "delete", "model": "alpha"})
    assert registry.has_model("alpha")
    assert path.read_bytes() == before


@pytest.mark.asyncio
async def test_waiting_load_does_not_resurrect_deleted_model(tmp_path: Path) -> None:
    """An acquisition queued during cleanup rechecks registration under the lock."""
    config = _config(tmp_path)
    registry = await _registry(config)
    cleaning = asyncio.Event()
    finish_cleanup = asyncio.Event()

    async def cleanup() -> None:
        cleaning.set()
        await finish_cleanup.wait()

    registry._handlers["alpha"] = SimpleNamespace(cleanup=cleanup)
    registry._on_demand_loaded.add("alpha")
    deletion = asyncio.create_task(
        ModelManager(config, registry).dispatch({"operation": "delete", "model": "alpha"})
    )
    await cleaning.wait()
    acquisition = asyncio.create_task(registry.ensure_on_demand_loaded("first"))
    await asyncio.sleep(0)
    finish_cleanup.set()
    await deletion
    with pytest.raises(KeyError, match="no longer registered"):
        await acquisition
    assert not registry.has_model("alpha")


@pytest.mark.asyncio
async def test_private_socket_discovers_port_and_deletes(tmp_path: Path, control_dir: Path) -> None:
    """Exercise the real local protocol, its permissions, and its cleanup."""
    config = _config(tmp_path)
    registry = await _registry(config)
    path = control_dir / "server-8123.sock"
    async with serve_model_management(config, registry):
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        server = await asyncio.to_thread(find_local_server, None)
        assert server["port"] == 8123
        result = await asyncio.to_thread(
            management_request, 8123, {"operation": "delete", "model": "alpha"}
        )
        assert result["files_deleted"] is False
        assert registry.list_model_ids() == ["beta"]
    assert not path.exists()


@pytest.mark.asyncio
async def test_discovery_refuses_multiple_servers(tmp_path: Path, control_dir: Path) -> None:
    """Omitting --port cannot select an arbitrary server."""
    first = _config(tmp_path, 8123)
    second = _config(tmp_path, 8124)
    async with (
        serve_model_management(first, await _registry(first)),
        serve_model_management(second, await _registry(second)),
    ):
        with pytest.raises(ModelManagementError, match="specify --port"):
            await asyncio.to_thread(find_local_server, None)
        selected = await asyncio.to_thread(find_local_server, 8124)
        assert selected["port"] == 8124


def test_config_still_requires_models_key(tmp_path: Path) -> None:
    """An omitted key remains a configuration error, unlike explicit models: []."""
    path = tmp_path / "missing.yaml"
    path.write_text("server:\n  port: 8123\n")
    with pytest.raises(ValueError, match="models"):
        load_config_from_yaml(str(path))


@pytest.mark.parametrize(
    ("arguments", "answer", "deletes"),
    [
        ([], "n\n", False),
        ([], "y\n", True),
        (["--yes"], None, True),
        (["--dry-run"], None, False),
    ],
)
def test_cli_confirmation_and_preview(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    answer: str | None,
    deletes: bool,
) -> None:
    """The CLI previews its target, preserves files, and honors cancellation."""
    from app import cli

    calls = []
    monkeypatch.setattr(
        cli,
        "find_local_server",
        lambda port, timeout: {
            "port": 8123,
            "config": "/models/config.yaml",
            "models": [{"id": "alpha", "aliases": ["first"]}],
        },
    )

    def request(port: int, payload: dict[str, Any], timeout: float) -> dict[str, str]:
        """Record the selected canonical target."""
        calls.append((port, payload))
        return {"model": "alpha", "backup": "/models/config.yaml.bak"}

    monkeypatch.setattr(cli, "management_request", request)
    result = CliRunner().invoke(cli.cli, ["delete", "model", "first", *arguments], input=answer)
    assert bool(calls) is deletes
    assert result.exit_code == (1 if answer == "n\n" else 0), result.output
    assert "Checkpoint and cache files will be kept" in result.output
    if deletes:
        assert calls == [(8123, {"operation": "delete", "model": "alpha"})]


def test_cli_port_and_unknown_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit ports reach discovery and unknown names cannot trigger deletion."""
    from app import cli

    ports = []

    def find(port: int | None, timeout: float) -> dict[str, Any]:
        """Return an empty registry for the selected port."""
        ports.append(port)
        return {"port": port, "config": "/config.yaml", "models": []}

    monkeypatch.setattr(cli, "find_local_server", find)
    result = CliRunner().invoke(cli.cli, ["delete", "model", "missing", "--port", "8124", "--yes"])
    assert ports == [8124]
    assert result.exit_code == 1
    assert "not registered" in result.output


def test_cli_updates_live_status_and_survives_restart(tmp_path: Path, control_dir: Path) -> None:
    """Use the actual CLI/socket/lifespan path with disposable model entries."""
    from app.cli import cli
    from app.server import setup_server

    config = _config(tmp_path)
    config.no_log_file = True
    with TestClient(setup_server(config).app) as client:
        result = CliRunner().invoke(cli, ["delete", "model", "alpha", "--yes"])
        assert result.exit_code == 0, result.output
        status = client.get("/v1/models/status").json()
        assert status["configured"] == 1
        assert [model["id"] for model in status["data"]] == ["beta"]

    restarted = load_config_from_yaml(config.source_path)
    restarted.no_log_file = True
    with TestClient(setup_server(restarted).app) as client:
        assert client.get("/v1/models/status").json()["configured"] == 1
        result = CliRunner().invoke(cli, ["delete", "model", "beta", "--yes"])
        assert result.exit_code == 0, result.output
        assert client.get("/v1/models/status").json()["configured"] == 0

    empty = load_config_from_yaml(config.source_path)
    empty.no_log_file = True
    with TestClient(setup_server(empty).app) as client:
        assert client.get("/v1/models/status").json()["data"] == []
    assert (tmp_path / "checkpoint/model.safetensors").read_bytes() == b"keep these model bytes"


@pytest.mark.asyncio
async def test_existing_control_file_is_never_overwritten(
    tmp_path: Path, control_dir: Path
) -> None:
    """An unexpected file at the socket path must survive a failed startup."""
    config = _config(tmp_path)
    path = control_dir / "server-8123.sock"
    path.write_text("do not replace")
    with pytest.raises(ModelManagementError, match="not a socket"):
        async with serve_model_management(config, await _registry(config)):
            pytest.fail("The listener must not replace a regular file")
    assert path.read_text() == "do not replace"


@pytest.mark.asyncio
async def test_startup_loaded_model_is_refused_without_interrupting_requests(
    tmp_path: Path,
) -> None:
    """Static workers lack request leases, so deletion must not stop them."""
    config = _config(tmp_path)
    registry = ModelRegistry()
    cleanup = AsyncMock()
    await registry.register_model("alpha", SimpleNamespace(cleanup=cleanup), "lm")
    before = Path(config.source_path).read_bytes()
    with pytest.raises(ValueError, match="startup-loaded"):
        await ModelManager(config, registry).dispatch({"operation": "delete", "model": "alpha"})
    cleanup.assert_not_awaited()
    assert Path(config.source_path).read_bytes() == before
