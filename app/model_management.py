"""Persist model removal through a private, per-user local control socket.

The public inference API does not gain configuration-write access. A local CLI
selects a running server by port, then contacts its owner-only Unix socket.
Model checkpoints and Hugging Face cache files are never modified.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from io import StringIO
import json
import os
from pathlib import Path
import socket
import stat
import tempfile
from typing import Any

from loguru import logger
from ruamel.yaml import YAML

from .config import MultiModelServerConfig, load_config_from_yaml
from .core.model_registry import ModelRegistry

DEFAULT_MANAGEMENT_TIMEOUT = 30.0
_MAX_MESSAGE_BYTES = 64 * 1024


class ModelManagementError(Exception):
    """A local management operation could not be completed."""


def runtime_directory() -> Path:
    """Return the private directory containing local server control sockets."""
    override = os.environ.get("MLX_SERVER_RUNTIME_DIR")
    return Path(override).expanduser() if override else Path.home() / ".cache/mlx-openai-server"


def _socket_path(port: int) -> Path:
    """Return the control socket corresponding to an inference port."""
    return runtime_directory() / f"server-{port}.sock"


def _atomic_write(path: Path, content: bytes, mode: int) -> None:
    """Replace a file with a fully written, permission-preserving temporary file."""
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def remove_from_config(config: MultiModelServerConfig, model_id: str) -> Path:
    """Remove one unchanged model entry and retain comments and a config backup.

    Parameters
    ----------
    config : MultiModelServerConfig
        The running server's configuration, including its source path.
    model_id : str
        Canonical model name to remove.

    Returns
    -------
    Path
        Backup containing the configuration immediately before this deletion.
    """
    if config.source_path is None:
        raise ModelManagementError("Persistent deletion requires a server launched with --config.")
    path = Path(config.source_path)
    before = path.read_bytes()
    current = load_config_from_yaml(str(path))
    original = next((model for model in config.models if model.served_model_name == model_id), None)
    matching = [
        (index, model)
        for index, model in enumerate(current.models)
        if model.served_model_name == model_id
    ]
    if len(matching) != 1 or matching[0][1] != original:
        raise ModelManagementError(
            f"Configuration for '{model_id}' changed on disk. Restart the server before deleting it."
        )

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    document = yaml.load(before.decode("utf-8"))
    del document["models"][matching[0][0]]
    output = StringIO()
    yaml.dump(document, output)
    if path.read_bytes() != before:
        raise ModelManagementError("The configuration changed during deletion; retry the command.")
    mode = stat.S_IMODE(path.stat().st_mode)
    backup = path.with_name(f"{path.name}.bak")
    _atomic_write(backup, before, mode)
    _atomic_write(path, output.getvalue().encode("utf-8"), mode)
    return backup


class ModelManager:
    """Dispatch local model-registration management for a running server."""

    def __init__(self, config: MultiModelServerConfig, registry: ModelRegistry) -> None:
        """Store the configuration and live registry managed by this socket."""
        self.config = config
        self.registry = registry

    async def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        """Describe the local server or persist deletion of one model."""
        operation = request.get("operation")
        if operation == "describe":
            return {
                "port": self.config.port,
                "pid": os.getpid(),
                "config": self.config.source_path,
                "models": [
                    {"id": name, "aliases": self.registry.list_aliases(name)}
                    for name in self.registry.list_model_ids()
                ],
            }
        if operation != "delete":
            raise ModelManagementError("Unknown model management operation.")
        model_id = request.get("model")
        if not isinstance(model_id, str) or not model_id:
            raise ModelManagementError("A model name is required.")

        backup: Path | None = None

        def persist(canonical: str) -> None:
            """Write the config while the registry holds its lifecycle lock."""
            nonlocal backup
            backup = remove_from_config(self.config, canonical)

        canonical = await self.registry.delete_model(model_id, persist)
        self.config.models = [
            model for model in self.config.models if model.served_model_name != canonical
        ]
        return {
            "status": "deleted",
            "model": canonical,
            "config": self.config.source_path,
            "backup": str(backup),
            "files_deleted": False,
        }

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Read one bounded JSON request and return one JSON response."""
        response: dict[str, Any] | None = None
        try:
            message = await asyncio.wait_for(reader.readline(), timeout=5)
            if not message:
                return
            request = json.loads(message)
            if not isinstance(request, dict):
                raise ModelManagementError("Expected a JSON object.")
            response = await self.dispatch(request)
        except (ModelManagementError, KeyError, ValueError, OSError, TimeoutError) as exc:
            message = str(exc.args[0]) if isinstance(exc, KeyError) else str(exc)
            response = {"error": message}
        except Exception:
            # A failed cleanup must return an error without tearing down the
            # control listener. Details stay in the server log.
            logger.exception("Local model deletion failed")
            response = {"error": "Model deletion failed; inspect the server log."}
        else:
            logger.debug(f"Completed local model management operation: {request.get('operation')}")
        finally:
            if response is not None:
                writer.write(json.dumps(response).encode("utf-8") + b"\n")
                try:
                    await writer.drain()
                except (ConnectionError, OSError):
                    pass
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass


@asynccontextmanager
async def serve_model_management(
    config: MultiModelServerConfig,
    registry: ModelRegistry,
) -> AsyncIterator[None]:
    """Expose configuration management only to the local server's user."""
    if config.source_path is None:
        yield
        return

    folder = runtime_directory()
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    if folder.stat().st_uid != os.getuid():
        raise ModelManagementError(f"Control directory is not owned by this user: {folder}")
    folder.chmod(0o700)
    path = _socket_path(config.port)
    if path.exists():
        if not stat.S_ISSOCK(path.lstat().st_mode):
            raise ModelManagementError(f"Control path already exists and is not a socket: {path}")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.settimeout(1.0)
            try:
                probe.connect(str(path))
            except ConnectionRefusedError:
                path.unlink()
            else:
                raise ModelManagementError(f"A local server already manages port {config.port}.")

    manager = ModelManager(config, registry)
    server = await asyncio.start_unix_server(
        manager.handle, path=str(path), limit=_MAX_MESSAGE_BYTES
    )
    path.chmod(0o600)
    inode = path.stat().st_ino
    try:
        async with server:
            yield
    finally:
        if path.exists() and path.stat().st_ino == inode:
            path.unlink()


def management_request(
    port: int, request: dict[str, Any], timeout: float = DEFAULT_MANAGEMENT_TIMEOUT
) -> dict[str, Any]:
    """Send one owner-local management request to the selected server."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(timeout)
            connection.connect(str(_socket_path(port)))
            connection.sendall(json.dumps(request).encode("utf-8") + b"\n")
            with connection.makefile("rb") as response_stream:
                message = response_stream.readline(_MAX_MESSAGE_BYTES + 1)
        if len(message) > _MAX_MESSAGE_BYTES:
            raise ModelManagementError("The server returned an oversized control response.")
        response = json.loads(message)
    except (OSError, ValueError) as exc:
        raise ModelManagementError(
            f"Could not manage the local server on port {port}. Start or restart it "
            f"with --config using the updated mlx-openai-server installation. ({exc})"
        ) from exc
    if not isinstance(response, dict):
        raise ModelManagementError("The server returned an invalid control response.")
    if "error" in response:
        raise ModelManagementError(str(response["error"]))
    return response


def find_local_server(
    port: int | None, timeout: float = DEFAULT_MANAGEMENT_TIMEOUT
) -> dict[str, Any]:
    """Find the selected local server, or the only active server when omitted."""
    if port is not None:
        return management_request(port, {"operation": "describe"}, timeout)

    servers = []
    for path in sorted(runtime_directory().glob("server-*.sock")):
        try:
            candidate = int(path.stem.removeprefix("server-"))
            server = management_request(candidate, {"operation": "describe"}, min(timeout, 1.0))
        except (ValueError, ModelManagementError):
            continue
        servers.append(server)
    if not servers:
        raise ModelManagementError(
            "No local model-management server found. Start or restart "
            "mlx-openai-server with --config."
        )
    if len(servers) != 1:
        ports = ", ".join(str(server["port"]) for server in servers)
        raise ModelManagementError(f"Multiple local servers are running ({ports}); specify --port.")
    return servers[0]
