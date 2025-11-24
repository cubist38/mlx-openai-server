"""Combined unit tests for hub runtime state helpers and cleanup.

This file merges the previous `test_hub_runtime_state.py` and
`test_hub_runtime_cleanup.py` tests so runtime persistence and the daemon's
cleanup behavior are verified together.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import socket

from fastapi.testclient import TestClient

from app.cli import _read_hub_runtime_state, _runtime_state_path, _write_hub_runtime_state
from app.hub.config import MLXHubConfig
from app.hub.daemon import create_app


def test_hub_runtime_state_written_and_read(tmp_path: Path) -> None:
    """Writing runtime state should create the file and it should be readable.

    The test binds a temporary listening socket on localhost to simulate a
    running daemon port and uses the current process PID so the liveness
    checks in `_read_hub_runtime_state` succeed.
    """

    # Configure a hub config that writes logs into the temporary directory
    config = MLXHubConfig(host="127.0.0.1", log_path=tmp_path)

    # Create a listening socket so the quick port check passes
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = sock.getsockname()[1]

    try:
        # Write runtime state using current pid and the listening port
        _write_hub_runtime_state(config, os.getpid(), port)

        path = _runtime_state_path(config)
        assert path.exists(), "runtime file should be created"

        runtime = _read_hub_runtime_state(config)
        assert isinstance(runtime, dict)
        assert runtime["pid"] == os.getpid()
        assert runtime["daemon_port"] == port

        # Closing the listening socket should make the quick port check fail
        sock.close()
        after = _read_hub_runtime_state(config)
        assert after is None, "runtime read should return None when port is closed"
    finally:
        # Defensive cleanup
        with contextlib.suppress(Exception):
            sock.close()


def _write_hub_yaml(path: Path) -> Path:
    data = {
        "log_path": str(path),
        "models": [{"name": "testmodel", "model_path": "dummy"}],
    }
    yaml_path = path / "hub.yaml"
    # JSON is valid YAML for our simple mapping; write it for the loader.
    yaml_path.write_text(json.dumps(data))
    return yaml_path


def test_runtime_file_removed_on_shutdown(tmp_path: Path) -> None:
    """Daemon lifespan should remove `hub_runtime.json` from its log path.

    We create a temporary hub YAML where `log_path` points at `tmp_path`,
    create a dummy `hub_runtime.json` there, start the daemon app with
    `TestClient` (which triggers lifespan events), and verify the file is
    removed after the client context exits.
    """

    yaml_path = _write_hub_yaml(tmp_path)

    runtime_file = tmp_path / "hub_runtime.json"
    runtime_file.write_text(json.dumps({"pid": 1, "daemon_port": 12345, "host": "127.0.0.1"}))

    assert runtime_file.exists()

    app = create_app(str(yaml_path))
    with TestClient(app) as client:
        # During app lifetime the runtime file should still exist
        assert runtime_file.exists()
        resp = client.get("/health")
        assert resp.status_code == 200

    # After exiting the client context the lifespan shutdown should have run
    assert not runtime_file.exists()
