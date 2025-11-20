"""Regression tests for the hub service IPC layer."""

from __future__ import annotations

from pathlib import Path
import shutil
from textwrap import dedent
import threading

import pytest

from app.config import MLXServerConfig
from app.hub.config import load_hub_config
from app.hub.manager import ManagedProcess, ProcessFactory
from app.hub.service import HubService, HubServiceClient, get_service_paths


class _StubProcess(ManagedProcess):
    """Lightweight ManagedProcess implementation used in service tests."""

    _pid_counter = iter(range(2000, 5000))

    def __init__(self, config: MLXServerConfig) -> None:
        self.config = config
        self.started = False
        self.stopped = False
        self.exit_code: int | None = None
        self._pid = next(self._pid_counter)

    def start(self) -> None:
        self.started = True

    def stop(self, timeout: float = 10.0) -> None:  # noqa: ARG002
        self.stopped = True
        if self.exit_code is None:
            self.exit_code = 0

    def poll(self) -> int | None:
        return self.exit_code

    @property
    def pid(self) -> int | None:
        return self._pid if self.started else None


class _StubFactory(ProcessFactory):
    """Collect stub process instances for inspection in tests."""

    def __init__(self) -> None:
        self.instances: dict[str, list[_StubProcess]] = {}

    def __call__(self, config: MLXServerConfig) -> _StubProcess:
        if config.name is None:  # pragma: no cover - defensive guard
            raise AssertionError("Hub models must have names")
        process = _StubProcess(config)
        self.instances.setdefault(config.name, []).append(process)
        return process


def _start_service(
    config_path: Path, factory: _StubFactory, monkeypatch: pytest.MonkeyPatch
) -> tuple[HubServiceClient, threading.Thread]:
    monkeypatch.setattr("app.hub.service._configure_service_logging", lambda *_, **__: None)
    monkeypatch.setattr("app.hub.service.HubService._install_signal_handlers", lambda self: None)

    service = HubService(config_path, process_factory=factory)
    ready = threading.Event()
    thread = threading.Thread(
        target=service.serve_forever, kwargs={"ready_event": ready}, daemon=True
    )
    thread.start()
    if not ready.wait(timeout=5):  # pragma: no cover - defensive guard
        raise AssertionError("Hub service did not start in time")

    config = load_hub_config(config_path)
    paths = get_service_paths(config)
    client = HubServiceClient(paths.socket_path)
    assert client.wait_until_available(timeout=5)
    return client, thread


def _write_hub_yaml(tmp_path: Path, *, test_id: str) -> tuple[Path, Path]:
    """Create a hub.yaml with a test-isolated log directory under /tmp."""

    log_dir = Path("/tmp") / f"hub-{test_id}"
    log_dir.mkdir(parents=True, exist_ok=True)
    hub_yaml = tmp_path / f"{test_id}.yaml"
    hub_yaml.write_text(
        dedent(
            f"""
            log_path: {log_dir}
            models:
              - name: alpha
                model_path: /models/alpha
                default: true
            """
        ).strip()
    )
    return hub_yaml, log_dir


def test_hub_service_reports_status_and_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The service should start configured models and report their status."""
    hub_yaml, log_dir = _write_hub_yaml(tmp_path, test_id="reload")
    factory = _StubFactory()
    client, thread = _start_service(hub_yaml, factory, monkeypatch)

    status = client.status()
    assert status["models"]
    assert status["models"][0]["name"] == "alpha"
    assert factory.instances["alpha"][0].started is True

    reload_result = client.reload()
    assert "started" in reload_result

    client.shutdown()
    thread.join(timeout=5)
    shutil.rmtree(log_dir, ignore_errors=True)


def test_hub_service_start_stop_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit start/stop commands should control individual models."""
    hub_yaml, log_dir = _write_hub_yaml(tmp_path, test_id="start-stop")
    factory = _StubFactory()
    client, thread = _start_service(hub_yaml, factory, monkeypatch)

    client.stop_model("alpha")
    assert factory.instances["alpha"][0].stopped is True

    client.start_model("alpha")
    assert len(factory.instances["alpha"]) == 2
    assert factory.instances["alpha"][1].started is True

    client.shutdown()
    thread.join(timeout=5)
    shutil.rmtree(log_dir, ignore_errors=True)
