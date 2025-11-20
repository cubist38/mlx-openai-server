"""Regression tests for the process-based HubManager."""

from __future__ import annotations

from collections import defaultdict
import itertools
from pathlib import Path

import yaml

from app.config import MLXServerConfig
from app.hub.manager import HubManager, ManagedProcess
from app.hub.observability import HubModelContext


def _write_hub_config(
    path: Path,
    *,
    log_path: Path,
    models: list[dict[str, object]],
    groups: list[dict[str, object]] | None = None,
) -> None:
    """Write a hub.yaml file containing the supplied models/groups."""

    payload: dict[str, object] = {
        "log_path": str(log_path),
        "models": models,
    }
    if groups:
        payload["groups"] = groups
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _model_entry(
    name: str,
    *,
    port: int,
    group: str | None = None,
    default: bool = False,
) -> dict[str, object]:
    """Return a minimal YAML entry for ``name``."""

    data: dict[str, object] = {
        "name": name,
        "model_path": f"/models/{name}",
        "port": port,
    }
    if group:
        data["group"] = group
    if default:
        data["default"] = True
    return data


class _StubProcess(ManagedProcess):
    """Minimal ManagedProcess implementation for unit tests."""

    _pid_counter = itertools.count(2000)

    def __init__(self, config: MLXServerConfig) -> None:
        self.config = config
        self.started = False
        self.stopped = False
        self.exit_code: int | None = None
        self._pid = next(self._pid_counter)

    def start(self) -> None:
        """Record that the fake process was started."""

        self.started = True

    def stop(self, timeout: float = 10.0) -> None:
        """Record that the fake process was stopped."""

        self.stopped = True
        if self.exit_code is None:
            self.exit_code = 0

    def poll(self) -> int | None:
        """Return the forced exit code, if any."""

        return self.exit_code

    @property
    def pid(self) -> int | None:
        """Return a deterministic PID once the process started."""

        return self._pid if self.started else None

    def force_exit(self, code: int) -> None:
        """Simulate the process exiting with ``code``."""

        self.exit_code = code


class _StubProcessFactory:
    """Callable factory that records spawned stub processes."""

    def __init__(self) -> None:
        self.instances: defaultdict[str, list[_StubProcess]] = defaultdict(list)

    def __call__(self, config: MLXServerConfig) -> _StubProcess:
        """Return a new stub process for ``config`` and record it."""

        if config.name is None:  # pragma: no cover - hub config guarantees names
            raise AssertionError("Hub models must have names")
        process = _StubProcess(config)
        self.instances[config.name].append(process)
        return process

    def last(self, name: str) -> _StubProcess:
        """Return the last spawned process for ``name``."""

        return self.instances[name][-1]


class _RecordingSink:
    """Capture observability events emitted by the HubManager."""

    def __init__(self) -> None:
        self.events: list[tuple[str, HubModelContext, int | None]] = []

    def model_started(self, ctx: HubModelContext, *, pid: int | None) -> None:
        self.events.append(("started", ctx, pid))

    def model_stopped(self, ctx: HubModelContext, *, exit_code: int | None) -> None:
        self.events.append(("stopped", ctx, exit_code))

    def model_failed(self, ctx: HubModelContext, *, exit_code: int | None) -> None:
        self.events.append(("failed", ctx, exit_code))


def test_hub_manager_reload_applies_model_diffs(tmp_path: Path) -> None:
    """Reloading should start, stop, and restart models as configs change."""

    config_path = tmp_path / "hub.yaml"
    log_path = tmp_path / "logs"
    factory = _StubProcessFactory()
    manager = HubManager(config_path, process_factory=factory)

    _write_hub_config(
        config_path,
        log_path=log_path,
        models=[
            _model_entry("alpha", port=8100, default=True),
            _model_entry("beta", port=8101, default=True),
        ],
    )

    result = manager.reload()
    assert set(result.started) == {"alpha", "beta"}
    assert result.stopped == []

    statuses = manager.get_status()
    assert {status.name for status in statuses} == {"alpha", "beta"}
    assert all(status.state == "running" for status in statuses)

    _write_hub_config(
        config_path,
        log_path=log_path,
        models=[_model_entry("alpha", port=8200, default=True)],
    )

    result = manager.reload()
    assert set(result.started) == {"alpha"}
    assert set(result.stopped) == {"alpha", "beta"}

    statuses = manager.get_status()
    assert [status.name for status in statuses] == ["alpha"]
    assert statuses[0].state == "running"
    assert factory.instances["beta"][0].stopped is True


def test_hub_manager_leaves_on_demand_models_idle(tmp_path: Path) -> None:
    """Models without the default flag should remain stopped after reload."""

    config_path = tmp_path / "hub.yaml"
    log_path = tmp_path / "logs"
    factory = _StubProcessFactory()
    manager = HubManager(config_path, process_factory=factory)

    _write_hub_config(
        config_path,
        log_path=log_path,
        models=[_model_entry("gamma", port=8700, default=False)],
    )

    result = manager.reload()
    assert result.started == []
    assert result.stopped == []
    assert result.unchanged == ["gamma"]
    assert manager.get_status() == []

    manager.start_model("gamma")
    statuses = manager.get_status()
    assert len(statuses) == 1
    assert statuses[0].name == "gamma"
    assert statuses[0].state == "running"


def test_hub_manager_reload_restarts_crashed_process(tmp_path: Path) -> None:
    """A crashed model should restart on the next reload."""

    config_path = tmp_path / "hub.yaml"
    log_path = tmp_path / "logs"
    factory = _StubProcessFactory()
    manager = HubManager(config_path, process_factory=factory)

    _write_hub_config(
        config_path,
        log_path=log_path,
        models=[_model_entry("alpha", port=8300, default=True)],
    )
    manager.reload()

    crashed = factory.last("alpha")
    crashed.force_exit(9)
    snapshot = manager.get_status()
    assert snapshot[0].state == "failed"

    result = manager.reload()
    assert set(result.started) == {"alpha"}
    assert set(result.stopped) == {"alpha"}
    assert len(factory.instances["alpha"]) == 2
    assert factory.last("alpha").started is True


def test_hub_manager_releases_group_slot_after_crash(tmp_path: Path) -> None:
    """Unexpected exits should free the group slot for future models."""

    config_path = tmp_path / "hub.yaml"
    log_path = tmp_path / "logs"
    factory = _StubProcessFactory()
    manager = HubManager(config_path, process_factory=factory)

    _write_hub_config(
        config_path,
        log_path=log_path,
        models=[_model_entry("alpha", port=8400, group="tier", default=True)],
        groups=[{"name": "tier", "max_loaded": 1}],
    )
    manager.reload()

    crashing = factory.last("alpha")
    crashing.force_exit(7)
    manager.get_status()

    _write_hub_config(
        config_path,
        log_path=log_path,
        models=[_model_entry("beta", port=8500, group="tier", default=True)],
        groups=[{"name": "tier", "max_loaded": 1}],
    )

    result = manager.reload()
    assert set(result.started) == {"beta"}
    assert set(result.stopped) == {"alpha"}


def test_hub_manager_ignores_group_limit_for_processes(tmp_path: Path) -> None:
    """Manual starts should always be allowed regardless of group max_loaded."""

    config_path = tmp_path / "hub.yaml"
    log_path = tmp_path / "logs"
    factory = _StubProcessFactory()
    manager = HubManager(config_path, process_factory=factory)

    _write_hub_config(
        config_path,
        log_path=log_path,
        models=[
            _model_entry("alpha", port=8600, group="tier", default=True),
            _model_entry("beta", port=8601, group="tier", default=False),
        ],
        groups=[{"name": "tier", "max_loaded": 1}],
    )

    manager.reload()
    manager.start_model("beta")
    assert len(factory.instances["beta"]) == 1


def test_hub_manager_emits_observability_events(tmp_path: Path) -> None:
    """Lifecycle hooks should surface start/stop/failure events with context."""

    config_path = tmp_path / "hub.yaml"
    log_path = tmp_path / "logs"
    sink = _RecordingSink()
    factory = _StubProcessFactory()
    manager = HubManager(
        config_path,
        process_factory=factory,
        observability_sink=sink,
    )

    _write_hub_config(
        config_path,
        log_path=log_path,
        models=[
            _model_entry("alpha", port=8610, default=True),
            _model_entry("beta", port=8611, default=True),
        ],
    )
    manager.reload()

    started = {ctx.name for kind, ctx, _ in sink.events if kind == "started"}
    assert started == {"alpha", "beta"}

    manager.stop_model("alpha")
    stopped = [ctx.name for kind, ctx, _ in sink.events if kind == "stopped"]
    assert "alpha" in stopped

    crashing = factory.last("beta")
    crashing.force_exit(9)
    manager.get_status()
    failed = [ctx.name for kind, ctx, _ in sink.events if kind == "failed"]
    assert failed[-1] == "beta"


def test_hub_manager_assigns_per_model_log_files(tmp_path: Path) -> None:
    """Default log paths should be namespaced per model."""

    config_path = tmp_path / "hub.yaml"
    log_path = tmp_path / "logs"
    factory = _StubProcessFactory()
    manager = HubManager(config_path, process_factory=factory)

    _write_hub_config(
        config_path,
        log_path=log_path,
        models=[_model_entry("alpha", port=8620, default=True)],
    )

    manager.reload()
    record = manager._records.get("alpha")  # noqa: SLF001 - internal state verification
    assert record is not None
    expected = log_path / "alpha.log"
    assert record.log_path == expected
    assert expected.parent == log_path
