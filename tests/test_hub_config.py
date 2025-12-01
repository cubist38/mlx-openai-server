"""Tests for hub configuration parsing helpers."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from app.hub.config import HubConfigError, MLXHubConfig, load_hub_config


@pytest.fixture(autouse=True)
def _stub_port_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tests do not rely on real network state."""
    monkeypatch.setattr("app.hub.config.is_port_available", lambda _host=None, _port=None: True)


def _write_yaml(path: Path, contents: str) -> Path:
    path.write_text(dedent(contents).strip(), encoding="utf-8")
    return path


def test_load_hub_config_creates_log_directory(tmp_path: Path) -> None:
    """Loading a hub config should expand and create the log directory."""
    log_dir = tmp_path / "logs" / "nested"
    hub_path = _write_yaml(
        tmp_path / "hub.yaml",
        f"""
        log_path: {log_dir}
        models:
          - name: alpha
            model_path: /models/alpha
        """,
    )

    config = load_hub_config(hub_path)

    assert config.log_path == log_dir
    assert log_dir.exists(), f"Log directory {log_dir} should exist"
    assert log_dir.is_dir(), f"Log path {log_dir} should be a directory"


def test_load_hub_config_rejects_invalid_model_slug(tmp_path: Path) -> None:
    """Model names must already be slug-compliant."""
    hub_path = _write_yaml(
        tmp_path / "hub.yaml",
        """
        models:
          - name: "invalid slug"
            model_path: /models/alpha
        """,
    )

    with pytest.raises(HubConfigError, match="model name"):
        load_hub_config(hub_path)


def test_load_hub_config_allows_group_reference_when_groups_missing(tmp_path: Path) -> None:
    """Referencing a group is permitted even when no group objects are defined."""
    hub_path = _write_yaml(
        tmp_path / "hub.yaml",
        """
        models:
          - name: alpha
            model_path: /models/a
            group: workers
        """,
    )

    config = load_hub_config(hub_path)

    assert isinstance(config, MLXHubConfig)
    assert config.models[0].group == "workers"


def test_models_without_ports_receive_unique_offsets(tmp_path: Path) -> None:
    """Models should default to sequential ports starting at the configured base."""
    hub_path = _write_yaml(
        tmp_path / "hub.yaml",
        """
        models:
          - name: alpha
            model_path: /models/a
          - name: beta
            model_path: /models/b
        """,
    )

    config = load_hub_config(hub_path)

    ports = [model.port for model in config.models]
    assert ports == [47850, 47851]  # Models start at model_starting_port


def test_models_honor_custom_starting_port(tmp_path: Path) -> None:
    """model_starting_port overrides the default sequential assignment."""
    hub_path = _write_yaml(
        tmp_path / "hub.yaml",
        dedent(
            """
            model_starting_port: 60000
            models:
              - name: alpha
                model_path: /models/a
              - name: beta
                model_path: /models/b
            """,
        ),
    )

    config = load_hub_config(hub_path)

    ports = [model.port for model in config.models]
    assert ports == [60000, 60001]  # Models start at model_starting_port


def test_starting_port_below_range_raises(tmp_path: Path) -> None:
    """model_starting_port must live inside the reserved TCP range."""
    hub_path = _write_yaml(
        tmp_path / "hub.yaml",
        dedent(
            """
            model_starting_port: 100
            models:
              - name: alpha
                model_path: /models/a
            """,
        ),
    )

    with pytest.raises(HubConfigError, match="model_starting_port"):
        load_hub_config(hub_path)


def test_starting_port_must_be_integer(tmp_path: Path) -> None:
    """Non-integer model_starting_port values should fail fast."""
    hub_path = _write_yaml(
        tmp_path / "hub.yaml",
        dedent(
            """
            model_starting_port: invalid
            models:
              - name: alpha
                model_path: /models/a
            """,
        ),
    )

    with pytest.raises(HubConfigError, match="integer"):
        load_hub_config(hub_path)


def test_model_port_conflict_raises(tmp_path: Path) -> None:
    """Explicitly reusing a port should raise a configuration error."""
    hub_path = _write_yaml(
        tmp_path / "hub.yaml",
        dedent(
            """
            models:
              - name: alpha
                model_path: /models/a
                port: 8100
              - name: beta
                model_path: /models/b
                port: 8100
            """,
        ),
    )

    with pytest.raises(HubConfigError, match="port"):
        load_hub_config(hub_path)


def test_model_port_conflict_with_controller_port(tmp_path: Path) -> None:
    """Ports cannot overlap with the controller's host/port."""
    hub_path = _write_yaml(
        tmp_path / "hub.yaml",
        dedent(
            """
            port: 8000
            models:
              - name: alpha
                model_path: /models/a
                port: 8000
            """,
        ),
    )

    with pytest.raises(HubConfigError, match="port"):
        load_hub_config(hub_path)


def test_auto_port_skips_in_use_candidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto allocation should probe ports and skip ones that appear busy."""
    calls: list[int] = []

    def _fake_probe(_host: str | None = None, port: int | None = None) -> bool:
        assert port is not None
        calls.append(port)
        return port != 47850

    monkeypatch.setattr("app.hub.config.is_port_available", _fake_probe)

    hub_path = _write_yaml(
        tmp_path / "hub.yaml",
        """
        models:
          - name: alpha
            model_path: /models/a
        """,
    )

    config = load_hub_config(hub_path)
    assert [model.port for model in config.models] == [47851]  # Model gets 47851
    assert calls == [
        47850,
        47851,
    ]  # Tries 47850 (busy), 47851 (free), model gets 47851


def test_explicit_port_in_use_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit ports should also be probed and rejected when busy."""

    def _fake_probe(_host: str | None = None, port: int | None = None) -> bool:
        return port != 60010  # Make 60010 unavailable, others available

    monkeypatch.setattr("app.hub.config.is_port_available", _fake_probe)

    hub_path = _write_yaml(
        tmp_path / "hub.yaml",
        """
        models:
          - name: alpha
            model_path: /models/a
            port: 60010
        """,
    )

    with pytest.raises(HubConfigError, match="in use"):
        load_hub_config(hub_path)


def test_auto_ports_reuse_persisted_assignment_when_busy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persisted ports should remain stable even when sockets are busy."""
    hub_path = _write_yaml(
        tmp_path / "hub.yaml",
        """
        models:
          - name: alpha
            model_path: /models/a
        """,
    )

    initial_config = load_hub_config(hub_path)
    assigned_port = initial_config.models[0].port
    assert assigned_port is not None

    busy_ports = {assigned_port}

    def _fake_probe(_host: str | None = None, port: int | None = None) -> bool:
        return port not in busy_ports

    monkeypatch.setattr("app.hub.config.is_port_available", _fake_probe)

    reloaded_without_persist = load_hub_config(hub_path)
    assert reloaded_without_persist.models[0].port != assigned_port

    reloaded_with_persist = load_hub_config(
        hub_path,
        persisted_ports={"alpha": assigned_port},
    )
    assert reloaded_with_persist.models[0].port == assigned_port


def test_group_idle_trigger_requires_max_loaded(tmp_path: Path) -> None:
    """idle_unload_trigger_min cannot be set without max_loaded."""
    hub_path = _write_yaml(
        tmp_path / "hub.yaml",
        """
        groups:
          - name: workers
            idle_unload_trigger_min: 5
        models:
          - name: alpha
            model_path: /models/a
        """,
    )

    with pytest.raises(HubConfigError, match="requires max_loaded"):
        load_hub_config(hub_path)


def test_group_idle_trigger_must_be_positive_integer(tmp_path: Path) -> None:
    """idle_unload_trigger_min must be a positive integer."""
    hub_path = _write_yaml(
        tmp_path / "hub.yaml",
        """
        groups:
          - name: workers
            max_loaded: 2
            idle_unload_trigger_min: 0
        models:
          - name: alpha
            model_path: /models/a
        """,
    )

    with pytest.raises(HubConfigError, match="idle_unload_trigger_min"):
        load_hub_config(hub_path)


def test_group_idle_trigger_loads_successfully(tmp_path: Path) -> None:
    """Valid idle_unload_trigger_min values should populate group configs."""
    hub_path = _write_yaml(
        tmp_path / "hub.yaml",
        """
        groups:
          - name: workers
            max_loaded: 2
            idle_unload_trigger_min: 15
        models:
          - name: alpha
            model_path: /models/a
            group: workers
        """,
    )

    config = load_hub_config(hub_path)

    assert config.groups[0].idle_unload_trigger_min == 15
