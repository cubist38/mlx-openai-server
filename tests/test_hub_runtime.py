"""Tests for hub runtime scaffolding and validation."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from app.config import MLXServerConfig
from app.hub.config import HubConfigError, MLXHubConfig, MLXHubGroupConfig, load_hub_config
from app.hub.runtime import HubRuntime


def _write_hub_file(base: Path, contents: str) -> Path:
    path = base / "hub.yaml"
    path.write_text(contents, encoding="utf-8")
    return path


def test_load_hub_config_rejects_unknown_group_when_declared(tmp_path: Path) -> None:
    """Models referencing undefined groups should error when groups are declared."""

    hub_path = _write_hub_file(
        tmp_path,
        dedent(
            """
            models:
              - name: foo
                model_path: /models/foo
                group: missing
            groups:
              - name: existing
                max_loaded: 1
            """
        ).strip(),
    )

    with pytest.raises(HubConfigError, match="not defined"):
        load_hub_config(hub_path)


def test_group_default_limit_not_enforced(tmp_path: Path) -> None:
    """Default process counts should not be limited by max_loaded."""

    hub_path = _write_hub_file(
        tmp_path,
        dedent(
            f"""
            log_path: {tmp_path / "logs"}
            models:
              - name: alpha
                model_path: /models/alpha
                group: constrained
                default: true
              - name: beta
                model_path: /models/beta
                group: constrained
                default: true
            groups:
              - name: constrained
                max_loaded: 1
            """
        ).strip(),
    )

    config = load_hub_config(hub_path)
    assert [model.name for model in config.models] == ["alpha", "beta"]


def test_hub_runtime_bootstrap_and_selection(tmp_path: Path) -> None:
    """HubRuntime should summarize models and honor selection filters."""

    models = [
        MLXServerConfig(model_path="/models/foo", name="foo", is_default_model=True),
        MLXServerConfig(model_path="/models/bar", name="bar", is_default_model=False),
    ]
    config = MLXHubConfig(
        log_path=tmp_path / "logs",
        models=models,
        groups=[MLXHubGroupConfig(name="general", max_loaded=None)],
    )

    runtime = HubRuntime(config)

    assert runtime.bootstrap_targets() == []

    summaries = runtime.describe_models(["bar"])
    assert len(summaries) == 1
    assert summaries[0]["name"] == "bar"

    with pytest.raises(HubConfigError, match="Unknown model"):
        runtime.describe_models(["missing"])


def test_hub_runtime_enforces_group_slots(tmp_path: Path) -> None:
    """HubRuntime should reserve group slots and block when caps reached."""

    models = [
        MLXServerConfig(model_path="/models/a", name="alpha", group="g", is_default_model=True),
        MLXServerConfig(model_path="/models/b", name="beta", group="g", is_default_model=True),
    ]
    config = MLXHubConfig(
        log_path=tmp_path / "logs",
        models=models,
        groups=[MLXHubGroupConfig(name="g", max_loaded=1)],
    )

    runtime = HubRuntime(config)
    assert runtime.bootstrap_targets() == []

    assert runtime.can_load("alpha") is True
    runtime.mark_loading("alpha")
    assert runtime.describe_models(["alpha"])[0]["status"] == "loading"

    assert runtime.can_load("beta") is False
    with pytest.raises(HubConfigError):
        runtime.mark_loading("beta")

    runtime.mark_failed("alpha", "init failed")
    assert runtime.describe_models(["alpha"])[0]["status"] == "failed"
    assert runtime.can_load("beta") is True
    runtime.mark_loading("beta")


def test_hub_runtime_load_and_unload_transitions(tmp_path: Path) -> None:
    """State transitions should update statuses and group usage."""

    config = MLXHubConfig(
        log_path=tmp_path / "logs",
        models=[MLXServerConfig(model_path="/m/foo", name="foo", group=None)],
    )
    runtime = HubRuntime(config)

    runtime.mark_loading("foo")
    runtime.mark_loaded("foo")
    assert runtime.describe_models(["foo"])[0]["status"] == "loaded"

    runtime.mark_unloaded("foo")
    assert runtime.describe_models(["foo"])[0]["status"] == "unloaded"
