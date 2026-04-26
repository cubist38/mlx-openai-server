"""Fail-first CLI parity tests for sampling-default flags."""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
import types
from typing import Any

from click.testing import CliRunner
import pytest

from app.config import MLXServerConfig, ModelEntryConfig


def _load_cli_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import ``app.cli`` with a lightweight ``app.main`` stub."""

    fake_main = types.ModuleType("app.main")
    fake_mflux = types.ModuleType("app.models.mflux")

    async def _placeholder_start(_config: Any) -> None:
        return None

    async def _placeholder_start_multi(_config: Any) -> None:
        return None

    fake_main.start = _placeholder_start
    fake_main.start_multi = _placeholder_start_multi
    fake_mflux.IMAGE_CONFIG_NAMES = ()
    monkeypatch.setitem(sys.modules, "app.main", fake_main)
    monkeypatch.setitem(sys.modules, "app.models.mflux", fake_mflux)
    monkeypatch.delitem(sys.modules, "app.cli", raising=False)
    cli_module = importlib.import_module("app.cli")
    return importlib.reload(cli_module)


def _load_main_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import ``app.main`` with a lightweight ``app.server`` stub."""

    fake_server = types.ModuleType("app.server")

    def _placeholder_setup_server(_config: Any) -> Any:
        return object()

    fake_server.setup_server = _placeholder_setup_server
    monkeypatch.setitem(sys.modules, "app.server", fake_server)
    monkeypatch.delitem(sys.modules, "app.main", raising=False)
    main_module = importlib.import_module("app.main")
    return importlib.reload(main_module)


def test_launch_accepts_repetition_penalty_and_passes_it_to_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI should expose ``--repetition-penalty`` and wire it into config defaults."""

    cli_module = _load_cli_module(monkeypatch)
    captured: dict[str, Any] = {}

    async def _fake_start_multi(config: Any) -> None:
        captured["config"] = config

    monkeypatch.setattr(cli_module, "start_multi", _fake_start_multi)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        [
            "launch",
            "--model-path",
            "dummy-model",
            "--repetition-penalty",
            "1.25",
        ],
    )

    assert result.exit_code == 0, result.output
    # Single-model ``launch`` is wrapped in a one-entry MultiModelServerConfig
    # so it runs through the subprocess-isolation path; the sampling default
    # now lives on the sole ModelEntryConfig.
    assert captured["config"].models[0].default_repetition_penalty == 1.25


def test_launch_leaves_sampling_defaults_unset_when_flags_are_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitted CLI sampling flags should not shadow model generation_config.json."""

    cli_module = _load_cli_module(monkeypatch)
    captured: dict[str, Any] = {}

    async def _fake_start_multi(config: Any) -> None:
        captured["config"] = config

    monkeypatch.setattr(cli_module, "start_multi", _fake_start_multi)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        [
            "launch",
            "--model-path",
            "dummy-model",
        ],
    )

    assert result.exit_code == 0, result.output
    model_config = captured["config"].models[0]
    assert model_config.default_temperature is None
    assert model_config.default_top_p is None
    assert model_config.default_top_k is None
    assert model_config.default_repetition_penalty is None
    assert model_config.default_max_tokens is None


def test_start_exports_repetition_penalty_before_server_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-model startup should export repetition-penalty before setup_server()."""

    main_module = _load_main_module(monkeypatch)

    captured_env: dict[str, str | None] = {}

    def _fake_setup_server(config: Any) -> Any:
        del config
        captured_env["DEFAULT_REPETITION_PENALTY"] = os.environ.get("DEFAULT_REPETITION_PENALTY")
        return object()

    class _FakeUvicornServer:
        def __init__(self, _config: Any) -> None:
            pass

        async def serve(self) -> None:
            return None

    monkeypatch.setattr(main_module, "setup_server", _fake_setup_server)
    monkeypatch.setattr(main_module.uvicorn, "Server", _FakeUvicornServer)
    monkeypatch.setattr(main_module, "print_startup_banner", lambda _config: None)
    monkeypatch.delenv("DEFAULT_REPETITION_PENALTY", raising=False)

    config = MLXServerConfig(
        model_path="dummy-model",
        model_type="lm",
        default_repetition_penalty=1.25,
    )

    asyncio.run(main_module.start(config))

    assert captured_env["DEFAULT_REPETITION_PENALTY"] == "1.25"


def test_launch_defaults_prompt_cache_size_to_ten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI should default ``--prompt-cache-size`` to 10."""

    cli_module = _load_cli_module(monkeypatch)
    captured: dict[str, Any] = {}

    async def _fake_start_multi(config: Any) -> None:
        captured["config"] = config

    monkeypatch.setattr(cli_module, "start_multi", _fake_start_multi)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.cli,
        [
            "launch",
            "--model-path",
            "dummy-model",
        ],
    )

    assert result.exit_code == 0, result.output
    # Single-model ``launch`` is wrapped into a one-entry multi-model config
    # (subprocess isolation), so the cache size now lives on the wrapped entry.
    assert captured["config"].models[0].prompt_cache_size == 10


def test_model_entry_extras_surfaces_non_default_batch_scheduler_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-model banners should show per-model batch scheduler overrides."""

    main_module = _load_main_module(monkeypatch)
    extras = dict(
        main_module._model_entry_extras(
            ModelEntryConfig(
                model_path="dummy-model",
                model_type="lm",
                batch_completion_size=16,
                batch_prefill_step_size=1024,
            )
        )
    )

    assert extras["batch_scheduler"] == "decode=16, prefill_step=1024"
