"""Fail-first tests for per-model sampling defaults in multi-model mode."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import types
from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import pytest

from app.config import MLXServerConfig, load_config_from_yaml
from app.schemas.openai import ChatCompletionRequest, Config, Message, ResponsesRequest

PER_MODEL_DEFAULTS_A: dict[str, int | float] = {
    "default_temperature": 1.0,
    "default_top_p": 0.95,
    "default_top_k": 35,
    "default_min_p": 0.0,
    "default_repetition_penalty": 1.11,
    "default_seed": 111,
    "default_max_tokens": 180000,
    "default_xtc_probability": 0.11,
    "default_xtc_threshold": 0.61,
    "default_presence_penalty": 0.31,
    "default_repetition_context_size": 41,
}

PER_MODEL_DEFAULTS_B: dict[str, int | float] = {
    "default_temperature": 0.7,
    "default_top_p": 0.9,
    "default_top_k": 20,
    "default_min_p": 0.05,
    "default_repetition_penalty": 1.22,
    "default_seed": 222,
    "default_max_tokens": 60000,
    "default_xtc_probability": 0.22,
    "default_xtc_threshold": 0.72,
    "default_presence_penalty": 0.42,
    "default_repetition_context_size": 52,
}

GLOBAL_ENV_DEFAULTS: dict[str, str] = {
    "DEFAULT_TEMPERATURE": "0.2",
    "DEFAULT_TOP_P": "0.5",
    "DEFAULT_TOP_K": "99",
    "DEFAULT_MIN_P": "0.9",
    "DEFAULT_REPETITION_PENALTY": "1.5",
    "DEFAULT_SEED": "999",
    "DEFAULT_MAX_TOKENS": "42",
    "DEFAULT_XTC_PROBABILITY": "0.77",
    "DEFAULT_XTC_THRESHOLD": "0.88",
    "DEFAULT_PRESENCE_PENALTY": "0.66",
    "DEFAULT_REPETITION_CONTEXT_SIZE": "123",
}


def _load_endpoints_module() -> Any:
    """Import ``app.api.endpoints`` with lightweight handler stubs."""

    fake_lm_module = types.ModuleType("app.handler.mlx_lm")
    fake_lm_module.MLXLMHandler = object

    fake_vlm_module = types.ModuleType("app.handler.mlx_vlm")
    fake_vlm_module.MLXVLMHandler = object

    module_names = [
        "app.handler.mlx_lm",
        "app.handler.mlx_vlm",
        "app.api.endpoints",
    ]
    original_modules: dict[str, types.ModuleType | None] = {
        name: sys.modules.get(name) for name in module_names
    }

    try:
        sys.modules["app.handler.mlx_lm"] = fake_lm_module
        sys.modules["app.handler.mlx_vlm"] = fake_vlm_module
        sys.modules.pop("app.api.endpoints", None)
        return importlib.import_module("app.api.endpoints")
    finally:
        sys.modules.pop("app.api.endpoints", None)
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class _FakeRegistry:
    """Minimal registry stub that resolves handlers by model id."""

    def __init__(
        self,
        handlers: dict[str, Any],
        on_demand_handlers: dict[str, Any] | None = None,
    ) -> None:
        self._handlers = handlers
        self._on_demand_handlers = on_demand_handlers or {}

    def get_handler(self, model_id: str) -> Any:
        return self._handlers[model_id]

    def is_on_demand(self, model_id: str) -> bool:
        return model_id in self._on_demand_handlers

    async def ensure_on_demand_loaded(self, model_id: str) -> Any:
        handler = self._on_demand_handlers[model_id]
        self._handlers[model_id] = handler
        return handler

    async def release_on_demand(self, model_id: str) -> None:
        if model_id in self._on_demand_handlers:
            self._handlers.pop(model_id, None)

    def list_model_ids(self) -> list[str]:
        return sorted(set(self._handlers.keys()) | set(self._on_demand_handlers.keys()))


def _make_raw_request(registry: _FakeRegistry | None, handler: Any | None = None) -> Any:
    """Build a minimal request-like object for endpoint unit tests."""

    return types.SimpleNamespace(
        app=types.SimpleNamespace(state=types.SimpleNamespace(registry=registry, handler=handler)),
        state=types.SimpleNamespace(request_id="req-test"),
    )


def _single_model_registry_case(case: str, handler: Any) -> _FakeRegistry | None:
    """Build registry shapes that may or may not actually own ``handler``."""

    if case == "no-registry":
        return None
    if case == "empty-registry":
        return _FakeRegistry({})
    if case == "other-model-only":
        return _FakeRegistry(
            {
                "model-a": types.SimpleNamespace(
                    handler_type="lm",
                    served_model_name="model-a",
                    model_path="other/model/path",
                )
            }
        )
    if case == "foreign-owned-handler-id":
        return _FakeRegistry(
            {
                handler.served_model_name: types.SimpleNamespace(
                    handler_type="lm",
                    served_model_name=handler.served_model_name,
                    model_path="other/model/path",
                )
            }
        )

    raise ValueError(f"Unknown single-model registry case: {case}")


def _single_model_handler_with_implicit_defaults() -> Any:
    """Build a handler-shaped object carrying implicit single-model defaults."""

    config = MLXServerConfig(model_path="dummy-model", model_type="lm")
    return types.SimpleNamespace(
        handler_type="lm",
        default_temperature=config.default_temperature,
        default_top_p=config.default_top_p,
        default_top_k=config.default_top_k,
        default_min_p=config.default_min_p,
        default_repetition_penalty=config.default_repetition_penalty,
        default_seed=config.default_seed,
        default_max_tokens=config.default_max_tokens,
        default_xtc_probability=config.default_xtc_probability,
        default_xtc_threshold=config.default_xtc_threshold,
        default_presence_penalty=config.default_presence_penalty,
        default_repetition_context_size=config.default_repetition_context_size,
    )


def test_load_config_from_yaml_preserves_per_model_sampling_defaults(tmp_path: Path) -> None:
    """Multi-model YAML should preserve per-model sampling-default overrides."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "models:\n"
        "  - model_path: /models/model-a\n"
        "    model_type: lm\n"
        "    served_model_name: model-a\n"
        "    default_temperature: 1.0\n"
        "    default_top_p: 0.95\n"
        "    default_top_k: 35\n"
        "    default_min_p: 0.0\n"
        "    default_repetition_penalty: 1.11\n"
        "    default_seed: 111\n"
        "    default_max_tokens: 180000\n"
        "    default_xtc_probability: 0.11\n"
        "    default_xtc_threshold: 0.61\n"
        "    default_presence_penalty: 0.31\n"
        "    default_repetition_context_size: 41\n"
        "  - model_path: /models/model-b\n"
        "    model_type: lm\n"
        "    served_model_name: model-b\n"
        "    default_temperature: 0.7\n"
        "    default_top_p: 0.9\n"
        "    default_top_k: 20\n"
        "    default_min_p: 0.05\n"
        "    default_repetition_penalty: 1.22\n"
        "    default_seed: 222\n"
        "    default_max_tokens: 60000\n"
        "    default_xtc_probability: 0.22\n"
        "    default_xtc_threshold: 0.72\n"
        "    default_presence_penalty: 0.42\n"
        "    default_repetition_context_size: 52\n",
        encoding="utf-8",
    )

    config = load_config_from_yaml(str(config_path))

    for key, expected in PER_MODEL_DEFAULTS_A.items():
        assert getattr(config.models[0], key) == expected
    for key, expected in PER_MODEL_DEFAULTS_B.items():
        assert getattr(config.models[1], key) == expected


def test_load_config_from_yaml_preserves_startup_timeout(tmp_path: Path) -> None:
    """Multi-model YAML should preserve per-model startup timeout overrides."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "models:\n"
        "  - model_path: /models/model-a\n"
        "    model_type: lm\n"
        "    served_model_name: model-a\n"
        "    startup_timeout: 3600\n",
        encoding="utf-8",
    )

    config = load_config_from_yaml(str(config_path))

    assert config.models[0].startup_timeout == 3600


@pytest.mark.asyncio
async def test_chat_completions_can_apply_different_defaults_per_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Different models should be able to supply different omitted sampling defaults."""

    endpoints_module = _load_endpoints_module()
    captured_requests: list[ChatCompletionRequest] = []

    async def _fake_process_text_request(
        _handler: Any,
        request: ChatCompletionRequest,
        request_id: str | None = None,
    ) -> JSONResponse:
        del request_id
        captured_requests.append(request.model_copy(deep=True))
        return JSONResponse(content={"ok": True})

    handler_a = types.SimpleNamespace(
        handler_type="lm", _uses_model_sampling_defaults=True, **PER_MODEL_DEFAULTS_A
    )
    handler_b = types.SimpleNamespace(
        handler_type="lm", _uses_model_sampling_defaults=True, **PER_MODEL_DEFAULTS_B
    )
    registry = _FakeRegistry({"model-a": handler_a, "model-b": handler_b})

    monkeypatch.setattr(endpoints_module, "process_text_request", _fake_process_text_request)
    for env_name, value in GLOBAL_ENV_DEFAULTS.items():
        monkeypatch.setenv(env_name, value)

    request_a = ChatCompletionRequest(
        model="model-a",
        messages=[Message(role="user", content="hello")],
        temperature=None,
        top_p=None,
        top_k=None,
        min_p=None,
        seed=None,
        repetition_penalty=None,
        max_completion_tokens=None,
        max_tokens=None,
        xtc_probability=None,
        xtc_threshold=None,
        presence_penalty=None,
        repetition_context_size=None,
    )
    request_b = ChatCompletionRequest(
        model="model-b",
        messages=[Message(role="user", content="hello")],
        temperature=None,
        top_p=None,
        top_k=None,
        min_p=None,
        seed=None,
        repetition_penalty=None,
        max_completion_tokens=None,
        max_tokens=None,
        xtc_probability=None,
        xtc_threshold=None,
        presence_penalty=None,
        repetition_context_size=None,
    )

    await endpoints_module.chat_completions(request_a, _make_raw_request(registry))
    await endpoints_module.chat_completions(request_b, _make_raw_request(registry))

    assert captured_requests[0].temperature == pytest.approx(
        PER_MODEL_DEFAULTS_A["default_temperature"]
    )
    assert captured_requests[0].top_p == pytest.approx(PER_MODEL_DEFAULTS_A["default_top_p"])
    assert captured_requests[0].top_k == PER_MODEL_DEFAULTS_A["default_top_k"]
    assert captured_requests[0].min_p == pytest.approx(PER_MODEL_DEFAULTS_A["default_min_p"])
    assert captured_requests[0].repetition_penalty == pytest.approx(
        PER_MODEL_DEFAULTS_A["default_repetition_penalty"]
    )
    assert captured_requests[0].seed == PER_MODEL_DEFAULTS_A["default_seed"]
    assert captured_requests[0].max_completion_tokens == PER_MODEL_DEFAULTS_A["default_max_tokens"]
    assert captured_requests[0].xtc_probability == pytest.approx(
        PER_MODEL_DEFAULTS_A["default_xtc_probability"]
    )
    assert captured_requests[0].xtc_threshold == pytest.approx(
        PER_MODEL_DEFAULTS_A["default_xtc_threshold"]
    )
    assert captured_requests[0].presence_penalty == pytest.approx(
        PER_MODEL_DEFAULTS_A["default_presence_penalty"]
    )
    assert (
        captured_requests[0].repetition_context_size
        == PER_MODEL_DEFAULTS_A["default_repetition_context_size"]
    )

    assert captured_requests[1].temperature == pytest.approx(
        PER_MODEL_DEFAULTS_B["default_temperature"]
    )
    assert captured_requests[1].top_p == pytest.approx(PER_MODEL_DEFAULTS_B["default_top_p"])
    assert captured_requests[1].top_k == PER_MODEL_DEFAULTS_B["default_top_k"]
    assert captured_requests[1].min_p == pytest.approx(PER_MODEL_DEFAULTS_B["default_min_p"])
    assert captured_requests[1].repetition_penalty == pytest.approx(
        PER_MODEL_DEFAULTS_B["default_repetition_penalty"]
    )
    assert captured_requests[1].seed == PER_MODEL_DEFAULTS_B["default_seed"]
    assert captured_requests[1].max_completion_tokens == PER_MODEL_DEFAULTS_B["default_max_tokens"]
    assert captured_requests[1].xtc_probability == pytest.approx(
        PER_MODEL_DEFAULTS_B["default_xtc_probability"]
    )
    assert captured_requests[1].xtc_threshold == pytest.approx(
        PER_MODEL_DEFAULTS_B["default_xtc_threshold"]
    )
    assert captured_requests[1].presence_penalty == pytest.approx(
        PER_MODEL_DEFAULTS_B["default_presence_penalty"]
    )
    assert (
        captured_requests[1].repetition_context_size
        == PER_MODEL_DEFAULTS_B["default_repetition_context_size"]
    )


@pytest.mark.asyncio
async def test_single_model_implicit_handler_defaults_do_not_shadow_env_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Implicit single-model config defaults should not override ambient DEFAULT_* env."""

    endpoints_module = _load_endpoints_module()
    captured_requests: list[ChatCompletionRequest] = []

    async def _fake_process_text_request(
        _handler: Any,
        request: ChatCompletionRequest,
        request_id: str | None = None,
    ) -> JSONResponse:
        del request_id
        captured_requests.append(request.model_copy(deep=True))
        return JSONResponse(content={"ok": True})

    handler = _single_model_handler_with_implicit_defaults()
    registry = _FakeRegistry({"local-text-model": handler})

    monkeypatch.setattr(endpoints_module, "process_text_request", _fake_process_text_request)
    for env_name, value in GLOBAL_ENV_DEFAULTS.items():
        monkeypatch.setenv(env_name, value)

    request = ChatCompletionRequest(
        messages=[Message(role="user", content="hello")],
        temperature=None,
        top_p=None,
        top_k=None,
        min_p=None,
        repetition_penalty=None,
        seed=None,
        max_completion_tokens=None,
        max_tokens=None,
        xtc_probability=None,
        xtc_threshold=None,
        presence_penalty=None,
        repetition_context_size=None,
    )

    await endpoints_module.chat_completions(request, _make_raw_request(registry))

    assert captured_requests[0].temperature == pytest.approx(
        float(GLOBAL_ENV_DEFAULTS["DEFAULT_TEMPERATURE"])
    )
    assert captured_requests[0].top_p == pytest.approx(float(GLOBAL_ENV_DEFAULTS["DEFAULT_TOP_P"]))
    assert captured_requests[0].top_k == int(GLOBAL_ENV_DEFAULTS["DEFAULT_TOP_K"])
    assert captured_requests[0].min_p == pytest.approx(float(GLOBAL_ENV_DEFAULTS["DEFAULT_MIN_P"]))
    assert captured_requests[0].repetition_penalty == pytest.approx(
        float(GLOBAL_ENV_DEFAULTS["DEFAULT_REPETITION_PENALTY"])
    )
    assert captured_requests[0].seed == int(GLOBAL_ENV_DEFAULTS["DEFAULT_SEED"])
    assert captured_requests[0].max_completion_tokens == int(
        GLOBAL_ENV_DEFAULTS["DEFAULT_MAX_TOKENS"]
    )
    assert captured_requests[0].xtc_probability == pytest.approx(
        float(GLOBAL_ENV_DEFAULTS["DEFAULT_XTC_PROBABILITY"])
    )
    assert captured_requests[0].xtc_threshold == pytest.approx(
        float(GLOBAL_ENV_DEFAULTS["DEFAULT_XTC_THRESHOLD"])
    )
    assert captured_requests[0].presence_penalty == pytest.approx(
        float(GLOBAL_ENV_DEFAULTS["DEFAULT_PRESENCE_PENALTY"])
    )
    assert captured_requests[0].repetition_context_size == int(
        GLOBAL_ENV_DEFAULTS["DEFAULT_REPETITION_CONTEXT_SIZE"]
    )


@pytest.mark.asyncio
async def test_chat_completions_explicit_request_values_override_model_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit client-supplied sampling params should win over model defaults."""

    endpoints_module = _load_endpoints_module()
    captured_requests: list[ChatCompletionRequest] = []

    async def _fake_process_text_request(
        _handler: Any,
        request: ChatCompletionRequest,
        request_id: str | None = None,
    ) -> JSONResponse:
        del request_id
        captured_requests.append(request.model_copy(deep=True))
        return JSONResponse(content={"ok": True})

    handler = types.SimpleNamespace(
        handler_type="lm", _uses_model_sampling_defaults=True, **PER_MODEL_DEFAULTS_A
    )
    registry = _FakeRegistry({"model-a": handler})

    monkeypatch.setattr(endpoints_module, "process_text_request", _fake_process_text_request)
    for env_name, value in GLOBAL_ENV_DEFAULTS.items():
        monkeypatch.setenv(env_name, value)

    request = ChatCompletionRequest(
        model="model-a",
        messages=[Message(role="user", content="hello")],
        temperature=0.33,
        top_p=0.44,
        top_k=12,
        min_p=0.22,
        repetition_penalty=1.23,
        seed=555,
        max_completion_tokens=77,
        xtc_probability=0.12,
        xtc_threshold=0.13,
        presence_penalty=0.14,
        repetition_context_size=15,
    )

    await endpoints_module.chat_completions(request, _make_raw_request(registry))

    assert captured_requests[0].temperature == pytest.approx(0.33)
    assert captured_requests[0].top_p == pytest.approx(0.44)
    assert captured_requests[0].top_k == 12
    assert captured_requests[0].min_p == pytest.approx(0.22)
    assert captured_requests[0].repetition_penalty == pytest.approx(1.23)
    assert captured_requests[0].seed == 555
    assert captured_requests[0].max_completion_tokens == 77
    assert captured_requests[0].xtc_probability == pytest.approx(0.12)
    assert captured_requests[0].xtc_threshold == pytest.approx(0.13)
    assert captured_requests[0].presence_penalty == pytest.approx(0.14)
    assert captured_requests[0].repetition_context_size == 15


@pytest.mark.asyncio
async def test_responses_can_apply_different_defaults_per_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Different models should be able to supply different omitted Responses defaults."""

    endpoints_module = _load_endpoints_module()

    handler_a = types.SimpleNamespace(
        handler_type="lm", _uses_model_sampling_defaults=True, **PER_MODEL_DEFAULTS_A
    )
    handler_b = types.SimpleNamespace(
        handler_type="lm", _uses_model_sampling_defaults=True, **PER_MODEL_DEFAULTS_B
    )
    for env_name, value in GLOBAL_ENV_DEFAULTS.items():
        monkeypatch.setenv(env_name, value)

    request_a = ResponsesRequest(
        model="model-a",
        input="hello",
        temperature=None,
        top_p=None,
        top_k=None,
        min_p=None,
        repetition_penalty=None,
        seed=None,
        max_output_tokens=None,
    )
    request_b = ResponsesRequest(
        model="model-b",
        input="hello",
        temperature=None,
        top_p=None,
        top_k=None,
        min_p=None,
        repetition_penalty=None,
        seed=None,
        max_output_tokens=None,
    )

    refined_request_a = endpoints_module.refine_responses_request(request_a, handler_a)
    refined_request_b = endpoints_module.refine_responses_request(request_b, handler_b)

    assert refined_request_a.temperature == pytest.approx(
        PER_MODEL_DEFAULTS_A["default_temperature"]
    )
    assert refined_request_a.top_p == pytest.approx(PER_MODEL_DEFAULTS_A["default_top_p"])
    assert refined_request_a.top_k == PER_MODEL_DEFAULTS_A["default_top_k"]
    assert refined_request_a.min_p == pytest.approx(PER_MODEL_DEFAULTS_A["default_min_p"])
    assert refined_request_a.repetition_penalty == pytest.approx(
        PER_MODEL_DEFAULTS_A["default_repetition_penalty"]
    )
    assert refined_request_a.seed == PER_MODEL_DEFAULTS_A["default_seed"]
    assert refined_request_a.max_output_tokens == PER_MODEL_DEFAULTS_A["default_max_tokens"]

    assert refined_request_b.temperature == pytest.approx(
        PER_MODEL_DEFAULTS_B["default_temperature"]
    )
    assert refined_request_b.top_p == pytest.approx(PER_MODEL_DEFAULTS_B["default_top_p"])
    assert refined_request_b.top_k == PER_MODEL_DEFAULTS_B["default_top_k"]
    assert refined_request_b.min_p == pytest.approx(PER_MODEL_DEFAULTS_B["default_min_p"])
    assert refined_request_b.repetition_penalty == pytest.approx(
        PER_MODEL_DEFAULTS_B["default_repetition_penalty"]
    )
    assert refined_request_b.seed == PER_MODEL_DEFAULTS_B["default_seed"]
    assert refined_request_b.max_output_tokens == PER_MODEL_DEFAULTS_B["default_max_tokens"]


@pytest.mark.asyncio
async def test_responses_single_model_implicit_handler_defaults_do_not_shadow_env_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Implicit single-model defaults should not override ambient env on Responses."""

    endpoints_module = _load_endpoints_module()

    handler = _single_model_handler_with_implicit_defaults()
    for env_name, value in GLOBAL_ENV_DEFAULTS.items():
        monkeypatch.setenv(env_name, value)

    request = ResponsesRequest(
        input="hello",
        temperature=None,
        top_p=None,
        top_k=None,
        min_p=None,
        repetition_penalty=None,
        seed=None,
        max_output_tokens=None,
    )

    refined_request = endpoints_module.refine_responses_request(request, handler)

    assert refined_request.temperature == pytest.approx(
        float(GLOBAL_ENV_DEFAULTS["DEFAULT_TEMPERATURE"])
    )
    assert refined_request.top_p == pytest.approx(float(GLOBAL_ENV_DEFAULTS["DEFAULT_TOP_P"]))
    assert refined_request.top_k == int(GLOBAL_ENV_DEFAULTS["DEFAULT_TOP_K"])
    assert refined_request.min_p == pytest.approx(float(GLOBAL_ENV_DEFAULTS["DEFAULT_MIN_P"]))
    assert refined_request.repetition_penalty == pytest.approx(
        float(GLOBAL_ENV_DEFAULTS["DEFAULT_REPETITION_PENALTY"])
    )
    assert refined_request.seed == int(GLOBAL_ENV_DEFAULTS["DEFAULT_SEED"])
    assert refined_request.max_output_tokens == int(GLOBAL_ENV_DEFAULTS["DEFAULT_MAX_TOKENS"])


@pytest.mark.asyncio
async def test_responses_omitted_model_uses_backward_compatible_fallback_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitted-model Responses requests should still fall back to app.state.handler."""

    endpoints_module = _load_endpoints_module()
    captured_requests: list[ResponsesRequest] = []
    captured_handlers: list[Any] = []

    async def _fake_process_text_responses_request(
        handler: Any,
        request: ResponsesRequest,
    ) -> JSONResponse:
        captured_handlers.append(handler)
        captured_requests.append(request.model_copy(deep=True))
        return JSONResponse(content={"ok": True})

    fallback_handler = types.SimpleNamespace(
        handler_type="lm",
        served_model_name="alias-a",
        _uses_model_sampling_defaults=True,
        **PER_MODEL_DEFAULTS_A,
    )
    registry_handler = types.SimpleNamespace(
        handler_type="lm", _uses_model_sampling_defaults=True, **PER_MODEL_DEFAULTS_B
    )
    registry = _FakeRegistry({"model-a": registry_handler})

    monkeypatch.setattr(
        endpoints_module, "process_text_responses_request", _fake_process_text_responses_request
    )

    request = ResponsesRequest(input="hello", model=None)
    response = await endpoints_module.responses_endpoint(
        request, _make_raw_request(registry, handler=fallback_handler)
    )

    assert isinstance(response, JSONResponse)
    assert captured_handlers == [fallback_handler]


@pytest.mark.asyncio
async def test_responses_omitted_model_does_not_fallback_to_non_text_handler() -> None:
    """Omitted Responses requests should not route to a non-text fallback handler."""

    endpoints_module = _load_endpoints_module()

    fallback_handler = types.SimpleNamespace(handler_type="whisper")
    registry_handler = types.SimpleNamespace(
        handler_type="lm", _uses_model_sampling_defaults=True, **PER_MODEL_DEFAULTS_A
    )
    registry = _FakeRegistry({"model-a": registry_handler})

    request = ResponsesRequest(input="hello", model=None)
    with pytest.raises(HTTPException) as exc_info:
        await endpoints_module.responses_endpoint(
            request, _make_raw_request(registry, handler=fallback_handler)
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail["error"]["type"] == "model_not_found"


@pytest.mark.asyncio
async def test_responses_omitted_model_non_text_fallback_uses_on_demand_default_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitted Responses should resolve ``local-text-model`` via on-demand load."""

    endpoints_module = _load_endpoints_module()
    captured_handlers: list[Any] = []

    async def _fake_process_text_responses_request(
        handler: Any,
        request: ResponsesRequest,
    ) -> JSONResponse:
        del request
        captured_handlers.append(handler)
        return JSONResponse(content={"ok": True})

    fallback_handler = types.SimpleNamespace(handler_type="whisper")
    on_demand_text_handler = types.SimpleNamespace(
        handler_type="lm",
        served_model_name=Config.TEXT_MODEL,
        model_path="mlx-community/model-text",
        debug=False,
    )
    registry = _FakeRegistry(
        {"model-a": types.SimpleNamespace(handler_type="lm")},
        on_demand_handlers={Config.TEXT_MODEL: on_demand_text_handler},
    )

    monkeypatch.setattr(
        endpoints_module,
        "process_text_responses_request",
        _fake_process_text_responses_request,
    )

    request = ResponsesRequest(input="hello", model=None)
    response = await endpoints_module.responses_endpoint(
        request, _make_raw_request(registry, handler=fallback_handler)
    )

    assert isinstance(response, JSONResponse)
    assert response.status_code == 200
    assert captured_handlers == [on_demand_text_handler]


@pytest.mark.asyncio
async def test_chat_completions_omitted_model_uses_backward_compatible_fallback_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitted-model chat requests should still fall back to app.state.handler."""

    endpoints_module = _load_endpoints_module()
    captured_requests: list[ChatCompletionRequest] = []
    captured_handlers: list[Any] = []

    async def _fake_process_text_request(
        handler: Any,
        request: ChatCompletionRequest,
        request_id: str | None = None,
    ) -> JSONResponse:
        del request_id
        captured_handlers.append(handler)
        captured_requests.append(request.model_copy(deep=True))
        return JSONResponse(content={"ok": True})

    fallback_handler = types.SimpleNamespace(
        handler_type="lm", _uses_model_sampling_defaults=True, **PER_MODEL_DEFAULTS_A
    )
    registry_handler = types.SimpleNamespace(
        handler_type="lm", _uses_model_sampling_defaults=True, **PER_MODEL_DEFAULTS_B
    )
    registry = _FakeRegistry({"model-a": registry_handler})

    monkeypatch.setattr(endpoints_module, "process_text_request", _fake_process_text_request)

    request = ChatCompletionRequest(messages=[Message(role="user", content="hello")])
    response = await endpoints_module.chat_completions(
        request, _make_raw_request(registry, handler=fallback_handler)
    )

    assert isinstance(response, JSONResponse)
    assert captured_handlers == [fallback_handler]


@pytest.mark.asyncio
async def test_chat_completions_omitted_model_does_not_fallback_to_non_chat_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitted chat requests should not route to a non-chat fallback handler."""

    endpoints_module = _load_endpoints_module()
    captured_handlers: list[Any] = []

    async def _fake_process_text_request(
        handler: Any,
        request: ChatCompletionRequest,
        request_id: str | None = None,
    ) -> JSONResponse:
        del request, request_id
        captured_handlers.append(handler)
        return JSONResponse(content={"ok": True})

    fallback_handler = types.SimpleNamespace(handler_type="embeddings")
    registry_handler = types.SimpleNamespace(
        handler_type="lm", _uses_model_sampling_defaults=True, **PER_MODEL_DEFAULTS_A
    )
    registry = _FakeRegistry({"model-a": registry_handler})

    monkeypatch.setattr(endpoints_module, "process_text_request", _fake_process_text_request)

    request = ChatCompletionRequest(messages=[Message(role="user", content="hello")])

    with pytest.raises(HTTPException) as exc_info:
        await endpoints_module.chat_completions(
            request, _make_raw_request(registry, handler=fallback_handler)
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail["error"]["type"] == "model_not_found"
    assert captured_handlers == []


@pytest.mark.asyncio
async def test_chat_completions_omitted_model_reports_resolved_fallback_model_id() -> None:
    """Omitted-model chat requests should report the resolved fallback handler's model id."""

    endpoints_module = _load_endpoints_module()

    async def _fake_generate_text_response(_chat_request: Any) -> dict[str, Any]:
        return {"response": "hello", "usage": None}

    fallback_handler = types.SimpleNamespace(
        handler_type="lm",
        served_model_name="alias-a",
        model_path="mlx-community/model-a-4bit",
        debug=False,
        generate_text_response=_fake_generate_text_response,
    )
    registry = _FakeRegistry({"alias-a": fallback_handler})

    request = ChatCompletionRequest(messages=[Message(role="user", content="hello")])
    response = await endpoints_module.chat_completions(
        request, _make_raw_request(registry, handler=fallback_handler)
    )

    assert isinstance(response, JSONResponse)
    payload = json.loads(response.body)
    assert payload["model"] == "alias-a"


@pytest.mark.asyncio
async def test_chat_completions_stream_omitted_model_reports_resolved_fallback_model_id() -> None:
    """Streamed omitted-model chat requests should report the resolved fallback handler id."""

    endpoints_module = _load_endpoints_module()

    async def _fake_generate_text_stream(_chat_request: Any) -> Any:
        yield "hello"

    fallback_handler = types.SimpleNamespace(
        handler_type="lm",
        served_model_name="alias-a",
        model_path="mlx-community/model-a-4bit",
        debug=False,
        generate_text_stream=_fake_generate_text_stream,
    )
    registry = _FakeRegistry({"alias-a": fallback_handler})

    request = ChatCompletionRequest(messages=[Message(role="user", content="hello")], stream=True)
    response = await endpoints_module.chat_completions(
        request, _make_raw_request(registry, handler=fallback_handler)
    )

    assert isinstance(response, StreamingResponse)
    body_chunks = [
        chunk.decode() if isinstance(chunk, bytes) else chunk
        async for chunk in response.body_iterator
    ]

    events: list[dict[str, Any]] = []
    for item in "".join(body_chunks).split("\n\n"):
        if not item.startswith("data: "):
            continue
        payload = item.removeprefix("data: ")
        if payload == "[DONE]":
            continue
        events.append(json.loads(payload))

    model_values = [event["model"] for event in events if "model" in event]
    assert model_values
    assert set(model_values) == {"alias-a"}


@pytest.mark.asyncio
async def test_chat_completions_explicit_legacy_alias_without_registry_entry_still_404s() -> None:
    """Explicit ``local-text-model`` should still be treated as an invalid model id."""

    endpoints_module = _load_endpoints_module()

    async def _fake_generate_text_response(_chat_request: Any) -> dict[str, Any]:
        return {"response": "hello", "usage": None}

    fallback_handler = types.SimpleNamespace(
        handler_type="lm",
        served_model_name="alias-a",
        model_path="mlx-community/model-a-4bit",
        debug=False,
        generate_text_response=_fake_generate_text_response,
    )
    registry = _FakeRegistry({"alias-a": fallback_handler})

    request = ChatCompletionRequest(
        model=Config.TEXT_MODEL,
        messages=[Message(role="user", content="hello")],
    )

    with pytest.raises(HTTPException) as exc_info:
        await endpoints_module.chat_completions(
            request, _make_raw_request(registry, handler=fallback_handler)
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail["error"]["type"] == "model_not_found"


@pytest.mark.asyncio
async def test_responses_omitted_model_reports_resolved_fallback_model_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitted-model Responses should report the resolved fallback handler's model id."""

    endpoints_module = _load_endpoints_module()

    async def _fake_generate_text_response(_chat_request: Any) -> dict[str, Any]:
        return {"response": "hello", "usage": None}

    fallback_handler = types.SimpleNamespace(
        handler_type="lm",
        served_model_name="alias-a",
        model_path="mlx-community/model-a-4bit",
        debug=False,
        generate_text_response=_fake_generate_text_response,
    )
    registry = _FakeRegistry({"alias-a": fallback_handler})

    request = ResponsesRequest(input="hello", model=None)
    response = await endpoints_module.responses_endpoint(
        request, _make_raw_request(registry, handler=fallback_handler)
    )

    assert isinstance(response, JSONResponse)
    payload = json.loads(response.body)
    assert payload["model"] == "alias-a"


@pytest.mark.asyncio
async def test_responses_stream_omitted_model_reports_resolved_fallback_model_id() -> None:
    """Streamed omitted-model Responses should report the resolved fallback handler id."""

    endpoints_module = _load_endpoints_module()

    async def _fake_generate_text_stream(_chat_request: Any) -> Any:
        yield "hello"

    fallback_handler = types.SimpleNamespace(
        handler_type="lm",
        served_model_name="alias-a",
        model_path="mlx-community/model-a-4bit",
        debug=False,
        generate_text_stream=_fake_generate_text_stream,
    )
    registry = _FakeRegistry({"alias-a": fallback_handler})

    request = ResponsesRequest(input="hello", model=None, stream=True)
    response = await endpoints_module.responses_endpoint(
        request, _make_raw_request(registry, handler=fallback_handler)
    )

    assert isinstance(response, StreamingResponse)
    body_chunks = [
        chunk.decode() if isinstance(chunk, bytes) else chunk
        async for chunk in response.body_iterator
    ]

    events_by_type: dict[str, dict[str, Any]] = {}
    for item in "".join(body_chunks).split("\n\n"):
        for line in item.splitlines():
            if not line.startswith("data: "):
                continue
            payload = line.removeprefix("data: ")
            event = json.loads(payload)
            event_type = event.get("type")
            if event_type in {"response.created", "response.completed"}:
                events_by_type[event_type] = event

    assert events_by_type["response.created"]["response"]["model"] == "alias-a"
    assert events_by_type["response.completed"]["response"]["model"] == "alias-a"


@pytest.mark.asyncio
async def test_responses_explicit_legacy_alias_without_registry_entry_still_404s() -> None:
    """Explicit ``local-text-model`` should still 404 for Responses when not registered."""

    endpoints_module = _load_endpoints_module()

    async def _fake_generate_text_response(_chat_request: Any) -> dict[str, Any]:
        return {"response": "hello", "usage": None}

    fallback_handler = types.SimpleNamespace(
        handler_type="lm",
        served_model_name="alias-a",
        model_path="mlx-community/model-a-4bit",
        debug=False,
        generate_text_response=_fake_generate_text_response,
    )
    registry = _FakeRegistry({"alias-a": fallback_handler})

    request = ResponsesRequest(input="hello", model=Config.TEXT_MODEL)

    with pytest.raises(HTTPException) as exc_info:
        await endpoints_module.responses_endpoint(
            request, _make_raw_request(registry, handler=fallback_handler)
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail["error"]["type"] == "model_not_found"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "registry_case",
    ["no-registry", "empty-registry", "other-model-only", "foreign-owned-handler-id"],
    ids=["no-registry", "empty-registry", "other-model-only", "foreign-owned-handler-id"],
)
async def test_responses_single_model_omitted_model_preserves_legacy_default_alias(
    registry_case: str,
) -> None:
    """Single-model omitted Responses requests should still report ``local-text-model``.

    The real single-model boot path does not currently attach
    ``handler.served_model_name``. This test includes one anyway so the contract
    stays pinned if a future handler or proxy refactor adds that field.
    """

    endpoints_module = _load_endpoints_module()

    async def _fake_generate_text_response(_chat_request: Any) -> dict[str, Any]:
        return {"response": "hello", "usage": None}

    # Guard against future single-model handler/proxy refactors that
    # expose a concrete model_id while the API contract stays legacy.
    single_handler = types.SimpleNamespace(
        handler_type="lm",
        served_model_name="actual-single-id",
        model_path="actual/model/path",
        debug=False,
        generate_text_response=_fake_generate_text_response,
    )
    registry = _single_model_registry_case(registry_case, single_handler)

    request = ResponsesRequest(input="hello", model=None)
    response = await endpoints_module.responses_endpoint(
        request, _make_raw_request(registry, handler=single_handler)
    )

    assert isinstance(response, JSONResponse)
    payload = json.loads(response.body)
    assert payload["model"] == Config.TEXT_MODEL


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "registry_case",
    ["no-registry", "empty-registry", "other-model-only", "foreign-owned-handler-id"],
    ids=["no-registry", "empty-registry", "other-model-only", "foreign-owned-handler-id"],
)
async def test_responses_stream_single_model_omitted_model_preserves_legacy_default_alias(
    registry_case: str,
) -> None:
    """Streamed single-model omitted Responses requests should still report ``local-text-model``.

    The real single-model boot path does not currently attach
    ``handler.served_model_name``. This test includes one anyway so the stream
    path stays aligned if that changes later.
    """

    endpoints_module = _load_endpoints_module()

    async def _fake_generate_text_stream(_chat_request: Any) -> Any:
        yield "hello"

    single_handler = types.SimpleNamespace(
        handler_type="lm",
        served_model_name="actual-single-id",
        model_path="actual/model/path",
        debug=False,
        generate_text_stream=_fake_generate_text_stream,
    )
    registry = _single_model_registry_case(registry_case, single_handler)

    request = ResponsesRequest(input="hello", model=None, stream=True)
    response = await endpoints_module.responses_endpoint(
        request, _make_raw_request(registry, handler=single_handler)
    )

    assert isinstance(response, StreamingResponse)
    body_chunks = [
        chunk.decode() if isinstance(chunk, bytes) else chunk
        async for chunk in response.body_iterator
    ]

    events_by_type: dict[str, dict[str, Any]] = {}
    for item in "".join(body_chunks).split("\n\n"):
        for line in item.splitlines():
            if not line.startswith("data: "):
                continue
            payload = line.removeprefix("data: ")
            event = json.loads(payload)
            event_type = event.get("type")
            if event_type in {"response.created", "response.completed"}:
                events_by_type[event_type] = event

    assert events_by_type["response.created"]["response"]["model"] == Config.TEXT_MODEL
    assert events_by_type["response.completed"]["response"]["model"] == Config.TEXT_MODEL


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "registry_case",
    ["no-registry", "empty-registry", "other-model-only", "foreign-owned-handler-id"],
    ids=["no-registry", "empty-registry", "other-model-only", "foreign-owned-handler-id"],
)
async def test_chat_completions_single_model_omitted_model_preserves_legacy_default_alias(
    registry_case: str,
) -> None:
    """Single-model omitted chat requests should still report ``local-text-model``."""

    endpoints_module = _load_endpoints_module()

    async def _fake_generate_text_response(_chat_request: Any) -> dict[str, Any]:
        return {"response": "hello", "usage": None}

    single_handler = types.SimpleNamespace(
        handler_type="lm",
        served_model_name="actual-single-id",
        model_path="actual/model/path",
        debug=False,
        generate_text_response=_fake_generate_text_response,
    )
    registry = _single_model_registry_case(registry_case, single_handler)

    request = ChatCompletionRequest(messages=[Message(role="user", content="hello")])
    response = await endpoints_module.chat_completions(
        request, _make_raw_request(registry, handler=single_handler)
    )

    assert isinstance(response, JSONResponse)
    payload = json.loads(response.body)
    assert payload["model"] == Config.TEXT_MODEL


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "registry_case",
    ["no-registry", "empty-registry", "other-model-only", "foreign-owned-handler-id"],
    ids=["no-registry", "empty-registry", "other-model-only", "foreign-owned-handler-id"],
)
async def test_chat_completions_stream_single_model_omitted_model_preserves_legacy_default_alias(
    registry_case: str,
) -> None:
    """Streamed single-model omitted chat requests should still report ``local-text-model``."""

    endpoints_module = _load_endpoints_module()

    async def _fake_generate_text_stream(_chat_request: Any) -> Any:
        yield "hello"

    single_handler = types.SimpleNamespace(
        handler_type="lm",
        served_model_name="actual-single-id",
        model_path="actual/model/path",
        debug=False,
        generate_text_stream=_fake_generate_text_stream,
    )
    registry = _single_model_registry_case(registry_case, single_handler)

    request = ChatCompletionRequest(messages=[Message(role="user", content="hello")], stream=True)
    response = await endpoints_module.chat_completions(
        request, _make_raw_request(registry, handler=single_handler)
    )

    assert isinstance(response, StreamingResponse)
    body_chunks = [
        chunk.decode() if isinstance(chunk, bytes) else chunk
        async for chunk in response.body_iterator
    ]

    events: list[dict[str, Any]] = []
    for item in "".join(body_chunks).split("\n\n"):
        if not item.startswith("data: "):
            continue
        payload = item.removeprefix("data: ")
        if payload == "[DONE]":
            continue
        events.append(json.loads(payload))

    model_values = [event["model"] for event in events if "model" in event]
    assert model_values
    assert set(model_values) == {Config.TEXT_MODEL}
