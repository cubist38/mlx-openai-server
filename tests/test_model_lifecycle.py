"""Tests for on-demand model lifecycle management."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from app.core.keep_alive import parse_keep_alive
from app.core.model_registry import ModelRegistry
from app.schemas.openai import Delta, Message


class _FakeHandlerProcessProxy:
    """In-memory stand-in for a model subprocess."""

    instances: ClassVar[list[_FakeHandlerProcessProxy]] = []

    def __init__(
        self,
        model_cfg_dict: dict[str, Any],
        model_type: str,
        model_path: str,
        served_model_name: str,
    ) -> None:
        self.model_cfg_dict = model_cfg_dict
        self.handler_type = model_type
        self.model_path = model_path
        self.served_model_name = served_model_name
        self.started = False
        self.cleaned = False
        self.cleanup_error: Exception | None = None
        self.pid = 1000 + len(self.instances)
        self.instances.append(self)

    async def start(self, queue_config: dict[str, Any]) -> None:
        """Record process startup."""
        self.queue_config = queue_config
        self.started = True

    async def cleanup(self) -> None:
        """Record process cleanup."""
        if self.cleanup_error is not None:
            raise self.cleanup_error
        self.cleaned = True


async def _register(
    registry: ModelRegistry,
    model_id: str,
    idle_timeout: str | float = 300,
) -> None:
    """Register one test model."""
    await registry.register_on_demand_model(
        model_id=model_id,
        model_cfg_dict={"model_path": f"/models/{model_id}"},
        model_type="lm",
        model_path=f"/models/{model_id}",
        context_length=4096,
        queue_config={"timeout": 30, "queue_size": 4},
        idle_timeout=idle_timeout,
    )


@pytest.fixture(autouse=True)
def fake_process_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace model subprocesses with deterministic in-memory handlers."""
    from app.core import handler_process

    _FakeHandlerProcessProxy.instances.clear()
    monkeypatch.setattr(handler_process, "HandlerProcessProxy", _FakeHandlerProcessProxy)


@pytest.mark.parametrize(
    ("value", "default", "expected"),
    [
        (None, 300.0, 300.0),
        (90, 300.0, 90.0),
        ("90", 300.0, 90.0),
        ("1h30m", 300.0, 5400.0),
        ("250ms", 300.0, 0.25),
        (-1, 300.0, None),
        ("-1", 300.0, None),
    ],
)
def test_parse_keep_alive(
    value: str | int | None,
    default: float,
    expected: float | None,
) -> None:
    """Supported durations should normalize to seconds."""
    assert parse_keep_alive(value, default) == expected


@pytest.mark.parametrize("value", ["", "1w", "1m bad", True, float("inf")])
def test_parse_keep_alive_rejects_invalid_values(value: Any) -> None:
    """Invalid retention values should fail before model loading."""
    with pytest.raises(ValueError):
        parse_keep_alive(value, 300.0)


def test_multi_model_config_parses_lifecycle_options(tmp_path: Path) -> None:
    """YAML should accept server capacity and duration strings."""
    from app.config import load_config_from_yaml

    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        """
server:
  max_loaded_models: 2
  model_load_timeout: 45
models:
  - model_path: /models/alpha
    served_model_name: alpha
    on_demand: true
    on_demand_idle_timeout: 5m
""".strip()
    )

    config = load_config_from_yaml(str(config_path))

    assert config.max_loaded_models == 2
    assert config.model_load_timeout == 45
    assert config.models[0].on_demand_idle_timeout == "5m"


def test_model_config_rejects_invalid_idle_duration() -> None:
    """Invalid model retention values should fail while parsing configuration."""
    from app.config import ModelEntryConfig

    with pytest.raises(ValueError, match="Invalid keep_alive duration"):
        ModelEntryConfig(
            model_path="/models/alpha",
            on_demand=True,
            on_demand_idle_timeout="5fortnights",
        )


def test_reasoning_response_aliases_are_synchronized() -> None:
    """Streaming and final messages should expose both reasoning field names."""
    message = Message(role="assistant", reasoning_content="analysis")
    delta = Delta(reasoning="streamed analysis")

    assert message.reasoning == "analysis"
    assert delta.reasoning_content == "streamed analysis"


@pytest.mark.asyncio
async def test_every_request_acquires_a_model_lease() -> None:
    """Already-loaded models must count every concurrent request."""
    registry = ModelRegistry()
    await _register(registry, "alpha")

    first = await registry.ensure_on_demand_loaded("alpha")
    second = await registry.ensure_on_demand_loaded("alpha")

    assert first is second
    assert registry.get_model_status()[0]["active_requests"] == 2

    await registry.release_on_demand("alpha", keep_alive=0)
    assert registry.get_model_status()[0]["active_requests"] == 1
    assert registry.get_model_status()[0]["loaded"] is True

    await registry.release_on_demand("alpha", keep_alive=0)
    assert registry.get_model_status()[0]["loaded"] is False
    assert first.cleaned is True


@pytest.mark.asyncio
async def test_idle_timeout_unloads_model() -> None:
    """An idle model should unload after its configured TTL."""
    registry = ModelRegistry()
    await _register(registry, "alpha", idle_timeout=0.01)

    handler = await registry.ensure_on_demand_loaded("alpha")
    await registry.release_on_demand("alpha")
    await asyncio.sleep(0.03)

    assert registry.get_model_status()[0]["loaded"] is False
    assert handler.cleaned is True


@pytest.mark.asyncio
async def test_negative_keep_alive_retains_idle_model() -> None:
    """Negative per-request retention should disable expiry."""
    registry = ModelRegistry()
    await _register(registry, "alpha", idle_timeout=0.01)

    await registry.ensure_on_demand_loaded("alpha")
    await registry.release_on_demand("alpha", keep_alive=-1)
    await asyncio.sleep(0.03)

    status = registry.get_model_status()[0]
    assert status["loaded"] is True
    assert status["expires_at"] is None
    await registry.unload_on_demand("alpha")


@pytest.mark.asyncio
async def test_lru_evicts_only_idle_models() -> None:
    """Capacity pressure should evict the least recently used idle process."""
    registry = ModelRegistry(max_loaded_models=2)
    for model_id in ("alpha", "beta", "gamma"):
        await _register(registry, model_id)

    alpha = await registry.ensure_on_demand_loaded("alpha")
    await registry.release_on_demand("alpha", keep_alive=-1)
    beta = await registry.ensure_on_demand_loaded("beta")
    await registry.release_on_demand("beta", keep_alive=-1)

    await registry.ensure_on_demand_loaded("gamma")
    status = {item["id"]: item for item in registry.get_model_status()}

    assert status["alpha"]["loaded"] is False
    assert status["beta"]["loaded"] is True
    assert status["gamma"]["state"] == "busy"
    assert alpha.cleaned is True
    assert beta.cleaned is False
    await registry.release_on_demand("gamma", keep_alive=0)
    await registry.unload_on_demand("beta")


@pytest.mark.asyncio
async def test_load_waits_while_all_slots_are_active() -> None:
    """A request should wait instead of evicting an actively streaming model."""
    registry = ModelRegistry(max_loaded_models=1, model_load_timeout=1)
    await _register(registry, "alpha")
    await _register(registry, "beta")

    alpha = await registry.ensure_on_demand_loaded("alpha")
    beta_task = asyncio.create_task(registry.ensure_on_demand_loaded("beta"))
    await asyncio.sleep(0.01)

    assert beta_task.done() is False
    assert alpha.cleaned is False

    await registry.release_on_demand("alpha", keep_alive=-1)
    beta = await asyncio.wait_for(beta_task, timeout=0.5)

    assert alpha.cleaned is True
    assert beta.served_model_name == "beta"
    await registry.release_on_demand("beta", keep_alive=0)


@pytest.mark.asyncio
async def test_explicit_unload_refuses_active_model() -> None:
    """The safe unload path must not interrupt an active request."""
    registry = ModelRegistry()
    await _register(registry, "alpha")
    await registry.ensure_on_demand_loaded("alpha")

    with pytest.raises(ValueError, match="active request"):
        await registry.unload_on_demand("alpha")

    await registry.release_on_demand("alpha", keep_alive=0)


@pytest.mark.asyncio
async def test_failed_cleanup_keeps_model_tracked_for_retry() -> None:
    """A failed process cleanup must not discard lifecycle state."""
    registry = ModelRegistry()
    await _register(registry, "alpha")
    handler = await registry.ensure_on_demand_loaded("alpha")
    handler.cleanup_error = RuntimeError("worker did not stop")

    with pytest.raises(RuntimeError, match="worker did not stop"):
        await registry.release_on_demand("alpha", keep_alive=0)

    status = registry.get_model_status()[0]
    assert status["loaded"] is True
    assert status["active_requests"] == 0
    assert status["last_error"] == "Unload failed: worker did not stop"

    handler.cleanup_error = None
    assert await registry.unload_on_demand("alpha") is True


@pytest.mark.asyncio
async def test_endpoint_resolution_acquires_loaded_on_demand_model() -> None:
    """Endpoint routing must acquire a lease even for a resident worker."""
    from app.api.endpoints import _release_on_demand, _resolve_handler

    registry = ModelRegistry()
    await _register(registry, "alpha")
    first_request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(registry=registry)),
        state=SimpleNamespace(),
    )
    second_request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(registry=registry)),
        state=SimpleNamespace(),
    )

    first = await _resolve_handler(first_request, model_id="alpha")
    second = await _resolve_handler(second_request, model_id="alpha")

    assert first is second
    assert registry.get_model_status()[0]["active_requests"] == 2

    await _release_on_demand(first_request)
    await _release_on_demand(second_request)
    assert registry.get_model_status()[0]["active_requests"] == 0
    await registry.unload_on_demand("alpha")


@pytest.mark.asyncio
async def test_model_management_endpoints() -> None:
    """Load, status, and unload endpoints should expose process lifecycle."""
    from app.api.endpoints import load_model, model_status, unload_model
    from app.schemas.openai import ModelLoadRequest, ModelUnloadRequest

    registry = ModelRegistry()
    await _register(registry, "alpha")
    raw_request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(registry=registry)),
        state=SimpleNamespace(),
    )

    loaded = await load_model(
        ModelLoadRequest(model="alpha", keep_alive=-1),
        raw_request,
    )
    status = await model_status(raw_request)
    unloaded = await unload_model(ModelUnloadRequest(model="alpha"), raw_request)

    assert loaded["status"] == "loaded"
    assert status["configured"] == 1
    assert status["loaded"] == 1
    assert status["data"][0]["pid"] == 1000
    assert unloaded == {"status": "unloaded", "model": "alpha"}


@pytest.mark.asyncio
async def test_stream_release_runs_only_after_body_closes() -> None:
    """A streaming response should hold its model lease until disconnect."""
    from fastapi.responses import StreamingResponse

    from app.api.endpoints import _defer_release_until_stream_end

    class _Registry:
        def __init__(self) -> None:
            self.releases: list[tuple[str, Any]] = []

        async def release_on_demand(
            self,
            model_id: str,
            keep_alive: Any = None,
            handler: Any = None,
        ) -> None:
            self.releases.append((model_id, keep_alive))

    async def body() -> Any:
        yield b"first"
        yield b"second"

    registry = _Registry()
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(registry=registry)),
        state=SimpleNamespace(on_demand_leases=[("alpha", "5m", None)]),
    )
    response = _defer_release_until_stream_end(StreamingResponse(body()), request)
    iterator = response.body_iterator

    assert await iterator.__anext__() == b"first"
    assert registry.releases == []

    await iterator.aclose()
    assert registry.releases == [("alpha", "5m")]


@pytest.mark.asyncio
async def test_client_disconnect_still_releases_the_model_lease() -> None:
    """A dropped stream must not leave the worker permanently busy.

    A disconnect cancels the response task group, so an unshielded release
    would be abandoned at its first suspension point and the reference count
    would never return to zero.
    """
    from fastapi.responses import StreamingResponse

    from app.api.endpoints import _defer_release_until_stream_end

    class _SlowRegistry:
        def __init__(self) -> None:
            self.releases: list[str] = []

        async def release_on_demand(
            self,
            model_id: str,
            keep_alive: Any = None,
            handler: Any = None,
        ) -> None:
            # Stand in for a contended lock or a worker shutdown: a real
            # suspension point inside the release path.
            await asyncio.sleep(0)
            self.releases.append(model_id)

    async def body() -> Any:
        for index in range(100):
            yield f"chunk-{index}".encode()
            await asyncio.sleep(0.01)

    registry = _SlowRegistry()
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(registry=registry)),
        state=SimpleNamespace(on_demand_leases=[("alpha", "5m", None)]),
    )
    response = _defer_release_until_stream_end(StreamingResponse(body()), request)

    sent: list[dict[str, Any]] = []
    dropped = asyncio.Event()

    async def receive() -> dict[str, Any]:
        await dropped.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)
        if len([item for item in sent if item["type"] == "http.response.body"]) == 2:
            dropped.set()

    await response({"type": "http", "method": "GET", "path": "/", "headers": []}, receive, send)
    await asyncio.sleep(0.05)

    assert registry.releases == ["alpha"]


@pytest.mark.asyncio
async def test_stale_lease_does_not_release_a_replacement_worker() -> None:
    """A lease orphaned by a forced unload must not free its replacement."""
    registry = ModelRegistry()
    await _register(registry, "alpha")
    orphaned = await registry.ensure_on_demand_loaded("alpha")

    assert await registry.unload_on_demand("alpha", force=True) is True

    replacement = await registry.ensure_on_demand_loaded("alpha")
    assert replacement is not orphaned

    await registry.release_on_demand("alpha", handler=orphaned)
    assert registry.get_model_status()[0]["active_requests"] == 1

    await registry.release_on_demand("alpha", keep_alive=0, handler=replacement)
    assert registry.get_model_status()[0]["loaded"] is False


@pytest.mark.asyncio
async def test_health_reports_machine_readable_status_and_counts() -> None:
    """``model_status`` must stay a fixed token, with the detail in counts.

    Monitoring and the live contract test both treat ``model_status`` as an
    enumerated value, so the loaded/configured detail belongs in its own fields
    rather than in an interpolated sentence.
    """
    from app.api.endpoints import health

    registry = ModelRegistry(max_loaded_models=2)
    await _register(registry, "alpha")
    await _register(registry, "beta")
    raw_request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(registry=registry)),
        state=SimpleNamespace(),
    )

    cold = await health(raw_request)
    assert cold.model_status == "ready"
    assert (cold.models_configured, cold.models_loaded) == (2, 0)

    await registry.ensure_on_demand_loaded("alpha")
    warm = await health(raw_request)

    assert warm.model_status == "ready"
    assert (warm.models_configured, warm.models_loaded) == (2, 1)
    await registry.release_on_demand("alpha", keep_alive=0)


@pytest.mark.asyncio
async def test_model_metadata_exposes_backend_and_context_length() -> None:
    """``/v1/models`` metadata must carry the fields clients rely on."""
    registry = ModelRegistry()
    await registry.register_on_demand_model(
        model_id="alpha",
        model_cfg_dict={"model_path": "/models/alpha"},
        model_type="lm",
        model_path="/models/alpha",
        context_length=8192,
        queue_config={"timeout": 30, "queue_size": 4},
        idle_timeout=300,
    )

    metadata = registry.list_models()[0]["metadata"]

    assert metadata["backend"] == "mlx"
    assert metadata["context_length"] == 8192


@pytest.mark.asyncio
async def test_load_endpoint_reports_the_settled_state() -> None:
    """``keep_alive=0`` releases straight away, so the reply must say so."""
    from app.api.endpoints import load_model
    from app.schemas.openai import ModelLoadRequest

    registry = ModelRegistry()
    await _register(registry, "alpha")
    raw_request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(registry=registry)),
        state=SimpleNamespace(),
    )

    result = await load_model(ModelLoadRequest(model="alpha", keep_alive=0), raw_request)

    assert result["status"] == "unloaded"
    assert result["model"]["loaded"] is False
