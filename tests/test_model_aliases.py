"""Tests for model versioning and aliases.

Aliases are resolved inside the registry rather than at each endpoint, so these
tests assert both that every registry entry point accepts an alias and that an
alias shares state with its canonical model instead of shadowing it.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from app.config import VALID_MODEL_TYPES, ModelEntryConfig, load_config_from_yaml
from app.core.model_registry import ModelRegistry


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
        self.cleaned = False
        self.pid = 2000 + len(self.instances)
        self.instances.append(self)

    async def start(self, queue_config: dict[str, Any]) -> None:
        """Record process startup."""
        self.queue_config = queue_config

    async def cleanup(self) -> None:
        """Record process cleanup."""
        self.cleaned = True


@pytest.fixture(autouse=True)
def fake_process_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace model subprocesses with deterministic in-memory handlers."""
    from app.core import handler_process

    _FakeHandlerProcessProxy.instances.clear()
    monkeypatch.setattr(handler_process, "HandlerProcessProxy", _FakeHandlerProcessProxy)


async def _register(
    registry: ModelRegistry,
    model_id: str,
    aliases: list[str] | None = None,
    version: str | None = None,
) -> None:
    """Register one on-demand test model."""
    await registry.register_on_demand_model(
        model_id=model_id,
        model_cfg_dict={"model_path": f"/models/{model_id}"},
        model_type="lm",
        model_path=f"/models/{model_id}",
        context_length=4096,
        queue_config={"timeout": 30, "queue_size": 4},
        idle_timeout=300,
        aliases=aliases,
        version=version,
    )


# ----------------------------------------------------------------------
# Configuration layer
# ----------------------------------------------------------------------


def test_alias_names_combines_version_and_explicit_aliases() -> None:
    """A declared version becomes a ``name:version`` alias alongside the rest."""
    entry = ModelEntryConfig(
        model_path="/models/coder",
        served_model_name="coder",
        version="2.1",
        aliases=["coder-stable", "default-coder"],
    )

    assert entry.alias_names() == ["coder:2.1", "coder-stable", "default-coder"]


def test_alias_names_excludes_canonical_name_and_duplicates() -> None:
    """Redundant names are dropped so registration stays a no-op for them."""
    entry = ModelEntryConfig(
        model_path="/models/coder",
        served_model_name="coder",
        aliases=["coder", "twin", "twin"],
    )

    assert entry.alias_names() == ["twin"]


def test_alias_names_is_empty_without_version_or_aliases() -> None:
    """Models that declare nothing extra register no aliases."""
    entry = ModelEntryConfig(model_path="/models/coder", served_model_name="coder")

    assert entry.alias_names() == []


def test_version_is_appended_to_the_resolved_default_name() -> None:
    """``served_model_name`` defaults to ``model_path``, and the tag follows it."""
    entry = ModelEntryConfig(model_path="org/repo", version="3")

    assert entry.alias_names() == ["org/repo:3"]


@pytest.mark.parametrize("version", ["", "   ", "1 0", "v1:2"])
def test_invalid_version_is_rejected(version: str) -> None:
    """A version must be a non-empty token without whitespace or ``:``."""
    with pytest.raises(ValueError, match="version"):
        ModelEntryConfig(model_path="/models/coder", version=version)


@pytest.mark.parametrize("alias", ["", "  ", "two words"])
def test_invalid_alias_is_rejected(alias: str) -> None:
    """An unusable alias fails at configuration time, not at request time."""
    with pytest.raises(ValueError, match="alias"):
        ModelEntryConfig(model_path="/models/coder", aliases=[alias])


def test_alias_may_contain_a_colon() -> None:
    """Explicit ``name:tag`` aliases are how a bare tag is pointed at a model."""
    entry = ModelEntryConfig(
        model_path="/models/coder",
        served_model_name="coder-v2",
        aliases=["coder:stable"],
    )

    assert entry.alias_names() == ["coder:stable"]


def test_yaml_config_parses_version_and_aliases(tmp_path: Path) -> None:
    """Both fields round-trip from YAML into the model entry."""
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        """
models:
  - model_path: /models/alpha
    served_model_name: alpha-v2
    version: "2"
    aliases: [alpha, alpha-latest]
""".strip()
    )

    config = load_config_from_yaml(str(config_path))
    entry = config.models[0]

    assert entry.version == "2"
    assert entry.aliases == ["alpha", "alpha-latest"]
    assert entry.alias_names() == ["alpha-v2:2", "alpha", "alpha-latest"]


def test_yaml_rejects_alias_colliding_with_another_model_name(tmp_path: Path) -> None:
    """An alias that shadows a real model would make one of them unreachable."""
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        """
models:
  - model_path: /models/alpha
    served_model_name: alpha
  - model_path: /models/beta
    served_model_name: beta
    aliases: [alpha]
""".strip()
    )

    with pytest.raises(ValueError, match="Alias 'alpha'"):
        load_config_from_yaml(str(config_path))


def test_yaml_rejects_model_name_colliding_with_an_earlier_alias(tmp_path: Path) -> None:
    """The collision is caught regardless of declaration order."""
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        """
models:
  - model_path: /models/beta
    served_model_name: beta
    aliases: [alpha]
  - model_path: /models/alpha
    served_model_name: alpha
""".strip()
    )

    with pytest.raises(ValueError, match="Duplicate served_model_name 'alpha'"):
        load_config_from_yaml(str(config_path))


def test_yaml_rejects_the_same_alias_on_two_models(tmp_path: Path) -> None:
    """Two models cannot share an alias; routing would be ambiguous."""
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        """
models:
  - model_path: /models/alpha
    served_model_name: alpha
    aliases: [shared]
  - model_path: /models/beta
    served_model_name: beta
    aliases: [shared]
""".strip()
    )

    with pytest.raises(ValueError, match="Alias 'shared'"):
        load_config_from_yaml(str(config_path))


def test_yaml_rejects_two_models_whose_version_tags_collide(tmp_path: Path) -> None:
    """Implicit ``name:version`` aliases are validated like explicit ones."""
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        """
models:
  - model_path: /models/alpha
    served_model_name: alpha
    version: "1"
  - model_path: /models/beta
    served_model_name: beta
    aliases: ["alpha:1"]
""".strip()
    )

    with pytest.raises(ValueError, match="Alias 'alpha:1'"):
        load_config_from_yaml(str(config_path))


@pytest.mark.asyncio
async def test_shipped_example_config_registers_without_collisions() -> None:
    """``examples/config.yaml`` is copied by users, so it must actually register.

    Parsing is not enough: aliases are claimed in the registry, which is where a
    name collision or an unroutable tag would surface as a startup crash.
    """
    config = load_config_from_yaml(
        str(Path(__file__).resolve().parents[1] / "examples/config.yaml")
    )
    registry = ModelRegistry()

    for entry in config.models:
        await registry.register_on_demand_model(
            model_id=entry.served_model_name,
            model_cfg_dict={"model_path": entry.model_path},
            model_type=entry.model_type,
            model_path=entry.model_path,
            context_length=entry.context_length,
            queue_config={"timeout": entry.queue_timeout, "queue_size": entry.queue_size},
            idle_timeout=300,
            aliases=entry.alias_names(),
            version=entry.version,
        )

    for entry in config.models:
        for alias in entry.alias_names():
            assert registry.resolve_model_id(alias) == entry.served_model_name

    assert registry.get_model_count() == len(config.models)


# ----------------------------------------------------------------------
# Registry resolution
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_alias_resolves_to_the_canonical_model_id() -> None:
    """Every alias form maps back to the one canonical identifier."""
    registry = ModelRegistry()
    await _register(registry, "alpha-v2", aliases=["alpha", "alpha-v2:2"], version="2")

    assert registry.resolve_model_id("alpha") == "alpha-v2"
    assert registry.resolve_model_id("alpha-v2:2") == "alpha-v2"
    assert registry.resolve_model_id("alpha-v2") == "alpha-v2"


@pytest.mark.asyncio
async def test_unknown_name_resolves_to_itself() -> None:
    """Resolution must not invent a model, so callers still raise their own 404."""
    registry = ModelRegistry()
    await _register(registry, "alpha")

    assert registry.resolve_model_id("nope") == "nope"
    assert registry.has_model("nope") is False


@pytest.mark.asyncio
async def test_lookups_accept_an_alias() -> None:
    """``has_model`` and ``is_on_demand`` are alias-aware."""
    registry = ModelRegistry()
    await _register(registry, "alpha-v2", aliases=["alpha"])

    assert registry.has_model("alpha") is True
    assert registry.is_on_demand("alpha") is True


@pytest.mark.asyncio
async def test_get_handler_accepts_an_alias() -> None:
    """A startup-loaded model is reachable through its alias."""
    registry = ModelRegistry()
    handler = object()
    await registry.register_model(
        model_id="alpha-v2",
        handler=handler,
        model_type="lm",
        context_length=4096,
        aliases=["alpha"],
        version="2",
    )

    assert registry.get_handler("alpha") is handler
    assert registry.get_handler("alpha-v2") is handler


@pytest.mark.asyncio
async def test_alias_and_canonical_share_one_worker_and_one_lease_count() -> None:
    """The decisive case: an alias must not open a second, parallel lifecycle.

    If resolution were applied at only some call sites, a request naming the
    alias would load its own worker and keep its own reference count, so the
    canonical model could be evicted while the alias was mid-generation.
    """
    registry = ModelRegistry()
    await _register(registry, "alpha-v2", aliases=["alpha"])

    via_alias = await registry.ensure_on_demand_loaded("alpha")
    via_canonical = await registry.ensure_on_demand_loaded("alpha-v2")

    assert via_alias is via_canonical
    assert len(_FakeHandlerProcessProxy.instances) == 1

    status = registry.get_model_status()
    assert len(status) == 1
    assert status[0]["id"] == "alpha-v2"
    assert status[0]["active_requests"] == 2

    # Releasing through either name decrements the same counter.
    await registry.release_on_demand("alpha", keep_alive=-1)
    assert registry.get_model_status()[0]["active_requests"] == 1
    await registry.release_on_demand("alpha-v2", keep_alive=-1)
    assert registry.get_model_status()[0]["active_requests"] == 0

    await registry.unload_on_demand("alpha")
    assert registry.get_model_status()[0]["loaded"] is False


@pytest.mark.asyncio
async def test_unload_through_an_alias_refuses_an_active_model() -> None:
    """Safety checks apply to the canonical model, not to the name used."""
    registry = ModelRegistry()
    await _register(registry, "alpha-v2", aliases=["alpha"])
    await registry.ensure_on_demand_loaded("alpha-v2")

    with pytest.raises(ValueError, match="active request"):
        await registry.unload_on_demand("alpha")

    await registry.release_on_demand("alpha", keep_alive=0)
    assert registry.get_model_status()[0]["loaded"] is False


@pytest.mark.asyncio
async def test_unloading_an_unknown_alias_raises_key_error() -> None:
    """An unregistered name is still a not-found, not a silent success."""
    registry = ModelRegistry()
    await _register(registry, "alpha-v2", aliases=["alpha"])

    with pytest.raises(KeyError, match="ghost"):
        await registry.unload_on_demand("ghost")


@pytest.mark.asyncio
async def test_ensure_loaded_reports_the_requested_name_when_unknown() -> None:
    """The error names what the client sent, not a resolved substitute."""
    registry = ModelRegistry()
    await _register(registry, "alpha-v2", aliases=["alpha"])

    with pytest.raises(KeyError, match="ghost"):
        await registry.ensure_on_demand_loaded("ghost")


@pytest.mark.asyncio
async def test_duplicate_alias_across_models_is_rejected() -> None:
    """The registry refuses an alias already owned by another model."""
    registry = ModelRegistry()
    await _register(registry, "alpha", aliases=["shared"])

    with pytest.raises(ValueError, match="Alias 'shared' is already registered"):
        await _register(registry, "beta", aliases=["shared"])


@pytest.mark.asyncio
async def test_alias_colliding_with_a_registered_model_is_rejected() -> None:
    """An alias may not shadow an existing canonical model id."""
    registry = ModelRegistry()
    await _register(registry, "alpha")

    with pytest.raises(ValueError, match="collides with registered model 'alpha'"):
        await _register(registry, "beta", aliases=["alpha"])


@pytest.mark.asyncio
async def test_registering_the_canonical_name_as_its_own_alias_is_a_no_op() -> None:
    """Self-aliasing is harmless and must not raise."""
    registry = ModelRegistry()
    await _register(registry, "alpha", aliases=["alpha"])

    assert registry.list_aliases("alpha") == []
    assert registry.resolve_model_id("alpha") == "alpha"


@pytest.mark.asyncio
async def test_unregister_model_drops_its_aliases() -> None:
    """A stale alias would resolve to a name that no longer exists."""
    registry = ModelRegistry()
    await registry.register_model(
        model_id="alpha-v2",
        handler=SimpleNamespace(),
        model_type="lm",
        aliases=["alpha"],
    )

    await registry.unregister_model("alpha")

    assert registry.resolve_model_id("alpha") == "alpha"
    assert registry.has_model("alpha") is False
    assert registry.list_aliases("alpha-v2") == []


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_and_metadata_expose_version_and_aliases() -> None:
    """Clients discover the tag and the alternative names from the API."""
    registry = ModelRegistry()
    await _register(registry, "alpha-v2", aliases=["alpha", "alpha-v2:2"], version="2")

    status = registry.get_model_status()[0]
    assert status["version"] == "2"
    assert status["aliases"] == ["alpha", "alpha-v2:2"]

    metadata = registry.list_models()[0]["metadata"]
    assert metadata["version"] == "2"
    assert metadata["aliases"] == ["alpha", "alpha-v2:2"]


@pytest.mark.parametrize("model_type", sorted(VALID_MODEL_TYPES))
@pytest.mark.asyncio
async def test_aliases_work_for_every_model_type(model_type: str) -> None:
    """Alias resolution happens in the registry, so it must not depend on the type.

    Image and audio backends cannot be exercised end to end without their
    weights, so every declared model type is registered here instead: a future
    type-specific branch in the registry would break this before it ships.
    """
    registry = ModelRegistry()
    await registry.register_on_demand_model(
        model_id="worker-v3",
        model_cfg_dict={"model_path": "/models/worker"},
        model_type=model_type,
        model_path="/models/worker",
        context_length=None,
        queue_config={"timeout": 30, "queue_size": 4},
        idle_timeout=300,
        aliases=["worker"],
        version="3",
    )

    assert registry.resolve_model_id("worker") == "worker-v3"
    assert registry.resolve_model_id("worker-v3:3") == "worker-v3"
    assert registry.has_model("worker") is True
    assert registry.is_on_demand("worker-v3:3") is True
    assert registry.list_model_ids() == ["worker-v3"]
    assert registry.get_metadata("worker").version == "3"


@pytest.mark.asyncio
async def test_aliases_do_not_inflate_model_listings() -> None:
    """Aliases are extra routes to one model, not extra models."""
    registry = ModelRegistry()
    await _register(registry, "alpha-v2", aliases=["alpha", "alpha-latest"], version="2")
    await _register(registry, "beta")

    assert registry.list_model_ids() == ["alpha-v2", "beta"]
    assert registry.get_model_count() == 2
    assert len(registry.list_models()) == 2


@pytest.mark.asyncio
async def test_version_alone_creates_a_routable_tag() -> None:
    """A version is a route as well as metadata, even with no explicit aliases.

    The registry derives ``<id>:<version>`` itself so a tag is addressable
    however the model was registered, not only via the YAML config path.
    """
    registry = ModelRegistry()
    await _register(registry, "alpha", version="1.4")

    status = registry.get_model_status()[0]
    assert status["version"] == "1.4"
    assert status["aliases"] == ["alpha:1.4"]
    assert registry.resolve_model_id("alpha:1.4") == "alpha"

    handler = await registry.ensure_on_demand_loaded("alpha:1.4")
    assert handler is not None
    await registry.release_on_demand("alpha:1.4", keep_alive=0)


# ----------------------------------------------------------------------
# Endpoint layer
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_endpoint_resolution_leases_the_canonical_model_via_alias() -> None:
    """Routing a request by alias must book a lease on the real model."""
    from app.api.endpoints import _release_on_demand, _resolve_handler

    registry = ModelRegistry()
    await _register(registry, "alpha-v2", aliases=["alpha"])
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(registry=registry)),
        state=SimpleNamespace(),
    )

    handler = await _resolve_handler(request, model_id="alpha")

    assert handler is not None
    assert registry.get_model_status()[0]["active_requests"] == 1

    await _release_on_demand(request)
    assert registry.get_model_status()[0]["active_requests"] == 0
    await registry.unload_on_demand("alpha-v2")


@pytest.mark.asyncio
async def test_load_and_unload_endpoints_accept_an_alias() -> None:
    """The lifecycle endpoints route by alias like the inference endpoints."""
    from app.api.endpoints import load_model, unload_model
    from app.schemas.openai import ModelLoadRequest, ModelUnloadRequest

    registry = ModelRegistry()
    await _register(registry, "alpha-v2", aliases=["alpha"], version="2")
    raw_request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(registry=registry)),
        state=SimpleNamespace(),
    )

    loaded = await load_model(ModelLoadRequest(model="alpha-v2:2", keep_alive=-1), raw_request)

    # The reported state is looked up in a list keyed by canonical id, so an
    # aliased request must be resolved first or it silently reports "unloaded"
    # with a null model payload.
    assert loaded["status"] == "loaded"
    assert loaded["model"] is not None
    assert loaded["model"]["id"] == "alpha-v2"
    assert registry.get_model_status()[0]["loaded"] is True

    unloaded = await unload_model(ModelUnloadRequest(model="alpha"), raw_request)
    assert unloaded == {"status": "unloaded", "model": "alpha"}
    assert registry.get_model_status()[0]["loaded"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["load", "unload"])
async def test_lifecycle_errors_are_not_doubly_quoted(endpoint: str) -> None:
    """``str(KeyError(...))`` reprs its argument, adding quotes clients would see."""
    from app.api.endpoints import load_model, unload_model
    from app.schemas.openai import ModelLoadRequest, ModelUnloadRequest

    registry = ModelRegistry()
    await _register(registry, "alpha")
    raw_request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(registry=registry)),
        state=SimpleNamespace(),
    )

    if endpoint == "load":
        response = await load_model(ModelLoadRequest(model="ghost"), raw_request)
    else:
        response = await unload_model(ModelUnloadRequest(model="ghost"), raw_request)

    message = json.loads(response.body)["error"]["message"]
    assert response.status_code == 404
    assert "ghost" in message
    assert not message.startswith(('"', "'"))


@pytest.mark.asyncio
async def test_unknown_model_still_returns_not_found() -> None:
    """Alias support must not turn a typo into a successful load."""
    from fastapi import HTTPException

    from app.api.endpoints import _resolve_handler

    registry = ModelRegistry()
    await _register(registry, "alpha-v2", aliases=["alpha"])
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(registry=registry)),
        state=SimpleNamespace(),
    )

    with pytest.raises(HTTPException) as excinfo:
        await _resolve_handler(request, model_id="alpha-v3")

    assert excinfo.value.status_code == 404
