"""Unit tests for ModelRegistry behavior."""

from __future__ import annotations

import asyncio

from app.core.model_registry import ModelRegistry


def test_registry_tracks_handlers_and_metadata() -> None:
    """Ensure the registry records status, handler references, and metadata."""
    asyncio.run(_exercise_registry())


async def _exercise_registry() -> None:
    registry = ModelRegistry()
    await registry.register_model(
        model_id="foo",
        handler=None,
        model_type="lm",
        context_length=2048,
        metadata_extras={"group": "default"},
    )

    models = registry.list_models()
    assert len(models) == 1
    first = models[0]
    assert first["id"] == "foo"
    assert first["metadata"]["status"] == "unloaded"
    assert first["metadata"]["group"] == "default"

    handler = object()
    await registry.update_model_state(
        "foo",
        handler=handler,
        status="initialized",
        metadata_updates={"model_path": "/models/foo"},
    )

    assert registry.get_handler("foo") is handler
    updated = registry.list_models()[0]
    assert updated["metadata"]["status"] == "initialized"
    assert updated["metadata"]["model_path"] == "/models/foo"

    await registry.update_model_state("foo", handler=None, status="unloaded")
    assert registry.get_handler("foo") is None

    await registry.unregister_model("foo")
    assert registry.get_model_count() == 0
