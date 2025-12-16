"""Unit tests covering hub lifecycle helpers and adapters."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core.hub_lifecycle import HubServiceAdapter, get_hub_lifecycle_service


class _InlineController:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def start_model(self, name: str) -> dict[str, str]:
        self.calls.append(("start", name))
        return {"status": "ok", "action": "start", "name": name}

    async def stop_model(self, name: str) -> dict[str, str]:
        self.calls.append(("stop", name))
        return {"status": "ok", "action": "stop", "name": name}

    async def load_model(self, name: str) -> dict[str, str]:
        self.calls.append(("load", name))
        return {"status": "ok", "action": "load", "name": name}

    async def unload_model(self, name: str) -> dict[str, str]:
        self.calls.append(("unload", name))
        return {"status": "ok", "action": "unload", "name": name}

    async def get_status(self) -> dict[str, list[str]]:
        return {"models": [call[1] for call in self.calls if call[0] == "start"]}


class _CallableBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def start_model(self, name: str) -> dict[str, str]:
        self.calls.append(("start", name))
        return {"status": "ok", "action": "start", "name": name}

    async def stop_model(self, name: str) -> dict[str, str]:
        self.calls.append(("stop", name))
        return {"status": "ok", "action": "stop", "name": name}

    def load_model(self, name: str) -> dict[str, str]:
        self.calls.append(("load", name))
        return {"status": "ok", "action": "load", "name": name}

    async def unload_model(self, name: str) -> dict[str, str]:
        self.calls.append(("unload", name))
        return {"status": "ok", "action": "unload", "name": name}

    async def get_status(self) -> dict[str, list[str]]:
        return {"models": [call[1] for call in self.calls if call[0] == "start"]}


@pytest.mark.asyncio
async def test_get_hub_lifecycle_service_returns_state_controller() -> None:
    """Hub lifecycle helper should prefer controllers stored on app.state."""
    controller = _InlineController()
    container = SimpleNamespace(state=SimpleNamespace(hub_controller=controller))

    resolved = get_hub_lifecycle_service(container)

    assert resolved is controller

    await resolved.start_model("alpha")
    await resolved.stop_model("alpha")
    assert controller.calls[:2] == [("start", "alpha"), ("stop", "alpha")]


@pytest.mark.asyncio
async def test_get_hub_lifecycle_service_checks_container_attributes() -> None:
    """Helper inspects direct container attributes when state is missing."""
    controller = _InlineController()
    container = SimpleNamespace(hub_service=controller)

    resolved = get_hub_lifecycle_service(container)

    assert resolved is controller
    await resolved.load_model("beta")
    assert controller.calls == [("load", "beta")]


@pytest.mark.asyncio
async def test_hub_service_adapter_normalizes_mixed_callables() -> None:
    """Adapters await async callables and pass through sync callables."""
    backend = _CallableBackend()
    adapter = HubServiceAdapter(
        start_model_fn=backend.start_model,
        stop_model_fn=backend.stop_model,
        load_model_fn=backend.load_model,
        unload_model_fn=backend.unload_model,
        status_fn=backend.get_status,
    )

    await adapter.start_model("alpha")
    await adapter.stop_model("alpha")
    await adapter.load_model("beta")
    await adapter.unload_model("beta")
    status = await adapter.get_status()

    assert backend.calls == [
        ("start", "alpha"),
        ("stop", "alpha"),
        ("load", "beta"),
        ("unload", "beta"),
    ]
    assert status == {"models": ["alpha"]}
