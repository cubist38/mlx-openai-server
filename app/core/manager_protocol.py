"""Manager protocol used across the core modules.

This module defines a lightweight typing contract for manager-like objects
that control VRAM residency and per-request sessions. Placing it under
`app.core` keeps the handler package focused on concrete MLX handler
implementations.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Any, Literal, Protocol


class ManagerProtocol(Protocol):
    """Protocol describing manager behaviour used by the ModelRegistry.

    Implementations must ensure their VRAM operations are safe to call
    concurrently and idempotent where appropriate.
    """

    def is_vram_loaded(self) -> bool:  # pragma: no cover - typing stub
        """Return True when the manager currently has model weights resident in VRAM."""

    async def ensure_vram_loaded(
        self,
        *,
        force: bool = False,
        timeout: float | None = None,
    ) -> None:  # pragma: no cover - typing stub
        """Ensure VRAM residency for this manager. Should be idempotent."""

    async def release_vram(
        self,
        *,
        timeout: float | None = None,
        trigger: Literal["manual", "auto"] = "manual",
    ) -> None:  # pragma: no cover - typing stub
        """Release VRAM resources held by this manager."""

    def request_session(
        self,
        *,
        ensure_vram: bool = True,
        ensure_timeout: float | None = None,
    ) -> AbstractAsyncContextManager[Any]:  # pragma: no cover - typing stub
        """Return an async context manager used for per-request sessions.

        The context manager should increment any internal active-request
        counters on enter and decrement on exit. If ``ensure_vram`` is True
        it should ensure VRAM residency before yielding.
        """
