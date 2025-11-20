"""Observability hooks for hub-managed model processes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from loguru import logger


@dataclass(slots=True)
class HubModelContext:
    """Static metadata describing a managed model."""

    name: str
    group: str | None
    log_path: str | None


class HubObservabilitySink(Protocol):
    """Protocol for observing model lifecycle events."""

    def model_started(self, ctx: HubModelContext, *, pid: int | None) -> None:
        """Record that ``ctx`` started running in ``pid``."""

    def model_stopped(self, ctx: HubModelContext, *, exit_code: int | None) -> None:
        """Record that ``ctx`` stopped with ``exit_code``."""

    def model_failed(self, ctx: HubModelContext, *, exit_code: int | None) -> None:
        """Record that ``ctx`` exited unexpectedly with ``exit_code``."""


class LoggingHubObservabilitySink:
    """Default sink that renders contextual log records for each event."""

    def __init__(self, base_logger: Any | None = None) -> None:
        self._logger = base_logger or logger

    def _bound(self, ctx: HubModelContext) -> Any:
        return self._logger.bind(
            hub_model=ctx.name,
            hub_group=ctx.group,
            hub_log_path=ctx.log_path,
        )

    def model_started(self, ctx: HubModelContext, *, pid: int | None) -> None:
        """Log that ``ctx`` started running."""

        bound = self._bound(ctx)
        bound.info(f"Started hub model (pid={pid or 'unknown'})")

    def model_stopped(self, ctx: HubModelContext, *, exit_code: int | None) -> None:
        """Log that ``ctx`` stopped cleanly (or was terminated)."""

        bound = self._bound(ctx)
        bound.info(f"Stopped hub model (exit_code={exit_code})")

    def model_failed(self, ctx: HubModelContext, *, exit_code: int | None) -> None:
        """Log that ``ctx`` crashed unexpectedly."""

        bound = self._bound(ctx)
        bound.warning(f"Hub model crashed (exit_code={exit_code})")
