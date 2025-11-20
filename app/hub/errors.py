"""Error types shared across hub subsystems."""

from __future__ import annotations


class HubControllerError(RuntimeError):
    """Raised when hub controller operations fail.

    Parameters
    ----------
    message : str
        Human-readable description of the failure.
    status_code : int | None, optional
        Optional HTTP status code surfaced to API clients.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class HubManagerError(RuntimeError):
    """Raised when the hub manager (process orchestrator) encounters a failure."""


class HubProcessError(HubManagerError):
    """Raised when spawning or stopping a managed model process fails."""
