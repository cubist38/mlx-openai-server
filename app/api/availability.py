"""Helpers for registry-backed availability enforcement."""

from __future__ import annotations

from http import HTTPStatus
from typing import cast

from fastapi import Request
from fastapi.responses import JSONResponse
from loguru import logger

from ..core.model_registry import ModelRegistry
from ..utils.errors import create_error_response


def get_model_registry(raw_request: Request) -> ModelRegistry | None:
    """Return the configured ``ModelRegistry`` instance when present.

    Parameters
    ----------
    raw_request : Request
        FastAPI request containing application state.

    Returns
    -------
    ModelRegistry or None
        Registry stored on ``app.state`` or ``None`` when unavailable.
    """

    registry = getattr(raw_request.app.state, "model_registry", None)
    if registry is None:
        return None
    return cast("ModelRegistry", registry)


def group_capacity_error(detail: str = "group busy") -> JSONResponse:
    """Return a standardized 429 response for group capacity violations.

    Parameters
    ----------
    detail : str, default "group busy"
        Human-readable detail appended to the status code string.

    Returns
    -------
    JSONResponse
        Response payload matching the existing rate limit error schema.
    """

    return JSONResponse(
        content=create_error_response(
            f"{HTTPStatus.TOO_MANY_REQUESTS}: {detail}",
            "rate_limit_error",
            HTTPStatus.TOO_MANY_REQUESTS,
        ),
        status_code=HTTPStatus.TOO_MANY_REQUESTS,
    )


def guard_model_availability(
    raw_request: Request,
    model_id: str | None,
    *,
    deny_detail: str = "group busy",
    log_context: str | None = None,
) -> JSONResponse | None:
    """Return an error when ``model_id`` is unavailable per cached policies.

    Parameters
    ----------
    raw_request : Request
        Incoming FastAPI request containing registry state.
    model_id : str or None
        Registry identifier being evaluated for availability.
    deny_detail : str, default "group busy"
        Detail string used when constructing the error payload.
    log_context : str or None, optional
        Friendly label rendered in structured log messages.

    Returns
    -------
    JSONResponse or None
        ``JSONResponse`` when the model is currently denied, otherwise ``None``.
    """

    if not model_id:
        return None

    registry = get_model_registry(raw_request)
    if registry is None:
        return None

    try:
        if not registry.has_model(model_id):
            return None
        if registry.is_model_available(model_id):
            return None
    except (
        KeyError,
        AttributeError,
    ) as exc:  # Expected when model is not present or registry stub lacks methods
        # Log at warning level to surface registry lookup issues. We intentionally
        # fall back to "allow" semantics when registry lookups fail so that
        # transient registry problems do not block requests; unexpected
        # exceptions should propagate to surface real bugs.
        logger.warning(
            f"Availability check could not complete for '{model_id}': {type(exc).__name__}: {exc}",
        )
        return None

    actor = log_context or "model"
    logger.info(f"{actor} '{model_id}' denied due to group capacity policy")
    return group_capacity_error(deny_detail)
