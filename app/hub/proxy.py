"""Lightweight HTTP proxy for forwarding OpenAI-style requests to hub workers.

This module exposes an APIRouter that proxies requests made to `/v1/*`
to a worker process managed by the hub daemon. The worker port is resolved
by querying the hub daemon `/hub/status` snapshot and selecting the named
model's reported `port` field.

Notes
- The proxy requires `app.state.hub_daemon_url` to be set (e.g. by the
  server factory when running in hub mode). If missing, the router will
  return a 503 response.
"""

from __future__ import annotations

import contextlib
import json
import time

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
from loguru import logger
from starlette.responses import Response as StarletteResponse

from ..utils.errors import create_error_response

proxy_router = APIRouter()

# Simple in-memory TTL cache key: model -> (host, port, ts)
CACHE_TTL_SECONDS = 5.0


def _get_cache(app: FastAPI) -> dict[str, tuple[str, int, float]]:
    cache = getattr(app.state, "hub_model_port_cache", None)
    if cache is None:
        cache = {}
        app.state.hub_model_port_cache = cache
    return cache


async def _resolve_worker_port(app: FastAPI, model_name: str) -> tuple[str, int] | None:
    """Query the hub daemon status and return (host, port) for model.

    Returns None when the daemon is unreachable or the model is not running.
    """
    base = getattr(app.state, "hub_daemon_url", None)
    if not base:
        return None

    # Short-circuit using a small TTL cache to avoid frequent daemon calls
    cache = _get_cache(app)
    entry = cache.get(model_name)
    if isinstance(entry, tuple):
        host_c, port_c, ts = entry
        if (time.time() - float(ts)) < CACHE_TTL_SECONDS:
            return host_c, int(port_c)

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{base.rstrip('/')}/hub/status")
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:
        logger.debug(f"Failed to fetch hub status from daemon at {base}: {exc}")
        return None

    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return None

    now_ts = time.time()
    # Populate cache for all reported models and return the requested one
    for entry in models:
        if not isinstance(entry, dict):
            continue
        nm = entry.get("id") or entry.get("name")
        if not nm:
            continue
        p = entry.get("metadata", {}).get("port") or entry.get("port")
        h = entry.get("metadata", {}).get("host") or "127.0.0.1"
        if p:
            with contextlib.suppress(Exception):
                cache[nm] = (h, int(p), now_ts)

    found = cache.get(model_name)
    if isinstance(found, tuple):
        h, p, _ = found
        return h, int(p)
    return None


def _json_error(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse(
        content=create_error_response(message, "invalid_request_error", status), status_code=status
    )


@proxy_router.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    response_model=None,
    include_in_schema=False,
)
async def proxy_v1(path: str, request: Request) -> StarletteResponse:
    """Proxy incoming /v1 requests to the selected worker process.

    The proxy attempts to determine the target model using the following
    heuristic (in order): query parameter `model`, JSON body field `model`.
    If the model cannot be determined the request fails with 400.
    """
    # Must have a configured hub daemon base URL
    base = getattr(request.app.state, "hub_daemon_url", None)
    if not base:
        return JSONResponse(
            content=create_error_response(
                "Hub daemon integration not configured for this server.",
                "service_unavailable",
                503,
            ),
            status_code=503,
        )

    # Try query param first
    model = request.query_params.get("model")

    body_bytes = None
    content_type = request.headers.get("content-type", "")
    if not model and content_type.startswith("application/json"):
        # Read body and attempt to parse
        body_bytes = await request.body()
        try:
            parsed = json.loads(body_bytes.decode("utf-8") or "{}")
            model = parsed.get("model")
        except Exception:
            model = None

    if not model:
        return _json_error(
            "Model selection is required when running in hub mode. Include the 'model' query param or JSON field."
        )

    resolved = await _resolve_worker_port(request.app, model)
    if not resolved:
        return JSONResponse(
            content=create_error_response(
                f"Model '{model}' is not started or hub daemon unreachable",
                "service_unavailable",
                503,
            ),
            status_code=503,
        )

    host, port = resolved
    target_base = f"http://{host}:{port}"
    target_url = f"{target_base.rstrip('/')}/v1/{path}"

    # Prepare headers (copy and sanitize)
    headers = dict(request.headers)
    headers.pop("host", None)
    # Let httpx compute content-length/transfer-encoding
    headers.pop("content-length", None)

    params = dict(request.query_params)

    # Read the body (preserve raw bytes for multipart/form-data)
    if body_bytes is None:
        body_bytes = await request.body()

    # Proxy the request and return the response (buffered to simplify tests).
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.request(
                request.method, target_url, headers=headers, params=params, content=body_bytes
            )
            status = resp.status_code
            response_headers = dict(resp.headers)
            content_bytes = resp.content

        # Filter hop-by-hop headers that should not be exposed
        for h in [
            "transfer-encoding",
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailer",
            "upgrade",
        ]:
            response_headers.pop(h, None)

        # If the worker returned JSON, return JSONResponse to preserve structure
        ctype = response_headers.get("content-type", "")
        if "application/json" in ctype:
            try:
                payload = json.loads(content_bytes.decode("utf-8") or "{}")
            except Exception:
                payload = None
            if isinstance(payload, dict):
                return JSONResponse(content=payload, status_code=status, headers=response_headers)

        # Fallback: return the raw bytes as a streaming response
        return StreamingResponse(
            iter([content_bytes]), status_code=status, headers=response_headers
        )
    except httpx.RequestError as exc:
        logger.debug(f"Proxy request to {target_url} failed: {exc}")
        return JSONResponse(
            content=create_error_response(
                f"Failed to contact worker for model '{model}': {exc}", "service_unavailable", 503
            ),
            status_code=503,
        )
