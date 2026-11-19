"""Read a running server's model state for the ``list`` CLI commands.

None of this can be answered from disk. A config file says which models are
*configured*; only the server process knows which are resident, which are
loading, how many requests each is serving and how long an idle one has left
before it is evicted. The server exposes that through
``GET /v1/models/status``, so this module holds the small HTTP client for it
plus the table renderer.

Keeping both here leaves :mod:`app.cli` to option wiring, and lets the
rendering be tested without a socket: :func:`format_status_table` is pure and
takes ``now`` so relative times are deterministic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from http import HTTPStatus
import time
from typing import Any

import httpx

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_TIMEOUT = 5.0
STATUS_PATH = "/v1/models/status"

# Longest error text kept in the table; the full string is still in --json.
_MAX_ERROR_CHARS = 160

# A server bound to a wildcard address is reached over loopback -- connecting
# to the wildcard itself is not portable, so treat these as "this machine".
_WILDCARD_HOSTS = frozenset({"", "0.0.0.0", "::", "[::]", "*"})  # noqa: S104

_HEADER_STATE = "STATE"
_HEADER_MODEL = "MODEL"
_HEADER_ALIASES = "ALIASES"


class StatusUnavailable(Exception):
    """Raised when a server's model status could not be read.

    The message is written for a person reading it in a terminal, so callers
    may surface ``str(exc)`` verbatim.
    """


def resolve_base_url(host: str | None, port: int | None) -> str:
    """Return the base URL to query for a given host and port.

    Parameters
    ----------
    host : str or None
        Host to contact. ``None``, an empty string and wildcard bind
        addresses all resolve to loopback, since that is where a server bound
        to a wildcard is actually reachable.
    port : int or None
        Port to contact, defaulting to :data:`DEFAULT_PORT`.

    Returns
    -------
    str
        A base URL with no trailing slash, e.g. ``http://127.0.0.1:8000``.
    """
    resolved_host = DEFAULT_HOST if host is None else host.strip()
    if resolved_host in _WILDCARD_HOSTS:
        resolved_host = DEFAULT_HOST
    if ":" in resolved_host and not resolved_host.startswith("["):
        # Bare IPv6 literal: a URL needs it bracketed to keep the port apart.
        resolved_host = f"[{resolved_host}]"
    resolved_port = DEFAULT_PORT if port is None else port
    return f"http://{resolved_host}:{resolved_port}"


def fetch_model_status(base_url: str, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Fetch the model status payload from a running server.

    Parameters
    ----------
    base_url : str
        Base URL of the server, as returned by :func:`resolve_base_url`.
    timeout : float, optional
        Seconds to wait for the response.

    Returns
    -------
    dict[str, Any]
        The decoded payload, guaranteed to carry a ``data`` list.

    Raises
    ------
    StatusUnavailable
        If the server cannot be reached, refuses the request, or answers with
        something other than the expected payload.
    """
    url = f"{base_url.rstrip('/')}{STATUS_PATH}"
    try:
        response = httpx.get(url, timeout=timeout)
    except httpx.RequestError as exc:
        raise StatusUnavailable(
            f"Could not reach a server at {base_url} ({exc}). Start one with "
            "`mlx-openai-server launch ...`, or point --host/--port at the right address."
        ) from exc

    if response.status_code == HTTPStatus.NOT_FOUND:
        raise StatusUnavailable(
            f"The server at {base_url} has no {STATUS_PATH} endpoint, so it predates "
            "model status reporting. Upgrade it to use this command."
        )
    if response.status_code >= HTTPStatus.BAD_REQUEST:
        raise StatusUnavailable(
            f"The server at {base_url} answered HTTP {response.status_code}: "
            f"{_error_message(response)}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise StatusUnavailable(
            f"The server at {base_url} answered {STATUS_PATH} with a non-JSON body."
        ) from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise StatusUnavailable(
            f"The server at {base_url} answered {STATUS_PATH} with an unexpected payload."
        )
    return payload


def format_duration(seconds: float | None) -> str:
    """Render a number of seconds as a compact duration.

    Parameters
    ----------
    seconds : float or None
        Duration to render. ``None`` becomes ``"-"`` and negative values are
        clamped to zero, so a timer that has just elapsed reads ``"0s"``
        rather than showing a negative countdown.

    Returns
    -------
    str
        One of ``"45s"``, ``"5m"``, ``"1m 58s"``, ``"2h"`` or ``"2h 5m"``.
    """
    if seconds is None:
        return "-"
    total = round(max(0.0, seconds))
    if total < 60:
        return f"{total}s"
    if total < 3600:
        minutes, remaining_seconds = divmod(total, 60)
        return f"{minutes}m" if remaining_seconds == 0 else f"{minutes}m {remaining_seconds}s"
    hours, remainder = divmod(total, 3600)
    minutes = remainder // 60
    return f"{hours}h" if minutes == 0 else f"{hours}h {minutes}m"


def format_status_table(
    payload: Mapping[str, Any],
    base_url: str | None = None,
    now: float | None = None,
) -> list[str]:
    """Render a status payload as aligned terminal lines.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Payload as returned by :func:`fetch_model_status`.
    base_url : str or None, optional
        Server URL to name in the summary line.
    now : float or None, optional
        Reference epoch seconds for the countdown column. Defaults to the
        current time; tests pass a fixed value.

    Returns
    -------
    list[str]
        Lines to print, without trailing newlines.
    """
    entries = [entry for entry in payload.get("data") or [] if isinstance(entry, Mapping)]
    moment = time.time() if now is None else now

    lines = [_summary_line(payload, entries, base_url)]
    if not entries:
        lines.append("No models are configured on this server.")
        return lines

    # The column only earns its width when something routes through it.
    show_aliases = any(entry.get("aliases") for entry in entries)
    rows = [_header_row(show_aliases)]
    rows.extend(_status_row(entry, moment, show_aliases) for entry in entries)

    lines.append("")
    lines.extend(_render_rows(rows))

    errors = [line for line in (_error_line(entry) for entry in entries) if line]
    if errors:
        lines.append("")
        lines.extend(errors)
    return lines


def _header_row(show_aliases: bool) -> list[str]:
    """Return the table header, with the alias column only when needed."""
    header = [_HEADER_STATE, _HEADER_MODEL, "TYPE", "PID", "ACTIVE", "KEEP-ALIVE", "EXPIRES"]
    if show_aliases:
        header.insert(2, _HEADER_ALIASES)
    return header


def _status_row(entry: Mapping[str, Any], now: float, show_aliases: bool) -> list[str]:
    """Return one table row for a single model entry."""
    is_loaded = bool(entry.get("loaded"))
    pid = entry.get("pid")
    row = [
        str(entry.get("state") or "unknown"),
        str(entry.get("id") or "-"),
        str(entry.get("type") or "-"),
        str(pid) if pid else "-",
        # Only a resident model can hold requests; "0" against an unloaded one
        # would suggest it is running and merely idle.
        str(entry.get("active_requests") or 0) if is_loaded else "-",
        _keep_alive_cell(_as_number(entry.get("default_keep_alive"))),
        _expires_cell(_as_number(entry.get("expires_at")), now),
    ]
    if show_aliases:
        aliases = entry.get("aliases")
        row.insert(2, ", ".join(str(alias) for alias in aliases) if aliases else "-")
    return row


def _render_rows(rows: Sequence[Sequence[str]]) -> list[str]:
    """Pad rows into aligned columns sized to their widest cell."""
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    return [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        for row in rows
    ]


def _summary_line(
    payload: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    base_url: str | None,
) -> str:
    """Return the counts line, falling back to counting the entries."""
    configured = _as_number(payload.get("configured"))
    loaded = _as_number(payload.get("loaded"))
    if configured is None:
        configured = len(entries)
    if loaded is None:
        loaded = sum(1 for entry in entries if entry.get("loaded"))
    summary = f"{int(configured)} configured, {int(loaded)} loaded"
    return f"{summary} · {base_url}" if base_url else summary


def _keep_alive_cell(keep_alive: float | None) -> str:
    """Render a model's default keep-alive, spelling out the pinned case."""
    if keep_alive is None:
        return "-"
    if keep_alive < 0:
        # Negative keep-alive means the model is never evicted for being idle.
        return "never"
    return format_duration(keep_alive)


def _expires_cell(expires_at: float | None, now: float) -> str:
    """Render the time left before an idle model is evicted."""
    if expires_at is None:
        return "-"
    return format_duration(expires_at - now)


def _error_line(entry: Mapping[str, Any]) -> str | None:
    """Return a one-line note about a model's last error, if it has one."""
    error = entry.get("last_error")
    if not error:
        return None
    first_line = str(error).strip().splitlines()[0]
    if len(first_line) > _MAX_ERROR_CHARS:
        first_line = f"{first_line[:_MAX_ERROR_CHARS]}…"
    return f"! {entry.get('id') or 'unknown'}: {first_line}"


def _error_message(response: httpx.Response) -> str:
    """Extract a readable message from an error response."""
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, Mapping):
        error = payload.get("error")
        if isinstance(error, Mapping) and error.get("message"):
            return str(error["message"])
    return response.text.strip() or "no details given"


def _as_number(value: Any) -> float | None:
    """Return ``value`` as a float, or ``None`` when it is not numeric.

    The payload comes from another process, so a missing or malformed field
    must degrade to a dash in the table rather than raise.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)
