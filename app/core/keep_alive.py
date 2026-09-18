"""Model ``keep_alive`` parsing helpers."""

from __future__ import annotations

import math
import re
from typing import TypeAlias

KeepAlive: TypeAlias = str | int | float | None

_DURATION_PART = re.compile(r"(?P<value>\d+(?:\.\d+)?)(?P<unit>ms|s|m|h|d)", re.IGNORECASE)
_UNIT_SECONDS = {
    "ms": 0.001,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
}


def parse_keep_alive(value: KeepAlive, default_seconds: float | None) -> float | None:
    """Convert a keep-alive value to seconds.

    ``None`` selects the configured default, zero requests immediate unload,
    and a negative value means that the model should remain loaded
    indefinitely. String durations may contain multiple components, such as
    ``"1h30m"``.

    Parameters
    ----------
    value : KeepAlive
        Number of seconds or a duration string using ``ms``, ``s``, ``m``,
        ``h``, or ``d``.
    default_seconds : float | None
        Value returned when ``value`` is ``None``. ``None`` means no expiry.

    Returns
    -------
    float | None
        Duration in seconds, or ``None`` when the model must not expire.

    Raises
    ------
    ValueError
        If the value is invalid, non-finite, or contains unsupported units.
    """
    if value is None:
        return default_seconds
    if isinstance(value, bool):
        raise ValueError("keep_alive must be a duration or a number of seconds")

    if isinstance(value, int | float):
        seconds = float(value)
    elif isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("keep_alive cannot be empty")
        try:
            seconds = float(normalized)
        except ValueError:
            seconds = _parse_duration(normalized)
    else:
        raise ValueError("keep_alive must be a duration or a number of seconds")

    if not math.isfinite(seconds):
        raise ValueError("keep_alive must be finite")
    if seconds < 0:
        return None
    return seconds


def _parse_duration(value: str) -> float:
    """Parse a duration composed of one or more unit-suffixed parts."""
    position = 0
    seconds = 0.0
    for match in _DURATION_PART.finditer(value):
        if match.start() != position:
            raise ValueError(f"Invalid keep_alive duration: {value!r}")
        seconds += float(match.group("value")) * _UNIT_SECONDS[match.group("unit").lower()]
        position = match.end()

    if position != len(value) or position == 0:
        raise ValueError(f"Invalid keep_alive duration: {value!r}")
    return seconds
