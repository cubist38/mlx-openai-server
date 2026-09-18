"""Relay of handler-subprocess log records into the main process's sinks.

Loguru sinks belong to the process that adds them, and every model handler runs
in its own ``spawn``-ed process (see :mod:`app.core.handler_process`). The sinks
installed by :func:`app.server.configure_logging` therefore do not exist in the
child, so anything a handler logs — a load failure, a cleanup error, the
traceback of an exception raised while serving a request — went to the child's
inherited stderr and never reached the configured log file. The parent's view of
a handler was limited to what it logs itself: spawned / ready / unloaded.

This module closes that gap over the IPC channel the proxy already drains:

* the child calls :func:`install_queue_sink`, replacing its own sinks with one
  that pushes each record onto the response queue as a plain picklable dict;
* the parent's reader thread passes every such message to :func:`emit_record`,
  which re-emits it through the main process's sinks with the record's original
  level, module, function and line, prefixed by the model it came from.

Records are filtered in the child against the level the main process was
configured with (:func:`get_log_level`), so nothing is shipped across the queue
only to be discarded on arrival.

Output written straight to file descriptor 2 by native code (MLX/Metal
warnings) cannot be intercepted this way and still appears on the server's
stderr.
"""

from __future__ import annotations

import sys
import traceback
from typing import Any

from loguru import logger

# ``type`` value marking a response-queue message as a forwarded log record.
LOG_MESSAGE_TYPE = "log"

_DEFAULT_LEVEL = "INFO"

# Level the main process's sinks were configured with, recorded by
# ``configure_logging`` so spawned children can filter at the source. Module
# state rather than a config field: children are spawned from several call
# sites, and this is by definition whatever the sinks actually got.
_log_level: str = _DEFAULT_LEVEL


def set_log_level(level: str | None) -> None:
    """Record the level the main process's sinks were configured with.

    Parameters
    ----------
    level : str | None
        Loguru level name; ``None`` or empty falls back to ``"INFO"``.
    """
    global _log_level  # noqa: PLW0603
    _log_level = (level or _DEFAULT_LEVEL).upper()


def get_log_level() -> str:
    """Return the level forwarded records are filtered against in children."""
    return _log_level


def install_queue_sink(response_queue: Any, level: str = _DEFAULT_LEVEL) -> None:
    """Replace this process's loguru sinks with one that forwards to the parent.

    Call this once, as early as possible, in a spawned handler process. Every
    subsequent ``logger`` call in the process (handler, inference worker, debug
    logging, third-party code using loguru) is turned into a ``"log"`` message
    on ``response_queue`` and re-emitted by the parent.

    Parameters
    ----------
    response_queue : Any
        The child's end of the response queue, already shared with the parent.
    level : str
        Minimum level to forward; records below it never touch the queue.
    """

    def _sink(message: Any) -> None:
        record = message.record
        text = record["message"]
        exception = record["exception"]
        if exception is not None:
            text += (
                "\n"
                + "".join(
                    traceback.format_exception(exception.type, exception.value, exception.traceback)
                ).rstrip()
            )

        payload = {
            "type": LOG_MESSAGE_TYPE,
            "level": record["level"].name,
            "message": text,
            "name": record["name"],
            "function": record["function"],
            "line": record["line"],
        }
        try:
            response_queue.put(payload)
        except Exception:  # noqa: BLE001 - parent gone or pipe broken
            # Last resort so a broken relay does not silence the record.
            sys.stderr.write(f"{record['level'].name} | {text}\n")

    logger.remove()
    logger.add(_sink, level=level, format="{message}")


def emit_record(payload: dict[str, Any], model_name: str | None = None) -> None:
    """Re-emit a forwarded child record through this process's sinks.

    Parameters
    ----------
    payload : dict[str, Any]
        A message built by the sink installed by :func:`install_queue_sink`.
    model_name : str | None
        Model the record came from; prefixed to the message (matching how
        subprocess load progress is reported) so a shared log file stays
        attributable.
    """
    message = str(payload.get("message", ""))
    if model_name:
        message = f"'{model_name}': {message}"

    # Restore the child's origin so the console format's name:function:line
    # points at the code that logged, not at this relay.
    patched = logger.patch(
        lambda record: record.update(
            name=payload.get("name") or record["name"],
            function=payload.get("function") or record["function"],
            line=payload.get("line") or record["line"],
        )
    )

    level = str(payload.get("level") or _DEFAULT_LEVEL).upper()
    try:
        patched.log(level, message)
    except ValueError:
        # A level registered only in the child process is unknown here.
        patched.log(_DEFAULT_LEVEL, message)
