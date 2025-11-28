"""MLX OpenAI Server package."""

# Suppress verbose banners or logs from external packages (e.g. mlx_vlm) which
# may emit import-time INFO logs that clutter CLI output. Use a logging.Filter
# attached to the package logger instead of monkey-patching Handler.emit.
import logging as _std_logging


class _SuppressMlxVlmFilter(_std_logging.Filter):
    """Filter out a specific noisy banner emitted by the `mlx_vlm` package.

    The filter returns False for LogRecords that originate from the
    `mlx_vlm` namespace and contain the known banner text, preventing them
    from being emitted by handlers. For other records it returns True.
    """

    def filter(self, record: _std_logging.LogRecord) -> bool:
        try:
            name = getattr(record, "name", "")
            msg = record.getMessage() if hasattr(record, "getMessage") else str(record.msg)
            if (
                name.startswith("mlx_vlm")
                and "This is a beta version of the video understanding" in msg
            ):
                return False
        except Exception:
            # If anything goes wrong while inspecting the record, allow it
            # through to avoid suppressing unrelated logs.
            return True
        return True


# Attach the filter to the mlx_vlm loggers so the banner is suppressed in a
# maintainable, testable way without monkey-patching core logging internals.
_filter = _SuppressMlxVlmFilter()
_std_logging.getLogger("mlx_vlm").addFilter(_filter)
_std_logging.getLogger("mlx_vlm.video_generate").addFilter(_filter)

_std_logging.getLogger("mlx_vlm").setLevel(_std_logging.WARNING)
_std_logging.getLogger("mlx_vlm.video_generate").setLevel(_std_logging.WARNING)
