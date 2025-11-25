"""MLX OpenAI Server package."""

# Suppress verbose banners or logs from external packages (e.g. mlx_vlm) which
# may emit import-time INFO logs that clutter CLI output.
import logging as _std_logging

# Silence the specific INFO banner from `mlx_vlm.video_generate` by
# patching StreamHandler.emit to ignore that message from the package's
# logger (this avoids editing third-party site packages directly).
from typing import Any

_orig_emit: Any = _std_logging.StreamHandler.emit


def _suppress_mlx_emit(self: _std_logging.Handler, record: _std_logging.LogRecord) -> None:
    try:
        name = getattr(record, "name", "")
        msg = record.getMessage() if hasattr(record, "getMessage") else str(record.msg)
        if (
            name.startswith("mlx_vlm")
            and "This is a beta version of the video understanding" in msg
        ):
            return
    except Exception:
        pass
    _orig_emit(self, record)


_std_logging.StreamHandler.emit = _suppress_mlx_emit  # type: ignore[method-assign]

_std_logging.getLogger("mlx_vlm").setLevel(_std_logging.WARNING)
_std_logging.getLogger("mlx_vlm.video_generate").setLevel(_std_logging.WARNING)
