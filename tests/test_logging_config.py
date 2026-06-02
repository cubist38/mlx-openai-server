"""Tests for :func:`app.server.configure_logging`.

These verify that the file sink is a faithful transcript of the console
(same fields, including ``name:function:line``), that file logging can be
disabled, and that rotation is opt-out for subprocess callers.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import re

from loguru import logger
import pytest

from app.server import configure_logging

# Matches one formatted record, e.g.:
#   2026-06-02 12:00:00 | INFO     | tests.test_logging_config:test_x:42 | ✦ hello
_RECORD_RE = re.compile(
    r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| "
    r"\w+\s+\| "
    r"[\w.]+:\w+:\d+ \| "
    r"✦ "
)


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    """Restore loguru to a clean state after each test."""
    yield
    logger.remove()


def test_file_sink_shares_console_format(tmp_path: Path) -> None:
    """The file sink records source location (name:function:line), not a stripped subset."""
    log_path = tmp_path / "app.log"
    configure_logging(log_file=str(log_path), log_level="INFO")

    logger.info("hello-from-test")
    logger.remove()  # flush + close the (enqueued) file sink

    content = log_path.read_text(encoding="utf-8")
    assert "hello-from-test" in content
    # Source location must be present so the file matches the console.
    assert _RECORD_RE.search(content), content
    assert "test_file_sink_shares_console_format" in content
    # File output must not contain raw color markup or ANSI escapes.
    assert "<green>" not in content
    assert "\x1b[" not in content


def test_no_log_file_disables_file_sink(tmp_path: Path) -> None:
    """``no_log_file=True`` must not create a log file."""
    log_path = tmp_path / "app.log"
    configure_logging(log_file=str(log_path), no_log_file=True)

    logger.info("should-not-be-written")
    logger.remove()

    assert not log_path.exists()


def test_log_level_filters_file_records(tmp_path: Path) -> None:
    """Records below the configured level are not written to the file."""
    log_path = tmp_path / "app.log"
    configure_logging(log_file=str(log_path), log_level="WARNING")

    logger.info("below-threshold")
    logger.warning("at-threshold")
    logger.remove()

    content = log_path.read_text(encoding="utf-8")
    assert "below-threshold" not in content
    assert "at-threshold" in content


def test_rotation_can_be_disabled_for_subprocesses(tmp_path: Path) -> None:
    """``enable_rotation=False`` still writes, just without rotation/retention."""
    log_path = tmp_path / "app.log"
    # Should not raise and should still produce output for child processes.
    configure_logging(log_file=str(log_path), enable_rotation=False)

    logger.info("child-process-log")
    logger.remove()

    assert "child-process-log" in log_path.read_text(encoding="utf-8")
