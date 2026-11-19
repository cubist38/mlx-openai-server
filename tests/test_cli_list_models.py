"""Tests for ``mlx-openai-server list models`` and its status client.

The command's whole job is to turn one HTTP response into readable lines, so
the tests split along that seam: the pure renderer is checked against fixed
payloads with a frozen clock, the client is checked against constructed
``httpx.Response`` objects, and the Click layer is checked for how it resolves
an address and what it exits with.
"""

from __future__ import annotations

import importlib
import json
from typing import Any

from click.testing import CliRunner
import httpx
import pytest

from app.cli_status import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    STATUS_PATH,
    StatusUnavailable,
    fetch_model_status,
    format_duration,
    format_status_table,
    resolve_base_url,
)

# Frozen reference time so every countdown in the expected output is exact.
NOW = 1_700_000_000.0

BASE_URL = "http://127.0.0.1:8123"


def _entry(**overrides: Any) -> dict[str, Any]:
    """Return a status entry shaped like the server's, with overrides applied."""
    entry: dict[str, Any] = {
        "id": "some-model",
        "type": "lm",
        "backend": "mlx",
        "context_length": None,
        "version": None,
        "aliases": [],
        "state": "unloaded",
        "loaded": False,
        "on_demand": True,
        "active_requests": 0,
        "model_path": "org/some-model",
        "pid": None,
        "loaded_at": None,
        "last_used": None,
        "expires_at": None,
        "default_keep_alive": 300.0,
        "last_error": None,
    }
    entry.update(overrides)
    return entry


def _payload(*entries: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Return a full status payload wrapping the given entries."""
    payload: dict[str, Any] = {
        "object": "list",
        "configured": len(entries),
        "loaded": sum(1 for entry in entries if entry.get("loaded")),
        "data": list(entries),
    }
    payload.update(overrides)
    return payload


LOADED_EMBEDDING = _entry(
    id="qwen3-embedding",
    type="embeddings",
    state="loaded",
    loaded=True,
    pid=4242,
    active_requests=0,
    default_keep_alive=120.0,
    expires_at=NOW + 118,
)

UNLOADED_MULTIMODAL = _entry(
    id="qwen-agentcoder",
    type="multimodal",
    state="unloaded",
    loaded=False,
    default_keep_alive=300.0,
)


class _Recorder:
    """Stand-in for ``httpx.get`` that records the call and replays a result."""

    def __init__(self, result: httpx.Response | Exception) -> None:
        self.result = result
        self.url: str | None = None
        self.timeout: float | None = None
        self.calls = 0

    def __call__(self, url: str, timeout: float | None = None) -> httpx.Response:
        self.calls += 1
        self.url = url
        self.timeout = timeout
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture
def cli_module():
    """Import ``app.cli`` the same way the console entry point does.

    Imported through :mod:`importlib` rather than at module scope because other
    tests in the suite delete ``app.main`` from ``sys.modules`` and re-import
    it, which can leave a stale second copy bound to the ``app`` package.
    """
    return importlib.import_module("app.cli")


def _patch_http(monkeypatch, result: httpx.Response | Exception) -> _Recorder:
    """Route the status client's HTTP call to a recorder.

    Patches ``httpx.get`` on the shared ``httpx`` module, so it takes effect no
    matter which copy of ``app.cli_status`` is doing the calling.
    """
    recorder = _Recorder(result)
    monkeypatch.setattr(httpx, "get", recorder)
    return recorder


class TestResolveBaseUrl:
    def test_defaults_to_loopback_and_default_port(self) -> None:
        assert resolve_base_url(None, None) == f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"

    def test_explicit_host_and_port_are_used(self) -> None:
        assert resolve_base_url("example.internal", 9000) == "http://example.internal:9000"

    def test_surrounding_whitespace_is_trimmed(self) -> None:
        assert resolve_base_url("  10.0.0.5  ", 8000) == "http://10.0.0.5:8000"

    @pytest.mark.parametrize("wildcard", ["0.0.0.0", "::", "[::]", "*", "", "   "])
    def test_wildcard_bind_addresses_become_loopback(self, wildcard: str) -> None:
        # A server bound to a wildcard is reachable on loopback; dialling the
        # wildcard itself is not portable, and a config file usually holds one.
        assert resolve_base_url(wildcard, 8123) == f"http://{DEFAULT_HOST}:8123"

    def test_bare_ipv6_literal_is_bracketed(self) -> None:
        assert resolve_base_url("::1", 8123) == "http://[::1]:8123"

    def test_already_bracketed_ipv6_is_left_alone(self) -> None:
        assert resolve_base_url("[fe80::1]", 8123) == "http://[fe80::1]:8123"


class TestFormatDuration:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (None, "-"),
            (0, "0s"),
            (1, "1s"),
            (45, "45s"),
            (59.4, "59s"),
            (59.6, "1m"),  # rounds up into the minute branch, never "60s"
            (60, "1m"),
            (118, "1m 58s"),
            (120, "2m"),
            (300, "5m"),
            (3599, "59m 59s"),
            (3600, "1h"),
            (7500, "2h 5m"),
            (-30, "0s"),  # an elapsed timer must not read as a negative wait
        ],
    )
    def test_rendering(self, seconds: float | None, expected: str) -> None:
        assert format_duration(seconds) == expected


class TestFormatStatusTable:
    def test_full_table_matches_expected_layout(self) -> None:
        lines = format_status_table(
            _payload(UNLOADED_MULTIMODAL, LOADED_EMBEDDING),
            base_url=BASE_URL,
            now=NOW,
        )

        assert lines == [
            "2 configured, 1 loaded · http://127.0.0.1:8123",
            "",
            "STATE     MODEL            TYPE        PID   ACTIVE  KEEP-ALIVE  EXPIRES",
            "unloaded  qwen-agentcoder  multimodal  -     -       5m          -",
            "loaded    qwen3-embedding  embeddings  4242  0       2m          1m 58s",
        ]

    def test_columns_widen_for_long_model_names(self) -> None:
        long_name = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
        lines = format_status_table(_payload(_entry(id=long_name)), now=NOW)

        header, row = lines[-2], lines[-1]
        assert long_name in row
        # Every cell after the widened one still starts at the same offset.
        assert header.index("TYPE") == row.index("lm")

    def test_summary_omits_url_when_not_given(self) -> None:
        lines = format_status_table(_payload(LOADED_EMBEDDING), now=NOW)

        assert lines[0] == "1 configured, 1 loaded"

    def test_counts_are_derived_when_the_server_omits_them(self) -> None:
        payload = {"data": [UNLOADED_MULTIMODAL, LOADED_EMBEDDING]}

        assert format_status_table(payload, now=NOW)[0] == "2 configured, 1 loaded"

    def test_no_models_says_so_instead_of_printing_an_empty_table(self) -> None:
        lines = format_status_table(_payload(), base_url=BASE_URL, now=NOW)

        assert lines == [
            "0 configured, 0 loaded · http://127.0.0.1:8123",
            "No models are configured on this server.",
        ]

    def test_unloaded_models_show_dashes_not_zeroes(self) -> None:
        # "0" in the ACTIVE column would imply a resident model sitting idle.
        row = format_status_table(_payload(UNLOADED_MULTIMODAL), now=NOW)[-1]

        assert row.split() == ["unloaded", "qwen-agentcoder", "multimodal", "-", "-", "5m", "-"]

    def test_busy_model_reports_its_request_count(self) -> None:
        row = format_status_table(
            _payload(_entry(state="busy", loaded=True, pid=99, active_requests=3)),
            now=NOW,
        )[-1]

        assert row.split()[:5] == ["busy", "some-model", "lm", "99", "3"]

    def test_alias_column_appears_only_when_a_model_has_aliases(self) -> None:
        without = format_status_table(_payload(_entry()), now=NOW)
        with_aliases = format_status_table(
            _payload(_entry(aliases=["chat", "chat:stable"])), now=NOW
        )

        assert "ALIASES" not in without[-2]
        assert "ALIASES" in with_aliases[-2]
        assert "chat, chat:stable" in with_aliases[-1]

    def test_models_without_aliases_get_a_dash_in_the_alias_column(self) -> None:
        lines = format_status_table(
            _payload(_entry(id="a", aliases=["chat"]), _entry(id="b")), now=NOW
        )

        # ALIASES sits next to MODEL: both name the routes into the model.
        assert lines[2].split() == [
            "STATE",
            "MODEL",
            "ALIASES",
            "TYPE",
            "PID",
            "ACTIVE",
            "KEEP-ALIVE",
            "EXPIRES",
        ]
        assert lines[-1].split() == ["unloaded", "b", "-", "lm", "-", "-", "5m", "-"]

    def test_negative_keep_alive_reads_as_never_expiring(self) -> None:
        row = format_status_table(_payload(_entry(default_keep_alive=-1)), now=NOW)[-1]

        assert "never" in row

    def test_elapsed_expiry_clamps_to_zero(self) -> None:
        row = format_status_table(
            _payload(_entry(loaded=True, pid=7, expires_at=NOW - 5)), now=NOW
        )[-1]

        assert row.endswith("0s")

    def test_last_error_is_reported_under_the_table(self) -> None:
        lines = format_status_table(
            _payload(_entry(id="broken", last_error="Model path does not exist")), now=NOW
        )

        assert lines[-1] == "! broken: Model path does not exist"
        assert lines[-2] == ""

    def test_multiline_error_is_reduced_to_its_first_line(self) -> None:
        traceback = "RuntimeError: no metal device\nTraceback (most recent call last):\n  ..."
        lines = format_status_table(_payload(_entry(last_error=traceback)), now=NOW)

        assert lines[-1] == "! some-model: RuntimeError: no metal device"

    def test_overlong_error_is_truncated(self) -> None:
        lines = format_status_table(_payload(_entry(last_error="x" * 500)), now=NOW)

        assert lines[-1].endswith("…")
        assert len(lines[-1]) < 220

    @pytest.mark.parametrize(
        "entry",
        [
            {},
            {"id": "x", "expires_at": "soon", "default_keep_alive": "5m"},
            {"id": "x", "pid": "not-a-pid", "active_requests": None, "loaded": True},
            {"id": None, "type": None, "state": None},
        ],
    )
    def test_malformed_entries_degrade_to_dashes(self, entry: dict[str, Any]) -> None:
        # The payload crosses a process boundary; a surprising field has to
        # render as "-" rather than raise in the middle of the table.
        lines = format_status_table(_payload(entry), now=NOW)

        assert len(lines) == 4  # summary, blank, header, one row

    def test_non_mapping_entries_are_skipped(self) -> None:
        payload = {"data": ["garbage", None, LOADED_EMBEDDING]}

        lines = format_status_table(payload, now=NOW)

        assert lines[0] == "1 configured, 1 loaded"
        assert len(lines) == 4

    def test_no_line_has_trailing_whitespace(self) -> None:
        lines = format_status_table(
            _payload(UNLOADED_MULTIMODAL, LOADED_EMBEDDING), base_url=BASE_URL, now=NOW
        )

        assert all(line == line.rstrip() for line in lines)


class TestFetchModelStatus:
    def test_returns_the_payload_and_calls_the_status_endpoint(self, monkeypatch) -> None:
        payload = _payload(LOADED_EMBEDDING)
        recorder = _patch_http(monkeypatch, httpx.Response(200, json=payload))

        assert fetch_model_status(BASE_URL, timeout=2.5) == payload
        assert recorder.url == f"{BASE_URL}{STATUS_PATH}"
        assert recorder.timeout == 2.5

    def test_trailing_slash_does_not_double_up(self, monkeypatch) -> None:
        recorder = _patch_http(monkeypatch, httpx.Response(200, json=_payload()))

        fetch_model_status(f"{BASE_URL}/")

        assert recorder.url == f"{BASE_URL}{STATUS_PATH}"

    def test_connection_failure_explains_how_to_fix_it(self, monkeypatch) -> None:
        _patch_http(monkeypatch, httpx.ConnectError("[Errno 61] Connection refused"))

        with pytest.raises(StatusUnavailable) as excinfo:
            fetch_model_status(BASE_URL)

        message = str(excinfo.value)
        assert BASE_URL in message
        assert "Connection refused" in message
        assert "launch" in message

    def test_timeout_is_reported_as_unavailable(self, monkeypatch) -> None:
        _patch_http(monkeypatch, httpx.ReadTimeout("timed out"))

        with pytest.raises(StatusUnavailable):
            fetch_model_status(BASE_URL, timeout=0.1)

    def test_missing_endpoint_points_at_the_server_being_old(self, monkeypatch) -> None:
        _patch_http(monkeypatch, httpx.Response(404, text="Not Found"))

        with pytest.raises(StatusUnavailable, match="predates"):
            fetch_model_status(BASE_URL)

    def test_error_envelope_message_is_surfaced(self, monkeypatch) -> None:
        _patch_http(
            monkeypatch,
            httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Model lifecycle management requires multi-model mode",
                        "type": "unsupported_request",
                    }
                },
            ),
        )

        with pytest.raises(StatusUnavailable) as excinfo:
            fetch_model_status(BASE_URL)

        message = str(excinfo.value)
        assert "requires multi-model mode" in message
        assert "400" in message
        # Unwrapped, not dumped: the raw envelope would drag the JSON keys along.
        assert '"error"' not in message
        assert "unsupported_request" not in message

    def test_plain_text_error_body_is_used_as_the_detail(self, monkeypatch) -> None:
        _patch_http(monkeypatch, httpx.Response(500, text="internal boom"))

        with pytest.raises(StatusUnavailable, match="internal boom"):
            fetch_model_status(BASE_URL)

    def test_empty_error_body_still_produces_a_message(self, monkeypatch) -> None:
        _patch_http(monkeypatch, httpx.Response(503, text=""))

        with pytest.raises(StatusUnavailable, match="no details given"):
            fetch_model_status(BASE_URL)

    def test_non_json_success_body_is_rejected(self, monkeypatch) -> None:
        _patch_http(monkeypatch, httpx.Response(200, content=b"<html>hello</html>"))

        with pytest.raises(StatusUnavailable, match="non-JSON"):
            fetch_model_status(BASE_URL)

    @pytest.mark.parametrize("body", [[], {"object": "list"}, {"data": "not-a-list"}, "text"])
    def test_unexpected_payload_shape_is_rejected(self, monkeypatch, body: Any) -> None:
        _patch_http(monkeypatch, httpx.Response(200, json=body))

        with pytest.raises(StatusUnavailable, match="unexpected payload"):
            fetch_model_status(BASE_URL)


class TestListModelsCommand:
    def test_registered_under_the_list_group(self, cli_module) -> None:
        assert "models" in cli_module.cli.commands["list"].commands

    def test_prints_the_table(self, cli_module, monkeypatch) -> None:
        recorder = _patch_http(
            monkeypatch,
            httpx.Response(200, json=_payload(UNLOADED_MULTIMODAL, LOADED_EMBEDDING)),
        )

        result = CliRunner().invoke(cli_module.cli, ["list", "models", "--port", "8123"])

        assert result.exit_code == 0, result.output
        assert "2 configured, 1 loaded" in result.output
        assert "qwen3-embedding" in result.output
        assert recorder.url == f"http://{DEFAULT_HOST}:8123{STATUS_PATH}"

    def test_json_flag_prints_the_raw_payload(self, cli_module, monkeypatch) -> None:
        payload = _payload(LOADED_EMBEDDING)
        _patch_http(monkeypatch, httpx.Response(200, json=payload))

        result = CliRunner().invoke(cli_module.cli, ["list", "models", "--json"])

        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == payload
        assert "STATE" not in result.output

    def test_unreachable_server_exits_non_zero(self, cli_module, monkeypatch) -> None:
        _patch_http(monkeypatch, httpx.ConnectError("[Errno 61] Connection refused"))

        result = CliRunner().invoke(cli_module.cli, ["list", "models"])

        assert result.exit_code == 1
        assert "Could not reach a server" in result.output

    def test_host_and_port_default_to_loopback(self, cli_module, monkeypatch) -> None:
        recorder = _patch_http(monkeypatch, httpx.Response(200, json=_payload()))

        result = CliRunner().invoke(cli_module.cli, ["list", "models"])

        assert result.exit_code == 0, result.output
        assert recorder.url == f"http://{DEFAULT_HOST}:{DEFAULT_PORT}{STATUS_PATH}"

    def test_timeout_flag_reaches_the_client(self, cli_module, monkeypatch) -> None:
        recorder = _patch_http(monkeypatch, httpx.Response(200, json=_payload()))

        result = CliRunner().invoke(cli_module.cli, ["list", "models", "--timeout", "0.25"])

        assert result.exit_code == 0, result.output
        assert recorder.timeout == 0.25

    def test_environment_variables_supply_the_address(self, cli_module, monkeypatch) -> None:
        recorder = _patch_http(monkeypatch, httpx.Response(200, json=_payload()))

        result = CliRunner().invoke(
            cli_module.cli,
            ["list", "models"],
            env={"MLX_SERVER_HOST": "10.0.0.5", "MLX_SERVER_PORT": "9100"},
        )

        assert result.exit_code == 0, result.output
        assert recorder.url == f"http://10.0.0.5:9100{STATUS_PATH}"

    def test_config_file_supplies_the_address(self, cli_module, monkeypatch, tmp_path) -> None:
        config = tmp_path / "config.yaml"
        config.write_text(
            "server:\n"
            "  host: 127.0.0.1\n"
            "  port: 8123\n"
            "models:\n"
            "  - model_path: org/model\n"
            "    model_type: lm\n"
        )
        recorder = _patch_http(monkeypatch, httpx.Response(200, json=_payload()))

        result = CliRunner().invoke(cli_module.cli, ["list", "models", "--config", str(config)])

        assert result.exit_code == 0, result.output
        assert recorder.url == f"http://127.0.0.1:8123{STATUS_PATH}"

    def test_explicit_flags_win_over_the_config_file(
        self, cli_module, monkeypatch, tmp_path
    ) -> None:
        config = tmp_path / "config.yaml"
        config.write_text(
            "server:\n"
            "  host: 127.0.0.1\n"
            "  port: 8123\n"
            "models:\n"
            "  - model_path: org/model\n"
            "    model_type: lm\n"
        )
        recorder = _patch_http(monkeypatch, httpx.Response(200, json=_payload()))

        result = CliRunner().invoke(
            cli_module.cli, ["list", "models", "--config", str(config), "--port", "9999"]
        )

        assert result.exit_code == 0, result.output
        assert recorder.url == f"http://{DEFAULT_HOST}:9999{STATUS_PATH}"

    def test_wildcard_host_in_a_config_is_dialled_over_loopback(
        self, cli_module, monkeypatch, tmp_path
    ) -> None:
        # The config default is a wildcard bind address, which is the common
        # case for a config written for a server rather than for a client.
        config = tmp_path / "config.yaml"
        config.write_text(
            "server:\n  port: 8123\nmodels:\n  - model_path: org/model\n    model_type: lm\n"
        )
        recorder = _patch_http(monkeypatch, httpx.Response(200, json=_payload()))

        result = CliRunner().invoke(cli_module.cli, ["list", "models", "--config", str(config)])

        assert result.exit_code == 0, result.output
        assert recorder.url == f"http://{DEFAULT_HOST}:8123{STATUS_PATH}"

    def test_missing_config_file_is_a_usage_error(self, cli_module, monkeypatch) -> None:
        recorder = _patch_http(monkeypatch, httpx.Response(200, json=_payload()))

        result = CliRunner().invoke(
            cli_module.cli, ["list", "models", "--config", "/nonexistent/config.yaml"]
        )

        assert result.exit_code == 2
        assert recorder.calls == 0

    def test_help_does_not_contact_the_server(self, cli_module, monkeypatch) -> None:
        recorder = _patch_http(monkeypatch, httpx.Response(200, json=_payload()))

        result = CliRunner().invoke(cli_module.cli, ["list", "models", "--help"])

        assert result.exit_code == 0
        assert recorder.calls == 0
        assert "--json" in result.output
