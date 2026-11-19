"""Handler subprocesses must log into the server's log file, not a lost stderr.

Loguru sinks are per-process, so a model handler running in its own spawned
process starts with nothing but loguru's stderr default: before the relay, the
only trace of a handler in ``logs/app.log`` was what the *parent* logged about
it (spawned / ready / unloaded), and a traceback raised while serving a request
went to a stderr nobody collects. These tests cover the two halves of the fix
(:mod:`app.core.log_relay`) plus the file-sink rotation settings that keep the
resulting — now busier — log file from growing without bound.
"""

from __future__ import annotations

import importlib
from typing import Any

from click.testing import CliRunner
from loguru import logger
import pytest

from app.config import ModelEntryConfig, load_config_from_yaml
from app.core import handler_process, log_relay
from app.core.handler_process import HandlerProcessProxy, _handler_worker
from app.core.log_relay import (
    LOG_MESSAGE_TYPE,
    emit_record,
    get_log_level,
    install_queue_sink,
    set_log_level,
)
from app.server import configure_logging, parse_log_retention, parse_log_rotation


def _namespace_of(func: Any) -> dict[str, Any]:
    """Return the globals a function actually reads.

    Other tests in the suite swap ``sys.modules`` entries for stubs, which can
    leave ``app.<name>`` pointing at a stale second copy of a module; patching
    the function's own globals is unambiguous whichever copy is in play.
    """
    return func.__globals__


class _CaptureQueue:
    """Stand-in for the child's end of the response queue."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def put(self, item: dict[str, Any]) -> None:
        """Record a message in FIFO order."""

        self.items.append(item)


class _BrokenQueue:
    """Response queue whose pipe is gone (parent died)."""

    def put(self, item: dict[str, Any]) -> None:
        """Fail the way a closed queue does."""

        raise OSError("handle is closed")


class _CollectingQueue:
    """Stand-in for a pending caller's ``asyncio.Queue``."""

    def __init__(self) -> None:
        self.items: list[Any] = []

    def put_nowait(self, item: Any) -> None:
        """Accept a routed response."""

        self.items.append(item)


class _ImmediateLoop:
    """Event-loop stand-in that runs the callback on the calling thread."""

    def call_soon_threadsafe(self, callback: Any, *args: Any) -> None:
        """Invoke ``callback`` right away instead of scheduling it."""

        callback(*args)


class _FakeProcess:
    """Records what the proxy would have spawned, without spawning it."""

    def __init__(self, target: Any, args: tuple[Any, ...], name: str) -> None:
        self.target = target
        self.args = args
        self.name = name
        self.pid = 4242
        self._alive = False

    def start(self) -> None:
        """Pretend the child came up."""

        self._alive = True

    def is_alive(self) -> bool:
        """Whether the fake child is still running."""

        return self._alive

    def terminate(self) -> None:
        """Stop the fake child."""

        self._alive = False

    def join(self, timeout: float | None = None) -> None:
        """No-op; the fake child has already stopped."""


class _FakeContext:
    """``multiprocessing`` context stand-in that captures ``Process`` calls."""

    def __init__(self, spawned: list[_FakeProcess]) -> None:
        self._spawned = spawned

    def Queue(self) -> Any:  # noqa: N802 - mirrors the mp context API
        """Return a thread queue; the reader thread only needs ``get``."""
        import queue as _queue

        return _queue.Queue()

    def Process(  # noqa: N802 - mirrors the mp context API
        self, target: Any, args: tuple[Any, ...], name: str
    ) -> _FakeProcess:
        """Record the spawn instead of performing it."""
        process = _FakeProcess(target, args, name)
        self._spawned.append(process)
        return process


@pytest.fixture
def preserve_sinks():
    """Restore the ambient loguru configuration after tests that replace it.

    Both halves of the relay reconfigure the global logger (the child sink calls
    ``logger.remove()``), which would otherwise leak into the rest of the suite.
    """
    yield
    logger.remove()
    log_relay.set_log_level("INFO")


@pytest.fixture
def parent_sink(preserve_sinks):
    """Install a capturing sink standing in for the main process's sinks."""
    records: list[Any] = []
    logger.remove()
    logger.add(records.append, level="DEBUG", format="{message}")
    return records


# ---------------------------------------------------------------------------
# Child side: records become queue messages
# ---------------------------------------------------------------------------


class TestChildSink:
    """What a handler subprocess puts on the response queue when it logs."""

    def test_record_is_forwarded_with_its_origin(self, preserve_sinks) -> None:
        queue = _CaptureQueue()
        install_queue_sink(queue, level="INFO")

        logger.warning("model failed to load")

        assert len(queue.items) == 1
        payload = queue.items[0]
        assert payload["type"] == LOG_MESSAGE_TYPE
        assert payload["level"] == "WARNING"
        assert payload["message"] == "model failed to load"
        # Origin travels with the record so the parent can report the real
        # module:function:line instead of the relay's.
        assert payload["name"] == __name__
        assert payload["function"] == "test_record_is_forwarded_with_its_origin"
        assert isinstance(payload["line"], int)

    def test_traceback_is_carried_in_the_message(self, preserve_sinks) -> None:
        # The whole point of the relay: an exception inside a handler process
        # must be readable in the server log, not only in the child's stderr.
        queue = _CaptureQueue()
        install_queue_sink(queue, level="INFO")

        try:
            raise ValueError("metal buffer exploded")
        except ValueError:
            logger.exception("generation failed")

        message = queue.items[0]["message"]
        assert message.startswith("generation failed")
        assert "ValueError: metal buffer exploded" in message
        assert "Traceback (most recent call last)" in message

    def test_records_below_the_level_never_cross_the_queue(self, preserve_sinks) -> None:
        # Filtering happens in the child so debug chatter (e.g. a model with
        # ``debug: true``) is not shipped only to be dropped on arrival.
        queue = _CaptureQueue()
        install_queue_sink(queue, level="INFO")

        logger.debug("per-token detail")
        logger.info("kept")

        assert [item["message"] for item in queue.items] == ["kept"]

    def test_a_broken_queue_falls_back_to_stderr(
        self, preserve_sinks, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # install_queue_sink removes the child's stderr sink, so a dead parent
        # must not silence the process entirely.
        install_queue_sink(_BrokenQueue(), level="INFO")

        logger.error("parent went away")

        assert "parent went away" in capsys.readouterr().err

    def test_the_payload_survives_a_real_queue(self, preserve_sinks) -> None:
        # The real queue pickles what it carries, so nothing unpicklable may end
        # up in the payload — loguru's own exception tuple, for one, is not.
        import multiprocessing as mp

        queue = mp.get_context("spawn").Queue()
        install_queue_sink(queue, level="INFO")
        try:
            try:
                raise RuntimeError("model died")
            except RuntimeError:
                logger.exception("crashed")

            payload = queue.get(timeout=10)
        finally:
            queue.close()
            queue.join_thread()

        assert payload["type"] == LOG_MESSAGE_TYPE
        assert payload["level"] == "ERROR"
        assert "RuntimeError: model died" in payload["message"]

    def test_the_sink_replaces_the_inherited_ones(self, preserve_sinks) -> None:
        # Otherwise every handler line would be logged twice: once relayed into
        # the server's sinks, once on the child's inherited stderr. A stand-in
        # for that inherited sink must stop receiving records.
        inherited: list[Any] = []
        logger.add(inherited.append, level="DEBUG", format="{message}")
        queue = _CaptureQueue()

        install_queue_sink(queue, level="INFO")
        logger.info("only once")

        assert inherited == []
        assert len(queue.items) == 1


# ---------------------------------------------------------------------------
# Parent side: queue messages become log records again
# ---------------------------------------------------------------------------


class TestParentEmit:
    """How the main process re-emits a forwarded record."""

    def test_message_is_prefixed_with_the_model(self, parent_sink) -> None:
        emit_record(
            {"type": LOG_MESSAGE_TYPE, "level": "INFO", "message": "loaded"},
            "qwen-agentcoder",
        )

        assert [str(record) for record in parent_sink] == ["'qwen-agentcoder': loaded\n"]

    def test_level_and_origin_are_preserved(self, parent_sink) -> None:
        emit_record(
            {
                "type": LOG_MESSAGE_TYPE,
                "level": "ERROR",
                "message": "boom",
                "name": "app.handler.mlx_lm",
                "function": "generate",
                "line": 512,
            },
            "small-chat",
        )

        record = parent_sink[0].record
        assert record["level"].name == "ERROR"
        assert record["name"] == "app.handler.mlx_lm"
        assert record["function"] == "generate"
        assert record["line"] == 512

    def test_without_a_model_name_the_message_is_untouched(self, parent_sink) -> None:
        emit_record({"type": LOG_MESSAGE_TYPE, "level": "INFO", "message": "loaded"})

        assert [str(record) for record in parent_sink] == ["loaded\n"]

    def test_a_record_without_an_origin_still_gets_through(self, parent_sink) -> None:
        # Origin fields are always sent, but a truncated or hand-built message
        # must not raise on the reader thread.
        emit_record({"type": LOG_MESSAGE_TYPE, "level": "INFO", "message": "bare"}, "m")

        record = parent_sink[0].record
        assert record["name"] == "app.core.log_relay"
        assert isinstance(record["line"], int)

    def test_unknown_level_does_not_lose_the_record(self, parent_sink) -> None:
        # A level registered only inside the child would otherwise raise
        # ValueError on the reader thread and drop the message.
        emit_record(
            {"type": LOG_MESSAGE_TYPE, "level": "TRACE_CUSTOM", "message": "odd"},
            "m",
        )

        assert parent_sink[0].record["level"].name == "INFO"
        assert "odd" in str(parent_sink[0])

    def test_braces_in_the_message_are_not_formatted(self, parent_sink) -> None:
        # Handler output routinely contains JSON (tool calls), and loguru would
        # raise KeyError if the relayed text were treated as a format string.
        emit_record(
            {
                "type": LOG_MESSAGE_TYPE,
                "level": "INFO",
                "message": '{"name": "list_files", "arguments": {}}',
            },
            "m",
        )

        assert '{"name": "list_files", "arguments": {}}' in str(parent_sink[0])


# ---------------------------------------------------------------------------
# Wiring: proxy plumbing between the two halves
# ---------------------------------------------------------------------------


class TestProxyWiring:
    """The proxy has to route log messages and tell children which level to use."""

    @staticmethod
    def _proxy() -> HandlerProcessProxy:
        model_cfg = ModelEntryConfig(
            model_path="dummy-model",
            model_type="lm",
            served_model_name="dummy-model",
        )
        return HandlerProcessProxy(
            model_cfg_dict=model_cfg.__dict__.copy(),
            model_type=model_cfg.model_type,
            model_path=model_cfg.model_path,
            served_model_name=model_cfg.served_model_name,
        )

    def test_reader_thread_emits_log_messages(self, parent_sink) -> None:
        proxy = self._proxy()
        proxy._running = True

        payload = {
            "type": LOG_MESSAGE_TYPE,
            "level": "WARNING",
            "message": "kv cache is full",
        }

        def _one_message(timeout: float = 0.0) -> dict[str, Any]:
            proxy._running = False
            return payload

        proxy._response_queue.get = _one_message  # type: ignore[method-assign]
        proxy._response_reader()

        assert [str(record) for record in parent_sink] == ["'dummy-model': kv cache is full\n"]

    def test_log_messages_are_not_delivered_to_pending_callers(self, parent_sink) -> None:
        # A log record has no request id; routing it as a response would hand a
        # waiting caller a message it cannot parse (or hijack "__ready__").
        proxy = self._proxy()
        proxy._running = True
        proxy._pending["__ready__"] = object()  # type: ignore[assignment]
        proxy._loop = object()  # type: ignore[assignment]

        def _one_message(timeout: float = 0.0) -> dict[str, Any]:
            proxy._running = False
            return {"type": LOG_MESSAGE_TYPE, "level": "INFO", "message": "hi"}

        proxy._response_queue.get = _one_message  # type: ignore[method-assign]
        # A real loop/queue would raise on use; reaching the log branch first is
        # what keeps this from blowing up.
        proxy._response_reader()

        assert len(parent_sink) == 1

    def test_a_failing_log_record_does_not_kill_the_reader(
        self, parent_sink, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The reader thread delivers every response for the model, so an
        # exception while re-emitting a record must not escape the loop and
        # leave later chunks (and every waiting request) stranded.
        proxy = self._proxy()
        proxy._running = True
        pending = _CollectingQueue()
        proxy._pending["req-1"] = pending  # type: ignore[assignment]
        proxy._loop = _ImmediateLoop()  # type: ignore[assignment]

        def _boom(payload: dict[str, Any], model_name: str | None = None) -> None:
            raise RuntimeError("sink is on fire")

        monkeypatch.setattr(handler_process, "emit_record", _boom)

        responses = [
            {"type": LOG_MESSAGE_TYPE, "level": "INFO", "message": "poison"},
            {"id": "req-1", "chunk": "still flowing"},
        ]

        def _next(timeout: float = 0.0) -> dict[str, Any]:
            response = responses.pop(0)
            if not responses:
                proxy._running = False
            return response

        proxy._response_queue.get = _next  # type: ignore[method-assign]
        proxy._response_reader()

        assert pending.items == [{"id": "req-1", "chunk": "still flowing"}]

    def test_child_is_told_the_configured_level(self, tmp_path, preserve_sinks) -> None:
        configure_logging(log_file=str(tmp_path / "app.log"), log_level="DEBUG")

        assert get_log_level() == "DEBUG"

    def test_worker_accepts_the_level_the_proxy_passes(self) -> None:
        # The spawn call site passes get_log_level() positionally; keep the
        # child's signature able to receive it.
        import inspect

        params = list(inspect.signature(_handler_worker).parameters)
        assert params[-1] == "log_level"

    @pytest.mark.asyncio
    async def test_spawn_and_respawn_both_hand_over_the_level(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A child that is not told the level would fall back to the default and
        # silently drop DEBUG records the operator asked for. The restart path
        # builds its own args tuple, so it is checked too.
        set_log_level("DEBUG")
        spawned: list[_FakeProcess] = []
        proxy = self._proxy()
        proxy._ctx = _FakeContext(spawned)  # type: ignore[assignment]

        async def _ready(self, ready_queue, timeout=300):  # noqa: ANN001, ARG001
            return {"success": True}

        monkeypatch.setattr(HandlerProcessProxy, "_wait_for_ready", _ready)

        await proxy.start({"queue_size": 1, "timeout": 5})
        await proxy._restart()
        proxy._running = False

        assert len(spawned) == 2
        for process in spawned:
            assert process.target is _handler_worker
            # (cfg, queue_config, request_q, response_q, control_q, log_level)
            assert len(process.args) == 6
            assert process.args[-1] == "DEBUG"


# ---------------------------------------------------------------------------
# Startup: the banner has to land in the file too
# ---------------------------------------------------------------------------


class TestStartupOrdering:
    """The banner is only in the log file if the sinks exist when it is logged."""

    @staticmethod
    def _patch(monkeypatch: pytest.MonkeyPatch, entrypoint: Any, banner: str) -> list[str]:
        """Record the order of setup / banner / serve for one entrypoint.

        The globals of the coroutine itself are patched rather than the module
        reached via ``app.main``: another test in the suite swaps
        ``sys.modules["app.main"]`` for a stub, which leaves the package
        attribute pointing at a second, stale copy of the module.
        """
        namespace = entrypoint.__globals__
        calls: list[str] = []

        class _FakeServer:
            def __init__(self, uvconfig: Any) -> None:
                self._uvconfig = uvconfig

            async def serve(self) -> None:
                calls.append("serve")

        monkeypatch.setitem(
            namespace, "setup_server", lambda config: calls.append("setup") or object()
        )
        monkeypatch.setitem(namespace, banner, lambda config: calls.append("banner"))
        monkeypatch.setattr(namespace["uvicorn"], "Server", _FakeServer)
        return calls

    @pytest.mark.asyncio
    async def test_single_model_sets_up_before_the_banner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.config import MLXServerConfig
        from app.main import start

        calls = self._patch(monkeypatch, start, "print_startup_banner")

        await start(MLXServerConfig(model_path="x"))

        assert calls == ["setup", "banner", "serve"]

    @pytest.mark.asyncio
    async def test_multi_model_sets_up_before_the_banner(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.config import MultiModelServerConfig
        from app.main import start_multi

        calls = self._patch(monkeypatch, start_multi, "print_multi_startup_banner")

        await start_multi(MultiModelServerConfig(models=[]))

        assert calls == ["setup", "banner", "serve"]


# ---------------------------------------------------------------------------
# Rotation: the relayed volume must not grow the file forever
# ---------------------------------------------------------------------------


class TestRotationSettings:
    """Rotation/retention are configurable and bounded by default."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (5, 5),
            ("5", 5),  # YAML/CLI hand us strings; a count must stay a count
            ("10 days", "10 days"),
            ("none", None),
            ("NONE", None),
            ("", None),
            (None, None),
        ],
    )
    def test_retention_parsing(self, value: Any, expected: Any) -> None:
        assert parse_log_retention(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("50 MB", "50 MB"), (" 1 day ", "1 day"), ("none", None), (None, None)],
    )
    def test_rotation_parsing(self, value: Any, expected: Any) -> None:
        assert parse_log_rotation(value) == expected

    def test_file_sink_gets_the_configured_rotation(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch, preserve_sinks
    ) -> None:
        added: list[dict[str, Any]] = []
        monkeypatch.setattr(
            log_relay.logger.__class__,
            "add",
            lambda self, sink, **kwargs: added.append({"sink": sink, **kwargs}) or 0,
        )

        configure_logging(
            log_file=str(tmp_path / "app.log"),
            log_rotation="10 MB",
            log_retention="3",
        )

        file_sink = added[-1]
        assert file_sink["sink"] == str(tmp_path / "app.log")
        assert file_sink["rotation"] == "10 MB"
        assert file_sink["retention"] == 3

    def test_defaults_are_bounded(self, tmp_path, preserve_sinks) -> None:
        # Without a size trigger the file grows for as long as the server runs,
        # which for an always-on daemon is forever; a count-based retention caps
        # total disk use at roughly (retention + 1) x rotation.
        from app.config import MLXServerConfig, MultiModelServerConfig

        for config in (
            MLXServerConfig(model_path="x"),
            MultiModelServerConfig(models=[]),
        ):
            assert config.log_rotation == "50 MB"
            assert config.log_retention == 5

    def test_cli_flags_reach_the_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A mismatch between the option name and the launch parameter would
        # leave the flags accepted but ignored.
        cli_module = importlib.import_module("app.cli")
        launch = cli_module.cli.commands["launch"]
        captured: dict[str, Any] = {}

        async def _fake_start_multi(config: Any) -> None:
            captured["config"] = config

        monkeypatch.setitem(_namespace_of(launch.callback), "start_multi", _fake_start_multi)

        result = CliRunner().invoke(
            cli_module.cli,
            [
                "launch",
                "--model-path",
                "dummy-model",
                "--log-rotation",
                "5 MB",
                "--log-retention",
                "2",
            ],
        )

        assert result.exit_code == 0, result.output
        assert captured["config"].log_rotation == "5 MB"
        assert captured["config"].log_retention == "2"

    def test_no_log_file_leaves_the_level_for_children(self, tmp_path, preserve_sinks) -> None:
        # File logging off still has to tell children the level, or a relayed
        # record would be filtered against the wrong one on the console sink.
        log_file = tmp_path / "app.log"

        configure_logging(log_file=str(log_file), no_log_file=True, log_level="WARNING")

        assert get_log_level() == "WARNING"
        assert not log_file.exists()

    def test_yaml_server_section_overrides(self, tmp_path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "server:\n"
            "  log_file: /tmp/mlx.log\n"
            "  log_rotation: 1 day\n"
            "  log_retention: 7\n"
            "models:\n"
            "  - model_path: mlx-community/Qwen3-0.6B-4bit\n"
            "    model_type: lm\n"
        )

        config = load_config_from_yaml(str(config_file))

        assert config.log_file == "/tmp/mlx.log"
        assert config.log_rotation == "1 day"
        assert config.log_retention == 7

    def test_set_log_level_normalizes(self) -> None:
        set_log_level("debug")
        assert get_log_level() == "DEBUG"
        set_log_level(None)
        assert get_log_level() == "INFO"
