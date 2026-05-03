"""Tests for prompt cache insertion behavior on cancellation and errors."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import suppress
import importlib
from pathlib import Path
import sys
import types
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException
import pytest


def _install_fake_mlx_cache_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a lightweight mlx_lm.models.cache stub used by prompt cache imports."""

    fake_mlx_lm = types.ModuleType("mlx_lm")
    fake_models = types.ModuleType("mlx_lm.models")
    fake_cache = types.ModuleType("mlx_lm.models.cache")

    def can_trim_prompt_cache(cache: list[Any]) -> bool:
        return bool(cache)

    def trim_prompt_cache(cache: list[Any], num_tokens: int) -> int:
        return min(num_tokens, len(cache))

    fake_cache.can_trim_prompt_cache = can_trim_prompt_cache
    fake_cache.trim_prompt_cache = trim_prompt_cache
    fake_models.cache = fake_cache
    fake_mlx_lm.models = fake_models

    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", fake_models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", fake_cache)


def _load_handler_class(monkeypatch: pytest.MonkeyPatch) -> type[Any]:
    """Import ``MLXLMHandler`` while stubbing MLX-backed imports for CI safety."""

    repo_root = Path(__file__).resolve().parents[1]

    # Bypass app/handler/__init__.py eager imports (which pull MLX-backed modules).
    fake_handler_pkg = types.ModuleType("app.handler")
    fake_handler_pkg.__path__ = [str(repo_root / "app" / "handler")]
    monkeypatch.setitem(sys.modules, "app.handler", fake_handler_pkg)

    # Stub model wrapper imported by app.handler.mlx_lm.
    fake_mlx_lm_model = types.ModuleType("app.models.mlx_lm")

    class _FakeMLXLM:
        pass

    fake_mlx_lm_model.MLX_LM = _FakeMLXLM
    monkeypatch.setitem(sys.modules, "app.models.mlx_lm", fake_mlx_lm_model)

    _install_fake_mlx_cache_module(monkeypatch)
    sys.modules.pop("app.handler.mlx_lm", None)
    handler_module = importlib.import_module("app.handler.mlx_lm")
    handler_module = importlib.reload(handler_module)
    return handler_module.MLXLMHandler


def _submit_stream_on_current_thread(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
    """Run a stream worker function and expose it as an async generator."""
    func = args[0]
    func_args = args[1:]

    async def _gen() -> AsyncGenerator[Any, None]:
        stream = func(*func_args, **kwargs)
        try:
            for item in stream:
                yield item
        finally:
            stream.close()

    return _gen()


@pytest.mark.asyncio
async def test_prompt_cache_inserted_on_stream_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that the prompt cache is inserted when the streaming generator
    is cancelled (i.e., closed via GeneratorExit) before completion.

    """
    # Mock dependencies
    mock_model = Mock()
    mock_model.create_input_prompt.return_value = "prompt"
    mock_model.encode_prompt.return_value = [1, 2, 3]
    mock_model.create_prompt_cache.return_value = Mock(name="cache_obj")

    chunk = Mock()
    chunk.text = "Hello"
    chunk.token = 4
    mock_model.side_effect = lambda *_args, **_kwargs: iter([chunk])

    mock_inference_worker = Mock()
    mock_inference_worker.submit_stream = Mock(side_effect=_submit_stream_on_current_thread)

    mock_prompt_cache = Mock()
    cached_prompt = [Mock(name="cache_obj")]
    mock_prompt_cache.fetch_nearest_cache.return_value = (cached_prompt, [])

    mlx_lm_handler = _load_handler_class(monkeypatch)
    handler = mlx_lm_handler.__new__(mlx_lm_handler)
    handler.model = mock_model
    handler.inference_worker = mock_inference_worker
    handler.prompt_cache = mock_prompt_cache
    handler.reasoning_parser_name = ""
    handler.tool_parser_name = ""
    handler.debug = False
    handler.model_path = "/fake"
    handler.model_created = 12345
    handler.model_type = "text"
    handler.enable_auto_tool_choice = False
    handler.message_converter = Mock()
    handler._generation_lock = __import__("threading").RLock()

    # Mock internal methods
    handler._prepare_text_request = AsyncMock(return_value=([], {}))
    handler.refine_messages = Mock(return_value=[])

    # Mock ParserManager to avoid parser logic
    mock_parsers_result = Mock()
    mock_parsers_result.is_unified = False
    mock_parsers_result.reasoning_parser = None
    mock_parsers_result.tool_parser = None
    with patch("app.handler.mlx_lm.ParserManager.create_parsers", return_value=mock_parsers_result):
        fake_request = Mock()
        gen = handler.generate_text_stream(fake_request)
        try:
            async for _ in gen:
                break
        finally:
            # Explicitly close the generator to trigger GeneratorExit.
            await gen.aclose()
            await asyncio.sleep(0)

        submitted = mock_inference_worker.submit_stream.call_args[0]
        assert submitted[0].__name__ == "_stream_with_lock_and_cache_persist"


@pytest.mark.asyncio
async def test_prompt_cache_inserted_on_normal_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that the prompt cache is inserted when the streaming generator completes normally."""
    mock_model = Mock()
    mock_model.create_input_prompt.return_value = "prompt"
    input_ids = [1, 2, 3]
    mock_model.encode_prompt.return_value = input_ids
    mock_model.create_prompt_cache.return_value = Mock(name="cache_obj")

    chunks = []
    for token in [4, 5, 6]:
        chunk = Mock()
        chunk.text = f"Token {token}"
        chunk.token = token
        if token == 6:
            chunk.prompt_tokens = 10
            chunk.generation_tokens = 20
        chunks.append(chunk)
    mock_model.side_effect = lambda *_args, **_kwargs: iter(chunks)

    mock_inference_worker = Mock()
    mock_inference_worker.submit_stream = Mock(side_effect=_submit_stream_on_current_thread)

    mock_prompt_cache = Mock()
    cached_prompt = [Mock(name="cache_obj")]
    mock_prompt_cache.fetch_nearest_cache.return_value = (cached_prompt, [])

    mlx_lm_handler = _load_handler_class(monkeypatch)
    handler = mlx_lm_handler.__new__(mlx_lm_handler)
    handler.model = mock_model
    handler.inference_worker = mock_inference_worker
    handler.prompt_cache = mock_prompt_cache
    handler.reasoning_parser_name = ""
    handler.tool_parser_name = ""
    handler.debug = False
    handler.model_path = "/fake"
    handler.model_created = 12345
    handler.model_type = "text"
    handler.enable_auto_tool_choice = False
    handler.message_converter = Mock()
    handler._generation_lock = __import__("threading").RLock()

    handler._prepare_text_request = AsyncMock(return_value=([], {}))
    handler.refine_messages = Mock(return_value=[])

    mock_parsers_result = Mock()
    mock_parsers_result.is_unified = False
    mock_parsers_result.reasoning_parser = None
    mock_parsers_result.tool_parser = None
    with patch("app.handler.mlx_lm.ParserManager.create_parsers", return_value=mock_parsers_result):
        fake_request = Mock()
        gen = handler.generate_text_stream(fake_request)
        try:
            async for _ in gen:
                pass
        finally:
            await gen.aclose()
            await asyncio.sleep(0)

        mock_prompt_cache.insert_cache.assert_called_once()
        call_args = mock_prompt_cache.insert_cache.call_args
        inserted_key = call_args[0][0]
        inserted_cache = call_args[0][1]
        assert inserted_key == [1, 2, 3, 4, 5, 6]
        assert inserted_cache is cached_prompt


@pytest.mark.asyncio
async def test_prompt_cache_inserted_on_cancellation_before_any_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that the prompt cache is inserted when the generator is closed before any chunk is processed, using only the initial input_ids."""
    mock_model = Mock()
    mock_model.create_input_prompt.return_value = "prompt"
    input_ids = [1, 2, 3]
    mock_model.encode_prompt.return_value = input_ids
    mock_model.create_prompt_cache.return_value = Mock(name="cache_obj")

    mock_model.side_effect = lambda *_args, **_kwargs: iter([])

    mock_inference_worker = Mock()
    mock_inference_worker.submit_stream = Mock(side_effect=_submit_stream_on_current_thread)

    mock_prompt_cache = Mock()
    cached_prompt = [Mock(name="cache_obj")]
    mock_prompt_cache.fetch_nearest_cache.return_value = (cached_prompt, [])

    mlx_lm_handler = _load_handler_class(monkeypatch)
    handler = mlx_lm_handler.__new__(mlx_lm_handler)
    handler.model = mock_model
    handler.inference_worker = mock_inference_worker
    handler.prompt_cache = mock_prompt_cache
    handler.reasoning_parser_name = ""
    handler.tool_parser_name = ""
    handler.debug = False
    handler.model_path = "/fake"
    handler.model_created = 12345
    handler.model_type = "text"
    handler.enable_auto_tool_choice = False
    handler.message_converter = Mock()
    handler._generation_lock = __import__("threading").RLock()

    handler._prepare_text_request = AsyncMock(return_value=([], {}))
    handler.refine_messages = Mock(return_value=[])

    mock_parsers_result = Mock()
    mock_parsers_result.is_unified = False
    mock_parsers_result.reasoning_parser = None
    mock_parsers_result.tool_parser = None
    with patch("app.handler.mlx_lm.ParserManager.create_parsers", return_value=mock_parsers_result):
        fake_request = Mock()
        gen = handler.generate_text_stream(fake_request)
        task = asyncio.create_task(gen.__anext__())
        await asyncio.sleep(0)
        task.cancel()
        with suppress(asyncio.CancelledError, StopAsyncIteration, GeneratorExit, HTTPException):
            await task
        await gen.aclose()

        mock_prompt_cache.insert_cache.assert_called_once()
        call_args = mock_prompt_cache.insert_cache.call_args
        inserted_key = call_args[0][0]
        inserted_cache = call_args[0][1]
        assert inserted_key == [1, 2, 3]
        assert inserted_cache is cached_prompt


@pytest.mark.asyncio
async def test_prompt_cache_inserted_on_cancellation_after_multiple_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that the prompt cache is inserted after processing multiple chunks and then cancelling, with the cache key including all processed tokens."""
    mock_model = Mock()
    mock_model.create_input_prompt.return_value = "prompt"
    input_ids = [1, 2, 3]
    mock_model.encode_prompt.return_value = input_ids
    mock_model.create_prompt_cache.return_value = Mock(name="cache_obj")

    chunks = []
    for token in [4, 5, 6]:
        chunk = Mock()
        chunk.text = f"Token {token}"
        chunk.token = token
        chunks.append(chunk)
    mock_model.side_effect = lambda *_args, **_kwargs: iter(chunks)

    mock_inference_worker = Mock()
    mock_inference_worker.submit_stream = Mock(side_effect=_submit_stream_on_current_thread)

    mock_prompt_cache = Mock()
    cached_prompt = [Mock(name="cache_obj")]
    mock_prompt_cache.fetch_nearest_cache.return_value = (cached_prompt, [])

    mlx_lm_handler = _load_handler_class(monkeypatch)
    handler = mlx_lm_handler.__new__(mlx_lm_handler)
    handler.model = mock_model
    handler.inference_worker = mock_inference_worker
    handler.prompt_cache = mock_prompt_cache
    handler.reasoning_parser_name = ""
    handler.tool_parser_name = ""
    handler.debug = False
    handler.model_path = "/fake"
    handler.model_created = 12345
    handler.model_type = "text"
    handler.enable_auto_tool_choice = False
    handler.message_converter = Mock()
    handler._generation_lock = __import__("threading").RLock()

    handler._prepare_text_request = AsyncMock(return_value=([], {}))
    handler.refine_messages = Mock(return_value=[])

    mock_parsers_result = Mock()
    mock_parsers_result.is_unified = False
    mock_parsers_result.reasoning_parser = None
    mock_parsers_result.tool_parser = None
    with patch("app.handler.mlx_lm.ParserManager.create_parsers", return_value=mock_parsers_result):
        fake_request = Mock()
        gen = handler.generate_text_stream(fake_request)
        try:
            chunk_count = 0
            async for _ in gen:
                chunk_count += 1
                if chunk_count == 2:
                    break
        finally:
            await gen.aclose()

        submitted = mock_inference_worker.submit_stream.call_args[0]
        assert submitted[0].__name__ == "_stream_with_lock_and_cache_persist"


def test_worker_stream_wrapper_persists_cache_on_close(monkeypatch: pytest.MonkeyPatch) -> None:
    """The worker-side stream wrapper persists cache when the stream is closed."""
    mlx_lm_handler = _load_handler_class(monkeypatch)
    handler = mlx_lm_handler.__new__(mlx_lm_handler)
    handler._generation_lock = __import__("threading").RLock()
    handler.prompt_cache = Mock()

    chunks = []
    for token in [4, 5]:
        chunk = Mock()
        chunk.token = token
        chunks.append(chunk)
    handler.model = Mock(side_effect=lambda *_args, **_kwargs: iter(chunks))

    cache = [Mock(name="cache_obj")]
    stream = handler._stream_with_lock_and_cache_persist(
        [1, 2, 3],
        cache,
        input_ids=[3],
        prompt_cache=cache,
        stream=True,
    )

    first = next(stream)
    assert first.token == 4
    stream.close()

    handler.prompt_cache.insert_cache.assert_called_once_with([1, 2, 3, 4], cache)
