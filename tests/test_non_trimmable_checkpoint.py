"""Tests for prefix cache checkpoint support for non-trimmable hybrid models.

Validates the sentinel-based message boundary computation and the
checkpoint prefill path that enables cache reuse on models where
``can_trim_prompt_cache`` returns ``False`` (e.g. Qwen3.5, Nemotron-H).
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types
from typing import Any
from unittest.mock import Mock

import pytest

# ---------------------------------------------------------------------------
# Stubs — avoid importing real MLX / GPU modules in CI
# ---------------------------------------------------------------------------


def _install_fake_mlx_cache_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    trimmable: bool = True,
) -> None:
    """Install an ``mlx_lm.models.cache`` stub.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.
    trimmable : bool
        Value returned by ``can_trim_prompt_cache``.
    """
    fake_mlx_lm = types.ModuleType("mlx_lm")
    fake_models = types.ModuleType("mlx_lm.models")
    fake_cache = types.ModuleType("mlx_lm.models.cache")

    fake_cache.can_trim_prompt_cache = lambda _cache: trimmable
    fake_cache.trim_prompt_cache = lambda cache, n: min(n, len(cache))
    fake_cache.make_prompt_cache = lambda *_a, **_k: []
    fake_models.cache = fake_cache
    fake_mlx_lm.models = fake_models

    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", fake_models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", fake_cache)


def _load_prompt_cache_class(
    monkeypatch: pytest.MonkeyPatch,
    *,
    trimmable: bool = True,
) -> type:
    """Import ``LRUPromptCache`` with the fake ``mlx_lm.models.cache``."""
    _install_fake_mlx_cache_module(monkeypatch, trimmable=trimmable)
    sys.modules.pop("app.utils.prompt_cache", None)
    mod = importlib.import_module("app.utils.prompt_cache")
    mod = importlib.reload(mod)
    return mod.LRUPromptCache


def _load_handler_class(monkeypatch: pytest.MonkeyPatch) -> type:
    """Import ``MLXLMHandler`` while stubbing MLX-backed imports for CI."""
    repo_root = Path(__file__).resolve().parents[1]

    fake_handler_pkg = types.ModuleType("app.handler")
    fake_handler_pkg.__path__ = [str(repo_root / "app" / "handler")]
    monkeypatch.setitem(sys.modules, "app.handler", fake_handler_pkg)

    fake_mlx_lm_model = types.ModuleType("app.models.mlx_lm")

    class _FakeMLXLM:
        pass

    fake_mlx_lm_model.MLX_LM = _FakeMLXLM
    monkeypatch.setitem(sys.modules, "app.models.mlx_lm", fake_mlx_lm_model)

    _install_fake_mlx_cache_module(monkeypatch, trimmable=False)
    sys.modules.pop("app.handler.mlx_lm", None)
    handler_module = importlib.import_module("app.handler.mlx_lm")
    handler_module = importlib.reload(handler_module)
    return handler_module.MLXLMHandler


# ---------------------------------------------------------------------------
# LRUPromptCache: shorter-path reuse for non-trimmable caches
# ---------------------------------------------------------------------------


class TestNonTrimmableCacheReuse:
    """Verify the 'shorter cache' trie path enables reuse when trimming is impossible."""

    def test_checkpoint_prefix_reused_on_divergent_suffix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A checkpoint at the system-prompt boundary enables shorter-path cache hits."""
        LRUPromptCache = _load_prompt_cache_class(monkeypatch, trimmable=False)
        cache = LRUPromptCache(max_size=10)

        prefix_tokens = [1, 2, 3, 4, 5]
        full_request_1 = [*prefix_tokens, 10, 11, 12]
        full_request_2 = [*prefix_tokens, 20, 21, 22]

        fake_prefix_cache = [Mock(nbytes=100, state=[])]
        fake_full_cache = [Mock(nbytes=200, state=[])]

        cache.insert_cache(prefix_tokens, fake_prefix_cache, cache_type="system")
        cache.insert_cache(full_request_1, fake_full_cache)

        result_cache, rest = cache.fetch_nearest_cache(full_request_2)

        assert result_cache is not None, "Expected shorter-path cache hit"
        assert rest == [20, 21, 22], "Remaining tokens should be the divergent suffix"

    def test_no_checkpoint_means_no_reuse_on_non_trimmable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a checkpoint, non-trimmable caches get no reuse on divergent suffixes."""
        LRUPromptCache = _load_prompt_cache_class(monkeypatch, trimmable=False)
        cache = LRUPromptCache(max_size=10)

        full_request_1 = [1, 2, 3, 10, 11]
        full_request_2 = [1, 2, 3, 20, 21]

        fake_cache = [Mock(nbytes=200, state=[])]
        cache.insert_cache(full_request_1, fake_cache)

        result_cache, rest = cache.fetch_nearest_cache(full_request_2)

        assert result_cache is None, "No shorter prefix cached, no trim possible"
        assert rest == full_request_2

    def test_exact_match_still_works_for_non_trimmable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exact token matches always work regardless of trimmability."""
        LRUPromptCache = _load_prompt_cache_class(monkeypatch, trimmable=False)
        cache = LRUPromptCache(max_size=10)

        tokens = [1, 2, 3, 4, 5]
        fake_cache = [Mock(nbytes=100, state=[])]
        cache.insert_cache(tokens, fake_cache)

        result_cache, rest = cache.fetch_nearest_cache(tokens)

        assert result_cache is not None
        assert rest == []


# ---------------------------------------------------------------------------
# _compute_checkpoint_boundary (sentinel substitution)
# ---------------------------------------------------------------------------


class TestComputeCheckpointBoundary:
    """Test the sentinel-based message boundary computation."""

    def _make_handler(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        """Create a handler with a mocked model for boundary tests."""
        MLXLMHandler = _load_handler_class(monkeypatch)
        handler = MLXLMHandler.__new__(MLXLMHandler)

        mock_model = Mock()

        def fake_create_input_prompt(messages: list[dict], kwargs: dict) -> str:
            kwargs.pop("_partial_mode", None)
            parts = [f"<|{m['role']}|>{m.get('content', '')}" for m in messages]
            parts.append("<|assistant|>")
            return "".join(parts)

        def fake_encode_prompt(prompt: str) -> list[int]:
            return [ord(c) for c in prompt]

        mock_model.create_input_prompt = fake_create_input_prompt
        mock_model.encode_prompt = fake_encode_prompt
        handler.model = mock_model
        return handler

    def test_boundary_found_for_two_messages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The boundary separates system tokens from user message content."""
        handler = self._make_handler(monkeypatch)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello world"},
        ]
        full_prompt = handler.model.create_input_prompt(messages, {})
        input_ids = handler.model.encode_prompt(full_prompt)

        boundary = handler._compute_checkpoint_boundary(messages, input_ids, {})

        assert boundary is not None
        assert 0 < boundary < len(input_ids)

        sentinel_messages = [*messages[:-1], {"role": "user", "content": "x"}]
        sentinel_prompt = handler.model.create_input_prompt(sentinel_messages, {})
        sentinel_ids = handler.model.encode_prompt(sentinel_prompt)

        for i in range(boundary):
            assert input_ids[i] == sentinel_ids[i], f"Mismatch at position {i}"

    def test_returns_none_for_single_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No checkpoint boundary when there's only one message."""
        handler = self._make_handler(monkeypatch)
        messages = [{"role": "user", "content": "Hello"}]
        full_prompt = handler.model.create_input_prompt(messages, {})
        input_ids = handler.model.encode_prompt(full_prompt)

        boundary = handler._compute_checkpoint_boundary(messages, input_ids, {})
        assert boundary is None

    def test_returns_none_for_assistant_last_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No checkpoint when the last message is from the assistant."""
        handler = self._make_handler(monkeypatch)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "assistant", "content": "I can help"},
        ]
        full_prompt = handler.model.create_input_prompt(messages, {})
        input_ids = handler.model.encode_prompt(full_prompt)

        boundary = handler._compute_checkpoint_boundary(messages, input_ids, {})
        assert boundary is None

    def test_boundary_with_multi_turn_conversation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Boundary is at the last user message, not earlier ones."""
        handler = self._make_handler(monkeypatch)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
        ]
        full_prompt = handler.model.create_input_prompt(messages, {})
        input_ids = handler.model.encode_prompt(full_prompt)

        boundary = handler._compute_checkpoint_boundary(messages, input_ids, {})

        assert boundary is not None

        prefix_prompt = handler.model.create_input_prompt(
            [*messages[:-1], {"role": "user", "content": "x"}], {}
        )
        prefix_ids = handler.model.encode_prompt(prefix_prompt)
        common = 0
        for a, b in zip(input_ids, prefix_ids, strict=False):
            if a != b:
                break
            common += 1
        assert boundary == common

    def test_boundary_handles_template_error_gracefully(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returns None when the chat template raises an exception."""
        handler = self._make_handler(monkeypatch)
        original = handler.model.create_input_prompt

        call_count = 0

        def failing_prompt(messages, kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise ValueError("Template error")
            return original(messages, kwargs)

        handler.model.create_input_prompt = failing_prompt
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hello"},
        ]
        full_prompt = original(messages, {})
        input_ids = handler.model.encode_prompt(full_prompt)

        handler.model.create_input_prompt = lambda m, k: (_ for _ in ()).throw(ValueError("boom"))
        boundary = handler._compute_checkpoint_boundary(messages, input_ids, {})
        assert boundary is None


# ---------------------------------------------------------------------------
# MLX_LM.__call__ checkpoint integration
# ---------------------------------------------------------------------------


class TestModelCheckpointPrefill:
    """Test that checkpoint_position / checkpoint_callback split prefill correctly.

    These tests mock ``_prefill_cache`` and ``stream_generate`` on a real-ish
    ``MLX_LM`` instance to verify the ``__call__`` method actually invokes the
    checkpoint path with the correct prefix/suffix split.
    """

    @staticmethod
    def _make_model(monkeypatch: pytest.MonkeyPatch) -> Any:
        """Build a minimal ``MLX_LM``-like object with mocked internals."""
        _install_fake_mlx_cache_module(monkeypatch, trimmable=False)

        # Stub heavy imports so MLX_LM can be constructed without GPU
        fake_mx = types.ModuleType("mlx.core")
        fake_mx.array = lambda x: x
        fake_mx.random = Mock()
        monkeypatch.setitem(sys.modules, "mlx", types.ModuleType("mlx"))
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

        fake_generate = types.ModuleType("mlx_lm.generate")

        class _FakeGenResponse:
            def __init__(self, text: str, token: int, finish_reason: str | None = None):
                self.text = text
                self.token = token
                self.finish_reason = finish_reason
                self.peak_memory = 0.1
                self.generation_tps = 10.0
                self.prompt_tps = 100.0
                self.prompt_tokens = 5
                self.generation_tokens = 1

        fake_generate.GenerationResponse = _FakeGenResponse
        fake_generate.stream_generate = Mock(
            return_value=iter(
                [
                    _FakeGenResponse("hi", 99, "stop"),
                ]
            )
        )
        monkeypatch.setitem(sys.modules, "mlx_lm.generate", fake_generate)

        fake_sample = types.ModuleType("mlx_lm.sample_utils")
        fake_sample.make_logits_processors = lambda **kw: []
        fake_sample.make_sampler = lambda **kw: Mock()
        monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", fake_sample)

        fake_utils = types.ModuleType("mlx_lm.utils")
        fake_utils._download = Mock()
        fake_utils.load = Mock()
        monkeypatch.setitem(sys.modules, "mlx_lm.utils", fake_utils)

        # Stub outlines and its submodules (TransformerTokenizer import chain)
        fake_outlines = types.ModuleType("outlines")
        fake_outlines_models = types.ModuleType("outlines.models")
        fake_outlines_transformers = types.ModuleType("outlines.models.transformers")
        fake_outlines_transformers.TransformerTokenizer = type("TransformerTokenizer", (), {})
        fake_outlines_proc = types.ModuleType("outlines.processors")
        fake_outlines_proc.JSONLogitsProcessor = Mock()
        monkeypatch.setitem(sys.modules, "outlines", fake_outlines)
        monkeypatch.setitem(sys.modules, "outlines.models", fake_outlines_models)
        monkeypatch.setitem(sys.modules, "outlines.models.transformers", fake_outlines_transformers)
        monkeypatch.setitem(sys.modules, "outlines.processors", fake_outlines_proc)

        # Re-import with stubs in place
        sys.modules.pop("app.utils.outlines_transformer_tokenizer", None)
        sys.modules.pop("app.models.mlx_lm", None)
        mod = importlib.import_module("app.models.mlx_lm")
        mod = importlib.reload(mod)
        MLX_LM = mod.MLX_LM

        # Construct without __init__ to avoid real model loading
        model = MLX_LM.__new__(MLX_LM)
        model.model = Mock()
        model.tokenizer = Mock()
        model.tokenizer.eos_token_id = 0
        model.tokenizer.encode = Mock(return_value=[10])
        model.draft_model = None
        model.draft_tokenizer = None
        model.num_draft_tokens = 2
        model.context_length = None
        model.pad_token_id = 0
        model.bos_token = "<s>"
        model.model_type = "test"
        model.debug = False
        model.generation_config = {}
        model.outlines_tokenizer = Mock()
        model._cache_is_trimmable = False
        model._num_model_cache_layers = 2
        model._prefill_cache = Mock()

        return model, fake_generate

    def test_prefill_called_with_correct_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``_prefill_cache`` receives exactly the prefix tokens up to checkpoint_position."""
        model, _ = self._make_model(monkeypatch)

        input_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        fake_cache = [Mock(state=[]), Mock(state=[])]
        saved_caches: list[Any] = []

        def cb(cache: Any) -> None:
            saved_caches.append("called")

        model(
            input_ids,
            prompt_cache=fake_cache,
            stream=False,
            checkpoint_position=5,
            checkpoint_callback=cb,
        )

        # _prefill_cache should be called with the prefix [1,2,3,4,5]
        model._prefill_cache.assert_called_once_with([1, 2, 3, 4, 5], fake_cache)
        assert saved_caches == ["called"]

    def test_stream_generate_receives_suffix_after_checkpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After checkpoint prefill, ``stream_generate`` gets only the remaining suffix tokens."""
        model, fake_generate = self._make_model(monkeypatch)

        input_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        fake_cache = [Mock(state=[]), Mock(state=[])]

        model(
            input_ids,
            prompt_cache=fake_cache,
            stream=False,
            checkpoint_position=5,
            checkpoint_callback=lambda _: None,
        )

        # stream_generate should receive the suffix [6, 7, 8]
        call_args = fake_generate.stream_generate.call_args
        actual_input_ids = (
            call_args[1].get("prompt") if "prompt" in (call_args[1] or {}) else call_args[0][2]
        )
        assert list(actual_input_ids) == [6, 7, 8]

    def test_no_prefill_when_position_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No checkpoint logic runs when position is None."""
        model, fake_generate = self._make_model(monkeypatch)

        input_ids = [1, 2, 3, 4, 5]
        fake_cache = [Mock(state=[]), Mock(state=[])]

        model(input_ids, prompt_cache=fake_cache, stream=False)

        model._prefill_cache.assert_not_called()
        # stream_generate should receive the full input_ids
        call_args = fake_generate.stream_generate.call_args
        actual_input_ids = call_args[0][2]
        assert list(actual_input_ids) == [1, 2, 3, 4, 5]

    def test_no_prefill_when_position_out_of_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No checkpoint when position >= len(input_ids)."""
        model, _ = self._make_model(monkeypatch)

        input_ids = [1, 2, 3]
        fake_cache = [Mock(state=[]), Mock(state=[])]

        model(
            input_ids,
            prompt_cache=fake_cache,
            stream=False,
            checkpoint_position=10,
            checkpoint_callback=lambda _: None,
        )

        model._prefill_cache.assert_not_called()

    def test_no_prefill_when_cache_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No checkpoint when prompt_cache is None."""
        model, _ = self._make_model(monkeypatch)

        input_ids = [1, 2, 3, 4, 5]

        model(
            input_ids,
            prompt_cache=None,
            stream=False,
            checkpoint_position=3,
            checkpoint_callback=lambda _: None,
        )

        model._prefill_cache.assert_not_called()


# ---------------------------------------------------------------------------
# _build_inference_context: checkpoint on shorter cache hits
# ---------------------------------------------------------------------------


class TestCheckpointOnShorterCacheHit:
    """Verify checkpoints are created even when a shorter cache hit exists.

    When a non-trimmable model gets a shorter cache hit, the checkpoint
    position must be adjusted relative to rest_input_ids so that the
    model's __call__ receives a valid offset into the suffix it processes.
    """

    def _make_handler(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        """Create a handler with a mocked model for checkpoint-on-hit tests."""
        MLXLMHandler = _load_handler_class(monkeypatch)
        handler = MLXLMHandler.__new__(MLXLMHandler)

        mock_model = Mock()

        def fake_create_input_prompt(messages: list[dict], kwargs: dict) -> str:
            kwargs.pop("_partial_mode", None)
            parts = [f"<|{m['role']}|>{m.get('content', '')}" for m in messages]
            parts.append("<|assistant|>")
            return "".join(parts)

        def fake_encode_prompt(prompt: str) -> list[int]:
            return [ord(c) for c in prompt]

        mock_model.create_input_prompt = fake_create_input_prompt
        mock_model.encode_prompt = fake_encode_prompt
        mock_model.cache_is_trimmable = False
        mock_model.create_prompt_cache = Mock(return_value=[Mock(state=[])])
        handler.model = mock_model
        handler.debug = False
        return handler

    def test_checkpoint_position_adjusted_for_shorter_hit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """checkpoint_position is relative to rest_input_ids, not full input_ids."""
        handler = self._make_handler(monkeypatch)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
        ]
        full_prompt = handler.model.create_input_prompt(messages, {})
        input_ids = handler.model.encode_prompt(full_prompt)

        boundary = handler._compute_checkpoint_boundary(messages, input_ids, {})
        assert boundary is not None

        # Simulate a shorter cache hit covering the first 10 tokens
        cached_prefix_len = 10
        rest_input_ids = input_ids[cached_prefix_len:]

        # The checkpoint position should be adjusted for the shorter hit
        assert boundary > cached_prefix_len, (
            "Boundary must be beyond cached prefix for this test to be meaningful"
        )
        expected_checkpoint_pos = boundary - cached_prefix_len
        assert 0 < expected_checkpoint_pos < len(rest_input_ids)

    def test_no_checkpoint_when_boundary_within_cached_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No checkpoint needed when boundary falls within the already-cached portion."""
        handler = self._make_handler(monkeypatch)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        full_prompt = handler.model.create_input_prompt(messages, {})
        input_ids = handler.model.encode_prompt(full_prompt)

        boundary = handler._compute_checkpoint_boundary(messages, input_ids, {})
        assert boundary is not None

        # If cached_prefix_len >= boundary, no checkpoint should be set
        cached_prefix_len = boundary + 5
        assert cached_prefix_len >= boundary, "Cached prefix covers the boundary"


class TestNonBatchExactCacheHit:
    """Verify fallback generation never receives an empty prompt suffix."""

    def _make_handler(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        """Create a handler with a fake prompt cache for exact-hit tests."""
        MLXLMHandler = _load_handler_class(monkeypatch)
        handler = MLXLMHandler.__new__(MLXLMHandler)
        handler.prompt_cache = Mock()
        return handler

    def test_discards_unbackable_exact_hit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Non-trimmable exact hits without a shorter checkpoint fall back to full prefill."""
        handler = self._make_handler(monkeypatch)
        input_ids = [1, 2, 3, 4]
        exact_cache = [Mock(state=[])]
        handler.prompt_cache.fetch_nearest_cache.return_value = (None, input_ids[:-1])

        cache, rest = handler._normalize_nonbatch_cache_hit(input_ids, exact_cache, [])

        assert cache is None
        assert rest == input_ids

    def test_uses_shorter_checkpoint_for_exact_hit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A shorter checkpoint can back off a non-trimmable exact cache hit."""
        handler = self._make_handler(monkeypatch)
        input_ids = [1, 2, 3, 4]
        exact_cache = [Mock(state=[])]
        prefix_cache = [Mock(state=[])]
        handler.prompt_cache.fetch_nearest_cache.return_value = (prefix_cache, [])

        cache, rest = handler._normalize_nonbatch_cache_hit(input_ids, exact_cache, [])

        assert cache is prefix_cache
        assert rest == [4]

    def test_trims_trimmable_exact_hit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Trimmable exact hits are backed off by one token in-place."""
        handler = self._make_handler(monkeypatch)
        cache_module = sys.modules["mlx_lm.models.cache"]
        trim_calls: list[tuple[Any, int]] = []
        cache_module.can_trim_prompt_cache = lambda _cache: True
        cache_module.trim_prompt_cache = lambda cache, n: trim_calls.append((cache, n))

        input_ids = [1, 2, 3, 4]
        exact_cache = [Mock(state=[])]

        cache, rest = handler._normalize_nonbatch_cache_hit(input_ids, exact_cache, [])

        assert cache is exact_cache
        assert rest == [4]
        assert trim_calls == [(exact_cache, 1)]
