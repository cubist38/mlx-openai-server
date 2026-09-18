"""Tests for the JSON-schema and KV-cache parameters handed to generation.

Both features are configured far from where they take effect: a request's
``response_format`` becomes the ``schema`` key that
:meth:`MLX_LM.build_logits_processors` turns into a constrained-decoding
processor, and the handler's KV quantization settings become the ``kv_bits``
family forwarded to ``stream_generate``. A silent rename on either side would
disable the feature without any error, so the wiring is asserted here.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.schemas.openai import ChatCompletionRequest, Message

# The loader stubs the MLX-heavy imports so the handler class can be built
# without a model; it is shared rather than duplicated.
from tests.test_chat_completions_prompt_history import _load_mlx_lm_handler_class

CITY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}


def _prepare(
    response_format: dict[str, Any] | None = None,
    *,
    kv_bits: int | None = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
) -> dict[str, Any]:
    """Return the model params produced for a minimal chat request."""
    handler_cls = _load_mlx_lm_handler_class()
    handler = handler_cls.__new__(handler_cls)
    handler.kv_bits = kv_bits
    handler.kv_group_size = kv_group_size
    handler.quantized_kv_start = quantized_kv_start

    request = ChatCompletionRequest(
        model="local-text-model",
        messages=[Message(role="user", content="Name a city.")],
        response_format=response_format,
    )

    _, model_params = asyncio.run(handler._prepare_text_request(request))
    return model_params


def test_json_schema_response_format_becomes_the_schema_param() -> None:
    """``build_logits_processors`` looks for ``schema``; nothing else enables it."""
    params = _prepare({"type": "json_schema", "json_schema": {"name": "c", "schema": CITY_SCHEMA}})

    assert params["schema"] == CITY_SCHEMA


def test_json_object_response_format_does_not_set_a_schema() -> None:
    """Only ``json_schema`` carries a schema; ``json_object`` has none to apply."""
    params = _prepare({"type": "json_object"})

    assert params.get("schema") is None


def test_absent_response_format_leaves_generation_unconstrained() -> None:
    """A plain request must not acquire a constrained-decoding processor."""
    params = _prepare(None)

    assert params.get("schema") is None


def test_json_schema_without_a_schema_body_is_tolerated() -> None:
    """A malformed ``response_format`` must not raise mid-request."""
    params = _prepare({"type": "json_schema"})

    assert params.get("schema") is None


@pytest.mark.parametrize("kv_bits", [None, 4, 8])
def test_kv_quantization_settings_reach_the_generation_params(kv_bits: int | None) -> None:
    """The handler's KV settings are forwarded verbatim, including ``None``."""
    params = _prepare(None, kv_bits=kv_bits, kv_group_size=32, quantized_kv_start=512)

    assert params["kv_bits"] == kv_bits
    assert params["kv_group_size"] == 32
    assert params["quantized_kv_start"] == 512
