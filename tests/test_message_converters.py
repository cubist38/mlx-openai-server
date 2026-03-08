"""Tests for message converter auto-detection."""

from __future__ import annotations

import pytest

from app.config import MLXServerConfig, ModelEntryConfig
from app.message_converters import (
    GLM4MoEMessageConverter,
    MessageConverterManager,
    resolve_message_converter_name,
)


@pytest.mark.parametrize(
    ("tool_parser_name", "expected_converter_name"),
    [
        ("glm4_moe", "glm4_moe"),
        ("qwen3_coder", "qwen3_coder"),
        ("minimax_m2", "minimax_m2"),
        ("qwen3", None),
        ("step_35", "step_35"),
    ],
)
def test_resolve_message_converter_name_from_tool_parser(
    tool_parser_name: str,
    expected_converter_name: str | None,
) -> None:
    """Resolve a converter automatically from the tool parser name."""
    assert (
        resolve_message_converter_name(tool_parser_name=tool_parser_name) == expected_converter_name
    )


def test_resolve_message_converter_name_from_reasoning_parser() -> None:
    """Fall back to the reasoning parser when no tool parser is supplied."""
    assert resolve_message_converter_name(reasoning_parser_name="glm4_moe") == "glm4_moe"


def test_create_converter_auto_detects_from_tool_parser() -> None:
    """Create a converter automatically from parser configuration."""
    converter = MessageConverterManager.create_converter(tool_parser_name="qwen3_coder")

    assert isinstance(converter, GLM4MoEMessageConverter)


def test_explicit_message_converter_takes_precedence() -> None:
    """Prefer an explicit converter over automatic detection."""
    assert (
        resolve_message_converter_name(
            converter_name="GLM4_MOE",
            tool_parser_name="qwen3_coder",
        )
        == "glm4_moe"
    )


def test_server_config_auto_detects_message_converter() -> None:
    """Populate the message converter automatically for single-model config."""
    config = MLXServerConfig(
        model_path="mlx-community/Qwen3-Coder-Next-4bit",
        tool_call_parser="qwen3_coder",
    )

    assert config.message_converter == "qwen3_coder"


def test_model_entry_config_auto_detects_message_converter() -> None:
    """Populate the message converter automatically for YAML model entries."""
    config = ModelEntryConfig(
        model_path="mlx-community/GLM-4.7-Flash-8bit",
        model_type="multimodal",
        tool_call_parser="glm4_moe",
    )

    assert config.message_converter == "glm4_moe"
