"""Registry contract tests for the mixed-think tool-handoff parser naming."""

from __future__ import annotations

from app import parsers
from app.parsers.function_parameter import FunctionParameterToolParser


def test_mixed_think_tool_handoff_reasoning_parser_is_exported() -> None:
    """The semantic reasoning parser class should be exported from app.parsers."""
    assert hasattr(
        parsers,
        "MixedThinkToolHandoffReasoningParser",
    ), "Expected semantic parser class export in app.parsers."


def test_mixed_think_tool_handoff_reasoning_key_maps_to_semantic_class() -> None:
    """Semantic reasoning parser key should resolve to semantic parser class."""
    assert "mixed_think_tool_handoff" in parsers.REASONING_PARSER_MAP
    semantic_cls = getattr(parsers, "MixedThinkToolHandoffReasoningParser")
    assert parsers.REASONING_PARSER_MAP["mixed_think_tool_handoff"] is semantic_cls


def test_step35_reasoning_alias_maps_to_semantic_class() -> None:
    """Legacy step_35 key should resolve to a compatibility subclass of semantic parser."""
    semantic_cls = getattr(parsers, "MixedThinkToolHandoffReasoningParser")
    step35_cls = parsers.REASONING_PARSER_MAP["step_35"]
    assert issubclass(step35_cls, semantic_cls)


def test_parser_manager_accepts_semantic_reasoning_name() -> None:
    """ParserManager should instantiate semantic reasoning parser via new key."""
    semantic_cls = getattr(parsers, "MixedThinkToolHandoffReasoningParser")
    result = parsers.ParserManager.create_parsers(
        reasoning_parser_name="mixed_think_tool_handoff",
        tool_parser_name="step_35",
    )
    assert isinstance(result.reasoning_parser, semantic_cls)
    assert isinstance(result.tool_parser, FunctionParameterToolParser)
