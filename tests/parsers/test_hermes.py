"""Tests for the Hermes parser."""

import json

from app.parsers.hermes import HermesReasoningParser, HermesToolParser


def test_hermes_reasoning_and_tool_parsing_streaming() -> None:
    """Test streaming parsing of reasoning and tool calls."""
    reasoning_parser = HermesReasoningParser()
    tool_parser = HermesToolParser()

    chunks = [
        "<think>I am ",
        "thinking about the",
        "problem",
        ".</think><tool_call>",
        '{"name": "tool_name",',
        '"arguments": {"argument_name": "argument_value"}}',
        "</tool_call>",
    ]
    after_thinking_close_content = None
    reasoning_results = []
    tool_call_results = []
    is_complete_flags = []

    for chunk in chunks:
        if chunk is None:
            continue
        if reasoning_parser:
            reasoning, is_complete = reasoning_parser.extract_reasoning_streaming(chunk)
            reasoning_results.append(reasoning)
            if is_complete:
                reasoning_parser = None
                if isinstance(reasoning, dict) and reasoning.get("after_reasoning_close_content"):
                    after_thinking_close_content = reasoning.get("after_reasoning_close_content")
            continue
        if after_thinking_close_content:
            chunk = after_thinking_close_content + chunk
            after_thinking_close_content = None
        if tool_parser:
            tool_calls, is_complete = tool_parser.extract_tool_calls_streaming(chunk)
            tool_call_results.append(tool_calls)
            is_complete_flags.append(is_complete)

    # Verify reasoning parser extracted content correctly
    assert len(reasoning_results) > 0
    # The parser returns individual chunks, not accumulated content
    # Check that we got reasoning results for the chunks with reasoning tags
    assert any(
        "reasoning_content" in result for result in reasoning_results if isinstance(result, dict)
    )
    # The final reasoning result should contain the closing tag and after content
    final_reasoning = reasoning_results[-1]
    assert isinstance(final_reasoning, dict)
    assert "reasoning_content" in final_reasoning
    assert "after_reasoning_close_content" in final_reasoning
    assert final_reasoning["after_reasoning_close_content"] == "<tool_call>"

    # Verify tool parser extracted content correctly
    assert len(tool_call_results) > 0
    # Find the complete tool call result
    complete_tool_call = None
    for result in tool_call_results:
        if isinstance(result, dict) and "tool_calls" in result:
            complete_tool_call = result
            break

    assert complete_tool_call is not None
    assert "tool_calls" in complete_tool_call
    assert len(complete_tool_call["tool_calls"]) == 1
    assert complete_tool_call["tool_calls"][0]["name"] == "tool_name"
    assert json.loads(complete_tool_call["tool_calls"][0]["arguments"]) == {
        "argument_name": "argument_value"
    }


if __name__ == "__main__":
    test_hermes_reasoning_and_tool_parsing_streaming()
