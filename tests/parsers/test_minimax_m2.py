"""Tests for the MiniMax M2 parser (tool parsing only)."""

import json

from app.parsers.minimax_m2 import MiniMaxM2ToolParser


def test_minimax_m2_tool_parsing_streaming() -> None:
    """Test streaming parsing of tool calls."""
    tool_parser = MiniMaxM2ToolParser()

    chunks = [
        "I am ",
        "thinking about the",
        "problem",
        ".</think><minimax:tool_call>",
        '<invoke name="tool_name">\n',
        '<parameter name="argument_name">argument_value</parameter>\n',
        '<parameter name="argument_name">argument_value</parameter>\n',
        "</invoke>\n",
        "</minimax:tool_call>",
    ]
    tool_call_results = []
    is_complete_flags = []

    for chunk in chunks:
        if chunk is None:
            continue
        if tool_parser:
            parsed_content, is_complete = tool_parser.extract_tool_calls_streaming(chunk)
            if parsed_content:
                tool_call_results.append(parsed_content)
                is_complete_flags.append(is_complete)
            if is_complete:
                tool_parser = None
            continue

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
    # Verify that arguments is a JSON string containing the expected parameter
    # Note: Since there are two parameters with the same name, the second overwrites the first
    assert json.loads(complete_tool_call["tool_calls"][0]["arguments"]) == {
        "argument_name": "argument_value"
    }


if __name__ == "__main__":
    test_minimax_m2_tool_parsing_streaming()
