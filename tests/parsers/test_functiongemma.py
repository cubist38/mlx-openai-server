"""Tests for the FunctionGemma parser."""

from app.parsers.functiongemma import FunctionGemmaToolParser


def test_functiongemma_tool_parsing_streaming() -> None:
    """Test streaming parsing of tool calls."""
    parser = FunctionGemmaToolParser()

    chunks = [
        "I will call a function",
        "<start_function_call>call:",
        "get_weather",
        "{param:<escape>value<escape>}",
        "<end_function_call>",
        "\n\n",
        "I will call another function",
        "<start_function_call>call:get_time{}<end_function_call>",
    ]
    tool_call_results = []
    is_complete_flags = []

    for chunk in chunks:
        tool_calls, is_complete = parser.extract_tool_calls_streaming(chunk)
        if tool_calls:
            tool_call_results.append(tool_calls)
            is_complete_flags.append(is_complete)

    # Verify tool parser extracted content correctly
    assert len(tool_call_results) > 0
    # Find the complete tool call results
    complete_tool_calls = []
    for result in tool_call_results:
        if isinstance(result, dict) and "tool_calls" in result:
            complete_tool_calls.extend(result["tool_calls"])

    assert len(complete_tool_calls) == 2
    # Verify first tool call
    assert complete_tool_calls[0]["name"] == "get_weather"
    assert complete_tool_calls[0]["arguments"] == '{"param": "value"}'
    # Verify second tool call
    assert complete_tool_calls[1]["name"] == "get_time"
    assert complete_tool_calls[1]["arguments"] == "{}"


if __name__ == "__main__":
    test_functiongemma_tool_parsing_streaming()

