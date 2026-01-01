"""Tests for the Harmony parser."""

from app.parsers.harmony import HarmonyParser


def test_harmony_reasoning_and_tool_parsing_streaming() -> None:
    """Test streaming parsing of reasoning and tool calls."""
    parser = HarmonyParser()

    chunks = [
        "<|channel|>",
        "analysis",
        "<|message|>",
        "We",
        " need",
        " to",
        " call",
        " get_weather",
        " function",
        " with",
        " city",
        " Tokyo",
        ".",
        "<|end|>",
        "<|start|>",
        "assistant",
        "<|channel|>",
        "commentary",
        " to=functions.get_weather ",
        "<|constrain|>",
        "json",
        "<|message|>",
        "{",
        "\"city\"",
        ":",
        "\"Tokyo\"",
        "}",
        "<|call|>"
    ]

    results = []
    complete_flags = []

    for chunk in chunks:
        parsed_content, is_complete = parser.parse_streaming(chunk)
        if parsed_content:
            results.append(parsed_content)
        complete_flags.append(is_complete)

    # Verify we got results
    assert len(results) > 0
    # Check that we got reasoning results for the chunks with reasoning tags
    assert any("reasoning_content" in result for result in results if isinstance(result, dict))

    # Verify final result contains tool calls
    final_result = results[-1]
    assert "tool_calls" in final_result
    assert final_result["tool_calls"] is not None
    assert len(final_result["tool_calls"]) == 1
    
    tool_call = final_result["tool_calls"][0]
    assert tool_call["name"] == "get_weather"
    assert "Tokyo" in tool_call["arguments"]

    # Verify stream completed
    assert complete_flags[-1] is True


def test_harmony_non_streaming_parse() -> None:
    """Test non-streaming parsing of reasoning and tool calls."""
    parser = HarmonyParser()

    # Complete message with reasoning and tool call
    text = '<|channel|>analysis<|message|>We need to call get_weather function with city Tokyo.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city":"Tokyo"}<|call|>'

    result = parser.parse(text)

    # Verify reasoning content
    assert result is not None
    assert "reasoning_content" in result
    assert result["reasoning_content"] is not None
    assert "need" in result["reasoning_content"].lower() or "call" in result["reasoning_content"].lower()

    # Verify tool calls
    assert "tool_calls" in result
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["name"] == "get_weather"
    assert "Tokyo" in result["tool_calls"][0]["arguments"]

if __name__ == "__main__":
    test_harmony_reasoning_and_tool_parsing_streaming()
    test_harmony_non_streaming_parse()
    print("All tests passed!")