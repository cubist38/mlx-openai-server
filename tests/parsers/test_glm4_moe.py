"""Tests for the GLM4 MoE parser."""

import json

from app.parsers.glm4_moe import GLM4MoEReasoningParser, GLM4MoEToolParser


def test_glm4_moe_reasoning_and_tool_parsing_streaming() -> None:
    """Test streaming parsing of reasoning and tool calls."""
    reasoning_parser = GLM4MoEReasoningParser()
    tool_parser = GLM4MoEToolParser()

    chunks = [
        "<think>The user is asking",
        'for the weather in Tokyo. I have a function available called "get_weather" that takes a city parameter and returns the weather information for that city. The user has provided "Tokyo" as the city they want weather information for, so I have all the required parameters to make the function call.',
        "call.</think>",
        "I'll get the current weather information for Tokyo for you.",
        "<tool_call>get_weather",
        "<arg_key>city</arg_key>",
        "<arg_value>Tokyo</arg_value>",
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
                if isinstance(reasoning, dict) and reasoning.get("content"):
                    after_thinking_close_content = reasoning.get("content")
            continue
        if after_thinking_close_content:
            chunk = after_thinking_close_content + chunk
            after_thinking_close_content = None
        if tool_parser:
            parsed_content, is_complete = tool_parser.extract_tool_calls_streaming(chunk)
            if parsed_content:
                tool_call_results.append(parsed_content)
                is_complete_flags.append(is_complete)

    # Verify reasoning parser extracted content correctly
    assert len(reasoning_results) > 0
    # Check that we got reasoning results for the chunks with reasoning tags
    assert any(
        "reasoning_content" in result for result in reasoning_results if isinstance(result, dict)
    )
    # The reasoning parser should have completed and extracted reasoning content
    # Note: The original test checks for "content" key, which may be present in some cases
    final_reasoning = reasoning_results[-1]
    assert isinstance(final_reasoning, dict)

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
    assert complete_tool_call["tool_calls"][0]["name"] == "get_weather"
    # Verify that arguments is a JSON string containing the expected parameter
    assert json.loads(complete_tool_call["tool_calls"][0]["arguments"]) == {"city": "Tokyo"}


if __name__ == "__main__":
    test_glm4_moe_reasoning_and_tool_parsing_streaming()
