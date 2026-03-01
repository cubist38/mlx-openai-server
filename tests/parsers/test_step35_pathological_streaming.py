"""Pathological streaming regressions for step_35 parser composition."""

from __future__ import annotations

from app.parsers.function_parameter import FunctionParameterToolParser
from app.parsers.qwen3_moe import Qwen3MoEReasoningParser


def _simulate_step35_handler_stream(
    chunks: list[str],
) -> tuple[list[str], list[dict[str, str]], list[str]]:
    """Mirror the handler's separate reasoning/tool parser streaming loop."""
    reasoning_parser = Qwen3MoEReasoningParser()
    tool_parser = FunctionParameterToolParser()

    after_reasoning_close_content = None
    is_first_chunk = True

    emitted_content: list[str] = []
    emitted_tool_calls: list[dict[str, str]] = []
    emitted_reasoning: list[str] = []

    for chunk in chunks:
        text = chunk
        if is_first_chunk:
            if reasoning_parser and reasoning_parser.needs_redacted_reasoning_prefix():
                text = reasoning_parser.get_reasoning_open() + text
            is_first_chunk = False

        if reasoning_parser:
            parsed_content, is_complete = reasoning_parser.extract_reasoning_streaming(text)
            if parsed_content:
                reasoning_piece = parsed_content.get("reasoning_content")
                if isinstance(reasoning_piece, str):
                    emitted_reasoning.append(reasoning_piece)
                after_reasoning_close_content = parsed_content.get(
                    "after_reasoning_close_content"
                )
            if is_complete:
                reasoning_parser = None
            if after_reasoning_close_content:
                text = after_reasoning_close_content
                after_reasoning_close_content = None
            else:
                continue

        if tool_parser:
            parsed_content, _is_complete = tool_parser.extract_tool_calls_streaming(text)
            if parsed_content:
                content = parsed_content.get("content")
                if isinstance(content, str) and content:
                    emitted_content.append(content)
                tool_calls = parsed_content.get("tool_calls")
                if isinstance(tool_calls, list):
                    emitted_tool_calls.extend(
                        tool_call for tool_call in tool_calls if isinstance(tool_call, dict)
                    )
            continue

        emitted_content.append(text)

    return emitted_content, emitted_tool_calls, emitted_reasoning


def test_step35_pathological_interleaving_tools_thinking_and_text() -> None:
    """Stress step_35 parsing with mixed text plus tool calls inside/outside think blocks."""
    chunks = [
        "Initial deliberate reasoning.</think>Narration-1 ",
        (
            "<tool_call><function=read_file><parameter=path>\"/tmp/a.txt\"</parameter>"
            "</function></tool_call> mid-text <think>second-block "
        ),
        (
            "inside-think tool: <tool_call><function=list_dir><parameter=path>\"/tmp\"</parameter>"
            "</function></tool_call> still-thinking </think> bridge "
        ),
        (
            "<tool_call><function=get_time><parameter=tz>\"UTC\"</parameter></function></tool_call>"
            "<tool_"
        ),
        (
            "call><function=get_weather><parameter=city>\"NYC\"</parameter></function></tool_call>"
            " tail-text <think>third-block "
        ),
        (
            "inner <tool_call><function=finalize><parameter=ok>true</parameter></function></tool_call>"
            " done </think> epilogue"
        ),
    ]

    emitted_content, emitted_tool_calls, emitted_reasoning = _simulate_step35_handler_stream(
        chunks
    )

    assert "".join(emitted_reasoning) == "Initial deliberate reasoning."

    expected_names = ["read_file", "list_dir", "get_time", "get_weather", "finalize"]
    observed_names = [tool_call.get("name") for tool_call in emitted_tool_calls]
    assert observed_names == expected_names

    flattened_content = "".join(emitted_content)
    assert "<tool_call>" not in flattened_content
    assert "</tool_call>" not in flattened_content
    assert "call><function=get_weather>" not in flattened_content
