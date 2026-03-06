"""Pathological streaming regressions for mixed-think tool-handoff composition."""

from __future__ import annotations

import json

from app.parsers import ParserManager
from app.parsers.abstract_parser import ReasoningParserState, ToolParserState


def _simulate_mixed_think_tool_handoff_handler_stream(
    chunks: list[str],
) -> tuple[list[str], list[dict[str, str]], list[str]]:
    """Mirror the handler's separate reasoning/tool parser streaming loop."""
    parsers_result = ParserManager.create_parsers(
        reasoning_parser_name="mixed_think_tool_handoff",
        tool_parser_name="step_35",
    )
    reasoning_parser = parsers_result.reasoning_parser
    tool_parser = parsers_result.tool_parser

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

        pending_texts = [text]
        while pending_texts:
            text = pending_texts.pop(0)

            if tool_parser and (
                tool_parser.state != ToolParserState.NORMAL or bool(tool_parser.buffer)
            ):
                parsed_content, _is_complete = tool_parser.extract_tool_calls_streaming(text)
                requeue_reasoning_tail = ""
                if (
                    reasoning_parser
                    and reasoning_parser.state == ReasoningParserState.FOUND_PREFIX
                    and tool_parser.state == ToolParserState.NORMAL
                    and tool_parser.buffer
                ):
                    requeue_reasoning_tail = tool_parser.buffer
                    tool_parser.buffer = ""
                if parsed_content:
                    tool_calls = parsed_content.get("tool_calls")
                    if isinstance(tool_calls, list):
                        emitted_tool_calls.extend(
                            tool_call for tool_call in tool_calls if isinstance(tool_call, dict)
                        )
                    content = parsed_content.get("content")
                    if isinstance(content, str) and content:
                        if requeue_reasoning_tail:
                            content = f"{content}{requeue_reasoning_tail}"
                            requeue_reasoning_tail = ""
                        if (
                            reasoning_parser
                            and reasoning_parser.state == ReasoningParserState.FOUND_PREFIX
                        ):
                            pending_texts.insert(0, content)
                        else:
                            emitted_content.append(content)
                if requeue_reasoning_tail:
                    pending_texts.insert(0, requeue_reasoning_tail)
                continue

            if reasoning_parser:
                parsed_content, is_complete = reasoning_parser.extract_reasoning_streaming(text)
                if parsed_content:
                    reasoning_piece = parsed_content.get("reasoning_content")
                    if isinstance(reasoning_piece, str):
                        emitted_reasoning.append(reasoning_piece)
                    passthrough_content = parsed_content.get("content")
                    if isinstance(passthrough_content, str) and passthrough_content:
                        emitted_content.append(passthrough_content)
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
                requeue_reasoning_tail = ""
                if (
                    reasoning_parser
                    and reasoning_parser.state == ReasoningParserState.FOUND_PREFIX
                    and tool_parser.state == ToolParserState.NORMAL
                    and tool_parser.buffer
                ):
                    requeue_reasoning_tail = tool_parser.buffer
                    tool_parser.buffer = ""
                if parsed_content:
                    tool_calls = parsed_content.get("tool_calls")
                    if isinstance(tool_calls, list):
                        emitted_tool_calls.extend(
                            tool_call for tool_call in tool_calls if isinstance(tool_call, dict)
                        )
                    content = parsed_content.get("content")
                    if isinstance(content, str) and content:
                        if requeue_reasoning_tail:
                            content = f"{content}{requeue_reasoning_tail}"
                            requeue_reasoning_tail = ""
                        if (
                            reasoning_parser
                            and reasoning_parser.state == ReasoningParserState.FOUND_PREFIX
                        ):
                            pending_texts.insert(0, content)
                        else:
                            emitted_content.append(content)
                if requeue_reasoning_tail:
                    pending_texts.insert(0, requeue_reasoning_tail)
                continue

            emitted_content.append(text)

    return emitted_content, emitted_tool_calls, emitted_reasoning


def test_mixed_think_tool_handoff_pathological_interleaving_tools_thinking_and_text() -> None:
    """Stress mixed-think parsing with mixed text plus tools inside/outside think blocks."""
    chunks = [
        "<thinking>Initial deliberate reasoning.</thinking>Narration-1 ",
        (
            '<tool_call><function=read_file><parameter=path>"/tmp/a.txt"</parameter>'
            "</function></tool_call> mid-text <think>second-block "
        ),
        (
            'inside-think tool: <tool_call><function=list_dir><parameter=path>"/tmp"</parameter>'
            "</function></tool_call> still-thinking </think> bridge "
        ),
        (
            '<tool_call><function=get_time><parameter=tz>"UTC"</parameter></function></tool_call>'
            "<tool_"
        ),
        (
            'call><function=get_weather><parameter=city>"NYC"</parameter></function></tool_call>'
            " tail-text <think>third-block "
        ),
        (
            "inner <tool_call><function=finalize><parameter=ok>true</parameter></function></tool_call>"
            " done </think> epilogue"
        ),
    ]

    emitted_content, emitted_tool_calls, emitted_reasoning = (
        _simulate_mixed_think_tool_handoff_handler_stream(chunks)
    )

    assert len("".join(emitted_reasoning)) > 0

    expected_names = ["read_file", "list_dir", "get_time", "get_weather", "finalize"]
    observed_names = [tool_call.get("name") for tool_call in emitted_tool_calls]
    assert observed_names == expected_names

    flattened_content = "".join(emitted_content)
    assert "<tool_call>" not in flattened_content
    assert "</tool_call>" not in flattened_content
    assert "call><function=get_weather>" not in flattened_content


def test_mixed_think_tool_handoff_transcript_like_thinking_block_hands_off_tool_call() -> None:
    """Tool calls inside ``<thinking>`` blocks should be handed off for parsing."""
    chunks = [
        "<thinking>\n",
        "The file is only 137 lines.\n",
        "Let me check the Makefile to understand test commands:<tool_call>\n",
        "<function=read_file>\n",
        "<parameter=path>\n",
        "custom_mlx_quant_tools/Makefile\n",
        "</parameter>\n",
        "</function>\n",
        "</tool_call>\n",
        "</thinking>\n",
    ]

    emitted_content, emitted_tool_calls, emitted_reasoning = (
        _simulate_mixed_think_tool_handoff_handler_stream(chunks)
    )

    assert len(emitted_reasoning) > 0
    assert len(emitted_tool_calls) == 1
    assert emitted_tool_calls[0]["name"] == "read_file"
    assert json.loads(emitted_tool_calls[0]["arguments"]) == {
        "path": "custom_mlx_quant_tools/Makefile"
    }

    flattened_content = "".join(emitted_content)
    assert "<function=read_file>" not in flattened_content
    assert "<tool_call>" not in flattened_content


def test_mixed_think_tool_handoff_tool_call_inside_thinking_keeps_trailing_reasoning_out_of_content() -> (
    None
):
    """Reasoning after an in-thinking tool call should not leak into normal content."""
    chunks = [
        (
            "<thinking>before-tool "
            '<tool_call><function=read_file><parameter=path>"/tmp/a.txt"</parameter></function></tool_call>'
            " after-tool </thinking>"
        ),
    ]

    emitted_content, emitted_tool_calls, emitted_reasoning = (
        _simulate_mixed_think_tool_handoff_handler_stream(chunks)
    )

    assert [tool_call.get("name") for tool_call in emitted_tool_calls] == ["read_file"]
    assert "".join(emitted_reasoning) == "before-tool  after-tool "

    flattened_content = "".join(emitted_content)
    assert flattened_content == ""
    assert "</thinking>" not in flattened_content
    assert "<tool_call>" not in flattened_content


def test_mixed_think_tool_handoff_split_tool_open_inside_thinking_does_not_leak_partial_marker() -> (
    None
):
    """A split ``<tool_call>`` opener inside thinking should not leak ``<tool_`` into reasoning."""
    chunks = [
        "<thinking>abc <tool_",
        (
            'call><function=read_file><parameter=path>"/tmp/a.txt"</parameter></function>'
            "</tool_call> xyz </thinking>"
        ),
    ]

    emitted_content, emitted_tool_calls, emitted_reasoning = (
        _simulate_mixed_think_tool_handoff_handler_stream(chunks)
    )

    assert [tool_call.get("name") for tool_call in emitted_tool_calls] == ["read_file"]
    assert "".join(emitted_reasoning) == "abc  xyz "
    assert "".join(emitted_content) == ""


def test_mixed_think_tool_handoff_split_reasoning_close_after_tool_handoff_routes_tail_to_content() -> (
    None
):
    """Split ``</thinking>`` markers after a tool handoff should still close reasoning correctly."""
    chunks = [
        (
            "<thinking>before "
            '<tool_call><function=read_file><parameter=path>"/tmp/a.txt"</parameter></function>'
            "</tool_call> after <"
        ),
        "/thi",
        "nki",
        "ng> tail",
    ]

    emitted_content, emitted_tool_calls, emitted_reasoning = (
        _simulate_mixed_think_tool_handoff_handler_stream(chunks)
    )

    assert [tool_call.get("name") for tool_call in emitted_tool_calls] == ["read_file"]
    assert "".join(emitted_reasoning) == "before  after "
    assert "".join(emitted_content) == " tail"
