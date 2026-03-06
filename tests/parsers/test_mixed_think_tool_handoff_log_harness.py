"""Structural replay harness for mixed-think in-reasoning tool-call parsing.

These tests intentionally keep semantic prose minimal and preserve only
structure that influences parsing behavior:
- reasoning wrappers
- tool-call XML blocks
- chunk boundaries at sensitive tag splits
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from tests.parsers.test_mixed_think_tool_handoff_pathological_streaming import (
    _simulate_mixed_think_tool_handoff_handler_stream,
)


@dataclass(frozen=True)
class _ReplayCase:
    """One reduced transcript shape to replay through the parser pipeline."""

    name: str
    chunks: tuple[str, ...]
    expected_tool_names: tuple[str, ...]
    expected_reasoning: str
    expected_content: str = ""


_REPLAY_CASES = (
    _ReplayCase(
        name="terminal_tool_before_close_same_chunk",
        chunks=(
            (
                "<thinking>R "
                '<tool_call><function=grep><parameter=regex>"x"</parameter></function></tool_call>'
                "</thinking>"
            ),
        ),
        expected_tool_names=("grep",),
        expected_reasoning="R ",
    ),
    _ReplayCase(
        name="terminal_tool_before_close_split_close_tag",
        chunks=(
            (
                "<thinking>R "
                '<tool_call><function=grep><parameter=regex>"x"</parameter></function></tool_call>'
                "</thi"
            ),
            "nking>",
        ),
        expected_tool_names=("grep",),
        expected_reasoning="R ",
    ),
    _ReplayCase(
        name="terminal_tool_before_close_split_tool_close",
        chunks=(
            (
                "<thinking>R "
                '<tool_call><function=grep><parameter=regex>"x"</parameter></function></tool_'
            ),
            "call></thinking>",
        ),
        expected_tool_names=("grep",),
        expected_reasoning="R ",
    ),
    _ReplayCase(
        name="terminal_tool_before_close_with_newlines",
        chunks=(
            "<thinking>\n",
            "R\n",
            "<tool_call>\n",
            "<function=grep>\n",
            "<parameter=regex>\n",
            '"x"\n',
            "</parameter>\n",
            "</function>\n",
            "</tool_call>\n",
            "</thinking>\n",
        ),
        expected_tool_names=("grep",),
        expected_reasoning="\nR\n\n",
        expected_content="\n",
    ),
    _ReplayCase(
        name="terminal_tool_before_close_followed_by_outside_text",
        chunks=(
            (
                "<thinking>R "
                '<tool_call><function=grep><parameter=regex>"x"</parameter></function></tool_call>'
                "</thinking> OUT"
            ),
        ),
        expected_tool_names=("grep",),
        expected_reasoning="R ",
        expected_content=" OUT",
    ),
    _ReplayCase(
        name="transcript_tail_terminal_read_file_before_close",
        chunks=(
            (
                "<thinking>R:"
                "<tool_call>\n"
                "<function=read_file>\n"
                "<parameter=path>\n"
                "app/api/endpoints.py\n"
                "</parameter>\n"
                "<parameter=start_line>\n"
                "692\n"
                "</parameter>\n"
                "<parameter=end_line>\n"
                "790\n"
                "</parameter>\n"
                "</function>\n"
                "</tool_call>\n"
                "</thinking>"
            ),
        ),
        expected_tool_names=("read_file",),
        expected_reasoning="R:\n",
    ),
    _ReplayCase(
        name="transcript_tail_terminal_read_file_split_at_final_close",
        chunks=(
            (
                "<thinking>R:"
                "<tool_call>\n"
                "<function=read_file>\n"
                "<parameter=path>\n"
                "app/api/endpoints.py\n"
                "</parameter>\n"
                "<parameter=start_line>\n"
                "692\n"
                "</parameter>\n"
                "<parameter=end_line>\n"
                "790\n"
                "</parameter>\n"
                "</function>\n"
                "</tool_call>\n"
                "</thi"
            ),
            "nking>",
        ),
        expected_tool_names=("read_file",),
        expected_reasoning="R:\n",
    ),
)


@pytest.mark.parametrize("case", _REPLAY_CASES, ids=lambda case: case.name)
def test_mixed_think_log_harness_structural_replays(case: _ReplayCase) -> None:
    """Replay reduced transcript structures and assert stable tool extraction."""
    emitted_content, emitted_tool_calls, emitted_reasoning = (
        _simulate_mixed_think_tool_handoff_handler_stream(list(case.chunks))
    )

    observed_tool_names = tuple(tool_call.get("name", "") for tool_call in emitted_tool_calls)
    assert observed_tool_names == case.expected_tool_names

    joined_reasoning = "".join(emitted_reasoning)
    joined_content = "".join(emitted_content)

    assert joined_reasoning == case.expected_reasoning
    assert joined_content == case.expected_content

    assert "<tool_call>" not in joined_reasoning
    assert "</tool_call>" not in joined_reasoning
    assert "<function=" not in joined_reasoning
    assert "<parameter=" not in joined_reasoning

    assert "<tool_call>" not in joined_content
    assert "</tool_call>" not in joined_content
    assert "<function=" not in joined_content
    assert "<parameter=" not in joined_content
