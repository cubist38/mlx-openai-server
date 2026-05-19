"""Regression tests for Qwen3.5 reasoning parser absorbing <tool_call> blocks.

Bug: When serving a Qwen3.5 model with tool_call_parser=qwen3_coder and
reasoning_parser=qwen3_5, the reasoning parser consumes the entire
<tool_call>...</tool_call> block as reasoning_content, leaving tool_calls empty.

Root cause: HermesReasoningParser.extract_reasoning_streaming accumulates all
text after <think> until </think> as reasoning. When the model emits
<tool_call> before </think>, it gets absorbed into reasoning_content.

Fix: Qwen35ReasoningParser overrides extract_reasoning_streaming (and
extract_reasoning) to detect <tool_call> inside the reasoning region and
emit it as after_reasoning_close_content instead.
"""

from __future__ import annotations

from app.parsers.function_parameter import FunctionParameterToolParser
from app.parsers.qwen3_5 import Qwen35ReasoningParser

# ---------------------------------------------------------------------------
# Canonical Qwen3.5 tool-call output patterns
# ---------------------------------------------------------------------------

# Pattern A: tool_call AFTER </think>  (well-formed — should already work)
PATTERN_A = (
    "<think>\n\n</think>\n\n"
    "<tool_call>\n"
    "<function=read>\n"
    "<parameter=filePath>\n/tmp/x\n</parameter>\n"
    "</function>\n"
    "</tool_call>"
)

# Pattern B: tool_call BEFORE </think>  (problematic — this is the bug)
PATTERN_B = (
    "<think>\n"
    "<tool_call>\n"
    "<function=read>\n"
    "<parameter=filePath>\n/tmp/x\n</parameter>\n"
    "</function>\n"
    "</tool_call>\n"
    "</think>"
)

# Pattern C: reasoning text, then tool_call, then </think>
PATTERN_C = (
    "<think>\n"
    "I need to read the file.\n"
    "<tool_call>\n"
    "<function=read>\n"
    "<parameter=filePath>\n/tmp/x\n</parameter>\n"
    "</function>\n"
    "</tool_call>\n"
    "</think>"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_streaming(
    chunks: list[str],
    reasoning_parser: Qwen35ReasoningParser,
    tool_parser: FunctionParameterToolParser,
) -> tuple[list[dict], list[dict]]:
    """Simulate the handler streaming loop used in mlx_lm.py.

    Returns (reasoning_results, tool_call_results).
    """
    reasoning_results: list[dict] = []
    tool_call_results: list[dict] = []

    after_close: str | None = None
    is_first_chunk = True
    rp: Qwen35ReasoningParser | None = reasoning_parser
    tp: FunctionParameterToolParser | None = tool_parser

    for chunk in chunks:
        if is_first_chunk and rp and rp.needs_redacted_reasoning_prefix():
            chunk = rp.get_reasoning_open() + chunk
            is_first_chunk = False

        if rp:
            parsed, is_complete = rp.extract_reasoning_streaming(chunk)
            if parsed and isinstance(parsed, dict):
                reasoning_results.append(parsed)
                after_close = parsed.get("after_reasoning_close_content")
            if is_complete:
                rp = None
            if after_close:
                chunk = after_close
                after_close = None
            else:
                continue

        if tp:
            parsed, is_complete = tp.extract_tool_calls_streaming(chunk)
            if parsed and isinstance(parsed, dict):
                tool_call_results.append(parsed)
            if is_complete:
                tp = None

    return reasoning_results, tool_call_results


# ---------------------------------------------------------------------------
# Non-streaming tests
# ---------------------------------------------------------------------------


class TestExtractReasoningNonStreaming:
    """Non-streaming path: extract_reasoning must not absorb <tool_call>."""

    def test_pattern_a_tool_after_think(self) -> None:
        """tool_call after </think> — after_reasoning_close_content should contain it."""
        parser = Qwen35ReasoningParser()
        result = parser.extract_reasoning(PATTERN_A)
        assert result is not None
        assert "reasoning_content" in result
        after = result.get("after_reasoning_close_content", "")
        assert "<tool_call>" in after, (
            f"Expected <tool_call> in after_reasoning_close_content, got: {after!r}"
        )
        assert "reasoning_content" not in result.get("reasoning_content", "") or (
            "<tool_call>" not in result["reasoning_content"]
        ), f"<tool_call> must not appear in reasoning_content, got: {result['reasoning_content']!r}"

    def test_pattern_b_tool_inside_think(self) -> None:
        """tool_call inside <think>...</think> — must NOT be absorbed as reasoning."""
        parser = Qwen35ReasoningParser()
        result = parser.extract_reasoning(PATTERN_B)
        assert result is not None
        # The <tool_call> block must not land in reasoning_content
        reasoning = result.get("reasoning_content", "")
        assert "<tool_call>" not in reasoning, (
            f"<tool_call> was absorbed into reasoning_content: {reasoning!r}"
        )
        # It must be available for downstream tool parsing
        after = result.get("after_reasoning_close_content", "")
        assert "<tool_call>" in after, (
            f"Expected <tool_call> in after_reasoning_close_content, got: {after!r}"
        )
        # The </think> close marker is consumed by the tool boundary and must
        # not leak into the after-close content (which is fed to the tool parser
        # and would otherwise surface as stray content in the response).
        assert "</think>" not in after, (
            f"</think> leaked into after_reasoning_close_content: {after!r}"
        )

    def test_pattern_c_reasoning_then_tool_inside_think(self) -> None:
        """Reasoning text + tool_call inside <think> — split at tool_call boundary."""
        parser = Qwen35ReasoningParser()
        result = parser.extract_reasoning(PATTERN_C)
        assert result is not None
        reasoning = result.get("reasoning_content", "")
        assert "<tool_call>" not in reasoning, (
            f"<tool_call> was absorbed into reasoning_content: {reasoning!r}"
        )
        after = result.get("after_reasoning_close_content", "")
        assert "<tool_call>" in after, (
            f"Expected <tool_call> in after_reasoning_close_content, got: {after!r}"
        )
        assert "</think>" not in after, (
            f"</think> leaked into after_reasoning_close_content: {after!r}"
        )


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------


class TestExtractReasoningStreaming:
    """Streaming path: extract_reasoning_streaming must not absorb <tool_call>."""

    def test_pattern_b_single_chunk(self) -> None:
        """Entire PATTERN_B in one chunk — tool_call not absorbed."""
        parser = Qwen35ReasoningParser()
        # Prepend <think> since needs_redacted_reasoning_prefix=True
        chunk = parser.get_reasoning_open() + PATTERN_B.removeprefix("<think>")
        results = []
        is_complete = False
        parsed, is_complete = parser.extract_reasoning_streaming(chunk)
        if parsed:
            results.append(parsed)

        reasoning_text = "".join(
            r.get("reasoning_content", "") for r in results if isinstance(r, dict)
        )
        after_text = "".join(
            r.get("after_reasoning_close_content", "") for r in results if isinstance(r, dict)
        )
        assert "<tool_call>" not in reasoning_text, (
            f"<tool_call> absorbed into reasoning_content: {reasoning_text!r}"
        )
        assert "<tool_call>" in after_text, (
            f"<tool_call> not in after_reasoning_close_content: {after_text!r}"
        )
        assert "</think>" not in after_text, (
            f"</think> leaked into after_reasoning_close_content: {after_text!r}"
        )

    def test_full_pipeline_pattern_a(self) -> None:
        """Pattern A (tool after </think>) — end-to-end streaming pipeline succeeds."""
        # Split at token boundaries to simulate real streaming
        full = PATTERN_A
        chunks = [full[i : i + 8] for i in range(0, len(full), 8)]
        # Remove the leading <think> since the handler prepends it
        first_token = chunks[0]
        if first_token.startswith("<think>"):
            chunks[0] = first_token[len("<think>") :]
        elif full.startswith("<think>"):
            chunks = [
                full[len("<think>") :][i : i + 8] for i in range(0, len(full) - len("<think>"), 8)
            ]

        reasoning_results, tool_call_results = _run_streaming(
            chunks, Qwen35ReasoningParser(), FunctionParameterToolParser()
        )

        tool_calls = []
        for r in tool_call_results:
            tool_calls.extend(r.get("tool_calls", []))

        assert len(tool_calls) == 1, f"Expected 1 tool call, got: {tool_call_results}"
        assert tool_calls[0]["name"] == "read"

    def test_full_pipeline_pattern_b(self) -> None:
        """Pattern B (tool inside <think>) — tool_call must reach tool parser."""
        full = PATTERN_B
        # Strip the leading <think> — handler re-adds it
        inner = full[len("<think>") :]
        chunks = [inner[i : i + 6] for i in range(0, len(inner), 6)]

        reasoning_results, tool_call_results = _run_streaming(
            chunks, Qwen35ReasoningParser(), FunctionParameterToolParser()
        )

        # tool_call must not be swallowed as reasoning
        reasoning_text = "".join(
            r.get("reasoning_content", "") for r in reasoning_results if isinstance(r, dict)
        )
        assert "<tool_call>" not in reasoning_text, (
            f"<tool_call> absorbed as reasoning: {reasoning_text!r}"
        )

        tool_calls = []
        for r in tool_call_results:
            tool_calls.extend(r.get("tool_calls", []))

        assert len(tool_calls) == 1, (
            f"Expected 1 tool call, got tool_call_results={tool_call_results!r}"
        )
        assert tool_calls[0]["name"] == "read"

    def test_full_pipeline_pattern_c(self) -> None:
        """Pattern C (reasoning + tool inside <think>) — reasoning captured, tool parsed."""
        full = PATTERN_C
        inner = full[len("<think>") :]
        chunks = [inner[i : i + 5] for i in range(0, len(inner), 5)]

        reasoning_results, tool_call_results = _run_streaming(
            chunks, Qwen35ReasoningParser(), FunctionParameterToolParser()
        )

        reasoning_text = "".join(
            r.get("reasoning_content", "") for r in reasoning_results if isinstance(r, dict)
        )
        assert "<tool_call>" not in reasoning_text, (
            f"<tool_call> absorbed as reasoning: {reasoning_text!r}"
        )
        # Some reasoning text should have been captured
        assert len(reasoning_text) > 0 or any(
            "reasoning_content" in r for r in reasoning_results if isinstance(r, dict)
        )

        tool_calls = []
        for r in tool_call_results:
            tool_calls.extend(r.get("tool_calls", []))

        assert len(tool_calls) == 1, (
            f"Expected 1 tool call, got tool_call_results={tool_call_results!r}"
        )
        assert tool_calls[0]["name"] == "read"

    def test_pure_thinking_delegates_to_parent_across_chunks(self) -> None:
        """No <tool_call> present: the override must fall through to the
        parent so a </think> split across chunks is still recognized.
        """
        # `<think>thinking</think>summary` — handler prepends <think>, so the
        # raw chunks look like `['thinking</th', 'ink>summary']`.
        chunks = ["thinking</th", "ink>summary"]
        parser = Qwen35ReasoningParser()
        # First chunk gets the synthetic <think> prefix
        first = parser.get_reasoning_open() + chunks[0]

        results: list[dict] = []
        parsed, _ = parser.extract_reasoning_streaming(first)
        if isinstance(parsed, dict):
            results.append(parsed)
        parsed, _ = parser.extract_reasoning_streaming(chunks[1])
        if isinstance(parsed, dict):
            results.append(parsed)

        reasoning_text = "".join(r.get("reasoning_content", "") for r in results)
        after_text = "".join(r.get("after_reasoning_close_content", "") for r in results)

        assert reasoning_text == "thinking", (
            f"Expected reasoning_content == 'thinking', got: {reasoning_text!r}"
        )
        assert after_text == "summary", (
            f"Expected after_reasoning_close_content == 'summary', got: {after_text!r}"
        )

    def test_chunk_boundary_split_on_tool_open_tag(self) -> None:
        """<tool_call> split across chunks at tag boundary — still detected."""
        # Simulate chunk split mid-tag: '<tool_' | 'call>'
        inner = (
            "\n"
            "I need to read the file.\n"
            "<tool_call>\n"
            "<function=read>\n"
            "<parameter=filePath>\n/tmp/x\n</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "</think>"
        )
        # Split at the '<tool_' boundary
        split_idx = inner.index("<tool_call>") + len("<tool_")
        chunks = [inner[:split_idx], inner[split_idx:]]

        reasoning_results, tool_call_results = _run_streaming(
            chunks, Qwen35ReasoningParser(), FunctionParameterToolParser()
        )

        reasoning_text = "".join(
            r.get("reasoning_content", "") for r in reasoning_results if isinstance(r, dict)
        )
        assert "<tool_call>" not in reasoning_text, (
            f"<tool_call> absorbed as reasoning: {reasoning_text!r}"
        )

        tool_calls = []
        for r in tool_call_results:
            tool_calls.extend(r.get("tool_calls", []))

        assert len(tool_calls) == 1, (
            f"Expected 1 tool call, got tool_call_results={tool_call_results!r}"
        )
        assert tool_calls[0]["name"] == "read"
