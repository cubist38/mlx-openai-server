from __future__ import annotations

from .abstract_parser import ReasoningParserState, _suffix_prefix_overlap
from .qwen3_moe import Qwen3MoEReasoningParser

REASONING_OPEN = "<think>"
REASONING_CLOSE = "</think>"
TOOL_OPEN = "<tool_call>"


class Qwen35ReasoningParser(Qwen3MoEReasoningParser):
    """Reasoning parser for Qwen3.5 model's reasoning response format.

    Handles the Qwen3.5 model's reasoning response format:
    reasoning_content</think>

    Unlike the base Qwen3MoEReasoningParser, this parser stops collecting
    reasoning content when it encounters a ``<tool_call>`` tag, even if the
    closing ``</think>`` has not yet been seen.  The model occasionally emits
    tool calls inside the ``<think>`` block; without this guard the entire
    ``<tool_call>...</tool_call>`` payload is absorbed into ``reasoning_content``
    and the downstream tool parser never receives it.
    """

    def __init__(
        self, reasoning_open: str = REASONING_OPEN, reasoning_close: str = REASONING_CLOSE
    ) -> None:
        """Initialize the Qwen3.5 reasoning parser with appropriate regex patterns."""
        super().__init__(reasoning_open=reasoning_open, reasoning_close=reasoning_close)
        self.tool_open = TOOL_OPEN

    def respects_enable_thinking(self) -> bool:
        """Check if the reasoning parser respects the enable_thinking flag.

        Returns
        -------
        bool
            True if the reasoning parser respects the enable_thinking flag, False otherwise.
        """
        return True

    def extract_reasoning(self, model_output: str) -> dict[str, str] | None:
        """Extract reasoning from complete output, stopping at the first ``<tool_call>``.

        If a ``<tool_call>`` tag appears inside the ``<think>`` block, the text
        from that point onward is returned as ``after_reasoning_close_content``
        so the tool parser can process it normally.

        Parameters
        ----------
        model_output : str
            Complete model output, possibly with a synthetic ``<think>`` prefix
            added by the handler (``needs_redacted_reasoning_prefix=True``).

        Returns
        -------
        dict[str, str] | None
            Same structure as ``HermesReasoningParser.extract_reasoning``.
        """
        think_start = model_output.find(self.reasoning_open)
        if think_start < 0:
            return {"content": model_output}

        inner_start = think_start + len(self.reasoning_open)
        inner_text = model_output[inner_start:]

        # Check whether a tool_call appears before the closing </think>
        think_end = inner_text.find(self.reasoning_close)
        tool_idx = inner_text.find(self.tool_open)

        if tool_idx >= 0 and (think_end < 0 or tool_idx < think_end):
            # tool_call appears before (or instead of) </think> — split here.
            # The reasoning region is terminated by the tool boundary, so any
            # subsequent </think> marker is consumed (must not surface as content).
            reasoning_content = inner_text[:tool_idx]
            after_reasoning_close_content = inner_text[tool_idx:].replace(
                self.reasoning_close, "", 1
            )
            return {
                "reasoning_content": reasoning_content,
                "after_reasoning_close_content": after_reasoning_close_content,
            }

        # Delegate to the parent for the normal </think>-terminated case
        return super().extract_reasoning(model_output)

    def extract_reasoning_streaming(self, chunk: str) -> tuple[dict[str, str] | str | None, bool]:
        """Extract reasoning from a streaming chunk, stopping at ``<tool_call>``.

        While accumulating content inside ``<think>`` (both the initial chunk
        that contains ``<think>`` and subsequent ``FOUND_PREFIX`` chunks), if the
        combined buffer contains a ``<tool_call>`` tag the parser immediately
        closes the reasoning region and returns everything from that tag onward
        as ``after_reasoning_close_content``.

        Parameters
        ----------
        chunk : str
            Chunk of model output to process.

        Returns
        -------
        tuple[dict[str, str] | str | None, bool]
            Same tuple contract as ``HermesReasoningParser.extract_reasoning_streaming``.
        """
        if self.state == ReasoningParserState.NORMAL and self.reasoning_open in chunk:
            # Entering the reasoning region in this chunk — intercept before falling
            # through to the parent so we can apply tool_open boundary detection.
            start_idx = chunk.find(self.reasoning_open)
            inner = chunk[start_idx + len(self.reasoning_open) :]

            # Full tool_call tag visible before </think>?
            tool_idx = inner.find(self.tool_open)
            think_end_idx = inner.find(self.reasoning_close)

            if tool_idx >= 0 and (think_end_idx < 0 or tool_idx < think_end_idx):
                # Whole tool_call tag landed in the same chunk — split immediately.
                # Strip </think> from the tail (consumed by tool boundary).
                self.state = ReasoningParserState.NORMAL
                self.buffer = ""
                return {
                    "reasoning_content": inner[:tool_idx],
                    "after_reasoning_close_content": inner[tool_idx:].replace(
                        self.reasoning_close, "", 1
                    ),
                }, True

            if think_end_idx < 0:
                # </think> not yet seen — look for a partial tool_open at the tail.
                overlap = _suffix_prefix_overlap(inner, self.tool_open)
                if overlap > 0:
                    safe_text = inner[:-overlap]
                    self.state = ReasoningParserState.FOUND_PREFIX
                    self.buffer = inner[-overlap:]
                    if safe_text:
                        return {"reasoning_content": safe_text}, False
                    return None, False
                # No tool tag risk — let the parent handle it (will set FOUND_PREFIX)

        elif self.state == ReasoningParserState.FOUND_PREFIX:
            combined = self.buffer + chunk

            # Full tool_call tag visible before </think>?
            tool_idx = combined.find(self.tool_open)
            think_end_idx = combined.find(self.reasoning_close)

            if tool_idx >= 0 and (think_end_idx < 0 or tool_idx < think_end_idx):
                reasoning_content = combined[:tool_idx]
                # Strip </think> from the tail (consumed by tool boundary).
                after_reasoning_close_content = combined[tool_idx:].replace(
                    self.reasoning_close, "", 1
                )
                self.buffer = ""
                self.state = ReasoningParserState.NORMAL
                return {
                    "reasoning_content": reasoning_content,
                    "after_reasoning_close_content": after_reasoning_close_content,
                }, True

            # No full tool_open yet — guard against partial tag at chunk boundary.
            # Only apply this guard when </think> is not yet in the combined buffer;
            # once we can see the close tag the parent logic handles termination.
            if think_end_idx < 0:
                overlap = _suffix_prefix_overlap(combined, self.tool_open)
                if overlap > 0:
                    safe_text = combined[:-overlap]
                    self.buffer = combined[-overlap:]
                    if safe_text:
                        return {"reasoning_content": safe_text}, False
                    return None, False

        # Fall through to the parent's implementation for all other cases
        return super().extract_reasoning_streaming(chunk)
