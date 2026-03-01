"""Reasoning parser for mixed think/thinking tags with tool handoff support."""

from __future__ import annotations

from .abstract_parser import AbstractReasoningParser, ReasoningParserState, _suffix_prefix_overlap

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
THINKING_OPEN = "<thinking>"
THINKING_CLOSE = "</thinking>"
TOOL_OPEN = "<tool_call>"


class MixedThinkToolHandoffReasoningParser(AbstractReasoningParser):
    """Reasoning parser for mixed think wrappers with in-reasoning tool handoff.

    Some models emit both ``<think>`` and ``<thinking>`` wrappers. In some
    transcripts, tool calls can appear before the reasoning close marker. This
    parser supports both wrappers and hands off to tool parsing when a
    ``<tool_call>`` marker appears.
    """

    def __init__(self) -> None:
        """Initialize parser state and supported reasoning marker pairs."""
        super().__init__(reasoning_open=THINKING_OPEN, reasoning_close=THINKING_CLOSE)
        self._open_to_close: dict[str, str] = {
            THINKING_OPEN: THINKING_CLOSE,
            THINK_OPEN: THINK_CLOSE,
        }
        self._active_close: str | None = None

    def needs_redacted_reasoning_prefix(self) -> bool:
        """Outputs for this format include explicit reasoning-open markers."""
        return False

    def _find_first_open(self, text: str) -> tuple[int, str] | None:
        """Return the earliest reasoning-open marker in ``text``."""
        first_idx: int | None = None
        first_open: str | None = None
        for open_marker in self._open_to_close:
            idx = text.find(open_marker)
            if idx == -1:
                continue
            if first_idx is None or idx < first_idx:
                first_idx = idx
                first_open = open_marker
        if first_idx is None or first_open is None:
            return None
        return first_idx, first_open

    def _select_boundary(self, text: str) -> tuple[int, str | None]:
        """Select boundary index/type for close marker or tool handoff."""
        close_idx = -1
        if self._active_close is not None:
            close_idx = text.find(self._active_close)
        tool_idx = text.find(TOOL_OPEN)

        if close_idx >= 0 and (tool_idx < 0 or close_idx < tool_idx):
            return close_idx, "close"
        if tool_idx >= 0:
            return tool_idx, "tool"
        return -1, None

    def _extract_from_reasoning_body_stream(
        self, reasoning_body: str
    ) -> tuple[dict[str, str] | None, bool]:
        """Extract streaming reasoning payload and determine completion."""
        boundary_idx, boundary_type = self._select_boundary(reasoning_body)
        if boundary_type == "close" and self._active_close is not None:
            reasoning_content = reasoning_body[:boundary_idx]
            after_reasoning_close_content = reasoning_body[boundary_idx + len(self._active_close) :]
            self.buffer = ""
            self.state = ReasoningParserState.NORMAL
            self._active_close = None
            return {
                "reasoning_content": reasoning_content,
                "after_reasoning_close_content": after_reasoning_close_content,
            }, True

        if boundary_type == "tool":
            reasoning_content = reasoning_body[:boundary_idx]
            after_reasoning_close_content = reasoning_body[boundary_idx:]
            # Any buffered overlap was consumed into ``reasoning_body`` above.
            # Clear it so stale marker fragments do not leak into later chunks.
            self.buffer = ""
            return {
                "reasoning_content": reasoning_content,
                "after_reasoning_close_content": after_reasoning_close_content,
            }, False

        overlaps = [_suffix_prefix_overlap(reasoning_body, TOOL_OPEN)]
        if self._active_close is not None:
            overlaps.append(_suffix_prefix_overlap(reasoning_body, self._active_close))
        overlap = max(overlaps)
        if overlap > 0:
            emitted_reasoning = reasoning_body[:-overlap]
            self.buffer = reasoning_body[-overlap:]
        else:
            emitted_reasoning = reasoning_body
            self.buffer = ""

        if emitted_reasoning:
            return {"reasoning_content": emitted_reasoning}, False
        return None, False

    def extract_reasoning(self, model_output: str) -> dict[str, str] | None:
        """Extract reasoning content from complete model output."""
        open_match = self._find_first_open(model_output)
        if open_match is None:
            return {"content": model_output}

        open_idx, open_marker = open_match
        close_marker = self._open_to_close[open_marker]
        reasoning_body = model_output[open_idx + len(open_marker) :]

        close_idx = reasoning_body.find(close_marker)
        tool_idx = reasoning_body.find(TOOL_OPEN)
        if close_idx >= 0 and (tool_idx < 0 or close_idx < tool_idx):
            return {
                "reasoning_content": reasoning_body[:close_idx],
                "after_reasoning_close_content": reasoning_body[close_idx + len(close_marker) :],
            }
        if tool_idx >= 0:
            return {
                "reasoning_content": reasoning_body[:tool_idx],
                "after_reasoning_close_content": reasoning_body[tool_idx:],
            }
        return {
            "reasoning_content": reasoning_body,
            "after_reasoning_close_content": "",
        }

    def extract_reasoning_streaming(self, chunk: str) -> tuple[dict[str, str] | str | None, bool]:
        """Extract reasoning content from streaming chunks."""
        if self.state == ReasoningParserState.NORMAL:
            combined = self.buffer + chunk
            open_match = self._find_first_open(combined)
            if open_match is not None:
                open_idx, open_marker = open_match
                self.state = ReasoningParserState.FOUND_PREFIX
                self._active_close = self._open_to_close[open_marker]
                passthrough = combined[:open_idx]
                reasoning_body = combined[open_idx + len(open_marker) :]
                parsed_payload, is_complete = self._extract_from_reasoning_body_stream(
                    reasoning_body
                )

                if parsed_payload is not None and passthrough:
                    merged = dict(parsed_payload)
                    merged_content = merged.get("content")
                    if merged_content:
                        merged["content"] = f"{passthrough}{merged_content}"
                    else:
                        merged["content"] = passthrough
                    return merged, is_complete
                if parsed_payload is not None:
                    return parsed_payload, is_complete
                if passthrough:
                    return {"content": passthrough}, False
                return None, False

            overlap = max(
                _suffix_prefix_overlap(combined, marker) for marker in self._open_to_close
            )
            if overlap > 0:
                passthrough = combined[:-overlap]
                self.buffer = combined[-overlap:]
            else:
                passthrough = combined
                self.buffer = ""
            if passthrough:
                return {"content": passthrough}, False
            return None, False

        combined = self.buffer + chunk
        return self._extract_from_reasoning_body_stream(combined)


# Backward-compatible alias for existing model-specific registry names/tests.
Step35ReasoningParser = MixedThinkToolHandoffReasoningParser
