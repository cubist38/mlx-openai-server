from __future__ import annotations

from .hermes import HermesReasoningParser

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
REASONING_OPEN = "<think>"
REASONING_CLOSE = "</think>"


class Qwen3MoEReasoningParser(HermesReasoningParser):
    """Reasoning parser for Qwen3 MoE model's reasoning response format.

    Handles the Qwen3 MoE model's reasoning response format:
    reasoning_content</think>
    """

    def __init__(self, reasoning_open: str = REASONING_OPEN, reasoning_close: str = REASONING_CLOSE) -> None:
        """Initialize the Qwen3 MoE reasoning parser with appropriate regex patterns."""
        super().__init__(reasoning_open=reasoning_open, reasoning_close=reasoning_close)

    def needs_redacted_reasoning_prefix(self) -> bool:
        """Check if the reasoning parser needs a redacted reasoning prefix.
        
        Returns
        -------
        bool
            True if the reasoning parser needs a redacted reasoning prefix, False otherwise.
        """
        return True