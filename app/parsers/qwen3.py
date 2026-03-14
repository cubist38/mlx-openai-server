from __future__ import annotations

from .hermes import HermesReasoningParser

REASONING_OPEN = "<think>"
REASONING_CLOSE = "</think>"


class Qwen3ReasoningParser(HermesReasoningParser):
    """Reasoning parser for Qwen3 MoE model's reasoning response format.

    Handles the Qwen3 model's reasoning response format:
    <think>reasoning_content</think>
    """

    def __init__(
        self, reasoning_open: str = REASONING_OPEN, reasoning_close: str = REASONING_CLOSE
    ) -> None:
        """Initialize the Qwen3 reasoning parser with appropriate regex patterns."""
        super().__init__(reasoning_open=reasoning_open, reasoning_close=reasoning_close)

    def respects_enable_thinking(self) -> bool:
        """Check if the reasoning parser respects the enable_thinking flag.

        Returns
        -------
        bool
            True if the reasoning parser respects the enable_thinking flag, False otherwise.
        """
        return True
