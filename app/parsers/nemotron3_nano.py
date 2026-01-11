from __future__ import annotations

from .function_parameter import FunctionParameterToolParser
from .hermes import HermesReasoningParser

REASONING_OPEN = "<think>"
REASONING_CLOSE = "</think>"


class Nemotron3NanoReasoningParser(HermesReasoningParser):
    """Parser for Nemotron3 Nano model's reasoning response format.

    Handles reasoning content in format: <think>reasoning_content</think>
    """

    def __init__(self) -> None:
        """Initialize the Nemotron3 Nano reasoning parser."""
        super().__init__(
            reasoning_open=REASONING_OPEN,
            reasoning_close=REASONING_CLOSE,
        )

    def needs_redacted_reasoning_prefix(self) -> bool:
        """Check if the reasoning parser needs a redacted reasoning prefix.

        Returns
        -------
        bool
            True - Nemotron3 Nano requires redacted reasoning prefix.
        """
        return True


Nemotron3NanoToolParser = FunctionParameterToolParser
