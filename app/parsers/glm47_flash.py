from __future__ import annotations

import re
import json

from .glm4_moe import GLM4MoEReasoningParser, GLM4MoEToolParser

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
REASONING_OPEN = "<think>"
REASONING_CLOSE = "</think>"


class GLM47FlashReasoningParser(GLM4MoEReasoningParser):
    """Reasoning parser for GLM4 MoE model's reasoning response format.

    Handles the GLM4 MoE model's reasoning response format:
    <think>reasoning_content</think>
    """

    def __init__(self, reasoning_open: str = REASONING_OPEN, reasoning_close: str = REASONING_CLOSE) -> None:
        """Initialize the Hermes4 reasoning parser with appropriate regex patterns."""
        super().__init__(reasoning_open=reasoning_open, reasoning_close=reasoning_close)
    
    def needs_redacted_reasoning_prefix(self) -> bool:
        """Check if the reasoning parser needs a redacted reasoning prefix.
        
        Returns
        -------
        bool
            True if the reasoning parser needs a redacted reasoning prefix, False otherwise.
        """
        return True 