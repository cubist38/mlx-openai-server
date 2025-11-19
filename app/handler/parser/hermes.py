from .base import BaseToolParser, BaseThinkingParser

THINKING_OPEN = "<think>"
THINKING_CLOSE = "</think>"
TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"

class HermesThinkingParser(BaseThinkingParser):
    """Parser for Hermes model's thinking response format."""
    
    def __init__(self):
        super().__init__(
            thinking_open=THINKING_OPEN,
            thinking_close=THINKING_CLOSE
        )

class HermesToolParser(BaseToolParser):
    """Parser for Hermes model's tool response format."""

    def __init__(self):
        super().__init__(
            tool_open=TOOL_OPEN,
            tool_close=TOOL_CLOSE
        )