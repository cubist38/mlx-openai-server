from .base import BaseToolParser, BaseThinkingParser

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
THINKING_OPEN = "<think>"
THINKING_CLOSE = "</think>"

class Qwen3NextToolParser(BaseToolParser):
    """Parser for Qwen3 MoE model's tool response format."""
    
    def __init__(self):
        super().__init__(
            tool_open=TOOL_OPEN,
            tool_close=TOOL_CLOSE   
        )

class Qwen3NextThinkingParser(BaseThinkingParser):
    """Parser for Qwen3 MoE model's thinking response format."""
    
    def __init__(self):
        super().__init__(
            thinking_open=THINKING_OPEN,
            thinking_close=THINKING_CLOSE
        )