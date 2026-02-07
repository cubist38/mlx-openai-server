from __future__ import annotations

import re

from .glm4_moe import GLM4MoEToolParser

TOOL_OPEN = "<minimax:tool_call>"
TOOL_CLOSE = "</minimax:tool_call>"


class MiniMaxM2ToolParser(GLM4MoEToolParser):
    """Tool parser for MiniMax M2 model's tool response format.

    Handles the MiniMax M2 model's tool response format:
    <minimax:tool_call>
    <invoke name="tool-name-1">
    <parameter name="param-key-1">param-value-1</parameter>
    <parameter name="param-key-2">param-value-2</parameter>
    ...
    </invoke>
    <minimax:tool_call>
    """

    def __init__(self, tool_open: str = TOOL_OPEN, tool_close: str = TOOL_CLOSE) -> None:
        """Initialize the MiniMax M2 tool parser with appropriate regex patterns."""
        super().__init__(tool_open=tool_open, tool_close=tool_close)
        
        self.func_call_regex = re.compile(
            r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.DOTALL
        )
        
        # Regex patterns for parsing MiniMax tool calls
        self.func_detail_regex = re.compile(
            r'<invoke name="([^"]+)"\s*>(.*)', re.DOTALL
        )
        self.func_arg_regex = re.compile(
            r'<parameter name="([^"]+)"\s*>(.*?)</parameter>', re.DOTALL
        )