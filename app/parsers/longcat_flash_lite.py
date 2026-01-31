from __future__ import annotations

import re
from .glm4_moe import GLM4MoEToolParser

TOOL_OPEN = "<longcat_tool_call>"
TOOL_CLOSE = "</longcat_tool_call>"

class LongCatFlashLiteToolParser(GLM4MoEToolParser):
    """Tool parser for LongCat Flash Lite model's tool response format.

    Handles the LongCat Flash Lite model's tool response format:
    <longcat_tool_call>{function-name}
    <longcat_arg_key>{arg-key-1}</longcat_arg_key>
    <longcat_arg_value>{arg-value-1}</longcat_arg_value>
    <longcat_arg_key>{arg-key-2}</longcat_arg_key>
    <longcat_arg_value>{arg-value-2}</longcat_arg_value>
    ...
    </longcat_tool_call>
    """

    def __init__(self, tool_open: str = TOOL_OPEN, tool_close: str = TOOL_CLOSE) -> None:
        """Initialize the LongCat Flash Lite tool parser with appropriate regex patterns."""
        super().__init__(tool_open=tool_open, tool_close=tool_close)
        
        self.func_call_regex = re.compile(r"<longcat_tool_call>.*?</longcat_tool_call>", re.DOTALL)
        self.func_detail_regex = re.compile(
            r"<longcat_tool_call>(.*?)(<longcat_arg_key>.*?)?</longcat_tool_call>", re.DOTALL
        )
        self.func_arg_regex = re.compile(
            r"<longcat_arg_key>(.*?)</longcat_arg_key>(?:\\n|\s)*<longcat_arg_value>(.*?)</longcat_arg_value>",
            re.DOTALL,
        )