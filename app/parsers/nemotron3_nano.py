from __future__ import annotations

import re

from .hermes import HermesReasoningParser
from .glm4_moe import GLM4MoEToolParser

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"

Nemotron3NanoReasoningParser = HermesReasoningParser

class Nemotron3NanoToolParser(GLM4MoEToolParser):
    """Tool parser for Nemotron3 Nano model's tool response format.

    Handles the Nemotron3 Nano model's tool response format:
    <tool_call>
    <function=function_name>
    <parameter=param_name>
    param_value
    </parameter>
    </function>
    </tool_call>
    """

    def __init__(self, tool_open: str = TOOL_OPEN, tool_close: str = TOOL_CLOSE) -> None:
        """Initialize the Nemotron3 Nano tool parser with appropriate regex patterns."""
        super().__init__(tool_open=tool_open, tool_close=tool_close)
        
        # Regex pattern to extract function name and content
        self.func_call_regex = re.compile(
            r"<function=([^>]+)>\s*(.*?)\s*</function>", 
            re.DOTALL
        )
        # Regex pattern to extract parameter key-value pairs
        self.func_arg_regex = re.compile(
            r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
            re.DOTALL
        )