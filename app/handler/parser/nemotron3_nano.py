import json
import re
from typing import Any, Dict, Optional
from .base import BaseToolParser, BaseThinkingParser

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
THINKING_OPEN = "<think>"
THINKING_CLOSE = "</think>"


class Nemotron3NanoThinkingParser(BaseThinkingParser):
    """Parser for Nemotron3 Nano model's thinking response format."""
    
    def __init__(self):
        super().__init__(
            thinking_open=THINKING_OPEN,
            thinking_close=THINKING_CLOSE
        )


class Nemotron3NanoToolParser(BaseToolParser):
    """Parser for Nemotron3 Nano model's tool response format.
    
    Handles tool calls in the format:
    <tool_call>
    <function=function_name>
    <parameter=param_name>
    param_value
    </parameter>
    </function>
    </tool_call>
    """
    
    def __init__(self):
        super().__init__(
            tool_open=TOOL_OPEN,
            tool_close=TOOL_CLOSE   
        )
        # Regex pattern to extract function name and content
        self.function_regex = re.compile(
            r"<function=([^>]+)>\s*(.*?)\s*</function>", 
            re.DOTALL
        )
        # Regex pattern to extract parameter key-value pairs
        self.parameter_regex = re.compile(
            r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
            re.DOTALL
        )
    
    def _parse_tool_content(self, tool_content: str) -> Optional[Dict[str, Any]]:
        """Parse Nemotron3 Nano's XML-style tool call format.
        
        Parameters
        ----------
        tool_content : str
            The content between <tool_call> tags.
            
        Returns
        -------
        Dict[str, Any] or None
            Dictionary containing 'name' and 'arguments', or None if parsing fails.
        """
        try:
            # Extract function name and its content
            function_match = self.function_regex.search(tool_content)
            if not function_match:
                return None
            
            function_name = function_match.group(1).strip()
            function_content = function_match.group(2)
            
            # Extract all parameters
            arguments = {}
            for param_match in self.parameter_regex.finditer(function_content):
                param_name = param_match.group(1).strip()
                param_value = param_match.group(2).strip()
                arguments[param_name] = param_value
            
            return {
                "name": function_name,
                "arguments": json.dumps(arguments)
            }
        except Exception as e:
            print(f"Error parsing Nemotron3 Nano tool call: {tool_content}, Error: {e}")
            return None