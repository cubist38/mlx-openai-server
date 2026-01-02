from __future__ import annotations

import re
import json

from .hermes import HermesReasoningParser
from .glm4_moe import GLM4MoEToolParser


REASONING_OPEN = "<think>"
REASONING_CLOSE = "</think>"
TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"

class Nemotron3NanoReasoningParser(HermesReasoningParser):
    """Parser for Nemotron3 Nano model's reasoning response format."""
    
    def __init__(self):
        super().__init__(
            reasoning_open=REASONING_OPEN,
            reasoning_close=REASONING_CLOSE
        )

    def needs_redacted_reasoning_prefix(self) -> bool:
        """Check if the reasoning parser needs a redacted reasoning prefix.
        
        Returns
        -------
        bool
            True if the reasoning parser needs a redacted reasoning prefix, False otherwise.
        """
        return True


class Nemotron3NanoToolParser(GLM4MoEToolParser):
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
        self.tool_regex = re.compile(
            r"<function=([^>]+)>\s*(.*?)\s*</function>",
            re.DOTALL
        )
        # Regex pattern to extract parameter key-value pairs
        self.parameter_regex = re.compile(
            r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
            re.DOTALL
        )

    def extract_tool_calls(self, model_output: str) -> dict[str, list] | None:
        """Extract tool calls from complete model output.
        
        Parameters
        ----------
        model_output : str
            Complete model output containing tool calls in XML-like format.
            
        Returns
        -------
        dict[str, list] | None
            Dictionary with 'tool_calls' key containing list of parsed tool calls,
            or None if no tool calls found. Each tool call has 'name' and 'arguments'.
        """
        matches = self.tool_regex.findall(model_output)
        if not matches:
            return {
                "content": model_output
            }
        
        tool_calls = []
        for match in matches:
            function_name = match[0].strip()
            function_content = match[1].strip()
            
            # Extract parameters from function content
            param_matches = self.parameter_regex.findall(function_content)
            arguments = {}
            for param_match in param_matches:
                param_name = param_match[0].strip()
                param_value = param_match[1].strip()
                
                # Try to parse the value as JSON (for numbers, booleans, objects, arrays)
                try:
                    arguments[param_name] = json.loads(param_value)
                except (json.JSONDecodeError, ValueError):
                    arguments[param_name] = param_value
            
            tool_calls.append({
                "name": function_name,
                "arguments": json.dumps(arguments),
            })
        
        return {"tool_calls": tool_calls}