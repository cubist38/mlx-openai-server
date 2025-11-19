import re
import json
from typing import Any, Dict, List, Optional, Tuple
from .base import BaseToolParser, BaseThinkingParser, BaseMessageConverter

TOOL_OPEN = "<minimax:tool_call>"
TOOL_CLOSE = "</minimax:tool_call>"
THINKING_OPEN = "<think>"
THINKING_CLOSE = "</think>"

class MinimaxThinkingParser(BaseThinkingParser):
    """Parser for MiniMax model's thinking response format."""
    
    def __init__(self):
        super().__init__(
            thinking_open=THINKING_OPEN,
            thinking_close=THINKING_CLOSE
        )

class MinimaxToolParser(BaseToolParser):
    """Parser for MiniMax model's tool response format with XML-style arguments."""
    
    def __init__(self):
        super().__init__(
            tool_open=TOOL_OPEN,
            tool_close=TOOL_CLOSE
        )
        # Regex patterns for parsing MiniMax tool calls
        self.func_detail_regex = re.compile(
            r'<invoke name="([^"]+)"\s*>(.*)', re.DOTALL
        )
        self.func_arg_regex = re.compile(
            r'<parameter name="([^"]+)"\s*>([^<]*)</parameter>', re.DOTALL
        )
        
    def _deserialize_value(self, value: str) -> Any:
        """Try to deserialize a value from string to appropriate Python type."""
        value = value.strip()
        
        # Try JSON parsing first
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Try literal eval for Python literals
        try:
            import ast
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        
        # Return as string if all else fails
        return value
    
    def _parse_tool_content(self, tool_content: str) -> Optional[Dict[str, Any]]:
        """
        Overrides the base method to parse MiniMax's specific tool call format.
        """
        try:
            # Extract function name and arguments section
            detail_match = self.func_detail_regex.search(tool_content)
            if not detail_match:
                return None
            
            func_name = detail_match.group(1).strip()
            args_section = detail_match.group(2)
            
            # Extract all key-value pairs
            arg_pairs = self.func_arg_regex.findall(args_section)
            
            arguments = {}
            for key, value in arg_pairs:
                arg_key = key.strip()
                arg_value = self._deserialize_value(value)
                arguments[arg_key] = arg_value
            
            # Build tool call object
            return {
                "name": func_name,
                "arguments": arguments
            }
        except Exception as e:
            print(f"Error parsing MiniMax tool call content: {tool_content}, Error: {e}")
            return None

class MiniMaxMessageConverter(BaseMessageConverter):
    """MiniMax-specific message format converter"""

    def _parse_arguments_string(self, arguments_str: str) -> Any:
        """Parse MiniMax-specific argument string format"""
        try:
            return json.loads(arguments_str)
        except json.JSONDecodeError:
            return arguments_str
