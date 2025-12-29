import json
import re
from typing import Any, Dict, Optional, Tuple
from .base import BaseToolParser, BaseThinkingParser

from loguru import logger

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
        self.tool_open=TOOL_OPEN

    def parse_stream(self, chunk: Optional[str] = None) -> Tuple[Optional[Any], bool]:
        """
        Parse streaming chunks for thinking content.
        
        Returns:
            Tuple[parsed_content, is_complete]: 
                - parsed_content: The parsed chunk (could be str, dict, or None)
                - is_complete: True if thinking section is complete
        """
        if chunk is None:
            return None, False
            
        logger.info(f"ThinkingParser::parseStream chunk={chunk} isThinking={self.is_thinking}")
        if not self.is_thinking:
            # Check if thinking_open is in the chunk
            if self.thinking_open in chunk:
                self.is_thinking = True
                start_idx = chunk.find(self.thinking_open)
                after_open = chunk[start_idx + len(self.thinking_open):]
                before_open = chunk[:start_idx]
                
                # Check if thinking_close is also in this chunk (both tags in same chunk)
                if self.thinking_close in after_open:
                    close_idx = after_open.find(self.thinking_close)
                    self.is_thinking = False
                    # Return content before open tag + content after close tag
                    after_close = after_open[close_idx + len(self.thinking_close):]
                    return (before_open + after_close) if (before_open + after_close) else None, True
                
                # Only opening tag found, return content before it (if any) and reasoning content after
                # If there's content after the opening tag, return it as reasoning_content
                if after_open:
                    return {
                        "reasoning_content": after_open
                    }, False
                # Just the opening tag with nothing after it
                return before_open if before_open else None, False
            # No thinking tag, return chunk as is
            return chunk, False
        
        # Currently in thinking mode
        if self.thinking_close in chunk:
            close_idx = chunk.find(self.thinking_close)
            reasoning_part = chunk[:close_idx]
            after_close = chunk[close_idx + len(self.thinking_close):]
            self.is_thinking = False
            
            # If there's reasoning content before the close tag, return it with completion signal
            if reasoning_part:
                result = {"reasoning_content": reasoning_part}
                # If there's also content after the close tag, include it as text
                if after_close:
                    result["content"] = after_close
                return result, True
            # Close tag found, thinking complete, return content after close tag (if any)
            return after_close if after_close else None, True

        if self.tool_open in chunk:
            close_idx = chunk.find(self.tool_open)
            reasoning_part = chunk[:close_idx]
            after_close = chunk[close_idx:]
            # after_close = chunk[close_idx + len(self.tool_open):]
            self.is_thinking = False
            
            logger.info(f"ThinkingParser::parseStream tool_open reasoning_part={reasoning_part} after_close={after_close}")
            result = {"content": after_close}
            # If there's reasoning content before the close tag, return it with completion signal
            if reasoning_part:
                result = {"reasoning_content": reasoning_part}
            return result, True
        
        # Still in thinking mode, return as reasoning content
        return {
            "reasoning_content": chunk
        }, False


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
        logger.info("Nemotron3NanoToolParser::__init__")
    
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

    def _set_content(self, res: Dict[str, Any], content: str) -> None:
        """Helper to set content only if non-empty."""
        if content:
          res["content"] = content
