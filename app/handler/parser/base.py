import json
import logging
from json_repair import repair_json
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BaseThinkingParser:
    def __init__(self, thinking_open: str, thinking_close: str):
        self.thinking_open = thinking_open
        self.thinking_close = thinking_close
        self.is_thinking = False

    def get_thinking_open(self):
        return self.thinking_open
    
    def get_thinking_close(self):
        return self.thinking_close

    def parse(self, content: str) -> Tuple[Optional[str], str]:
        start_thinking = content.find(self.thinking_open)
        if start_thinking == -1:
            return None, content
        
        thinking_open_len = len(self.thinking_open)
        thinking_close_len = len(self.thinking_close)
        start_content = start_thinking + thinking_open_len
        end_thinking = content.find(self.thinking_close, start_content)
        
        if end_thinking == -1:
            return None, content
        
        thinking_content = content[start_content:end_thinking].strip()
        remaining_content = content[end_thinking + thinking_close_len:].strip()
        return thinking_content, remaining_content
        
    
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
        
        # Still in thinking mode, return as reasoning content
        return {
            "reasoning_content": chunk
        }, False

class ParseToolState:
    NORMAL = 0
    FOUND_PREFIX = 1
  
class BaseToolParser:
    def __init__(self, tool_open: str, tool_close: Optional[str] = None):
        self.tool_open = tool_open
        self.tool_close = tool_close 
        self.buffer = ""
        self.state = ParseToolState.NORMAL
        # Pre-calculate lengths for performance
        self._tool_open_len = len(tool_open)
        self._tool_close_len = len(tool_close) if tool_close else 0

    def get_tool_open(self):
        return self.tool_open
    
    def get_tool_close(self):
        return self.tool_close
    
    def _set_content(self, res: Dict[str, Any], content: str) -> None:
        """Helper to set content only if non-empty."""
        res["content"] = content if content else None
    
    def _parse_tool_content(self, tool_content: str) -> Optional[Dict[str, Any]]:
        """
        Parses the content of a tool call. Subclasses can override this method
        to support different content formats (e.g., XML, YAML).
        Args:
            tool_content: The string content extracted from between the tool tags.
        Returns:
            A dictionary representing the parsed tool call, or None if parsing fails.
        """

        try:
            repaired_json = repair_json(tool_content)
            return json.loads(repaired_json)
        except json.JSONDecodeError:
            raise

    def parse(self, content: str) -> Tuple[Optional[List[Dict[str, Any]]], str]:
        tool_calls = []
        remaining_parts = []
        
        if self.tool_open not in content:
            return [], content
        
        tool_open_len = len(self.tool_open)
        tool_close_len = len(self.tool_close)
        pos = 0
        
        while True:
            start_tool = content.find(self.tool_open, pos)
            if start_tool == -1:
                # No more tool calls, add remaining content
                if pos < len(content):
                    remaining_parts.append(content[pos:].strip())
                break
            
            # Add content before tool call
            if start_tool > pos:
                remaining_parts.append(content[pos:start_tool].strip())
            
            # Find closing tag
            search_start = start_tool + tool_open_len
            end_tool = content.find(self.tool_close, search_start)
            if end_tool == -1:
                # Unclosed tool tag, add remaining content and break
                remaining_parts.append(content[pos:].strip())
                break
            
            # Extract and parse tool content
            tool_content = content[search_start:end_tool].strip()
            try:
                json_output = self._parse_tool_content(tool_content)
                tool_calls.append(json_output)
            except json.JSONDecodeError:
                print("Error parsing tool call: ", tool_content)
                # Continue processing remaining content after error
                remaining_parts.append(content[pos:].strip())
                break
            
            # Move position past the closing tag
            pos = end_tool + tool_close_len
        
        remaining_content = " ".join(filter(None, remaining_parts))
        return tool_calls, remaining_content
    
    def parse_stream(self, chunk: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        Parse streaming chunks for tool calls.
        Args:
            chunk: The text chunk to parse, or None for empty chunks
        Returns:
            Tuple[parsed_content, is_complete]: 
                - parsed_content: The parsed chunk (could be str, dict)
                - is_complete: True if tool call is complete
        """
        res = {
            "name": None,
            "arguments": None,
            "content": None,
        }
        if chunk is None:
            return None, True

        start_tool_index = chunk.find(self.tool_open)

        if start_tool_index != -1:
            # Reset state and buffer when entering FOUND_PREFIX
            self.state = ParseToolState.FOUND_PREFIX
            self.buffer = ""
            self._set_content(res, chunk[:start_tool_index])
            self.buffer += chunk[start_tool_index + self._tool_open_len:]
            return res, False

        if self.state == ParseToolState.FOUND_PREFIX:
            end_tool_index = chunk.find(self.tool_close)
            if end_tool_index != -1:
                tool_call_content = self.buffer + chunk[:end_tool_index]
                try:
                    json_output = self._parse_tool_content(tool_call_content)
                except json.JSONDecodeError:
                    logger.error("Error parsing tool call: %s", tool_call_content)
                    return res, False
                res["name"] = str(json_output["name"])
                res["arguments"] = str(json_output["arguments"])
                # Calculate remaining content once and reset state
                remaining = chunk[end_tool_index + self._tool_close_len:]
                self.buffer = remaining
                self._set_content(res, remaining)
                self.state = ParseToolState.NORMAL
                return res, True
            else:
                self.buffer += chunk
                return res, False
            
        self._set_content(res, chunk)
        return res, True

"""
Base Message Converter
Provides generic conversion from OpenAI API message format to model-compatible format.
"""
class BaseMessageConverter:
    """Base message format converter class"""

    def convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert message format to be compatible with specific model chat templates"""
        converted_messages = []

        for message in messages:
            converted_message = self._convert_single_message(message)
            if converted_message:
                converted_messages.append(converted_message)

        return converted_messages

    def _convert_single_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a single message"""
        if not isinstance(message, dict):
            return message

        # Convert function.arguments from string to object in tool_calls
        tool_calls = message.get("tool_calls")
        if tool_calls and isinstance(tool_calls, list):
            self._convert_tool_calls(tool_calls)

        return message

    def _convert_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> None:
        """Convert arguments format in tool calls"""
        for tool_call in tool_calls:
            if isinstance(tool_call, dict) and "function" in tool_call:
                function = tool_call["function"]
                if isinstance(function, dict) and "arguments" in function:
                    arguments = function["arguments"]
                    if isinstance(arguments, str):
                        function["arguments"] = self._parse_arguments_string(arguments)

    def _parse_arguments_string(self, arguments_str: str) -> Any:
        """Parse arguments string to object, can be overridden by subclasses"""
        try:
            return json.loads(arguments_str)
        except json.JSONDecodeError:
            return arguments_str