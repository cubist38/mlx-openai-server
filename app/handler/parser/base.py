import json
from json_repair import repair_json
from typing import Any, Dict, List, Optional, Tuple


class BaseThinkingParser:
    def __init__(self, thinking_open: str, thinking_close: str):
        self.thinking_open = thinking_open
        self.thinking_close = thinking_close
        self.is_thinking = False

    def parse(self, content: str) -> Tuple[Optional[str], str]:
        if self.thinking_open in content:
            start_thinking = content.find(self.thinking_open)
            end_thinking = content.find(self.thinking_close)
            if end_thinking != -1:
                return content[start_thinking + len(self.thinking_open):end_thinking].strip(), content[end_thinking + len(self.thinking_close):].strip()
        return None, content
    
    def parse_stream(self, chunk: Optional[str] = None) -> Tuple[Optional[Any], bool]:
        """
        Parse streaming chunks for thinking content.
        
        Returns:
            Tuple[parsed_content, is_complete]: 
                - parsed_content: The parsed chunk (could be str, dict, or None)
                - is_complete: True if thinking section is complete
        """
        if not self.is_thinking:
            if chunk == self.thinking_open:
                self.is_thinking = True
                return None, False
            return chunk, False
        if chunk == self.thinking_close:
            self.is_thinking = False
            return None, True
        
        return {
            "reasoning_content": chunk
        }, False

class ParseToolState:
    NORMAL = 0
    FOUND_PREFIX = 1
  
class BaseToolParser:
    def __init__(self, tool_open: str, tool_close: str):
        self.tool_open = tool_open
        self.tool_close = tool_close
        self.buffer = ""
        self.state = ParseToolState.NORMAL

    def get_tool_open(self):
        return self.tool_open
    
    def get_tool_close(self):
        return self.tool_close
    
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
        remaining_content = ""
        start = 0
        while True:
            start_tool = content.find(self.tool_open, start)
            if start_tool == -1:
                break
            remaining_content += content[:start_tool].strip()
            end_tool = content.find(self.tool_close, start_tool + len(self.tool_open))
            if end_tool == -1:
                break
            tool_content = content[start_tool + len(self.tool_open):end_tool].strip()

            try:
                json_output = self._parse_tool_content(tool_content)
                tool_calls.append(json_output)
            except json.JSONDecodeError:
                print("Error parsing tool call: ", tool_content)
                break
            content = content[end_tool + len(self.tool_close):].strip()
        return tool_calls, remaining_content
    
    def parse_stream(self, chunk: Optional[str] = None) -> Tuple[Optional[Any], bool]:
        """
        Parse streaming chunks for tool calls.
        
        Returns:
            Tuple[parsed_content, is_complete]: 
                - parsed_content: The parsed chunk (could be str, dict, or None)
                - is_complete: True if tool call is complete
        """
        if chunk is None:
            return None, True
        
        if self.tool_open in chunk:
            self.state = ParseToolState.FOUND_PREFIX
            start_tool_index = chunk.find(self.tool_open)
            end_tool_index = chunk.find(self.tool_close)
            if end_tool_index != -1:
                self.buffer = chunk[start_tool_index + len(self.tool_open):end_tool_index]
                self.state = ParseToolState.NORMAL
                try:
                    json_output = self._parse_tool_content(self.buffer)
                except json.JSONDecodeError:
                    print("Error parsing tool call: ", self.buffer)
                    return None, True
                return {
                    "name": json_output["name"],
                    "arguments": json.dumps(json_output["arguments"])
                }, True

            self.buffer += chunk[start_tool_index + len(self.tool_open):]
            
            return chunk[:start_tool_index], False

        if self.state == ParseToolState.FOUND_PREFIX:
            end_tool_index = chunk.find(self.tool_close)
            if end_tool_index != -1:
                self.buffer += chunk[:end_tool_index]
                try:
                    json_output = self._parse_tool_content(self.buffer)
                except json.JSONDecodeError:
                    print("Error parsing tool call: ", self.buffer)
                    return None, False
                return {
                    "name": json_output["name"],
                    "arguments": json.dumps(json_output["arguments"])
                }, True
            else:
                self.buffer += chunk
                return None, False
            
        return chunk, False

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