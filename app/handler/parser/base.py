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
    def __init__(self, tool_open: str, tool_close: str):
        self.tool_open = tool_open
        self.tool_close = tool_close
        self.buffer = ""
        self.state = ParseToolState.NORMAL

    def get_tool_open(self):
        return self.tool_open
    
    def get_tool_close(self):
        return self.tool_close
    
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
                repaired_json = repair_json(tool_content)
                json_output = json.loads(repaired_json)  
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
                    repaired_json = repair_json(self.buffer)
                    json_output = json.loads(repaired_json)
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
                    repaired_json = repair_json(self.buffer)
                    json_output = json.loads(repaired_json)
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
