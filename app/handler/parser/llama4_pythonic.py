import ast
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseToolParser

logger = logging.getLogger(__name__)

class Llama4PythonicToolParser(BaseToolParser):
    """
    Toolcall parser for Llama4 that produce tool calls in a pythonic style
    Use --enable-auto-tool-choice --tool-call-parser llama4_pythonic
    """

    # Regex to match the tool call pattern: [call(arg=val), ...]
    TOOL_CALL_REGEX = re.compile(
        r"\[([a-zA-Z]+\w*\(([a-zA-Z]+\w*=.*,\s*)*([a-zA-Z]+\w*=.*\s)?\),\s*)*([a-zA-Z]+\w*\(([a-zA-Z]+\w*=.*,\s*)*([a-zA-Z]+\w*=.*\s*)?\)\s*)+\]",
        re.DOTALL,
    )

    def __init__(self):
        # We set tool_open and tool_close for reference, but we override parse/parse_stream
        # to handle the specific pythonic list format which might contain nested brackets.
        super().__init__(tool_open="[", tool_close="]")
        self.buffer = ""
        self.parsing_tool = False

    def _handle_single_tool(self, node: ast.Call) -> Dict[str, Any]:
        """Extract function name and arguments from an AST Call node."""
        if isinstance(node.func, ast.Name):
            function_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            function_name = node.func.attr
        else:
            function_name = str(node.func)

        arguments = {}
        for keyword in node.keywords:
            try:
                # Try to evaluate the value as a literal
                arguments[keyword.arg] = ast.literal_eval(keyword.value)
            except Exception:
                # Fallback: unparse the node to get the string representation
                try:
                    # ast.unparse is available in Python 3.9+
                    arguments[keyword.arg] = ast.unparse(keyword.value)
                except AttributeError:
                    # Fallback for older python versions or if unparse fails
                    arguments[keyword.arg] = str(keyword.value)
        
        return {
            "name": function_name,
            "arguments": arguments
        }

    def parse(self, content: str) -> Tuple[Optional[List[Dict[str, Any]]], str]:
        """
        Extract the tool calls from a complete model response.
        Collects content before and after tool calls as remaining_content.
        """
        remaining_parts = []
        tool_calls = []
        
        # remove <|python_start|> and <|python_end|>
        if content.startswith("<|python_start|>"):
            content = content[len("<|python_start|>") :]
            content = content.replace("<|python_end|>", "")

        # Find the tool call (starts with '[' and ends with ']')
        start_idx = content.find(self.tool_open)
        if start_idx == -1:
            # No tool call found
            return [], content
        
        # Add content before tool call to remaining_parts
        if start_idx > 0:
            before_content = content[:start_idx].strip()
            if before_content:
                remaining_parts.append(before_content)
        
        # Find the closing bracket
        end_idx = content.rfind(self.tool_close)
        if end_idx == -1 or end_idx <= start_idx:
            # No proper closing bracket found
            return [], content
        
        # Extract the tool call content
        tool_content = content[start_idx:end_idx + 1]
        
        # Add content after tool call to remaining_parts
        if end_idx + 1 < len(content):
            after_content = content[end_idx + 1:].strip()
            if after_content:
                remaining_parts.append(after_content)
        
        # Check if it matches the tool call pattern
        is_tool_call_pattern = self.TOOL_CALL_REGEX.match(tool_content) is not None

        if not is_tool_call_pattern:
            return [], content

        try:
            # Parse the content as a Python expression
            module = ast.parse(tool_content)
            parsed = getattr(module.body[0], "value", None)
            
            # Verify it's a list of calls
            if isinstance(parsed, ast.List) and all(
                isinstance(e, ast.Call) for e in parsed.elts
            ):
                tool_calls = [
                    self._handle_single_tool(e)
                    for e in parsed.elts
                ]
                remaining_content = " ".join(filter(None, remaining_parts))
                return tool_calls, remaining_content
            else:
                return [], content
        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return [], content

    def parse_stream(self, chunk: Optional[str] = None) -> Tuple[Optional[Any], bool]:
        """
        Parse streaming chunks for tool calls.
        We buffer content starting with '[' and attempt to parse it as a complete list of tools.
        
        Returns:
            Tuple[parsed_content, is_complete]: 
                - parsed_content: The parsed chunk (could be str, list of tools, or None)
                - is_complete: True if tool call is complete
        """
        if chunk is None:
            return None, False

        # If we are not currently parsing a tool, check if this chunk starts one
        if not self.parsing_tool:
            if self.tool_open in chunk:
                self.parsing_tool = True
                start_idx = chunk.find(self.tool_open)
                after_open = chunk[start_idx + len(self.tool_open):]
                before_open = chunk[:start_idx]
                
                # Check if tool_close is also in this chunk (both tags in same chunk)
                if self.tool_close in after_open:
                    close_idx = after_open.find(self.tool_close)
                    self.parsing_tool = False
                    # Extract the complete tool call content
                    tool_content = self.tool_open + after_open[:close_idx + len(self.tool_close)]
                    tools, remaining = self.parse(tool_content)
                    if tools:
                        # Return content before tool + content after tool (if any)
                        after_close = after_open[close_idx + len(self.tool_close):]
                        combined = (before_open + after_close).strip()
                        return tools if not combined else (combined, tools), True
                    # Failed to parse, treat as normal text
                    return chunk, False
                
                # Only opening tag found, buffer content after it
                self.buffer = after_open
                # Return content before opening tag (if any)
                return before_open if before_open else None, False
            # No tool tag, return chunk as is
            return chunk, False
        
        # Currently in parsing mode, append to buffer
        self.buffer += chunk
        
        # Check if we reached the end of the list
        if self.tool_close in chunk:
            close_idx = chunk.find(self.tool_close)
            # Try to parse the complete buffered content
            complete_tool = self.tool_open + self.buffer[:self.buffer.find(self.tool_close) + len(self.tool_close)]
            tools, remaining = self.parse(complete_tool)
            
            if tools:
                self.parsing_tool = False
                after_close = self.buffer[self.buffer.find(self.tool_close) + len(self.tool_close):]
                self.buffer = ""
                # Return tools with completion signal, include content after close tag if any
                return tools if not after_close else (after_close, tools), True
            else:
                # Parsing failed, might not be complete yet
                # Keep buffering
                return None, False
        
        # Still buffering, no close tag found
        return None, False