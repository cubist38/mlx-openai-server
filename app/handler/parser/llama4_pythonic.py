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
        """
        # remove <|python_start|> and <|python_end|>
        if content.startswith("<|python_start|>"):
            content = content[len("<|python_start|>") :]
            content = content.replace("<|python_end|>", "")

        # Check if it matches the tool call pattern
        is_tool_call_pattern = self.TOOL_CALL_REGEX.match(content) is not None

        if not is_tool_call_pattern:
            return [], content

        try:
            # Parse the content as a Python expression
            module = ast.parse(content)
            parsed = getattr(module.body[0], "value", None)
            
            # Verify it's a list of calls
            if isinstance(parsed, ast.List) and all(
                isinstance(e, ast.Call) for e in parsed.elts
            ):
                tool_calls = [
                    self._handle_single_tool(e)
                    for e in parsed.elts
                ]
                return tool_calls, "" # All content consumed as tool calls
            else:
                return [], content
        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return [], content

    def parse_stream(self, chunk: Optional[str] = None) -> Tuple[Optional[Any], bool]:
        """
        Parse streaming chunks for tool calls.
        We buffer content starting with '[' and attempt to parse it as a complete list of tools.
        """
        if chunk is None:
            return None, False

        # If we are not currently parsing a tool, check if this chunk starts one
        if not self.parsing_tool:
            if chunk.strip().startswith(self.tool_open):
                self.parsing_tool = True
                self.buffer = chunk
                # If the chunk itself is a complete tool call (rare in streaming but possible)
                if self.buffer.strip().endswith(self.tool_close):
                    tools, remaining = self.parse(self.buffer)
                    if tools:
                        self.parsing_tool = False
                        self.buffer = ""
                        return tools, True
                return None, False
            else:
                # Not a tool call, return as is
                return chunk, False
        
        # We are in parsing mode, append to buffer
        self.buffer += chunk
        
        # Check if we reached the end of the list
        # This is a heuristic: if it ends with ']', we try to parse.
        # Note: This might fail if ']' is inside a string, but parse() will handle validity.
        if self.buffer.strip().endswith(self.tool_close):
            tools, remaining = self.parse(self.buffer)
            if tools:
                self.parsing_tool = False
                self.buffer = ""
                return tools, True
            else:
                # Parsing failed, maybe it's not complete yet or it's just text that looks like a list.
                # If it's really long and still failing, maybe we should give up?
                # For now, we keep buffering.
                pass
        
        return None, False