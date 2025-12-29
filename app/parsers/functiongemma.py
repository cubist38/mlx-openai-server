from __future__ import annotations

import re
import json

from .abstract_parser import AbstractToolParser, ToolParserState

TOOL_OPEN = "<start_function_call>"
TOOL_CLOSE = "<end_function_call>"


class FunctionGemmaToolParser(AbstractToolParser):
    """Tool parser for Google's FunctionGemma model (google/functiongemma-270m-it).

    Handles the FunctionGemma function call format:
    <start_function_call>call:func_name{param:<escape>value<escape>}<end_function_call>
    """

    def __init__(self) -> None:
        """Initialize the FunctionGemma tool parser with appropriate regex patterns."""
        super().__init__(tool_open=TOOL_OPEN, tool_close=TOOL_CLOSE)
        # Regex patterns
        self.tool_call_regex = re.compile(
            r"<start_function_call>call:(\w+)\{(.*?)\}<end_function_call>"
            r"|<start_function_call>call:(\w+)\{(.*)",
            re.DOTALL,
        )
        self.arg_regex = re.compile(
            r"(\w+):<escape>(.*?)<escape>",
            re.DOTALL,
        )

    def extract_tool_calls(self, model_output: str) -> dict[str, list] | None:
        """Extract tool calls from complete model output.
        
        Parameters
        ----------
        model_output : str
            Complete model output containing tool calls.
            
        Returns
        -------
        dict[str, list] | None
            Dictionary with 'tool_calls' key containing list of parsed tool calls,
            or None if no tool calls found. Each tool call has 'name' and 'arguments'.
        """
        matches = self.tool_call_regex.findall(model_output)
        if not matches:
            return {
                "content": model_output
            }
        tool_calls = []
        for match in matches:
            function_name = match[0]
            args_str = match[1]
            args_matches = self.arg_regex.findall(args_str)
            args_dict = {key: value for key, value in args_matches}
            tool_calls.append({"name": function_name, "arguments": json.dumps(args_dict)})
        return {
            "tool_calls": tool_calls,
        }
        

    def extract_tool_calls_streaming(
        self, chunk: str
    ) -> tuple[dict[str, list] | str | None, bool]:
        """Extract tool calls from streaming chunks.
        
        Parameters
        ----------
        chunk : str
            Chunk of model output to process.
            
        Returns
        -------
        tuple[dict[str, list] | str | None, bool]
            Tuple of (extracted_content, is_complete) where:
            - extracted_content: Tool calls dict if complete, chunk if passthrough, None if buffering
            - is_complete: True if chunk should be sent, False if still buffering
        """
        if self.tool_open in chunk:
            self.state = ToolParserState.FOUND_PREFIX

            if self.tool_close in chunk:
                self.buffer += chunk
                result = self.extract_tool_calls(self.buffer)
                self.buffer = ""
                self.state = ToolParserState.NORMAL
                return result, True
            else:
                self.buffer += chunk
                return None, False

        if self.state == ToolParserState.FOUND_PREFIX:
            if self.tool_close in chunk:
                self.buffer += chunk
                result = self.extract_tool_calls(self.buffer)
                self.buffer = ""
                self.state = ToolParserState.NORMAL
                return result, True
            else:
                self.buffer += chunk
                return None, False

        return chunk, True


if __name__ == "__main__":
    parser = FunctionGemmaToolParser()
    tool_call_regex = parser.tool_call_regex
    arg_regex = parser.arg_regex
    # model_output = "I will call a function<start_function_call>call:get_weather{param:<escape>value<escape>}<end_function_call>\n\n I will call another function<start_function_call>call:get_time{}<end_function_call>"
    # tool_calls = parser.extract_tool_calls(model_output)
    # print(tool_calls)
    chunks = [
        "I will call a function",
        "<start_function_call>call:",
        "get_weather",
        "{param:<escape>value<escape>}",
        "<end_function_call>",
        "\n\n",
        "I will call another function",
        "<start_function_call>call:get_time{}<end_function_call>",
    ]
    for chunk in chunks:
        tool_calls, is_complete = parser.extract_tool_calls_streaming(chunk)
        print("Tool calls: ", tool_calls)
        print("Is complete: ", is_complete)