from .base import BaseToolParser, BaseThinkingParser

THINKING_OPEN = "[THINK]"
THINKING_CLOSE = "[/THINK]"
TOOL_OPEN = "[TOOL_CALLS]"
TOOL_CLOSE = None
ARGUMENT_OPEN = "[ARGS]"

class Ministral3ToolState:
    NORMAL = 0
    FOUND_PREFIX = 1
    FOUND_ARGUMENTS = 2

class Ministral3ToolParser(BaseToolParser):
    """Parser for Ministral3 model's tool response format."""

    def __init__(self):
        super().__init__(tool_open=TOOL_OPEN, tool_close=TOOL_CLOSE)
        self.argument_open = ARGUMENT_OPEN
        self.ministral3_tool_state = Ministral3ToolState.NORMAL

    def parse(self, content: str) -> Tuple[Optional[Dict[str, Any]], str]:
        tool_calls = []
        remaining_parts = []
        
        if self.tool_open not in content:
            return [], content

        tool_open_len = len(self.tool_open)
        argument_open_len = len(self.argument_open)
        pos = 0
        tool_name = None
        tool_arguments = None

        while True:
            start_tool = content.find(self.tool_open, pos)
            if start_tool == -1:
                # No more tool calls, add remaining content
                if pos < len(content):
                    remaining_parts.append(content[pos:].strip())
                break

            # finding argument open tag
            start_argument = content.find(self.argument_open, pos)
            if start_argument == -1:
                # No more arguments, add remaining content
                if pos < len(content):
                    remaining_parts.append(content[pos:].strip())
                break

            # Tool name should between tool open and argument open like [TOOL_CALLS]get_weather[ARGS] -> get_weather
            tool_name = content[start_tool + tool_open_len:start_argument].strip()

            # find end of argument by closing bracket
            end_argument = content.find("}", start_argument + argument_open_len)
            if end_argument == -1:
                # No more arguments, add remaining content
                if pos < len(content):
                    remaining_parts.append(content[pos:].strip())
                break

            # Arguments should between argument open and argument close like [ARGS]{"city": "Tokyo"} -> {"city": "Tokyo"}

            tool_arguments = content[start_argument + argument_open_len:end_argument].strip()

            # the string with "name" and "arguments"
            tool_content = f'{{"name": "{tool_name}", "arguments": {tool_arguments}}}'

            try:
                json_output = self._parse_tool_content(tool_content)
            except json.JSONDecodeError:
                print("Error parsing tool call: ", tool_content)
                # Continue processing remaining content after error
                remaining_parts.append(content[pos:].strip())
                break

            tool_calls.append(json_output)
            # Move position past the argument open tag
            pos = end_argument + 1

        remaining_content = " ".join(filter(None, remaining_parts))
        return tool_calls, remaining_content

    def parse_stream(self, chunk: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], bool]:
        if chunk is None:
            return None, True

        if self.tool_open in chunk:
            self.ministral3_tool_state = Ministral3ToolState.FOUND_PREFIX
            start_tool_index = chunk.find(self.tool_open)

            return chunk[:start_tool_index], False

        if self.ministral3_tool_state == Ministral3ToolState.FOUND_PREFIX:

            if self.argument_open in chunk:
                self.ministral3_tool_state = Ministral3ToolState.FOUND_ARGUMENTS
            

class Ministral3ThinkingParser(BaseThinkingParser):
    """Parser for Ministral3 model's thinking response format."""
    
    def __init__(self):
        super().__init__(
            thinking_open=THINKING_OPEN,
            thinking_close=THINKING_CLOSE
        )