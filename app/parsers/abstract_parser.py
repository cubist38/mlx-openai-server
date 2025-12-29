class AbstractReasoningParser:
    """
    Abstract reasoning parser class that should not be used directly. Provided
    properties and methods should be used in
    derived classes.
    """

    def __init__(self, reasoning_open: str, reasoning_close: str):
        self.reasoning_open = reasoning_open
        self.reasoning_close = reasoning_close

    def get_reasoning_open(self):
        return self.reasoning_open

    def get_reasoning_close(self):
        return self.reasoning_close

    def extract_reasoning(self, model_output: str):
        raise NotImplementedError(
            "AbstractReasoningParser.extract_reasoning has not been implemented!"
        )
    
    def extract_reasoning_streaming(self, chunk: str):
        raise NotImplementedError(
            "AbstractReasoningParser.extract_reasoning_stream has not been implemented!"
        )

class ToolParserState:
    NORMAL = 0
    FOUND_PREFIX = 1
    FOUND_ARGUMENTS = 2

class AbstractToolParser:
    """
    Abstract tool parser class that should not be used directly. Provided
    properties and methods should be used in
    derived classes.
    """

    def __init__(self, tool_open: str, tool_close: str, state: ToolParserState = ToolParserState.NORMAL):
        self.tool_open = tool_open
        self.tool_close = tool_close

        # use for streaming parsing
        self.state = state
        self.buffer = ""

    def get_tool_open(self):
        return self.tool_open

    def get_tool_close(self):
        return self.tool_close

    def extract_tool_calls(self, model_output: str):
        raise NotImplementedError(
            "AbstractToolParser.extract_tool_calls has not been implemented!"
        )

    def extract_tool_calls_streaming(self, chunk: str):
        raise NotImplementedError(
            "AbstractToolParser.extract_tool_calls_streaming has not been implemented!"
        )
  