from __future__ import annotations

from enum import Enum


class ReasoningParserState(Enum):
    """State constants for reasoning parser streaming operations."""

    NORMAL = "normal"
    FOUND_PREFIX = "found_prefix"


class AbstractReasoningParser:
    """Abstract reasoning parser class that should not be used directly.
    
    Provided properties and methods should be used in derived classes to parse
    reasoning content from model outputs.
    """

    def __init__(
        self,
        reasoning_open: str,
        reasoning_close: str,
        state: ReasoningParserState = ReasoningParserState.NORMAL,
    ) -> None:
        """Initialize the reasoning parser.
        
        Parameters
        ----------
        reasoning_open : str
            Opening tag/marker for reasoning content.
        reasoning_close : str
            Closing tag/marker for reasoning content.
        state : ReasoningParserState, optional
            Initial parser state, by default ReasoningParserState.NORMAL.
        """
        self.reasoning_open = reasoning_open
        self.reasoning_close = reasoning_close
        self.state = state
        self.buffer = ""

    def get_reasoning_open(self) -> str:
        """Get the opening tag for reasoning content.
        
        Returns
        -------
        str
            The opening tag string.
        """
        return self.reasoning_open

    def get_reasoning_close(self) -> str:
        """Get the closing tag for reasoning content.
        
        Returns
        -------
        str
            The closing tag string.
        """
        return self.reasoning_close

    def needs_redacted_reasoning_prefix(self) -> bool:
        """Check if the reasoning parser needs a redacted reasoning prefix.
        
        Returns
        -------
        bool
            True if the reasoning parser needs a redacted reasoning prefix, False otherwise.
        """
        return False

    def has_special_parsing(self) -> bool:
        """Check if the reasoning parser has special parsing logic.
        
        Returns
        -------
        bool
            True if the reasoning parser has special parsing logic, False otherwise.
        """
        return False

    def respects_enable_thinking(self) -> bool:
        """Check if the reasoning parser respects the enable_thinking flag.
        
        Returns
        -------
        bool
            True if the reasoning parser respects the enable_thinking flag, False otherwise.
        """
        return False

    def extract_reasoning(self, model_output: str) -> dict[str, str] | None:
        """Extract reasoning content from complete model output.
        
        Parameters
        ----------
        model_output : str
            Complete model output to parse.
            
        Returns
        -------
        dict[str, str] | None
            Dictionary with 'reasoning' key containing extracted content,
            or None if no reasoning found.
            
        Raises
        ------
        NotImplementedError
            This method must be implemented by derived classes.
        """
        raise NotImplementedError(
            "AbstractReasoningParser.extract_reasoning has not been implemented!"
        )

    def extract_reasoning_streaming(
        self, chunk: str
    ) -> tuple[dict[str, str] | str | None, bool]:
        """Extract reasoning content from streaming chunks.
        
        Parameters
        ----------
        chunk : str
            Chunk of model output to process.
            
        Returns
        -------
        tuple[dict[str, str] | str | None, bool]
            Tuple of (extracted_content, is_complete) where:
            - extracted_content: Reasoning dict, passthrough chunk, or None
            - is_complete: True if chunk should be sent, False if buffering
            
        Raises
        ------
        NotImplementedError
            This method must be implemented by derived classes.
        """
        raise NotImplementedError(
            "AbstractReasoningParser.extract_reasoning_streaming has not been implemented!"
        )


class ToolParserState(Enum):
    """State constants for tool parser streaming operations."""

    NORMAL = "normal"
    FOUND_PREFIX = "found_prefix"
    FOUND_ARGUMENTS = "found_arguments"


class AbstractToolParser:
    """Abstract tool parser class that should not be used directly.
    
    Provided properties and methods should be used in derived classes to parse
    tool calls from model outputs.
    """

    def __init__(
        self,
        tool_open: str,
        tool_close: str,
        state: ToolParserState = ToolParserState.NORMAL,
    ) -> None:
        """Initialize the tool parser.
        
        Parameters
        ----------
        tool_open : str
            Opening tag/marker for tool calls.
        tool_close : str
            Closing tag/marker for tool calls.
        state : ToolParserState, optional
            Initial parser state, by default ToolParserState.NORMAL.
        """
        self.tool_open = tool_open
        self.tool_close = tool_close
        self.state = state
        self.buffer = ""

    def get_tool_open(self) -> str:
        """Get the opening tag for tool calls.
        
        Returns
        -------
        str
            The opening tag string.
        """
        return self.tool_open

    def get_tool_close(self) -> str:
        """Get the closing tag for tool calls.
        
        Returns
        -------
        str
            The closing tag string.
        """
        return self.tool_close

    def extract_tool_calls(self, model_output: str) -> dict[str, list] | None:
        """Extract tool calls from complete model output.
        
        Parameters
        ----------
        model_output : str
            Complete model output to parse.
            
        Returns
        -------
        dict[str, list] | None
            Dictionary with 'tool_calls' key containing list of tool calls,
            or None if no tool calls found.
            
        Raises
        ------
        NotImplementedError
            This method must be implemented by derived classes.
        """
        raise NotImplementedError(
            "AbstractToolParser.extract_tool_calls has not been implemented!"
        )

    def extract_tool_calls_streaming(
        self, chunk: str
    ) -> tuple[dict[str, list] | str | None, bool]:
        """Extract tool calls from streaming chunks.
        
        Default implementation that buffers content between tool_open and tool_close
        tags. Subclasses can override this for custom streaming behavior.
        
        Parameters
        ----------
        chunk : str
            Chunk of model output to process.
            
        Returns
        -------
        tuple[dict[str, list] | str | None, bool]
            Tuple of (extracted_content, is_complete) where:
            - extracted_content: Tool calls dict, passthrough chunk, or None
            - is_complete: True if chunk should be sent, False if buffering
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

        return {"content": chunk}, False