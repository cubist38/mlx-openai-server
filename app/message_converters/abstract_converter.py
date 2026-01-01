from __future__ import annotations

from typing import Any

class AbstractMessageConverter:
    """Abstract message converter class that should not be used directly.
    
    Provided properties and methods should be used in derived classes to convert
    messages to be compatible with specific model chat templates.
    """

    def convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert messages to be compatible with specific model chat templates"""
        raise NotImplementedError(
            "AbstractMessageConverter.convert_messages has not been implemented!"
        )