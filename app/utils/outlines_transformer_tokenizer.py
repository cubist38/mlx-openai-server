"""Custom tokenizer class for outlines integration."""

from outlines.models.transformers import TransformerTokenizer

from .dill import Hasher


class OutlinesTransformerTokenizer(TransformerTokenizer):
    """
    Update the outlines TransformerTokenizer to use our own Hasher class, so that we don't need the datasets dependency.

    This class and the external dependency can be removed when the following import is deleted
    https://github.com/dottxt-ai/outlines/blob/69418d/outlines/models/transformers.py#L117
    """

    def __hash__(self) -> int:
        """Return a hash based on the tokenizer using custom hasher."""
        return hash(Hasher.hash(self.tokenizer))
