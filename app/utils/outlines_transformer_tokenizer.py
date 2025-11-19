"""Custom tokenizer class extending outlines TransformerTokenizer with custom hashing."""

from outlines.models.transformers import TransformerTokenizer

from .dill import Hasher


class OutlinesTransformerTokenizer(TransformerTokenizer):
    """
    Update the outlines TransformerTokenizer to use our own Hasher class, so that we don't need the datasets dependency.

    This class and the external dependency can be removed when the following import is deleted
    https://github.com/dottxt-ai/outlines/blob/69418d/outlines/models/transformers.py#L117
    """

    def __hash__(self):
        """Return hash of the tokenizer using custom Hasher.

        Returns
        -------
        int
            Hash value of the tokenizer.
        """
        return hash(Hasher.hash(self.tokenizer))
