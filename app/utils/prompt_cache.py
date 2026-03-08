# modified from https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/server.py

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import Any

from loguru import logger
from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache


class LRUPromptCache:
    """LRU cache for MLX prompt KV caches.

    The cache stores token sequences in a trie so it can efficiently find exact
    matches, shorter prefixes, and longer cached sequences that can be trimmed
    down to a requested prefix. Entries are evicted using an LRU policy with
    optional byte-based trimming.

    Parameters
    ----------
    max_size : int, optional
        Maximum number of cache entries to retain, by default 10.
    max_bytes : int, optional
        Maximum total bytes to retain across cached prompt caches, by default a
        practically unbounded value.
    """

    @dataclass
    class CacheEntry:
        """Stored prompt cache entry.

        Parameters
        ----------
        prompt_cache : list[Any]
            Prompt cache object stored for the token sequence.
        nbytes : int
            Approximate number of bytes consumed by the prompt cache.
        """

        prompt_cache: list[Any]
        nbytes: int

    class CacheOrder:
        """Track cache recency while prioritizing checkpoint retention."""

        def __init__(self) -> None:
            """Initialize checkpoint-aware LRU queues."""
            self._lru_checkpoints: deque[tuple[int, ...]] = deque()
            self._lru: deque[tuple[int, ...]] = deque()

        def __len__(self) -> int:
            """Return the total number of tracked cache entries."""
            return len(self._lru) + len(self._lru_checkpoints)

        def push(self, tokens: tuple[int, ...], checkpoint: bool = False) -> None:
            """Append an entry to the appropriate LRU queue."""
            queue = self._lru_checkpoints if checkpoint else self._lru
            queue.append(tokens)

        def remove(self, tokens: tuple[int, ...]) -> None:
            """Remove an entry from whichever queue currently contains it."""
            try:
                self._lru.remove(tokens)
            except ValueError:
                self._lru_checkpoints.remove(tokens)

        def pop(self) -> tuple[int, ...]:
            """Pop the least-recently-used entry, balancing checkpoints."""
            if len(self._lru) >= len(self._lru_checkpoints):
                return self._lru.popleft()
            return self._lru_checkpoints.popleft()

    @dataclass
    class SearchResult:
        """Result of searching the trie for a token sequence.

        Parameters
        ----------
        exact : list[int] | None
            Exact matching token sequence, if found.
        shorter : list[int] | None
            Shorter prefix match, if found.
        longer : list[int] | None
            Longer sequence containing the query as a prefix, if found.
        common_prefix : int
            Length of common prefix with matching cache entries.
        """

        exact: list[int] | None
        shorter: list[int] | None
        longer: list[int] | None
        common_prefix: int

    def __init__(self, max_size: int = 10, max_bytes: int = 1 << 63) -> None:
        """Initialize the LRU prompt cache."""
        self.max_size = max_size
        self.max_bytes = max_bytes
        self._cache: dict[int, Any] = {}
        self._lru = self.CacheOrder()
        self._n_bytes = 0

    def __len__(self) -> int:
        """Return the number of cached sequences."""
        return len(self._lru)

    @property
    def nbytes(self) -> int:
        """Return the approximate total bytes held by cached entries."""
        return self._n_bytes

    def _prompt_cache_nbytes(self, prompt_cache: list[Any]) -> int:
        """Estimate the size in bytes of a prompt cache.

        Parameters
        ----------
        prompt_cache : list[Any]
            Prompt cache layers to size.

        Returns
        -------
        int
            Best-effort byte count for the provided prompt cache.
        """
        total = 0
        for layer in prompt_cache:
            try:
                total += int(getattr(layer, "nbytes", 0))
            except (TypeError, ValueError):
                continue
        return total

    def _search(self, tokens_ids: list[int]) -> SearchResult:
        """Search the cache for a prompt cache.

        Parameters
        ----------
        tokens_ids : list[int]
            Token sequence to search for.
        Returns
        -------
        SearchResult
            Matching information for exact, shorter, or longer cache hits.
        """
        if not self._cache:
            return self.SearchResult(None, None, None, 0)

        current = self._cache
        last_cache_index = -1
        index = 0

        while index < len(tokens_ids) and tokens_ids[index] in current:
            current = current[tokens_ids[index]]
            if "cache" in current:
                last_cache_index = index
            index += 1

        if last_cache_index == len(tokens_ids) - 1:
            return self.SearchResult(tokens_ids, None, None, 0)

        shorter = None
        if last_cache_index > 0:
            shorter = tokens_ids[: last_cache_index + 1]

        longer = None
        common_prefix = index
        if index > 0:
            best = None
            stack = [(current, [])]
            while stack:
                current, extra = stack.pop()
                if "cache" in current:
                    if best is None or len(extra) < len(best):
                        best = extra
                else:
                    for tok in current:
                        stack.append((current[tok], extra + [tok]))
            if best is not None:
                longer = tokens_ids[:index] + best

        return self.SearchResult(None, shorter, longer, common_prefix)


    def fetch_nearest_cache(
        self,
        tokens_ids: list[int],
    ) -> tuple[list[Any] | None, list[int]]:
        """Fetch the nearest matching cache for the given token sequence.

        Parameters
        ----------
        tokens_ids : list[int]
            Token sequence to find a cache for.
        Returns
        -------
        tuple[list[Any] | None, list[int]]
            Tuple of (prompt_cache, remaining_tokens). If no cache found,
            returns (None, original_tokens).
        """
        result = self._search(tokens_ids)
        if result.exact is not None:
            cache_entry = self._get(result.exact)
            return copy.deepcopy(cache_entry.prompt_cache), []

        short_length = len(result.shorter) if result.shorter is not None else 0
        if result.longer is not None and result.common_prefix > short_length:
            cache_entry = self._get(result.longer)
            if can_trim_prompt_cache(cache_entry.prompt_cache):
                cache = copy.deepcopy(cache_entry.prompt_cache)
                prefix = min(len(tokens_ids) - 1, result.common_prefix)
                num_to_trim = len(result.longer) - prefix
                trim_prompt_cache(cache, num_to_trim)
                return cache, tokens_ids[prefix:]

        if short_length > 0:
            cache_entry = self._get(result.shorter)
            return copy.deepcopy(cache_entry.prompt_cache), tokens_ids[short_length:]

        return None, tokens_ids

    def _get(self, tokens_ids: list[int]) -> CacheEntry:
        """Retrieve a cache entry without removing it.

        Parameters
        ----------
        tokens_ids : list[int]
            Token sequence identifying the cache entry.
        Returns
        -------
        CacheEntry
            The cache entry at this location.

        Raises
        ------
        KeyError
            If the token sequence is not in the cache.
        """
        current = self._cache
        for tok in tokens_ids:
            current = current[tok]
        return current["cache"]

    def _delete(self, tokens_ids: list[int]) -> None:
        """Delete a cache entry and clean up empty trie nodes.

        Parameters
        ----------
        tokens_ids : list[int]
            Token sequence identifying the cache entry to delete.
        """
        path = [self._cache]
        for tok in tokens_ids:
            path.append(path[-1][tok])

        cache_bytes = path[-1]["cache"].nbytes
        self._n_bytes -= cache_bytes
        del path[-1]["cache"]
        for i in reversed(range(len(tokens_ids))):
            d_prev, d, t = path[i], path[i + 1], tokens_ids[i]
            if len(d) > 0:
                break
            del d_prev[t]

    def insert_cache(
        self,
        tokens_ids: list[int],
        prompt_cache: list[Any],
        checkpoint: bool = False,
    ) -> None:
        """Insert or update a cache entry.

        Parameters
        ----------
        tokens_ids : list[int]
            Token sequence identifying this cache entry.
        prompt_cache : list[Any]
            The prompt cache data to store.
        checkpoint : bool, optional
            Whether to keep this entry in the checkpoint-priority queue, by
            default ``False``.
        """
        tokens_tuple = tuple(tokens_ids)

        is_trimmable = can_trim_prompt_cache(prompt_cache)
        current = self._cache
        for index, tok in enumerate(tokens_ids):
            if tok not in current:
                current[tok] = {}
            if is_trimmable and "cache" in current:
                self._n_bytes -= current["cache"].nbytes
                del current["cache"]
                self._lru.remove(tuple(tokens_ids[:index]))
            current = current[tok]

        if "cache" in current:
            self._n_bytes -= current["cache"].nbytes
            self._lru.remove(tokens_tuple)
        cache_bytes = self._prompt_cache_nbytes(prompt_cache)
        current["cache"] = self.CacheEntry(prompt_cache, cache_bytes)
        self._n_bytes += cache_bytes

        self._lru.push(tokens_tuple, checkpoint=checkpoint)

        if len(self._lru) > self.max_size:
            oldest_tokens = self._lru.pop()
            self._delete(list(oldest_tokens))

        while self._n_bytes > self.max_bytes and len(self._lru) > 1:
            oldest_tokens = self._lru.pop()
            self._delete(list(oldest_tokens))

    def trim_to(self, *, n_sequences: int | None = None, n_bytes: int | None = None) -> None:
        """Trim the cache down to sequence and/or byte limits.

        Parameters
        ----------
        n_sequences : int | None, optional
            Maximum number of sequences to retain, by default ``None``.
        n_bytes : int | None, optional
            Maximum number of bytes to retain, by default ``None``.
        """
        max_sequences = max(0, n_sequences) if n_sequences is not None else 1 << 63
        max_bytes = max(0, n_bytes) if n_bytes is not None else 1 << 63

        while len(self._lru) > max_sequences:
            oldest_tokens = self._lru.pop()
            self._delete(list(oldest_tokens))

        while self._n_bytes > max_bytes:
            oldest_tokens = self._lru.pop()
            self._delete(list(oldest_tokens))

    def log_cache_stats(self) -> None:
        """Log the current cache size, bytes, and latest checkpoint token count."""
        latest_checkpoint_tokens = (
            len(self._lru._lru_checkpoints[-1]) if self._lru._lru_checkpoints else 0
        )
        logger.info(
            "KV Caches: {} seq, {:.2f} GB, latest user cache {} tokens",
            len(self),
            self.nbytes / 1e9,
            latest_checkpoint_tokens,
        )

if __name__ == "__main__":
    from app.models.mlx_lm import MLX_LM
    model_path = "mlx-community/Qwen3-Coder-Next-8bit"
    draft_model_path = "mlx-community/Qwen3-Coder-Next-4bit"
    model = MLX_LM(model_path, draft_model_path)
    prompt_cache = LRUPromptCache()

    import time

    start_time = time.time()
    first_token = True

    prompt_1 = "Hello, how are you? I'm fine, thank you."
    input_prompt = model.create_input_prompt([{"role": "user", "content": prompt_1}], {})
    input_ids = model.encode_prompt(input_prompt)

    cache, rest_input_ids = prompt_cache.fetch_nearest_cache(input_ids)
    if cache is None:
        cache = model.create_prompt_cache()
    # Use full input_ids for cache_key, not rest_input_ids
    cache_key = input_ids[:]

    response_1 = model(rest_input_ids, cache, stream=True)
    for chunk in response_1:
        if chunk:
            if first_token:
                print("TIME TO FIRST TOKEN", time.time() - start_time)
                first_token = False
            cache_key.append(chunk.token)


    prompt_cache.insert_cache(cache_key, cache)

    start_time = time.time()
    first_token = True
    prompt_2 = "Hello, how are you? I'm fine, thank you."
    input_prompt_2 = model.create_input_prompt([{"role": "user", "content": prompt_2}], {})
    input_ids_2 = model.encode_prompt(input_prompt_2)
    cache, rest_input_ids_2 = prompt_cache.fetch_nearest_cache(input_ids_2)
   
    if cache is None:
        cache = model.create_prompt_cache()
    # Use full input_ids for cache_key, not rest_input_ids
    cache_key_2 = input_ids_2[:]

    start_time = time.time()
    response_2 = model(rest_input_ids_2, cache, stream=True)
    raw_text = ""
    for chunk in response_2:
        if chunk:
            if first_token:
                print("TIME TO FIRST TOKEN", time.time() - start_time)
                first_token = False
            raw_text += chunk.text
            cache_key_2.append(chunk.token)

    print("RAW TEXT", raw_text)

    prompt_cache.insert_cache(cache_key_2, cache)