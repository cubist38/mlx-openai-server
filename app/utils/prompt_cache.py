# copied from https://github.com/ml-explore/mlx-lm/blob/d3dc2e3f337679cb666382c94b583566d2f492b2/mlx_lm/server.py

from __future__ import annotations

import copy
from typing import Any
from collections import deque
from dataclasses import dataclass
from mlx_lm.models.cache import (
    can_trim_prompt_cache,
    trim_prompt_cache,
)


class LRUPromptCache:
    """LRU cache for prompt caches using a trie-like structure.
    
    This cache stores prompt token sequences in a trie (prefix tree) structure,
    allowing efficient lookup of exact matches, shorter prefixes, and longer
    sequences. Uses LRU eviction policy to maintain a bounded cache size.
    
    Attributes
    ----------
    max_size : int
        Maximum number of cache entries to store before eviction
    """

    @dataclass
    class CacheEntry:
        """Entry stored in the cache.
        
        Attributes
        ----------
        prompt_cache : list[Any]
            The cached prompt data structure
        count : int
            Reference count for this cache entry
        """
        prompt_cache: list[Any]
        count: int

    @dataclass
    class SearchResult:
        """Result of searching for a token sequence in the cache.
        
        Attributes
        ----------
        exact : list[int] | None
            Exact matching token sequence, if found
        shorter : list[int] | None
            Shorter prefix match, if found
        longer : list[int] | None
            Longer sequence containing the query as prefix, if found
        common_prefix : int
            Length of common prefix with cache entries
        """
        exact: list[int] | None
        shorter: list[int] | None
        longer: list[int] | None
        common_prefix: int

    def __init__(self, max_size: int = 10) -> None:
        """Initialize the LRU prompt cache.
        
        Parameters
        ----------
        max_size : int, optional
            Maximum number of cache entries, by default 10
        """
        self.max_size = max_size
        self._cache: dict[int, Any] = {}
        self._lru: deque[tuple[int, ...]] = deque()

    def _search(self, tokens_ids: list[int]) -> SearchResult:
        """Search the cache for a prompt cache.
        
        Traverses the trie to find exact matches, shorter prefixes, or longer
        sequences that contain the query as a prefix.
        
        Parameters
        ----------
        tokens_ids : list[int]
            Token sequence to search for
            
        Returns
        -------
        SearchResult
            Contains exact, shorter, or longer matches and common prefix length
        """
        current = self._cache
        last_cache_index = -1
        index = 0

        # Traverse the trie as far as possible
        while index < len(tokens_ids) and tokens_ids[index] in current:
            current = current[tokens_ids[index]]
            if "cache" in current:
                last_cache_index = index
            index += 1

        # Exact match - no need to search for longer or shorter caches
        if last_cache_index == len(tokens_ids) - 1:
            return self.SearchResult(tokens_ids, None, None, 0)

        # Find the shorter cache (fixed: should include index 0)
        shorter = None
        if last_cache_index >= 0:
            shorter = tokens_ids[: last_cache_index + 1]

        # Find the shorter cache
        shorter = None
        if last_cache_index > 0:
            shorter = tokens_ids[: last_cache_index + 1]

        # Check for caches that are longer
        longer = None
        common_prefix = index
        if index > 0 and last_cache_index <= 0:
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

            longer = tokens_ids[:index] + best

        return self.SearchResult(None, shorter, longer, common_prefix)


    def fetch_nearest_cache(
        self, tokens_ids: list[int]
    ) -> tuple[list[Any] | None, list[int]]:
        """Fetch the nearest matching cache for the given token sequence.
        
        Searches for exact, shorter, or longer cache entries and returns the
        best match along with remaining tokens that aren't cached.
        
        Parameters
        ----------
        tokens_ids : list[int]
            Token sequence to find a cache for
            
        Returns
        -------
        tuple[list[Any] | None, list[int]]
            Tuple of (prompt_cache, remaining_tokens). If no cache found,
            returns (None, original_tokens).
        """
        result = self._search(tokens_ids)
        
        # Exact match - return cache with no remaining tokens
        if result.exact is not None:
            cache_entry = self._extract(result.exact)
            return cache_entry.prompt_cache, []

        # Shorter prefix match - return cache with remaining suffix
        if result.shorter is not None:
            cache_entry = self._extract(result.shorter)
            prefix_len = len(result.shorter)
            return cache_entry.prompt_cache, tokens_ids[prefix_len:]

        # Longer sequence match - try to trim it down to our prefix
        if result.longer is not None:
            cache_entry = self._get(result.longer)
            if can_trim_prompt_cache(cache_entry.prompt_cache):
                # Deep copy is necessary to avoid modifying the cached version
                trimmed_entry = self.CacheEntry(
                    copy.deepcopy(cache_entry.prompt_cache),
                    1,
                )
                prefix = min(len(tokens_ids) - 1, result.common_prefix)
                print("PREFIX", prefix)
                num_to_trim = len(result.longer) - prefix
                trim_prompt_cache(trimmed_entry.prompt_cache, num_to_trim)
                return trimmed_entry.prompt_cache, tokens_ids[prefix:]

        # No match found
        return None, tokens_ids

    def _get(self, tokens_ids: list[int]) -> CacheEntry:
        """Retrieve a cache entry without removing it.
        
        Parameters
        ----------
        tokens_ids : list[int]
            Token sequence identifying the cache entry
            
        Returns
        -------
        CacheEntry
            The cache entry at this location
            
        Raises
        ------
        KeyError
            If the token sequence is not in the cache
        """
        current = self._cache
        for tok in tokens_ids:
            current = current[tok]
        return current["cache"]

    def _delete(self, tokens_ids: list[int]) -> None:
        """Delete a cache entry and clean up empty trie nodes.
        
        Removes the cache entry and then walks back up the trie, removing
        empty nodes to keep the tree structure clean.
        
        Parameters
        ----------
        tokens_ids : list[int]
            Token sequence identifying the cache entry to delete
        """
        # Build path to the cache entry
        path = [self._cache]
        for tok in tokens_ids:
            path.append(path[-1][tok])
        
        # Delete the cache entry
        del path[-1]["cache"]
        
        # Walk back up and remove empty nodes
        for i in reversed(range(len(tokens_ids))):
            d_prev, d, t = path[i], path[i + 1], tokens_ids[i]
            # Stop if node still has children or other data
            if len(d) > 0:
                break
            del d_prev[t]

    def _extract(self, tokens_ids: list[int]) -> CacheEntry:
        """Extract a cache entry, potentially removing it if count reaches zero.
        
        If the entry has a reference count > 1, decrements the count and returns
        a deep copy. If count == 1, removes the entry entirely and returns it.
        
        Parameters
        ----------
        tokens_ids : list[int]
            Token sequence identifying the cache entry
            
        Returns
        -------
        CacheEntry
            The extracted cache entry (either original or a copy)
        """
        cache_entry = self._get(tokens_ids)
        
        # If this is the last reference, remove from cache and LRU
        if cache_entry.count == 1:
            self._delete(tokens_ids)
            # FIXED: Convert list to tuple for deque.remove()
            self._lru.remove(tuple(tokens_ids))
            return cache_entry

        # Otherwise, decrement count and return a copy
        cache_entry.count -= 1
        return self.CacheEntry(
            copy.deepcopy(cache_entry.prompt_cache),
            1,
        )

    def insert_cache(
        self, tokens_ids: list[int], prompt_cache: list[Any]
    ) -> None:
        """Insert or update a cache entry.
        
        If the entry already exists, increments its reference count and moves
        it to the end of the LRU queue. If the cache is full, evicts the least
        recently used entry.
        
        Parameters
        ----------
        tokens_ids : list[int]
            Token sequence identifying this cache entry
        prompt_cache : list[Any]
            The prompt cache data to store
        """
        # Convert to tuple for LRU tracking (lists aren't hashable)
        tokens_tuple = tuple(tokens_ids)
        
        # Navigate/create trie path
        current = self._cache
        for tok in tokens_ids:
            if tok not in current:
                current[tok] = {}
            current = current[tok]

        # Update existing or create new entry
        if "cache" in current:
            current["cache"].count += 1
            # FIXED: Use tuple for deque.remove()
            self._lru.remove(tokens_tuple)
        else:
            current["cache"] = self.CacheEntry(prompt_cache, 1)

        # Move to end of LRU (most recently used)
        self._lru.append(tokens_tuple)
        
        # Evict oldest if over capacity
        if len(self._lru) > self.max_size:
            oldest_tokens = self._lru.popleft()
            # Convert back to list for _delete
            self._delete(list(oldest_tokens))

if __name__ == "__main__":
    from app.models.mlx_lm import MLX_LM
    model = MLX_LM("mlx-community/MiniMax-M2.1-4bit")
    prompt_cache = LRUPromptCache()
    prompt_1 = "Hello, how are you? I'm fine, thank you."
    input_prompt = model.create_input_prompt([{"role": "user", "content": prompt_1}], {})
    input_ids = model.encode_prompt(input_prompt)
    # input_ids = [10, 25708, 1010, 11, 1010, 10, 3263, 1010, 58324, 4433, 1294, 1032, 1049, 1048, 6619, 11, 1010, 10, 1503, 19464, 1010, 12, 1010]
    cache, rest_input_ids = prompt_cache.fetch_nearest_cache(input_ids)
    if cache is None:
        cache = model.create_prompt_cache()
    cache_key = rest_input_ids[:]

    print("REST INPUT IDS", rest_input_ids[:100])

    import time
    start_time = time.time()

    response_1 = model(rest_input_ids, cache, stream=True)
    for chunk in response_1:
        if chunk:
            cache_key.append(chunk.token)

    print(f"Time taken: {time.time() - start_time} seconds")

    prompt_cache.insert_cache(cache_key, cache)

    prompt_2 = "Hello, how are you? I'm fine, thank you."
    input_prompt_2 = model.create_input_prompt([{"role": "user", "content": prompt_2}], {})
    input_ids_2 = model.encode_prompt(input_prompt_2)
    # input_ids_2 = [10, 25708, 1010, 11, 1010, 10, 3263, 1010, 58324, 4433, 1294, 1032, 1049, 1048, 6619, 1044, 1294, 12859, 4453, 11, 1010, 10, 1503, 19464, 1010, 12, 1010]
    cache, rest_input_ids_2 = prompt_cache.fetch_nearest_cache(input_ids_2)
   
    print("REST INPUT IDS 2", rest_input_ids_2[:100])
    if cache is None:
        cache = model.create_prompt_cache()
    cache_key_2 = rest_input_ids_2[:]

    start_time = time.time()
    response_2 = model(rest_input_ids_2, cache, stream=True)
    for chunk in response_2:
        if chunk:
            cache_key_2.append(chunk.token)

    print(f"Time taken: {time.time() - start_time} seconds")

    prompt_cache.insert_cache(cache_key_2, cache)