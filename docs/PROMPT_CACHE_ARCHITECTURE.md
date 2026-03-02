# LRU Prompt Cache Architecture

**Document Version:** 1.0
**Last Updated:** 2025-01-20
**Related Components:** `app/utils/prompt_cache.py`, `app/handler/mlx_lm.py`

---

## 1. Overview

### 1.1 Purpose

The **LRU Prompt Cache** is a performance optimization system for the MLX OpenAI Server that reduces computational overhead in transformer-based language model inference by reusing previously computed **KV (key-value) caches** from the model's attention mechanism.

When generating text, transformer models compute attention over all previous tokens, storing intermediate **key** and **value** vectors in a KV cache. For prompts that share common prefixes (e.g., repeated system prompts, similar user queries), the server can reuse these cached attention states instead of recomputing them, resulting in significant speed improvements—especially for long prompts.

### 1.2 Scope

The cache is **only enabled for language models** (`model_type = "lm"`). It is controlled by the `--prompt-cache-size` CLI parameter (default: 10 entries). Each model handler instance maintains its own independent cache; in multi-model mode, each model gets a separate cache.

---

## 2. Core Data Structures

The cache is implemented in `app/utils/prompt_cache.py` as the `LRUPromptCache` class. It combines three key structures:

### 2.1 Trie (Prefix Tree)

The trie stores token sequences as nested dictionaries:

```
_cache: dict[int, Any]
```

- Each node represents a token ID
- Child nodes are stored as `node[token_id] = child_dict`
- A node may contain a `"cache"` key holding a `CacheEntry`
- This structure enables **efficient prefix matching**: given a sequence of token IDs, we can traverse the trie to find the longest prefix that has a cached KV state

**Example:**
```
_cache = {
    123: {  # token "Hello"
        456: {  # token ","
            789: {  # token "how"
                "cache": CacheEntry(...)  # cached KV for "Hello, how"
            },
            101: {  # token "hi"
                "cache": CacheEntry(...)  # cached KV for "Hello, hi"
            }
        }
    }
}
```

### 2.2 LRU Queue

```
_lru: deque[tuple[int, ...]]
```

A double-ended queue tracking cache keys in **least-recently-used order**:
- New entries are appended to the right (most recently used)
- When capacity is exceeded, entries are popped from the left (least recently used)
- Each element is a **tuple** of token IDs (hashable) corresponding to a cache key

### 2.3 CacheEntry

```python
@dataclass
class CacheEntry:
    prompt_cache: list[Any]  # The actual KV cache data from the model
    count: int               # Reference count (supports sharing)
```

- `prompt_cache`: The opaque KV cache structure returned by `model.create_prompt_cache()` and consumed by the model during generation
- `count`: Tracks how many active requests are using this cache. Enables **safe sharing** and **copy-on-write** semantics

---

## 3. Core Algorithms

### 3.1 Search (`_search`)

**Input:** `tokens_ids: list[int]` - the full prompt token sequence
**Output:** `SearchResult` containing:
- `exact: list[int] | None` - exact match in the cache
- `shorter: list[int] | None` - longest prefix that has a cache (length > 0)
- `longer: list[int] | None` - shortest cached sequence that extends the query (contains the query as prefix)
- `common_prefix: int` - length of shared prefix with any cache entry

**Algorithm:**
1. Traverse the trie as far as possible following `tokens_ids`
2. Track the last node that contains a `"cache"` entry (`last_cache_index`)
3. If we consumed all tokens (`last_cache_index == len(tokens_ids) - 1`), we have an **exact match**
4. Otherwise:
   - **Shorter match**: If `last_cache_index > 0`, the prefix up to that index is cached
   - **Longer match**: If we stopped before finding any cache (`last_cache_index <= 0`), perform a DFS from the current node to find the shortest descendant cache entry

**Why "shorter" requires `last_cache_index > 0`?**
A cache at depth 1 (single token) provides minimal benefit and may represent a degenerate case. The policy favors healthier cache hits (prefixes of length ≥ 2).

### 3.2 Fetch Nearest Cache (`fetch_nearest_cache`)

**Input:** `tokens_ids: list[int]` - prompt token sequence
**Output:** `(prompt_cache, remaining_tokens)`

**Strategy (in order of preference):**

1. **Exact match** → return cached KV, `remaining_tokens = []`
2. **Shorter prefix match** → return cached KV for prefix, `remaining_tokens = tokens_ids[prefix_len:]`
3. **Longer sequence match** → attempt to **trim** the cached KV to the shared prefix, then return it with the remaining suffix tokens
4. **No match** → return `(None, tokens_ids)`

**Trimming logic:**
- Check `can_trim_prompt_cache(cache_entry)` to see if the cache supports trimming
- If yes, **extract** the cache (which decrements refcount or removes it)
- Compute `prefix = min(len(tokens_ids) - 1, result.common_prefix)`
- Trim `num_to_trim = len(longer_sequence) - prefix`
- Call `trim_prompt_cache(cache_entry.prompt_cache, num_to_trim)`

**Why extract before trimming?**
When reusing a cache entry that will be modified (trimmed), we must either:
- Remove it if it's the last reference (to avoid corrupting the original)
- Or copy it if shared, then decrement the original's reference count

This prevents memory leaks when the same cache is reused across multiple requests (e.g., a "swipe" where the same prompt is regenerated).

### 3.3 Insert Cache (`insert_cache`)

**Input:** `tokens_ids: list[int]` (full sequence: prompt + generated tokens), `prompt_cache: list[Any]`

**Algorithm:**
1. Convert `tokens_ids` to a tuple `tokens_tuple` for LRU tracking (lists aren't hashable)
2. **Navigate/create** the trie path for all tokens:
   ```python
   for tok in tokens_ids:
       if tok not in current:
           current[tok] = {}
       current = current[tok]
   ```
3. **Update or create** the cache entry at the leaf:
   - If `"cache"` exists: `count += 1`, remove `tokens_tuple` from `_lru` (will re-add at end)
   - Else: create new `CacheEntry(prompt_cache, 1)`
4. Append `tokens_tuple` to `_lru` (mark as most recently used)
5. **Evict if over capacity**:
   - `oldest_tokens = _lru.popleft()`
   - Call `_delete(list(oldest_tokens))` to remove from trie

### 3.4 Eviction (`_delete`)

Removes a cache entry and **prunes empty trie nodes**:

1. Build a path from root to the leaf node
2. Delete the `"cache"` key from the leaf
3. Walk backwards up the path:
   - If a node has no children, delete it from its parent
   - Stop when encountering a node that still has children or other data

This keeps the trie compact and prevents memory leaks from orphaned nodes.

### 3.5 Extract (`_extract`)

**Purpose:** Safely retrieve a cache entry for use, handling reference counting.

**Input:** `tokens_ids: list[int]`
**Output:** `CacheEntry` (either original or a deep copy)

**Logic:**
- Retrieve entry via `_get`
- If `count == 1`:
  - Remove from trie via `_delete`
  - Remove from LRU via `_lru.remove(tuple(tokens_ids))`
  - Return the original (now detached)
- If `count > 1`:
  - Decrement `count`
  - Return `CacheEntry(copy.deepcopy(prompt_cache), 1)` — a **fresh copy** with count 1

The copy ensures that if the cache is modified during generation (and then re-inserted), other requests still referencing the original aren't affected.

---

## 4. Control & Information Flow

### 4.1 Request Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                     HTTP Request arrives                    │
└───────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│  MLXLMHandler.generate_text_stream/response               │
│  1. Encode prompt → input_ids (list[int])                 │
│  2. cache, rest = prompt_cache.fetch_nearest_cache(input_ids) │
│  3. Submit to inference_worker:                           │
│     - model.generate(rest_input_ids, cache)               │
│  4. Stream/await response, collect generated tokens      │
│  5. cache_key = input_ids + generated_tokens              │
│  6. prompt_cache.insert_cache(cache_key, cache)          │
│  7. Return response to client                            │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Cache Key Semantics

The **cache key** is the **complete token sequence that produced the KV cache**, i.e.:

```
cache_key = prompt_token_ids + generated_token_ids
```

This is crucial:
- When a cache is **reused** (via fetch), we start with a partially-filled KV state
- After generation completes, the KV cache now corresponds to the **full** sequence (prompt + all outputs)
- Inserting with the full key ensures that future requests that exactly match this full sequence can reuse the complete KV state

**Pitfall avoided:** Using `rest_input_ids` as the cache key would cause memory leaks. For example:
- Cache A: `[A, B]` exists
- New request: prompt `[A, B, C]` → finds shorter match `[A, B]`, uses it, generates `X`
- If we inserted with key `[C, X]`, the original `[A, B]` entry remains in the trie (unused), and a new entry `[C, X]` is added. Over time, many partial entries accumulate.
- Correct: Insert with key `[A, B, C, X]` updates the existing branch in the trie.

### 4.3 Data Flow Diagram

```
┌─────────────────┐
│   Request 1     │
│  Prompt: "A B" │
└────────┬────────┘
         │
         ▼
  Encode → [tok_A, tok_B]
         │
         ▼
  fetch_nearest_cache([tok_A, tok_B])
         │
         ├─ Cache miss → (None, [tok_A, tok_B])
         │    Create fresh KV cache
         │
         ▼
  Generate tokens: [tok_C, tok_D]
         │
         ▼
  KV cache now covers: [tok_A, tok_B, tok_C, tok_D]
         │
         ▼
  insert_cache([tok_A, tok_B, tok_C, tok_D], kv_cache)
         │
         ▼
  Trie: [tok_A]→[tok_B]→[tok_C]→[tok_D]→(CacheEntry)

─────────────────────────────────────────────────────────

┌─────────────────┐
│   Request 2     │
│  Prompt: "A B C"│
└────────┬────────┘
         │
         ▼
  Encode → [tok_A, tok_B, tok_C]
         │
         ▼
  fetch_nearest_cache([tok_A, tok_B, tok_C])
         │
         ├─ Exact match? No (key is length 4)
         ├─ Shorter prefix? Yes: [tok_A, tok_B] (length 2)
         │    Return cache from that node
         │    rest_input_ids = [tok_C]
         │
         ▼
  Generate: [tok_E]
         │
         ▼
  KV cache now covers: [tok_A, tok_B, tok_C, tok_E]
         │
         ▼
  insert_cache([tok_A, tok_B, tok_C, tok_E], kv_cache)
         │
         ├─ Traverse/create: [tok_A]→[tok_B]→[tok_C]→[tok_E]
         ├─ Node for [tok_A]→[tok_B] still exists but its
         │   cache was extracted during fetch (count went to 0)
         └─ New leaf cache entry replaces it
```

---

## 5. Design Rationale

### 5.1 Why a Trie?

- **Prefix matching**: The fundamental operation is "find the longest cached prefix of this prompt". A trie provides O(L) lookup where L is prompt length, independent of cache size.
- **Shared prefixes**: Multiple prompts that share beginnings (e.g., "Write a poem about X", "Write a story about Y") naturally share trie nodes, saving memory.
- **Flexible reuse**: Supports three match types (exact, shorter, longer) with a single structure.

### 5.2 Why LRU Eviction?

- **Bounded memory**: On Apple Silicon, memory is limited. LRU ensures the cache doesn't grow indefinitely.
- **Temporal locality**: In practice, recent prompts are more likely to be reused (e.g., iterative refinement, conversational turn-taking). LRU captures this pattern.
- **Simplicity**: LRU with a deque is straightforward to implement and has O(1) operations per insert/evict.

### 5.3 Reference Counting

Allows **safe sharing** of cache entries across concurrent requests:
- Multiple requests can fetch the same cache (e.g., identical prompts)
- The first request to complete will **extract** (remove) the cache if its count is 1
- Subsequent requests still hold references (count > 1) and will get a **deep copy** when they extract
- This prevents a race where two requests try to update the same cache entry simultaneously

**Example:**
```
Cache entry K with count = 2 (two requests using it)
Request A finishes → extract() sees count=2 → decrement to 1, return copy
Request B finishes → extract() sees count=1 → delete original, return original
insert_cache(K_modified) → creates new entry
```

### 5.4 Trimming Support

When a cached sequence **extends** the current prompt (longer match), we can "trim" the cache to the shared prefix instead of discarding it entirely. This requires cooperation from the underlying MLX model's cache structure (`mlx_lm.models.cache`).

**Benefit:** If we have cached KV for tokens `[A, B, C, D]` and the new prompt is `[A, B, X]`, we can:
- Reuse the KV for `[A, B]` from the longer cache
- Trim away the extra `[C, D]` part
- Avoid creating a completely fresh cache

This captures cases where a previous generation was longer than the current prompt, but they share a prefix.

---

## 6. Usage in MLXLMHandler

### 6.1 Initialization

```python
self.prompt_cache = LRUPromptCache(max_size=prompt_cache_size)
```

The cache size is set via `--prompt-cache-size` (default 10). Each handler instance has its own cache.

### 6.2 Request Handling

Both `generate_text_stream` and `generate_text_response` follow the same pattern:

```python
# 1. Encode prompt
input_ids = self.model.encode_prompt(input_prompt)  # List[int]

# 2. Fetch cache
cache, rest_input_ids = self.prompt_cache.fetch_nearest_cache(input_ids)
cache_key = input_ids[:]  # Full prompt, for later insertion

if cache is None:
    cache = self.model.create_prompt_cache()

# 3. Generate (streaming or blocking)
response = self.inference_worker.submit(...,
    input_ids=rest_input_ids,
    prompt_cache=cache,
    ...
)

# 4. Update cache key with generated tokens
cache_key += response.tokens  # or accumulate from chunks

# 5. Insert updated cache
self.prompt_cache.insert_cache(cache_key, cache)
```

**Key points:**
- The model is **never called with the full `input_ids`**; only the uncached remainder (`rest_input_ids`)
- The cache object is mutated during generation (KV cache grows as new tokens are produced)
- After generation, the cache represents the KV state for the **full** sequence `cache_key`
- Inserting with the full key updates the trie path appropriately (replacing evicted entries along that path if they existed)

### 6.3 Debug Logging

When `--debug` is enabled, the handler logs:
- Cache hit statistics (cached tokens vs. remaining tokens)
- Raw responses and generation metrics
- Parser events (for reasoning/tool parsing, not directly cache-related)

---

## 7. Performance Characteristics

### 7.1 Time Complexity

- **`_search`**: O(L) where L = length of `tokens_ids`. Single pass down the trie, plus optional DFS for longer match (worst-case O(total_cached_nodes) but cache size is bounded).
- **`fetch_nearest_cache`**: O(L) dominated by `_search`
- **`insert_cache`**: O(L) to traverse/create path + O(1) LRU operations + potential O(L) eviction (delete path)
- **`_delete`**: O(L) to walk up and prune

All operations are effectively constant-time for typical prompt lengths (hundreds to thousands of tokens) because the cache size (max 10 by default) bounds the trie depth and breadth.

### 7.2 Space Complexity

- **Trie**: Each node is a dict mapping token IDs to child dicts. In worst case (no shared prefixes), each cache entry adds L nodes where L is sequence length. With sharing, space is reduced.
- **LRU deque**: O(N) where N = number of cache entries (≤ max_size)
- **CacheEntry**: The `prompt_cache` is the largest component, holding model-specific KV tensors. Its size is roughly O(context_length × num_layers × hidden_dim × 2). With `prompt_cache_size = 10`, at most 10 such structures coexist.

**Memory bound:** The `max_size` parameter directly limits the number of KV cache objects. However, the trie itself (token IDs) is small compared to the KV tensors.

### 7.3 Trade-offs

| Aspect | Design Choice | Reasoning |
|--------|--------------|-----------|
| Match policy | Prefer exact > shorter > longer | Maximizes reuse; longer requires trim support |
| Shorter threshold | `last_cache_index > 0` (depth ≥ 2) | Avoids degenerate 1-token hits |
| Eviction | Strict LRU | Simple, works well for temporal locality |
| Reference counting | Manual (increment on insert, decrement on extract) | Enables safe sharing across concurrent requests |
| Cache key | Full sequence (prompt + generation) | Ensures complete coverage for future exact matches |
| Insert behavior | Overwrite existing path | If same prefix re-appears, new generation updates it |

---

## 8. Edge Cases & Gotchas

### 8.1 Memory Leak Prevention

The original code had a bug where `rest_input_ids` was used as the cache key, causing accumulation of orphaned shorter cache entries. This was fixed by using the **full `input_ids`** as the cache key.

**Correct:**
```python
cache_key = input_ids[:]  # Full prompt
# after generation:
cache_key += generated_tokens
insert_cache(cache_key, cache)
```

**Incorrect (leaky):**
```python
cache_key = rest_input_ids  # Only uncached suffix
```

### 8.2 Trimming Requirement

The "longer match" path only succeeds if `can_trim_prompt_cache(cache_entry)` returns True. Not all model cache implementations support trimming. If trimming is unsupported, the longer match is ignored and the system falls back to no cache.

### 8.3 Concurrency Model

- Each `MLXLMHandler` instance has **one** `LRUPromptCache`
- Requests are processed **sequentially** per model instance because they go through an `InferenceWorker` with a bounded queue and limited concurrency (`max_concurrency`)
- The cache itself has **no internal locking**; it's safe because only one request accesses it at a time (the inference worker processes requests one-by-one for a given handler)
- If multiple handler instances are created (e.g., multi-model mode with the same model ID), each has its own cache—no sharing across processes

### 8.4 Cache Invalidation

There is **no explicit invalidation**. Entries are evicted automatically by LRU when capacity is exceeded. There is no time-based expiration or manual purge.

### 8.5 Empty Cache After Extract

When `extract()` removes the last reference to a cache entry, it also deletes the corresponding leaf node from the trie and prunes empty ancestors. This ensures the trie doesn't retain dead branches.

---

## 9. Configuration

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| `prompt_cache_size` | `--prompt-cache-size` | `10` | Maximum number of KV cache entries to retain. Set to 0 to disable caching. |
| `max_concurrency` | `--max-concurrency` | `1` | Number of concurrent inference tasks. Does not affect cache size but determines how many requests might be in flight. |

**Guidance:**
- **Smaller cache (1-5)**: For memory-constrained environments or models with huge KV caches (large context)
- **Larger cache (20-50)**: For workloads with many varied prompts that benefit from prefix reuse (e.g., serving an API with diverse queries)
- The optimal size depends on prompt diversity and available memory. KV caches can consume significant RAM (especially for 32K+ context lengths).

---

## 10. Future Enhancements

Potential improvements identified in the codebase:

1. **Adaptive cache sizing**: Dynamically adjust based on memory pressure or hit rate
2. **Cache hit metrics**: Expose hit/miss statistics via `/v1/models` or a dedicated endpoint
3. **Shared cache across models**: If multiple models share the same tokenizer, could share cache entries (requires normalization of token IDs)
4. **Distributed cache**: For multi-process deployments (e.g., multiple server instances), a shared cache could reduce duplication
5. **Configurable eviction policy**: alternatives to LRU (e.g., LFU, ARC) might work better for some workloads
6. **Cache prewarming**: Allow loading common prompt prefixes into the cache at startup

---

## 11. References

- **Implementation:** `app/utils/prompt_cache.py`
- **Usage:** `app/handler/mlx_lm.py` (methods `generate_text_stream`, `generate_text_response`)
- **Configuration:** `app/config.py` (`MLXServerConfig.prompt_cache_size`)
- **Origin:** Adapted from `mlx-lm` (https://github.com/ml-explore/mlx-lm/blob/.../mlx_lm/server.py)

---

## 12. Glossary

- **KV cache**: The key-value vectors stored during transformer attention; allows incremental generation without recomputing attention for previous tokens.
- **Trie**: A tree data structure where each path from root to leaf represents a sequence (here, token IDs).
- **LRU**: Least Recently Used—eviction policy that discards the least recently accessed item.
- **Cache hit**: When a requested prompt (or prefix) is found in the cache.
- **Cache miss**: No matching prefix found; a new KV cache must be created from scratch.
- **Shorter match**: A cached prefix that is shorter than the current prompt.
- **Longer match**: A cached sequence that extends the current prompt (the cache is longer than needed).
- **Trimming**: Reducing a KV cache to a shorter prefix by discarding unnecessary entries.
- **Reference counting**: Tracking how many consumers hold a reference to a cache entry, enabling safe sharing and copy-on-write.
