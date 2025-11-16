# Fusion Phase 0 – Engine Warm-Up (mlx-openai-server-lab)

## 1. Repo Role in Fusion

- **Tier:** 3 – Compute Fabric / Engine
- **Purpose:** Serve MLX models behind an OpenAI-compatible HTTP API for:
  - Chat completions
  - Embeddings
  - Vision / image / audio
  - (Later) RAG helpers: `/rag_query`, `/rag_upsert`, `/rag_stats`, …

This repo does **no** long-term state:
- No MongoDB
- No sessions
- No Kanban / tasks
- Pure, stateless compute.

---

## 2. High-Level Code Map  (to be filled by Claude)

- Entrypoint(s):
  - `...`
- Core server / routing files:
  - `...`
- Model loading / registry:
  - `...`
- Config & env handling:
  - `...`
- Tests / examples:
  - `...`

---

## 3. Models & Serving Strategy  (Tier 0 / 1 / 2)

- **Tier 0 – Embeddings**
  - Default model: `vegaluisjose/mlx-rag` (BERT encoder)
  - Lifetime: global singleton, kept warm.

- **Tier 1 – Chat**
  - Default model: `mlx-community/Phi-3-mini-4k-instruct-unsloth-4bit`
  - Lifetime: global singleton, kept warm.

- **Tier 2 – Heavy Models**
  - Examples: GPT-OSS-20B, Flux image, Whisper large, …
  - Mechanism: LRU cache with `maxsize = 2`; on-demand load / eviction.

*(Claude: confirm where this is implemented and link to files.)*

---

## 4. API Surface Relevant to Fusion

Existing / planned endpoints:

- `/v1/chat/completions` → maps to `llm_chat`
- `/v1/embeddings` → maps to `embeddings_create`
- `/health` → readiness / liveness
- (Future) `/rag_query`, `/rag_upsert`, `/rag_stats` wrappers

For each endpoint, document:

| Endpoint | Method | Handler file | Notes |
|---------|--------|--------------|-------|
|         |        |              |       |

---

## 5. Concurrency & Stability Defaults

- `max_concurrency = 1`
- `queue_size = 100`
- `queue_timeout = 300s`

*(Claude: locate where these are configured and document paths.)*

---

## 6. Phase 0 TODOs for this repo

- [ ] Complete section 2 (code map) with real file paths.
- [ ] Complete section 4 (API surface) as a table.
- [ ] Link this document from root `README.md` under “Fusion Architecture”.
- [ ] Add any obvious refactors / dead code notes as checkboxes here.
