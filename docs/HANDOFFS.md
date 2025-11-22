# HANDOFFS – Session Log

This document tracks session-to-session handoffs for the `mlx-openai-server-lab` fusion engine project. Each session appends a new entry with discoveries, actions, and next steps.

---

## Session 1: Phase 0 – Engine Warm-Up
**Date**: 2025-11-16
**Branch**: `claude/phase0-engine-warmup-012M2QbGXpRukoMy8x4jyVrg`
**Goal**: Map the existing codebase and document architecture for Phase 1 transformation

### Discoveries

**Architecture Overview:**
- FastAPI-based OpenAI-compatible API server running on Apple Silicon (MLX)
- Stateless design: no database, no persistent storage, all config via CLI
- Single-model-per-instance: model loaded at startup, no runtime switching
- Async request queue with semaphore-based concurrency control
- Supports 6 model types: lm, multimodal, embeddings, whisper, image-generation, image-edit

**Key Components Identified:**
1. **Entrypoints**: CLI via Click (`app/cli.py:145`) or argparse (`app/main.py:50`), script entry in `pyproject.toml:56`
2. **Model Loading**: Model wrappers in `app/models/` (mlx_lm, mlx_vlm, etc.), handlers in `app/handler/` wrapping models with queue logic
3. **Configuration**: All via CLI args (no config files), defaults from env vars in `app/models/mlx_lm.py:15-21`
4. **HTTP Routing**: Single router in `app/api/endpoints.py` with 8 endpoints (health, models, queue stats, chat, embeddings, images, audio)
5. **Concurrency**: `app/core/queue.py` RequestQueue with `asyncio.Semaphore`, defaults: max_concurrency=1, timeout=300s, queue_size=100

**Current Limitations:**
- **No model registry**: Can't switch models without restarting server
- **No persistent queue**: Queue is in-memory, lost on restart
- **No state management**: Completely stateless (good for Tier 3, but needs Tier 2 integration)
- **Sequential by default**: max_concurrency=1 means only one request processed at a time
- **No multi-model support**: Can only load one model per server instance
- **No job tracking**: No request history, logs, or status persistence

**Code Quality Observations:**
- Well-structured, clear separation of concerns (models, handlers, API)
- Good error handling and logging (loguru)
- Memory-conscious: explicit garbage collection and MLX cache clearing
- OpenAI compatibility: Follows OpenAI API schemas closely
- Extensible: Easy to add new model types via handler pattern

### Actions Taken

1. ✅ Scanned repository structure (42 Python files, 8 handler types, 5 model wrappers)
2. ✅ Identified main entrypoints: `app/cli.py`, `app/main.py`, `pyproject.toml` script entry
3. ✅ Mapped model loading flow: `app/models/` → `app/handler/` → `app/main.py` lifespan
4. ✅ Documented config handling: CLI args only, no config files
5. ✅ Catalogued HTTP endpoints: 8 routes in `app/api/endpoints.py`
6. ✅ Analyzed concurrency system: RequestQueue with semaphore in `app/core/queue.py`
7. ✅ Created `docs/FUSION_PHASE0.md` with comprehensive architecture documentation:
   - Executive summary
   - High-level code map with exact file paths and line numbers
   - Dataflow diagram
   - Complete API surface table
   - Concurrency configuration details
   - Model lifecycle and memory management
   - Dependencies and tech stack
   - Gaps and future work (TODOs for Phase 1)
8. ✅ Created `docs/HANDOFFS.md` (this file) for session tracking

### Next Actions for Phase 1: Fusion Engine Transformation

**Goal**: Transform this stateless inference server into a **multi-model fusion engine** that integrates with Tier 2 (MCP) for state management and orchestration.

#### Priority 1: Model Registry & Management
1. **Implement model registry** (`app/core/model_registry.py`):
   - Registry class to track loaded models by ID
   - Support multiple models loaded simultaneously
   - Model metadata: type, capabilities, context length, VRAM usage
   - Model loading/unloading API (hot-swap without restart)

2. **Add model routing logic** (`app/api/endpoints.py`):
   - Route requests to appropriate model by `model` parameter
   - Validate model exists before processing request
   - Return 404 if model not found

3. **Create model management endpoints**:
   - `POST /v1/models/load` – Load a new model
   - `DELETE /v1/models/{model_id}/unload` – Unload a model
   - `GET /v1/models/{model_id}/info` – Get model metadata
   - `GET /v1/models/{model_id}/stats` – Get model usage stats

#### Priority 2: Persistent Queue & Job Tracking
4. **Integrate MongoDB for job tracking**:
   - Job schema: request ID, model ID, status, timestamps, input/output
   - Status enum: queued, processing, completed, failed, cancelled
   - Store request metadata (no full request body for privacy)

5. **Implement job status API**:
   - `GET /v1/jobs/{job_id}` – Get job status
   - `GET /v1/jobs` – List jobs (with filters: status, model, time range)
   - `DELETE /v1/jobs/{job_id}` – Cancel a job

6. **Persist queue to Redis** (optional):
   - Replace in-memory queue with Redis-backed queue
   - Survive restarts without losing pending requests
   - Enable distributed queue for multi-node setup

#### Priority 3: Tier 2 Integration
7. **Define Tier 2 ↔ Tier 3 protocol**:
   - MCP (Tier 2) sends job requests to Tier 3 via HTTP
   - Tier 3 reports job status back to MCP (webhooks or polling)
   - Shared MongoDB for state synchronization

8. **Add health metrics endpoint** (`/v1/health/metrics`):
   - VRAM usage (current, peak)
   - Model inference latency (p50, p95, p99)
   - Queue depth and throughput
   - Active models and concurrency

9. **Implement callback/webhook system**:
   - Tier 3 notifies Tier 2 when job completes
   - POST to configurable webhook URL with job result
   - Retry logic for failed notifications

#### Priority 4: Observability & Performance
10. **Add Prometheus metrics export** (`/metrics`):
    - Request rate, error rate, latency
    - Queue stats (depth, wait time)
    - Model stats (inference time, VRAM usage)
    - Memory stats (gc count, cache clears)

11. **Implement request ID tracking**:
    - Generate unique request ID for each request
    - Pass request ID to Tier 2 for correlation
    - Include request ID in all logs

12. **Optimize concurrency defaults**:
    - Benchmark different concurrency levels for common models
    - Document recommended concurrency by model size
    - Consider adaptive concurrency (auto-tune based on VRAM/latency)

#### Priority 5: Code Refactoring (Low Priority)
13. **Extract config to dataclass** (optional):
    - Centralize all config in `app/core/config.py`
    - Replace arg parsing with pydantic settings
    - Support config file loading (YAML/TOML)

14. **Add integration tests**:
    - Test multi-model loading and routing
    - Test job tracking and status updates
    - Test Tier 2 integration (mock MCP)

15. **Document Tier 2 ↔ Tier 3 contract**:
    - API spec for job submission
    - Job status schema
    - Webhook payload format

### Risks & Open Questions

**Risks:**
1. **VRAM constraints**: Loading multiple models simultaneously may exceed available VRAM
   - *Mitigation*: Implement model LRU eviction, lazy loading, or VRAM monitoring
2. **Concurrency complexity**: Managing multiple models with different concurrency limits
   - *Mitigation*: Per-model queue configuration, global semaphore for VRAM
3. **State synchronization**: Keeping Tier 2 and Tier 3 state consistent
   - *Mitigation*: Single source of truth (MongoDB), atomic operations, idempotency

**Open Questions:**
1. Should Tier 3 own the model registry, or should Tier 2 dictate which models to load?
   - *Recommendation*: Tier 2 owns configuration, Tier 3 manages lifecycle
2. How to handle model loading failures (e.g., out of VRAM)?
   - *Recommendation*: Return 503 Service Unavailable, log error, notify Tier 2
3. Should job history be stored indefinitely, or pruned after N days?
   - *Recommendation*: Configurable TTL (e.g., 7 days), archive to S3/disk if needed
4. Should Tier 3 support streaming to Tier 2, or only non-streaming?
   - *Recommendation*: Support both, use SSE for streaming, JSON for non-streaming
5. How to handle model version updates (e.g., model repo changes)?
   - *Recommendation*: Treat as new model ID, allow side-by-side deployment, manual cutover

### Files Changed
- ✅ Created `docs/FUSION_PHASE0.md` (architecture documentation)
- ✅ Created `docs/HANDOFFS.md` (session log, this file)

### Files to Change in Phase 1
- `app/core/model_registry.py` (NEW) – Model registry implementation
- `app/core/job_tracker.py` (NEW) – MongoDB job tracking
- `app/api/endpoints.py` – Add model management and job status endpoints
- `app/main.py` – Update lifespan to support multi-model loading
- `app/handler/*.py` – Update handlers to report job status
- `app/schemas/openai.py` – Add job status schemas
- `README.md` – Update with new multi-model capabilities
- `docs/TIER2_INTEGRATION.md` (NEW) – Document Tier 2 ↔ Tier 3 protocol

---

## Session 2: Phase 0 – JIT + Auto-Unload Wiring
**Date**: 2025-11-21
**Branch**: `Implement-JIT-and-Auto-Unload`
**Goal**: Finish the Phase 0 work required to expose LazyHandlerManager + idle auto-unload through the API surface, health endpoint, and docs.

### Discoveries
- Handler lifecycle plumbing (LazyHandlerManager + IdleAutoUnloadController) was already present in `app/server.py`, but API routes still referenced `app.state.handler` directly, preventing JIT loading.
- `/health` returned `503` whenever the handler was unloaded, which made JIT unusable for liveness probes.
- Model metadata cache in `app.state.model_metadata` only tracked a timestamp, so `/v1/models` had to hit the handler each time.

### Actions Taken
- Added helper utilities inside `app/api/endpoints.py` that request handlers through the LazyHandlerManager (`ensure_loaded`) and surface consistent JSON errors when loading fails.
- Updated `/health`, `/v1/models`, `/v1/queue/stats`, chat, embeddings, image, and audio routes to rely on the new helpers so the model loads on demand.
- Changed the health endpoint to return `status="ok"` with `model_status="unloaded"` whenever JIT is enabled and the handler is idle.
- Cached canonical model metadata at startup and refresh it whenever the handler loads/unloads so `/v1/models` can respond instantly even if the handler is unloaded.
- Registered `RequestTrackingMiddleware` by default to ensure request IDs are always populated.
- Added regression tests (`tests/test_health_endpoint.py`) covering the new health responses.
- Documented the `--jit` and `--auto-unload-minutes` flags plus the updated health semantics in the README.

### Next Actions
- Extend the cached metadata structure to support multiple models once the registry work begins.
- Consider lightweight status objects for `/v1/queue/stats` so we don’t need to load the handler for purely informational queries.
- Monitor startup latency when JIT auto-loads on the first request and add user-facing logs/metrics if necessary.

### Risks & Open Questions
- If a handler fails to load (e.g., VRAM exhaustion), repeated requests will retry instantiation; we should add backoff or clearer error surfacing if this becomes noisy.
- Streaming endpoints now call `ensure_loaded` per request; if future multi-model support appears, we’ll need to route by `model` parameter before touching the manager.

### Files Changed
- `app/api/endpoints.py`
- `app/server.py`
- `tests/test_health_endpoint.py`
- `README.md`
