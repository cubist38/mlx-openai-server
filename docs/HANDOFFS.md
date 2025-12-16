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
-- Handler lifecycle plumbing (LazyHandlerManager + CentralIdleAutoUnloadController) was already present in `app/server.py`, but API routes still referenced `app.state.handler` directly, preventing JIT loading.
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

---

## Session 3: Phase 1 – Hub Bootstrap
**Date**: 2025-11-22  
**Branch**: `Create-hub`  
**Goal**: Stand up the hub configuration surface (YAML + CLI scaffolding) as the first step toward multi-model orchestration.

### Discoveries
- Hub requirements introduce several net-new concerns: slugged model names, per-model logging, group-level throttling, and YAML-based defaults.
- Existing `MLXServerConfig` needed additional metadata (`name`, `group`, `is_default_model`) so hub models can reuse the same dataclass without a parallel schema.
- There was no YAML dependency in the project, so PyYAML was added to `pyproject.toml` for safe config parsing.

### Actions Taken
1. ✅ Added `PyYAML` dependency and created the `app/hub/` package scaffold.
2. ✅ Implemented `app/hub/config.py` with:
   - `MLXHubConfig` / `MLXHubGroupConfig` dataclasses
   - Slug validation helpers and auto-created log directories
   - Default log path resolution plus automatic per-model log-file selection when `log_file`/`no_log_file` are omitted.
3. ✅ Extended `MLXServerConfig` to accept `name`, `group`, and `is_default_model`, keeping backwards compatibility with the existing CLI.
4. ✅ Introduced an experimental `mlx-openai-server hub` CLI group with `start` (placeholder) and `status` subcommands that parse hub.yaml and print per-model summaries.
5. ✅ Updated the README with a new “Hub Mode (experimental)” section describing the snake_case YAML schema, `log_path`, and example configuration; noted that orchestration will follow.
6. ✅ Logged the new work in this handoff file for continuity.
7. ✅ Wired the FastAPI lifespan into the `ModelRegistry`, added richer metadata/status propagation, and created regression tests for the registry/hub runtime helpers.

### Next Actions
- Wire the hub runtime into `app/server.py` so requests can be routed by `model` while respecting group `max_loaded` caps.
- Implement background logging redirection (per-model log sinks) plus `/hub` HTTP endpoints and flash-message HTML.
- Expand the CLI to send start/stop/load/unload operations to the hub controller once it exists.
- Add regression tests for YAML parsing, CLI error cases, and the upcoming hub control plane.

### Risks & Open Questions
- Multi-model routing requires changes to `_get_handler_or_error` and the FastAPI lifespan to select handlers by name without regressing single-model behavior.
- Per-model logging needs context-aware log sinks so concurrent requests do not interleave entries across files.
- CLI semantics for `start` vs `load` vs `unload` still need to be finalized once the runtime semantics are implemented.

### Files Changed
- `pyproject.toml`
- `app/config.py`
- `app/cli.py`
- `app/hub/__init__.py`
- `app/hub/config.py` (new)
- `README.md`

---

## Session 4: Phase 1 – Hub Status Surface
**Date**: 2025-11-22  
**Branch**: `Create-hub`  
**Goal**: Expose a server-side `/hub/status` snapshot that surfaces registry data for the CLI and upcoming HTML dashboard.

### Discoveries
- The new `ModelRegistry` introduced in Session 3 is already wired into the FastAPI lifespan, which makes it trivial to expose a read-only view without altering handler logic.
- Legacy single-model launches (without the registry) still populate `app.state.model_metadata`, so the HTTP endpoint can degrade gracefully by reusing that cache.

### Actions Taken
1. ✅ Added `HubStatusCounts` and `HubStatusResponse` schemas to `app/schemas/openai.py` to describe the JSON envelope shared by the CLI and dashboard.
2. ✅ Implemented `GET /hub/status` in `app/api/endpoints.py` which reads from the registry, computes loaded/registered counts, and deduplicates warnings when falling back to cached metadata.
3. ✅ Expanded `tests/test_health_endpoint.py` with hub-status regression coverage to ensure the endpoint prefers registry data and degrades cleanly without it.
4. ✅ Documented the new API in the README (including a `curl` example) to guide early adopters.

### Next Actions
- Use the `/hub/status` payload to back the future `/hub` HTML dashboard and CLI streaming updates.
- Start wiring hub orchestration (group-aware model loading/unloading) so the registry reflects multiple `LazyHandlerManager` instances instead of a single model.
- Extend the CLI to poll `/hub/status` once the background controller exists, replacing the current `hub start` placeholder output.

### Files Changed
- `app/api/endpoints.py`
- `app/schemas/openai.py`
- `tests/test_health_endpoint.py`
- `README.md`

---

## Session 5: Phase 1 – Hub Reservation Logic
**Date**: 2025-11-22  
**Branch**: `Create-hub`  
**Goal**: Add real state transitions and group-slot accounting to `HubRuntime`, then expose the behavior through the existing `hub start` command as the first tangible orchestration step.

### Discoveries
- Group limits are easiest to enforce at the runtime planner level; tracking how many models are in `loading`/`loaded` states per group gives us deterministic capacity enforcement before handlers exist.
- Auto-starting default models during `hub start` provides users with actionable insight (which worker processes will be hot vs on-demand) even before the controller loads handlers.

### Actions Taken
1. ✅ Extended `HubRuntime` with `can_load`, `mark_loading`, `mark_loaded`, `mark_failed`, and `mark_unloaded`, plus timestamp tracking and per-group usage counters.
2. ✅ Taught the hub manager to auto-start default models (subject to group limits) so the service spins up only the workers operators flag as defaults.
3. ✅ Created new regression tests in `tests/test_hub_runtime.py` that cover slot enforcement and transition sequencing to guard future controller work.

### Next Actions
- Implement a long-lived hub controller that drives these transitions by actually instantiating/tearing down handlers.
- Surface the richer runtime state via `/hub/status` once multiple models are registered.
- Connect the reservation logic to the ModelRegistry so reserved models appear immediately in the status API.

### Files Changed
- `app/hub/runtime.py`
- `app/cli.py`
- `tests/test_hub_runtime.py`

---

## Session 6: Phase 1 – Hub CLI & Docs Refresh
**Date**: 2025-11-22  
**Branch**: `Create-hub`  
**Goal**: Document the finalized CLI flows and make the new per-request model requirement explicit for hub deployments.

### Discoveries
- Users were still following the earlier guidance to launch hub mode via `launch --hub-config`, so the README needed to highlight that the `hub` subcommand owns all hub workflows now.
- Early adopters ran into 400 responses after enabling hub mode because existing API examples implied the `model` field was optional; the docs now need to clarify that a model name from `hub.yaml` is mandatory.

### Actions Taken
1. ✅ Expanded the README hub section with a bulletized CLI quick-reference covering `hub status` filtering, `--config` overrides, and `hub start` launch flows.
2. ✅ Added a new "Selecting models when using hub mode" subsection that explains the required `model` field, points to `hub status`/`/hub/status` for discovery, and differentiates single-model launches.
3. ✅ Updated the OpenAI client example to remind readers to switch the `model` argument when targeting a hub deployment and recorded the work in this handoff log.

### Next Actions
- Add dedicated CLI usage docs (or `--help` excerpts) once the controller exposes additional verbs such as `hub start-model`/`hub stop-model` and the complementary `hub load-model`/`hub unload-model` pair.
- Continue refreshing the README as hub orchestration becomes the default experience so the distinction between single-model and hub launches remains obvious.

### Files Changed
- `README.md`
- `docs/HANDOFFS.md`

---

## Session 7: Phase 1 – Hub CLI Actions & API Hooks
**Date**: 2025-11-22  
**Branch**: `Create-hub`  
**Goal**: Let operators load/unload models and watch live hub status from the CLI by plumbing new HTTP endpoints through FastAPI.

### Discoveries
- The CLI already had helper plumbing for `/hub/status`, so exposing new commands only required lightweight HTTP helpers plus controller endpoints.
- We needed dedicated API routes for start/stop actions because the controller previously only ran in-process; adding canonical `/hub/models/{model}/start` and `/hub/models/{model}/stop` endpoints (which the CLI commands `hub start-model` and `hub stop-model` call) keeps the flow uniform for future automation.

### Actions Taken
1. ✅ Added `HubModelActionRequest/Response` schemas and new FastAPI routes to proxy controller `load_model`/`unload_model` actions with proper error handling.
2. ✅ Extended the CLI with `hub start-model`, `hub stop-model`, and `hub watch` commands, wiring them to the controller endpoints alongside live status rendering and shared HTTP helpers.
    - Note: The CLI now uses the hub daemon HTTP API (no IPC shim). Start the daemon for local testing with:
       - `uvicorn app.hub.daemon:create_app --host 127.0.0.1 --port 8001`
3. ✅ Created integration and CLI regression tests to cover the new endpoints and commands, ensuring HTTP calls are formed correctly and controller failures propagate.
4. ✅ Updated the README to document the new commands so users know how to manage running hubs interactively.

### Next Actions
- Add richer error messaging/structured output for `hub watch` (e.g., columnized tables) once more metadata is available.
- Consider authenticated variants of the new endpoints before exposing the hub server on shared networks.

### Files Changed
- `app/api/endpoints.py`
- `app/cli.py`
- `app/schemas/openai.py`
- `README.md`
- `docs/HANDOFFS.md`
- `tests/test_hub_integration.py`
- `tests/test_cli_hub_actions.py` (new)

---

## Session 8: Hub HTML Status Page
**Date**: 2025-11-22  
**Branch**: `Create-hub`  
**Goal**: Ship a browser-friendly `/hub` dashboard that visualizes the existing `/hub/status` feed while respecting the `enable_status_page` toggle.

### Discoveries
- The FastAPI app already exposes `/hub/status` plus the CLI watcher, so the HTML page can rely entirely on that JSON without new controller dependencies.
- `enable_status_page` was wired through configuration but unused; the new route now enforces the flag so operators can disable the surface when needed.

### Actions Taken
1. ✅ Added a static `/hub` HTML route that renders a lightweight dashboard, polls `/hub/status`, and surfaces warnings, counts, and model metadata.
2. ✅ Created regression tests that verify the page loads (and is gated by `enable_status_page`).
3. ✅ Updated the README and docs to describe the dashboard, revised the config table entry, and added usage instructions.

### Next Actions
- Continue with the observability/logging polish workstream outlined in the roadmap (per-model log bindings, structured output, etc.).


### Files Changed
- `app/api/endpoints.py`
- `tests/test_hub_integration.py`
- `README.md`
- `docs/HANDOFFS.md`

---

## Session 9: Phase 1 – Daemon Docs Refresh
**Date**: 2025-12-16  
**Branch**: `Create-hub`  
**Goal**: Remove the last HubService references from the docs and point operators to the daemon-only HTTP API.

### Actions Taken
1. ✅ Updated `docs/HUB_MODE.md` so the API table documents `/hub/reload`, `/hub/shutdown`, and the remaining `/hub/models/*` routes instead of the removed `/hub/service/*` shim. Added explicit guidance that every CLI/dashboard action now calls the FastAPI daemon directly.
2. ✅ Refreshed the README "Hub vs Launch Modes" section with a note explaining that the FastAPI hub daemon (implemented in `app/hub/daemon.py`) is the only control surface—there is no IPC client or HubService shim.
3. ✅ Logged this work here so future sessions know the documentation already reflects the daemon-only architecture.

### Next Actions
- Spot-check smaller docs/examples for straggling references to the shim while focusing on Plan Part C cleanup tasks.

---

## Hub Daemon Migration — Single-daemon approach (summary)

The hub architecture is now a single long-lived FastAPI "hub daemon" that owns process supervision, memory/runtime state, and an HTTP control plane. The daemon exposes a canonical API rooted under `/hub/*` (examples: `/hub/status`, `/hub/models/{name}/start`, `/hub/models/{name}/load`) and is implemented in `app/hub/daemon.py`.

Key points:
- Ownership: the daemon is the sole owner of model process lifecycle (spawn/monitor/terminate), handler memory load/unload state, and per-model metadata.
- Control API: operators and the CLI MUST interact with the daemon via HTTP. There is no backwards compatibility shim; code that previously imported hub internals must be updated to call the daemon HTTP API.
- Consolidation: previous multi-process orchestration logic (for example `app/hub/controller.py`, `app/hub/manager.py`, `app/hub/runtime.py`) has been consolidated into the daemon. Those files are retained only for historical reference; runtime logic lives in `app/hub/daemon.py` and the `HubSupervisor` abstraction.

See `TEMP_PROJECT_STEPS.md` at the repository root for a detailed, step-by-step migration checklist, verification commands, and suggested tests to add or update.

---
