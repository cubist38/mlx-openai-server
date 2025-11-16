# FUSION_PHASE0 – Engine Warm-Up

**Repo**: `mlx-openai-server-lab`
**Role**: Tier 3 Compute Fabric / Stateless MLX Inference Engine
**Date**: 2025-11-16

---

## 1. Executive Summary

This repository implements a **stateless, OpenAI-compatible HTTP server** that exposes MLX-based models (text, multimodal, embeddings, audio, image generation) on Apple Silicon. It is designed to be a drop-in replacement for OpenAI API endpoints running entirely locally.

**Key Architecture:**
- FastAPI-based HTTP server with uvicorn
- Asynchronous request queue with semaphore-based concurrency control
- Model handlers wrapping MLX libraries (mlx-lm, mlx-vlm, mlx-whisper, mlx-embeddings, mflux)
- Streaming and non-streaming inference support
- Zero persistent state (all stateless)

**Current Limitations:**
- Single model per server instance (model loaded at startup, not hot-swappable)
- No model registry or multi-model management
- No persistent queue or job tracking
- No database or state storage
- Concurrency defaults to 1 (sequential processing)

---

## 2. High-Level Code Map

### Entrypoints
| Path | Function | Description |
|------|----------|-------------|
| `app/cli.py:74` | `cli()` | Click CLI group, main entry point |
| `app/cli.py:145` | `launch()` | CLI command that starts the server |
| `app/main.py:50` | `parse_args()` | Alternative argparse-based CLI parser |
| `app/main.py:189` | `setup_server()` | Creates FastAPI app and uvicorn config |
| `app/main.py:82` | `create_lifespan()` | Factory for FastAPI lifespan context manager |
| `pyproject.toml:56` | Script entry | `mlx-openai-server` console script → `app.cli:cli` |

### Model Loading & Registry
| Path | Class/Function | Description |
|------|----------------|-------------|
| `app/models/mlx_lm.py:23` | `MLX_LM.__init__()` | Loads language model via `mlx_lm.utils.load()` |
| `app/models/mlx_lm.py:136` | `MLX_LM.__call__()` | Inference entry point (streaming/non-streaming) |
| `app/models/mlx_vlm.py` | `MLX_VLM` | Multimodal model wrapper (mlx-vlm) |
| `app/models/mlx_embeddings.py` | `MLXEmbeddings` | Embeddings model wrapper |
| `app/models/mlx_whisper.py` | `MLXWhisper` | Audio transcription model wrapper |
| `app/models/mflux.py` | `MFlux` | Flux image generation model wrapper |
| `app/handler/mlx_lm.py:16` | `MLXLMHandler` | Handler that wraps MLX_LM with queue |
| `app/handler/mlx_vlm.py` | `MLXVLMHandler` | Handler for multimodal models |
| `app/handler/mlx_embeddings.py` | `MLXEmbeddingsHandler` | Handler for embeddings |
| `app/handler/mlx_whisper.py` | `MLXWhisperHandler` | Handler for Whisper |
| `app/handler/mflux.py` | `MLXFluxHandler` | Handler for Flux image generation |
| `app/main.py:88-149` | Lifespan handler selection | Model type → Handler instantiation |

**Model Selection Logic:**
- Model type specified via `--model-type` CLI arg (choices: lm, multimodal, image-generation, image-edit, embeddings, whisper)
- Appropriate handler instantiated in `create_lifespan()` based on model type
- Model loaded synchronously at server startup
- **No runtime model switching** – one model per server instance

### Configuration & Environment
| Path | Config Method | Description |
|------|---------------|-------------|
| `app/cli.py:146-258` | Click options | All server config via CLI flags |
| `app/main.py:51-75` | Argparse | Alternative CLI arg parsing |
| `app/models/mlx_lm.py:15-21` | Environment variables | Default sampling params (TEMPERATURE, TOP_P, etc.) |
| `app/cli.py:11` | `Config` class | Configuration container for server parameters |

**Key Config Parameters:**
- `--model-path`: Path to model directory or HuggingFace identifier
- `--model-type`: Model type (lm, multimodal, embeddings, etc.)
- `--context-length`: Max context length (default: 32768)
- `--port`: Server port (default: 8000)
- `--host`: Server host (default: 0.0.0.0)
- `--max-concurrency`: Max concurrent requests (default: 1)
- `--queue-timeout`: Request timeout in seconds (default: 300)
- `--queue-size`: Max pending requests (default: 100)

**No config files** – all configuration via CLI arguments.

### HTTP Routing & API
| Path | Route | Handler |
|------|-------|---------|
| `app/api/endpoints.py:34` | `GET /health` | Health check (no dependencies) |
| `app/api/endpoints.py:41` | `GET /v1/models` | List available models |
| `app/api/endpoints.py:58` | `GET /v1/queue/stats` | Queue statistics |
| `app/api/endpoints.py:82` | `POST /v1/chat/completions` | Chat completions (streaming/non-streaming) |
| `app/api/endpoints.py:102` | `POST /v1/embeddings` | Generate embeddings |
| `app/api/endpoints.py:116` | `POST /v1/images/generations` | Generate images (Flux) |
| `app/api/endpoints.py:141` | `POST /v1/images/edits` | Edit images (Flux kontext) |
| `app/api/endpoints.py:166` | `POST /v1/audio/transcriptions` | Transcribe audio (Whisper) |
| `app/main.py:207` | Router inclusion | `app.include_router(router)` |

**Request Flow:**
1. FastAPI receives HTTP request
2. Request validated against Pydantic schemas (`app/schemas/openai.py`)
3. Handler retrieved from `app.state.handler`
4. Request enqueued in handler's RequestQueue
5. Worker processes request when semaphore available
6. Response streamed or returned as JSON

### Queue & Concurrency
| Path | Class/Function | Description |
|------|----------------|-------------|
| `app/core/queue.py:33` | `RequestQueue` | Async queue with semaphore concurrency control |
| `app/core/queue.py:49` | `semaphore` | `asyncio.Semaphore(max_concurrency)` |
| `app/core/queue.py:50` | `queue` | `asyncio.Queue(maxsize=queue_size)` |
| `app/core/queue.py:55` | `start()` | Starts worker loop |
| `app/core/queue.py:109` | `_worker_loop()` | Main worker that dequeues and processes |
| `app/core/queue.py:129` | `_process_request()` | Processes single request with timeout |
| `app/core/queue.py:174` | `enqueue()` | Adds request to queue |
| `app/core/queue.py:211` | `submit()` | Enqueue + wait for result |
| `app/main.py:151-155` | Handler initialization | Queue config passed to handler |

**Concurrency Model:**
- Semaphore limits concurrent model inference tasks
- Queue holds pending requests (FIFO)
- Each request gets a Future that resolves when processed
- Timeout enforced at request level (default 300s)
- Memory cleanup via garbage collection every 10 requests

---

## 3. Dataflow Diagram

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTP POST /v1/chat/completions
       ▼
┌─────────────────────────────────────┐
│  FastAPI Router (endpoints.py)      │
│  - Validates request (Pydantic)     │
│  - Extracts handler from app.state  │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Handler (e.g., MLXLMHandler)       │
│  - generate_text_stream()           │
│  - Prepares request data            │
└──────┬──────────────────────────────┘
       │ queue.submit(request_id, data)
       ▼
┌─────────────────────────────────────┐
│  RequestQueue (queue.py)            │
│  - Enqueues request                 │
│  - Worker acquires semaphore        │
└──────┬──────────────────────────────┘
       │ _process_request()
       ▼
┌─────────────────────────────────────┐
│  MLX_LM Model (mlx_lm.py)           │
│  - __call__() invokes mlx-lm        │
│  - stream_generate() or generate()  │
└──────┬──────────────────────────────┘
       │ Response generator/string
       ▼
┌─────────────────────────────────────┐
│  Handler (post-processing)          │
│  - Parser (tool calls, reasoning)   │
│  - Yield chunks or final response   │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  FastAPI Router                     │
│  - StreamingResponse (SSE)          │
│  - or JSONResponse                  │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────┐
│   Client    │
└─────────────┘
```

---

## 4. API Surface – Existing Endpoints

| Method | Endpoint | Purpose | Handler Method | Notes |
|--------|----------|---------|----------------|-------|
| GET | `/health` | Health check | N/A | Always returns 200 OK, no dependencies |
| GET | `/v1/models` | List available models | `handler.get_models()` | Returns single model loaded at startup |
| GET | `/v1/queue/stats` | Queue statistics | `handler.get_queue_stats()` | Returns queue size, active requests, max concurrency |
| POST | `/v1/chat/completions` | Chat completions | `handler.generate_text_stream()` or `handler.generate_text_response()` | Streaming: SSE, Non-streaming: JSON |
| POST | `/v1/embeddings` | Generate embeddings | `handler.generate_embeddings_response()` | Returns list of embeddings |
| POST | `/v1/images/generations` | Generate images | `handler.generate_image()` | Flux models only, requires mflux |
| POST | `/v1/images/edits` | Edit images | `handler.edit_image()` | Flux kontext only, requires mflux |
| POST | `/v1/audio/transcriptions` | Transcribe audio | `handler.generate_transcription_response()` | Whisper models only |

**Compatibility:**
- Implements OpenAI API format for requests and responses
- Supports standard parameters: temperature, top_p, max_tokens, stream, etc.
- Tool calling via `tools` parameter (with auto-detection of tool call parsers)
- Function calling format matches OpenAI
- Streaming uses SSE (Server-Sent Events) with `data: [DONE]` terminator

**Missing OpenAI Endpoints:**
- No `/v1/completions` (legacy, only chat completions)
- No `/v1/files`, `/v1/fine-tuning`, `/v1/moderations`, etc.
- No multi-model management (only single model at startup)

---

## 5. Concurrency & Queue Configuration

### Where Limits Are Set

| Component | Location | Default | Configurable Via |
|-----------|----------|---------|------------------|
| **Max Concurrency** | `app/core/queue.py:46` | 1 | `--max-concurrency` CLI arg |
| **Queue Size** | `app/core/queue.py:48` | 100 | `--queue-size` CLI arg |
| **Request Timeout** | `app/core/queue.py:47` | 300s | `--queue-timeout` CLI arg |
| **Semaphore** | `app/core/queue.py:49` | `asyncio.Semaphore(max_concurrency)` | Derived from max-concurrency |
| **Async Queue** | `app/core/queue.py:50` | `asyncio.Queue(maxsize=queue_size)` | Derived from queue-size |

### How It Works

1. **Semaphore Control**: `app/core/queue.py:138`
   - `async with self.semaphore:` blocks until a slot is available
   - Ensures at most `max_concurrency` requests are being processed simultaneously
   - Default = 1 (sequential processing)

2. **Queue Buffering**: `app/core/queue.py:199`
   - Requests enqueued into `asyncio.Queue` with max size `queue_size`
   - If queue full, raises `asyncio.QueueFull` (returns HTTP 429)
   - Worker loop dequeues and processes FIFO

3. **Timeout Enforcement**: `app/core/queue.py:142-145`
   - Each request wrapped in `asyncio.wait_for(processor(data), timeout=self.timeout)`
   - If exceeded, request.set_exception(TimeoutError)
   - Default = 300s (5 minutes)

4. **Memory Management**:
   - `app/core/queue.py:171-172`: Garbage collection every 10 processed requests
   - `app/main.py:232-235`: Memory cleanup every 50 HTTP requests
   - `app/models/mlx_lm.py:126`: MLX cache cleared after embeddings
   - Explicit cleanup in request queue on shutdown

### Current Bottlenecks

- **Single concurrency by default**: Only 1 request processed at a time
- **No persistent queue**: Queue is in-memory, lost on restart
- **No priority system**: FIFO only, no request prioritization
- **No request preemption**: Long-running requests block queue
- **No distributed processing**: Single-node only

### Recommendations for Phase 1

- Increase default `max_concurrency` based on model size and VRAM
- Add request priority/preemption logic
- Implement persistent queue (Redis/MongoDB) for job tracking
- Add model pool management for multi-model serving
- Expose concurrency metrics via `/v1/queue/stats`

---

## 6. Model Lifecycle & Memory

### Loading (Startup)
- `app/main.py:85-156`: Lifespan context manager loads model on startup
- Model loaded **synchronously** during FastAPI lifespan startup
- No async loading or lazy initialization
- Model stored in `app.state.handler`

### Inference (Runtime)
- `app/models/mlx_lm.py:136`: `__call__()` method invokes MLX generation
- `app/models/mlx_lm.py:180`: Prompt cache created per request
- Streaming uses generator pattern (low memory footprint)
- Non-streaming buffers entire response

### Cleanup (Shutdown)
- `app/main.py:169-182`: Lifespan shutdown calls `handler.cleanup()`
- `app/handler/mlx_lm.py:352-366`: Handler cleanup stops queue
- `app/core/queue.py:69-107`: Queue cleanup cancels pending requests
- `app/main.py:181`: MLX cache cleared, garbage collection forced

### Memory Management
- `mx.clear_cache()` called periodically and on cleanup
- `gc.collect()` forced after inference and every N requests
- Request data explicitly deleted after processing
- No model unloading/reloading – model stays in memory for server lifetime

---

## 7. Dependencies & Tech Stack

### Core Libraries (pyproject.toml:24-41)
- **FastAPI** (0.115.14): HTTP server framework
- **uvicorn** (0.35.0): ASGI server
- **mlx-lm** (0.28.3): Text-only language models
- **mlx-vlm** (0.3.6): Multimodal vision-language models
- **mlx-whisper** (0.4.3): Audio transcription
- **mlx-embeddings** (0.0.4): Text embeddings
- **mflux** (optional): Flux image generation
- **outlines** (1.1.1): Structured output (JSON schema)
- **loguru** (0.7.3): Logging
- **click** (8.2.1): CLI framework

### Python Version
- Requires: >=3.11, <3.13
- Optimized for Apple Silicon (MLX framework)

---

## 8. Gaps & Future Work (TODOs for Phase 1)

### Model Management
- [ ] Multi-model registry (load multiple models, route by model ID)
- [ ] Model hot-swapping (load/unload without restart)
- [ ] Model metadata API (capabilities, context length, etc.)
- [ ] Model versioning and aliases

### Queue & Concurrency
- [ ] Persistent queue (survive restarts)
- [ ] Request priority/preemption
- [ ] Distributed queue (multi-node)
- [ ] Adaptive concurrency (auto-tune based on load)
- [ ] Request batching for embeddings

### State & Integration
- [ ] Database integration (MongoDB for Tier 2 MCP)
- [ ] Job tracking (status, history, logs)
- [ ] Tier 2 ↔ Tier 3 protocol (state handoff)
- [ ] Health metrics (latency, throughput, VRAM usage)

### API Enhancements
- [ ] Batch endpoints (multiple requests in one call)
- [ ] Model warmup API (preload without request)
- [ ] Cancel endpoint (abort in-flight requests)
- [ ] WebSocket streaming (alternative to SSE)

### Observability
- [ ] Prometheus metrics export
- [ ] OpenTelemetry traces
- [ ] Structured logging (JSON)
- [ ] Request ID tracking across tiers

---

## 9. References

- **Main Entrypoint**: `app/cli.py:145` (`launch` command)
- **Server Setup**: `app/main.py:189` (`setup_server`)
- **API Routes**: `app/api/endpoints.py`
- **Queue Logic**: `app/core/queue.py`
- **Model Wrappers**: `app/models/*.py`
- **Handlers**: `app/handler/*.py`
- **Schemas**: `app/schemas/openai.py`
- **README**: Project overview, usage examples
- **pyproject.toml**: Dependencies, scripts, metadata
