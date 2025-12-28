# Hub Mode Deep Dive

This document expands on the **Hub Mode** section in the project README and describes how the background manager service, CLI, and FastAPI endpoints interact. Use it as a reference when wiring custom tooling, authoring deployment docs, or onboarding new contributors.

## Architecture Overview

1. **HubManager (process orchestrator)**
   - Loads `hub.yaml`, validates model/group definitions, and spawns a dedicated subprocess per model using the standard single-model server entrypoint. Group `max_loaded` limits are enforced later by the in-process runtime when a handler tries to load into memory.
   - Persists telemetry (`hub-manager.log`, `<model>.log`, `hub-manager.pid`, and `hub-manager.sock`) inside the configured `log_path`.
   - Emits structured observability events via `HubObservabilitySink` so you can stream process lifecycle updates to third-party collectors.
2. **Hub daemon (HTTP control plane)**
   - The legacy IPC-based `HubService` was replaced by a single FastAPI-backed hub daemon. The daemon owns process supervision, memory lifecycle, and an HTTP control surface under `/hub/*`.
   - The CLI and FastAPI routes interact with the daemon via HTTP (for example, `GET /hub/status`, `POST /hub/reload`, `POST /hub/models/{model}/start`).
   - There is no `/hub/service/*` shim or IPC socket client anymore; every operator flow calls the daemon's HTTP API directly.
   - Supported actions include health checks, status queries, reloads, model start/stop, memory load/unload, and graceful shutdowns. Every public control surface maps to one of these HTTP endpoints.
3. **Shared lifecycle helpers**
   - `app/core/hub_lifecycle.py` exports `HubLifecycleService`, `HubServiceAdapter`, and `get_hub_lifecycle_service()` so FastAPI routes, CLI helpers, and background tasks all speak the same interface. Tests can stub a single service object and rely on the helper to inject it into `app.state` the same way the daemon does.
   - `RegistrySyncService` centralizes registry mutations (handler load/unload, worker telemetry). Both the daemon and the single-model server reuse it to keep metadata identical across processes.
   - `app/core/hub_status.py` contains `build_group_state()`, which normalizes the `/hub/status` group summaries for the CLI, HTML dashboard, and API consumers.
4. **Daemon client pattern**
   - Instead of an IPC client, callers now issue HTTP requests to the daemon. Tests and tools should call the daemon endpoints or stub the async HTTP helper used by `app.api.hub_routes`.
5. **FastAPI `/hub` surface**
   - Exposes JSON APIs plus the HTML dashboard. Operators can trigger the same start/reload/stop/load/unload actions directly from the browser while receiving flash/toast feedback.

   **Routing note:** The `/hub` admin surface is implemented in `app/api/hub_routes.py` and is intended to be mounted only by the hub daemon. The single-model launch process exposes only the OpenAI-compatible `/v1/...` endpoints. This ensures a single canonical implementation for `/hub` and prevents duplicate handlers across processes.

## Hub YAML Quick Reference

```yaml
host: 0.0.0.0           # optional, defaults to 0.0.0.0
port: 8000              # optional, defaults to 8000
model_starting_port: 47850  # optional, defaults to 47850 for auto-assigned workers
log_level: INFO         # optional, defaults to INFO
log_path: ~/mlx-openai-server/logs  # auto-created if missing
enable_status_page: true

models:
  - name: alpha                 # required slug (letters, numbers, -, _)
    model_path: /models/alpha   # required
    model_type: lm              # any MLXServerConfig option works
    default: true               # optional, auto-start the worker process at startup
    group: tier_one             # optional slug reference

  - name: beta
    model_path: /models/beta
    group: tier_one

# Optional memory-throttling groups (max_loaded >= 1)
groups:
  - name: tier_one
    max_loaded: 1
```

Validation highlights:

- Model and group names must already be slug-compliant; invalid values raise `HubConfigError`.
- Default models inside a group may exceed `max_loaded`; every worker can still start, but only `max_loaded` handlers from that group can be loaded into memory at once.
- `log_path` is expanded and created automatically, ensuring log files and socket/PID artifacts share the same directory tree.
- Referencing a group without defining it is permitted when you plan to add the group later; set `groups` explicitly to enforce caps.
- Auto-assigned model ports start at `model_starting_port` (defaults to `47850`). The loader probes each candidate socket and skips busy ports. Override `port` explicitly when you need fixed values, but never reuse the controller's `host:port` or another model's port.

## CLI Workflows & Flash Messaging

| Command | Description |
| --- | --- |
| `hub start` | Launches the HubManager service (or prints current status if already up) and ensures the FastAPI controller is running so `/hub`, `/hub/status`, and OpenAI endpoints stay online. Shows flash messages such as `[ok] Hub manager is now running`. |
| `hub status [MODEL ...]` | Reloads `hub.yaml`, syncs with the service, and prints per-model summaries. |
| `hub reload` | Forces the service to reload YAML and emits a flash summary of started/stopped/unchanged models. |
| `hub stop` | Reloads one last time, requests shutdown, and confirms via flash message. |
| `hub start-model MODEL [...]` | Reloads then calls `start_model` for each name to ensure the worker process is running. |
| `hub stop-model MODEL [...]` | Reloads then calls `stop_model` for each name. |
| `hub load-model MODEL [...]` | Talks to the controller to instantiate handlers in memory for running workers; returns OpenAI-style 429 errors when a group's `max_loaded` cap is saturated. |
| `hub unload-model MODEL [...]` | Tells the controller to dispose of in-memory handlers while leaving the worker alive. |
| `hub watch [--interval N]` | Streams `/hub/status` snapshots with uptime, exit codes, and log filenames. |

Flash helper tones (`info`, `success`, `warning`, `error`) mirror the HTML dashboard styles so operators see consistent feedback across surfaces.

## Availability Cache & Degraded Telemetry

- The FastAPI controller keeps a shared `ModelRegistry` in sync with hub group policies. Every handler attach/detach, `/hub reload`, or daemon status poll recomputes the allowed set and caches it as `available_model_ids`.
- OpenAI-compatible endpoints honor this cache when listing models or acquiring handlers. During a daemon outage the controller reuses the last successful snapshot so `/v1/models` keeps returning a consistent filtered list instead of exposing stale entries.
- Hub-facing actions (`hub start-model`, `hub load-model`, `hub vram load`, `/hub/models/{model}/start`, etc.) call the same guard helper. If a group is already at its `max_loaded` ceiling and no member satisfies the optional `idle_unload_trigger_min`, the request fails fast with the existing OpenAI-style `429: Group capacity exceeded. Unload another model or wait for auto-unload.` response instead of invoking the controller/daemon.
- Because the CLI, dashboard, and OpenAI APIs all reference the same cache, operators see identical availability decisions no matter which surface they use.

## VRAM Lifecycle & Model Visibility

The hub implements a clear VRAM lifecycle that separates **visibility** (what appears in `/v1/models`) from **VRAM residency** (what's loaded in memory). Understanding these distinctions is critical for configuring groups and managing memory effectively.

### Visibility Rules

| State | Process | VRAM | `/v1/models` | Accepts Requests |
|-------|---------|------|--------------|-----------------|
| Started | Running | Loaded | ✅ Visible | ✅ Yes |
| Started | Running | Unloaded (JIT) | ✅ Visible | ✅ Yes (auto-loads) |
| Stopped | Not running | Not loaded | ❌ Hidden | ❌ No |

**Core principle**: Model visibility is determined by the `started` flag. Once started, a model remains visible until explicitly stopped, even if handlers are unloaded from memory. However, group capacity rules may temporarily hide unloaded models when the group is at capacity (see below).

### Load vs Start Operations

- **start-model**: Launches worker process + marks as started + (optionally) loads into VRAM
- **stop-model**: Terminates worker + unloads VRAM + marks as stopped (removes from `/v1/models`)
- **load-model**: Loads handlers into VRAM for an already-started model (may return 429 if group capacity exceeded)
- **unload-model**: Releases VRAM while keeping worker running and model visible

### Configuration Validation

The hub enforces these rules at config load time:

- **JIT requirement for auto-unload**: `auto_unload_minutes` requires `jit_enabled: true`
- **Group trigger dependency**: `idle_unload_trigger_min` can only be set when `max_loaded` is also defined
- **Capacity minimum**: `max_loaded` must be ≥ 1 if specified

Invalid configurations raise `HubConfigError` or `ConfigError` before the hub starts.

### Group Capacity Enforcement

Groups control simultaneous VRAM occupancy. Behavior differs based on `idle_unload_trigger_min`:

#### Without `idle_unload_trigger_min`

Simple capacity cap with visibility gating:

1. **Hard limit**: Only `max_loaded` models can be loaded at once
2. **Load blocking**: HTTP 429 when attempting to exceed capacity
3. **Dynamic visibility**:
   - When loaded count < `max_loaded`: All started models visible in `/v1/models`
   - When loaded count = `max_loaded`: Only loaded models visible; unloaded models hidden
   - After unload reduces count: All started models become visible again

**Example**: Group `tier_one` with `max_loaded: 1` and models [A, B, C] all started:
```
Initial state (none loaded):
  - /v1/models shows: [A, B, C]
  - Load A → succeeds, /v1/models shows: [A]
  - Load B → returns 429, /v1/models still: [A]
  - Unload A → /v1/models shows: [A, B, C]
  - Load B → succeeds, /v1/models shows: [B]
```

#### With `idle_unload_trigger_min`

Intelligent eviction with idle-based visibility:

1. **Capacity limit**: Still enforces `max_loaded` simultaneous models
2. **Eviction check**: When at capacity, checks if any loaded model idle ≥ threshold
3. **Automatic unload**: Evicts longest-idle model meeting threshold to free capacity
4. **Conditional visibility**:
   - When loaded count < `max_loaded`: All started models visible
   - When at `max_loaded` AND all idle < threshold: Only loaded models visible
   - When at `max_loaded` AND ≥1 idle ≥ threshold: All started models visible (eviction possible)
5. **429 fallback**: Returns error only when at capacity with no eviction candidates

**Example**: Group `high_memory` with `max_loaded: 2`, `idle_unload_trigger_min: 15` and models [X, Y, Z] all started:
```
Scenario 1 - No eviction candidates:
  - Load X (idle=0) + Y (idle=0) → both loaded, /v1/models: [X, Y]
  - Load Z → returns 429 (no models idle ≥15 min)

Scenario 2 - Eviction candidate available:
  - X loaded (idle=20 min), Y loaded (idle=5 min)
  - /v1/models shows: [X, Y, Z] (one model idle ≥15 min)
  - Load Z → X auto-unloaded, Z loaded, /v1/models: [Y, Z]
  - Load X → Y still active, returns 429
  - Wait 10 min (Y now idle=15 min)
  - /v1/models shows: [X, Y, Z]
  - Load X → Y auto-unloaded, X loaded, succeeds
```

### Auto-Unload Behavior

Per-model `auto_unload_minutes` provides scheduled memory management:

- **Requirements**: Must have `jit_enabled: true` (enforced during config validation)
- **Trigger**: Background timer starts after request completion
- **Timer reset**: Every request resets the idle counter
- **Action**: When timer expires, handlers unload from VRAM automatically
- **Visibility preserved**: Worker stays running, model remains in `/v1/models`
- **JIT reload**: Next request triggers automatic load
- **Independent operation**: Works separately from group `idle_unload_trigger_min`

**Key distinction**:
- `auto_unload_minutes`: Proactive scheduled unload (background timer)
- `idle_unload_trigger_min`: Reactive threshold-based eviction (capacity-driven)

### Manual Load/Unload

Regardless of JIT or group settings, manual operations always work:

- CLI: `hub load-model <name>`, `hub unload-model <name>`
- API: `POST /hub/models/{model}/load`, `POST /hub/models/{model}/unload`
- Respects group capacity (may return 429 or trigger eviction)
- Does not require JIT to be enabled

### Implementation Notes

- Visibility is determined by syncing `HubSupervisor` memory state into `ModelRegistry` before filtering
- `get_available_model_ids()` respects the `started` flag plus group load-state visibility rules
- `build_group_state()` counts only loaded models while including all group members
- Tests: `test_v1_models_filters_based_on_supervisor_memory_loaded`, `test_v1_models_hides_stopped_models`

## HTML Dashboard Details

- Polls `/hub/status` every five seconds by default and renders the same data used by the CLI.
- Service controls (Start/Reload/Stop) sit beside the status pill and counts area.
- Each model row exposes a **Process** control cluster (Start/Stop) plus a **Memory** cluster (Load/Unload) that only appears after the worker is running; buttons disable themselves based on live status to prevent conflicting actions.
- Flash banners appear below the header to highlight the last action, while toast notifications appear in the lower-right corner for transient success/failure updates.
- When the HubManager is offline, the dashboard degrades gracefully, shows warnings, and continues to surface cached metadata if available.

## API Endpoints

| Endpoint | Method | Description |
| --- | --- | --- |
| `/hub/status` | GET | Reloads `hub.yaml`, queries the HubManager, and returns counts/models/warnings. Falls back to cached metadata with warnings when the service is offline. |
| `/hub` | GET | Serves the HTML dashboard when `enable_status_page` is true; returns 404 otherwise. |
| `/hub/reload` | POST | Calls the controller's `reload_config()` and returns the diff (`started/stopped/unchanged`). |
| `/hub/shutdown` | POST | Requests a graceful shutdown of managed models; pass `?exit=1` or `X-Hub-Exit: 1` to terminate the daemon after cleanup. |
| `/hub/models/{model}/start` | POST | Validates availability and calls `start_model`; HTTP 429 indicates group capacity exhaustion. (CLI: `hub start-model`) |
| `/hub/models/{model}/stop` | POST | Tells the controller to stop the worker and schedule VRAM unload. (CLI: `hub stop-model`) |
| `/hub/models/{model}/load` | POST | Passes the request to the controller so it can instantiate the handler locally (VRAM load). |
| `/hub/models/{model}/unload` | POST | Requests the controller tear down the in-memory handler and free resources (VRAM unload). |

All responses use the same OpenAI-style `error` envelope (`type`, `message`, `code`) so upstream tooling can reuse existing error handling paths.

## Health & Telemetry

- `/health` reflects handler status when running in hub mode (loaded/unloaded) and degrades to HTTP 503 when neither a handler nor handler manager is present.
- `hub watch` and `/hub/status` include exit codes for stopped/failed models so you can quickly spot crashes.
- Logs rotate automatically (25 MB per file for the service, 500 MB for app logs) and are kept inside the hub `log_path`.

For more implementation details, inspect:

- `app/hub/config.py` – YAML loader, slug validation, log-path normalization.
- `app/hub/manager.py` – Process orchestration, group accounting, observability events.
- `app/hub/daemon.py` – FastAPI-based hub daemon, process supervision, and HTTP control plane.
- `app/core/hub_lifecycle.py` – Shared lifecycle services (`HubLifecycleService`, `HubServiceAdapter`, `RegistrySyncService`).
- `app/core/hub_status.py` – Group/status formatting helpers consumed by `/hub/status` and the CLI.
- `app/api/hub_routes.py` – FastAPI endpoints powering `/hub`, `/hub/status`, and service/model controls.
