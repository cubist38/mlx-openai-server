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
