# Hub Mode Deep Dive

This document expands on the **Hub Mode** section in the project README and describes how the background manager service, CLI, and FastAPI endpoints interact. Use it as a reference when wiring custom tooling, authoring deployment docs, or onboarding new contributors.

## Architecture Overview

1. **HubManager (process orchestrator)**
   - Loads `hub.yaml`, validates model/group definitions, and spawns a dedicated subprocess per model using the standard single-model server entrypoint. Group `max_loaded` limits are enforced later by the in-process runtime when a handler tries to load into memory.
   - Persists telemetry (`hub-manager.log`, `<model>.log`, `hub-manager.pid`, and `hub-manager.sock`) inside the configured `log_path`.
   - Emits structured observability events via `HubObservabilitySink` so you can stream process lifecycle updates to third-party collectors.
2. **HubService (IPC server)**
   - Wraps `HubManager` behind a UNIX domain socket so the CLI, FastAPI routes, and HTML dashboard all share the same control plane.
   - "IPC" stands for **Inter-Process Communication** — the mechanisms processes use to exchange data and coordinate operations. In this project, IPC is implemented via a UNIX domain socket and Python's `multiprocessing.connection` APIs.
   - Supports `ping`, `status`, `reload`, `start_model`, `stop_model`, and `shutdown` actions. Every public control surface ultimately maps to one of these subcommands.
3. **HubServiceClient (IPC client)**
   - Lightweight helper used by FastAPI (`app/api/hub_routes.py`) and the CLI (`app/cli.py`).
   - Always reloads `hub.yaml` before servicing requests to keep the HubManager in sync with on-disk changes.
4. **FastAPI `/hub` surface**
   - Exposes JSON APIs plus the HTML dashboard. Operators can trigger the same start/reload/stop/load/unload actions directly from the browser while receiving flash/toast feedback.

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
| `/hub/service/start` | POST | Starts the HubManager if it is not running. Returns PID details or HTTP 400 if the config is missing. |
| `/hub/service/reload` | POST | Runs `reload()` inside the service and returns the diff (`started/stopped/unchanged`). |
| `/hub/service/stop` | POST | Requests shutdown, returning HTTP 503 if no manager is running. |
| `/hub/models/{model}/start-model` | POST | Reloads, then issues `start_model`. HTTP 429 indicates group capacity exhaustion. |
| `/hub/models/{model}/stop-model` | POST | Reloads, then issues `stop_model`. |
| `/hub/models/{model}/load-model` | POST | Passes the request to the controller so it can instantiate the handler locally. |
| `/hub/models/{model}/unload-model` | POST | Requests the controller tear down the in-memory handler and free resources. |

All responses use the same OpenAI-style `error` envelope (`type`, `message`, `code`) so upstream tooling can reuse existing error handling paths.

## Health & Telemetry

- `/health` reflects handler status when running in hub mode (loaded/unloaded) and degrades to HTTP 503 when neither a handler nor handler manager is present.
- `hub watch` and `/hub/status` include exit codes for stopped/failed models so you can quickly spot crashes.
- Logs rotate automatically (25 MB per file for the service, 500 MB for app logs) and are kept inside the hub `log_path`.

For more implementation details, inspect:

- `app/hub/config.py` – YAML loader, slug validation, log-path normalization.
- `app/hub/manager.py` – Process orchestration, group accounting, observability events.
- `app/hub/service.py` – IPC service/client plus logging setup.
- `app/api/hub_routes.py` – FastAPI endpoints powering `/hub`, `/hub/status`, and service/model controls.
