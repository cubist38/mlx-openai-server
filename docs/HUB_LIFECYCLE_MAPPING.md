# Hub Lifecycle Convergence Notes

This document captures plan-streamlineHubLifecycle Part A step 1 by mapping the overlapping lifecycle flows between the daemon-driven `HubSupervisor` and the single-model server bootstrap. It focuses on where both paths already share behavior and where shared utilities need to stabilize worker, registry, and idle-controller logic.

## Lifecycle Map

| Area | HubSupervisor path | Single-server path | Shared needs surfaced |
| --- | --- | --- | --- |
| Manager bootstrap & logging | `_build_manager_for_record()` prepares `LazyHandlerManager`, file logging, and metadata before `start_model()` wires registry IDs and workers ([app/hub/daemon.py](app/hub/daemon.py#L523-L1014)). | `create_lifespan()` instantiates `LazyHandlerManager`, attaches on-change hooks, and records metadata plus handlers in the registry ([app/server.py](app/server.py#L1061-L1269)). | Need a shared factory that standardizes manager creation, per-model log sinks, and initial registry payloads so daemon/server cannot drift on defaults (log level, queue sizing, model identifiers). |
| Registry synchronization | `_execute_load_operation()` and `_execute_unload_operation()` push handler + metadata changes, while `_start_worker()` / `_stop_worker()` update worker ports inside registry entries ([app/hub/daemon.py](app/hub/daemon.py#L574-L775)). | `_sync_registry_update()` inside `create_lifespan()` keeps registry metadata (VRAM timestamps, status, model path) authoritative ([app/server.py](app/server.py#L1165-L1219)). | Introduce a registry helper (Plan suggestion: `RegistrySyncService`) that encapsulates metadata merging, worker port management, and VRAM timestamps so both flows call the same code for handler-loaded, unloaded, and worker lifecycle events. |
| Worker lifecycle | `start_model()` bootstraps sidecar `SidecarWorker` instances and keeps `_start_worker()` idempotent; `_stop_worker()` clears worker ports and surfaces errors ([app/hub/daemon.py](app/hub/daemon.py#L574-L1045)). | Single-model server has no sidecar worker abstraction, but relies on handler residency states to expose `/v1/models` details ([app/server.py](app/server.py#L1061-L1344)). | Need a shared coordinator class (Plan callout: `SidecarWorkerCoordinator`) so daemon and any future server modules interact with workers via the same interface and propagate registry metadata consistently even when only one worker path exists today. |
| Idle controller & auto-unload | `HubSupervisor` accepts an injected `idle_controller`, uses it while building status snapshots, and enforces group eviction thresholds via `_ensure_group_capacity()` ([app/hub/daemon.py](app/hub/daemon.py#L705-L928) and [get_status() snapshot](app/hub/daemon.py#L1484-L1687)). | `CentralIdleAutoUnloadController` watches the registry, registers callbacks via the lifespan, and provides `get_expected_unload_timestamp()` for downstream reporting ([app/server.py](app/server.py#L827-L1054)). | Need a common idle/unload service so daemon status, CLI, and registry polling surface the same unload timestamps and eviction logic. Today duplicate timers lead to diverging `unload_timestamp` semantics between hub daemon snapshots and single-server metadata. |
| Status & snapshot assembly | `get_status()` composes `models` plus group payloads, probes workers, and injects idle controller timestamps ([app/hub/daemon.py](app/hub/daemon.py#L1484-L1775)). | Hub-aware server builds `/hub/status` payloads by combining config + registry snapshots inside `_build_models_from_config()` ([app/api/hub_routes.py](app/api/hub_routes.py#L238-L333)). | Need a shared formatter that converts supervisor/live registry data into the canonical snapshot, ensuring daemon UI, CLI, and hub routes reuse identical timestamp fields (e.g., `last_activity_ts`, `unload_timestamp`, group membership lists). |

## Cross-cutting Observations

- Both flows gate JIT load/unload decisions through `LazyHandlerManager`, but the daemon wraps it manually while the server wires it through FastAPI lifespan. Consolidating construction + teardown will eliminate branching logic around logging, registry hooks, and idle callbacks.
- Registry metadata schemas already overlap (worker ports, VRAM timestamps, group metadata), yet each code path populates fields differently. Centralizing the write paths will unblock Plan Part A step 2 without risking regressions in `/v1/models` responses.
- The daemon status builder currently probes workers for memory state, while the server trusts registry data. Sharing a formatter plus probe helper will let the server UI depend on the same load/unload timestamps that the daemon surfaces after `_probe_worker_for_memory()`.
- CLI, daemon UI, and API route handlers all need a unified `HubLifecycleService` wrapper that exposes `start/load/unload/stop/status` and internally orchestrates registry + worker helpers. This document identifies where that orchestration logic already exists so the next steps can safely relocate it under `app/core/`.

## Hub Service Shim Retirement Checklist

1. ✅ Removed unused shim endpoints in [app/api/hub_routes.py](app/api/hub_routes.py#L1-L600); dashboard buttons now call the canonical daemon routes described in Plan Part B/C.
2. ✅ Dropped shim-only schemas in [app/schemas/openai.py](app/schemas/openai.py) so only daemon payloads mirroring `HubSupervisor` snapshots remain.
3. ✅ Updated the dashboard template [templates/hub_status.html.jinja](templates/hub_status.html.jinja) so start/stop/load/unload buttons call `/hub/models/{name}/...` and all UI labels reference the daemon instead of the removed service shim.
4. ✅ Replaced `test_hub_service.py` with daemon-focused API coverage (`tests/test_hub_daemon_routes.py`) exercising `/hub/models/*`, `/hub/reload`, and `/hub/shutdown`.
5. ✅ Refreshed hub documentation ([docs/HUB_MODE.md](docs/HUB_MODE.md), [docs/HANDOFFS.md](docs/HANDOFFS.md), README sections) to describe the daemon-only control plane now that the shim is gone.

## VRAM Lifecycle Implementation

This section documents the stable VRAM lifecycle implementation following the plan-stableVramLifecycle convergence:

### Configuration Validation (Step 1 Complete)

**Location**: `app/config.py`, `app/hub/config.py`

**Rules enforced**:
- `auto_unload_minutes` requires `jit: true` (raises `ConfigError`)
- Group `idle_unload_trigger_min` requires `max_loaded` to be set (raises `HubConfigError`)
- `max_loaded` must be ≥ 1 if specified

**Tests**: `tests/test_config.py`, `tests/test_hub_config.py`

### Registry & Handler Coordination (Step 2 Complete)

**Location**: `app/core/model_registry.py`, `app/core/manager_protocol.py`

**Key changes**:
- Explicit manager protocol separates manual vs automated unload triggers
- Registry availability cache tracks started-not-loaded models as available
- Idle timestamp plumbing preserved; provenance tracking added

**Visibility logic**:
- `get_available_model_ids()` returns models where `started=True`
- Group filtering applies additional load-state visibility rules
- Tests: `test_model_registry.py`

### Supervisor/Daemon Flows (Step 3 Complete)

**Location**: `app/hub/daemon.py`, `app/hub/worker.py`

**Behavior**:
- `HubSupervisor.start_model` always records `started` flag and spins up workers
- `HubSupervisor.stop_model` clears `started` flag via `RegistrySyncService`
- Started-but-unloaded models retain visibility metadata
- Group capacity enforcement uses 10s availability cache

**Error responses**:
- HTTP 429 when group at capacity with no eviction candidates
- Message: "Group capacity exceeded. Unload another model or wait for auto-unload."

**Tests**: `test_hub_daemon.py`, `test_control_plane.py`

### API Visibility (Step 4 Complete)

**Location**: `app/api/endpoints.py`, `app/core/hub_status.py`

**Implementation**:
- `/v1/models` syncs supervisor memory state into registry before filtering
- Uses `get_available_model_ids()` which respects `started` flag
- `build_group_state()` counts only loaded models while including all members

**Visibility rules**:
- Stopped models: Never appear
- Started models: Appear by default
- Group at capacity without idle candidates: Only loaded models visible
- Group at capacity with idle ≥ threshold: All started models visible

**Tests**: `test_endpoints_models.py` (`test_v1_models_filters_based_on_supervisor_memory_loaded`, `test_v1_models_hides_stopped_models`)

### Manual Controls (Step 5 Complete)

**Location**: `app/cli.py`, `app/api/hub_routes.py`

**Commands**:
- `hub load-model <name>` / `POST /hub/models/{model}/load`
- `hub unload-model <name>` / `POST /hub/models/{model}/unload`

**Behavior**:
- Works regardless of JIT setting
- Calls daemon API and surfaces validation errors
- Hub routes enforce group constraints via `_guard_hub_action_availability`
- Returns 429 when group policies violated

**Tests**: `test_cli_hub_actions.py`, `test_control_plane.py`

### Documentation (Step 6 Complete)

**Updated files**:
- `README.md`: Added comprehensive "VRAM Lifecycle & Model Visibility" section
- `docs/HUB_MODE.md`: Expanded lifecycle rules with detailed examples
- `docs/HUB_LIFECYCLE_MAPPING.md`: This section documents implementation

**Test coverage**:
- `tests/test_hub_model_lifecycle.py`: Lifecycle state transitions
- `tests/test_routes_by_mode.py`: Mode-specific visibility behavior
- `tests/test_endpoints_models.py`: `/v1/models` filtering
- `tests/test_control_plane.py`: Group capacity and eviction

### Key Principles

1. **Visibility = Started**: Model appears in `/v1/models` when `started=True`, independent of VRAM state (with group visibility exceptions)
2. **Load ≠ Start**: Loading is VRAM operation; starting is process lifecycle
3. **Group Visibility Gating**: Groups may hide unloaded models when at capacity based on idle state
4. **Independent Timers**: `auto_unload_minutes` (scheduled) vs `idle_unload_trigger_min` (on-demand)
5. **Manual Override**: Load/unload always available regardless of JIT configuration
