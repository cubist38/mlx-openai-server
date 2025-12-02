## Plan: Group Availability Enforcement

Implement runtime filtering so OpenAI API + hub operations only expose models allowed by their group’s current VRAM state, caching the last computed availability for degraded telemetry and ensuring hub-triggered launches honor the same rules.

### Steps
1. Extend `app/core/model_registry.py` (`ModelRegistry`, group snapshot helpers) to track per-group `loaded_count`, idle durations, and in-memory `available_model_ids`, plus getters for the last computed filtered set.
2. Invoke the availability refresh helper whenever handlers load/unload or metadata changes by updating `app/server.py` (LazyHandlerManager `_update_handler`, central controller notifications) and `_hub_sync_once` in `app/api/hub_routes.py`.
3. Filter OpenAI model listings and request handling in `app/api/endpoints.py` (`list_models`, `_get_handler_or_error`, hub controller acquire) using the cached availability, falling back to the last known set if hub telemetry is unavailable.
4. Enforce the same availability rules for hub admin actions in `app/api/hub_routes.py` (`hub_start_model`, `hub_load_model`, `hub_service` helpers) so disallowed operations reuse the current “group full/busy” error path.
5. Update tests (`tests/test_model_registry.py`, `tests/test_hub_service.py`, `tests/test_cli_hub_status.py`) to cover max-loaded vs. idle-trigger scenarios, cached availability persistence, and hub request failures when models become unavailable.

### Further Considerations
1. Persist availability only in memory; on restart the filtered list resets until new telemetry arrives.
2. Reuse existing “group full/busy” error payloads so client behavior stays consistent.
