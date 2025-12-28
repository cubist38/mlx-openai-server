## Plan: Stable VRAM Lifecycle

Clarify and enforce the VRAM lifecycle so started-but-unloaded models stay visible while idle eviction, JIT, and auto-unload interact predictably. Central tasks: While keeping the current idle timestamp source, tighten config validation, update registry/state machines to respect the new rules, adjust availability filtering and CLI/web controls, and expand regression tests to pin the desired UX. Add a README-level “VRAM lifecycle” subsection and syncing supporting docs. Work spans stricter config validation, registry/controller enforcement, API visibility, manual controls, documentation, and tests.

### Initial Request

I need you to fix and/or redesign the load/unload into vram processes. For this program, load/unload refers to loading/unloading into vram which is separate from starting/stopping a model. When models are manually loaded from the hub webpage or jit auto-loaded, they now seem to unload very quickly after loading, sometimes even before a single query is run.

#### The desired functionality

* Stopped models do not appear in /v1/models (the OpenAI API)
* In general, started models (whether loaded or unloaded) do appear in /v1/models. Additional limitations are described in the groups bullets below.
* When jit is enabled on a model, the models don't load on start
* auto-unload can only be set for a model where jit is enabled
* For a model with auto-unload set, the model unloads after it has been idle for auto-unload minutes
* Regardless of jit or auto-unload being set, models can be manually loaded/unloaded via the cli and the hub webpage
* Models can be a part of groups and max_loaded and idle_unload_trigger_min are group options
* idle_unload_trigger_min can only be set in a group where max_loaded is set
* If idle_unload_trigger_min is not set and max_loaded is set
  * Only the max_loaded number of models in a group can be actively loaded at a time and if additional models in a group try to load a 429 error is thrown
  * Once max_loaded models are loaded, the other unloaded models do not show in /v1/models. Once the number of loaded models is less than max_loaded, all of the started models in the group show in /v1/models
* If idle_unload_trigger_min is set: 
  * Only the max_loaded number of models in a group can be actively loaded at a time. 
  * If another model in a group tries to load where max_loaded models are already loaded, if one or more loaded models in a group has been idle >= idle_unload_trigger_min, then the model idle for the longest is unloaded and the requested models is allowed to load.
  * Once max_loaded models are loaded and if all models are idle < idle_unload_trigger_min, the other unloaded models do not show in /v1/models. Once one or more loaded models in a group has been idle >= idle_unload_trigger_min or the number of loaded models is less than max_loaded, all of the started models in the group show in /v1/models
* The auto-unload and idle_unload_trigger_min are separate from each other and each operate independently.

### Guidelines

* Follow AGENTS.md
* Update this prompt file as the plan progresses. Mark steps as complete and add any additional steps or changes as they are determined.
* There do not need to be any legacy or backward compatibility considerations for this plan.
* These changes should not affect the existing functionality of the `launch` command.
* Once all steps are complete, mark the entire plan as complete.

### Steps

1. ✅ Harden config validation in app/config.py and app/hub/config.py so `jit` gates startup loading, `auto_unload` requires `jit`, and group `{max_loaded, idle_unload_trigger_min}` constraints raise early when invalid.
2. ✅ Evolve registry + handler coordination in app/core/model_registry.py and app/core/manager_protocol.py to preserve the existing idle timestamp plumbing while separating manual load/unload, auto-unload timers, and group-driven eviction paths.
	- [x] Define explicit manager protocol hooks for "manual" vs "automated" unload triggers so registry can record provenance without conflating timestamps.
	- [x] Ensure registry availability cache tracks started-not-loaded models as available while keeping stopped models hidden; update helper methods/tests accordingly.
3. ✅ Update supervisor/daemon flows in app/hub/daemon.py, app/hub/worker.py, and related controller glue so started-but-unloaded models remain visible on `/v1/models`, stopped ones never surface, and group capacity rules (429 vs longest-idle eviction) respect the 10 s availability cache.
	- [x] Ensure `HubSupervisor.start_model` always records the registry `started` flag and spins up workers even when a manager already exists so started-but-unloaded models retain visibility metadata.
	- [x] Ensure `HubSupervisor.stop_model` clears the registry `started` flag via `RegistrySyncService` so fully stopped models disappear from `/v1/models`.
4. ✅ Adjust `/v1/models` exposure and hub status reporting in app/api/endpoints.py and app/core/hub_status.py to honor the new visibility matrix without altering the confirmed idle timestamp source.
	- [x] Updated `/v1/models` endpoint to sync supervisor memory state into registry before filtering, ensuring group availability logic sees current VRAM residency.
	- [x] Changed filtering logic to use `get_available_model_ids()` which respects `started` flag, making all started models visible in `/v1/models` listing.
	- [x] Confirmed `build_group_state()` already correctly counts only loaded models while including all group members regardless of load state.
	- [x] Added comprehensive tests (`test_v1_models_filters_based_on_supervisor_memory_loaded`, `test_v1_models_hides_stopped_models`) to verify visibility matrix.
5. ✅ Ensure CLI + hub web routes (app/cli.py, app/api/hub_routes.py) allow manual load/unload regardless of JIT, surfacing validation errors when policies are violated.
	- [x] Verified CLI commands (`hub load-model`, `hub unload-model`) call daemon API and surface 429 errors when group policies are violated.
	- [x] Confirmed hub routes (`/hub/models/{model}/load`, `/hub/models/{model}/unload`) call `_guard_hub_action_availability` which enforces group constraints for load actions.
	- [x] Ran tests (`test_cli_hub_actions.py`, `test_control_plane.py`) to confirm proper validation and error surfacing.
6. ✅ Add a dedicated "VRAM lifecycle" subsection to README.md (linking to CLI flags and hub policies) and mirror key points in docs/HUB_MODE.md plus docs/HUB_LIFECYCLE_MAPPING.md; expand lifecycle/unit tests (e.g., tests/test_hub_model_lifecycle.py, tests/test_routes_by_mode.py) to cover visibility, eviction ordering, and error scenarios.
	- [x] Added comprehensive VRAM lifecycle documentation to README.md with detailed visibility rules, configuration validation, and group behavior examples
	- [x] Expanded docs/HUB_MODE.md with detailed scenarios for groups with/without idle_unload_trigger_min
	- [x] Updated docs/HUB_LIFECYCLE_MAPPING.md with implementation notes documenting the completed steps and key principles

---

## Plan Complete ✅

All steps have been completed successfully. The VRAM lifecycle is now stable and well-documented with:

- Strict configuration validation ensuring JIT gates auto-unload and group constraints are properly enforced
- Clear separation of started/stopped (visibility) vs loaded/unloaded (VRAM residency)
- Group capacity enforcement with optional intelligent eviction based on idle thresholds
- Comprehensive documentation in README.md, HUB_MODE.md, and HUB_LIFECYCLE_MAPPING.md
- Full test coverage of visibility matrix, group behavior, and eviction scenarios

The desired functionality has been achieved with no backward compatibility concerns affecting the `launch` command.
