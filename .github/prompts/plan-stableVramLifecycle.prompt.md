## Plan: Stable VRAM Lifecycle

Clarify and enforce the VRAM lifecycle so started-but-unloaded models stay visible while idle eviction, JIT, and auto-unload interact predictably. Central tasks: While keeping the current idle timestamp source, tighten config validation, update registry/state machines to respect the new rules, adjust availability filtering and CLI/web controls, and expand regression tests to pin the desired UX. Add a README-level “VRAM lifecycle” subsection and syncing supporting docs. Work spans stricter config validation, registry/controller enforcement, API visibility, manual controls, documentation, and tests.

### Instructions

* Follow AGENTS.md
* Update this prompt file as the plan progresses. Mark steps as complete and add any additional steps or changes as they are determined.
* There do not need to be any legacy or backward compatibility considerations for this plan.
* These changes should not affect the existing functionality of the `launch` command.
* Once all steps are complete, mark the entire plan as complete.

### Steps
1. Harden config validation in [app/config.py](app/config.py) and [app/hub/config.py](app/hub/config.py) so `jit` gates startup loading, `auto_unload` requires `jit`, and group `{max_loaded, idle_unload_trigger_min}` constraints raise early when invalid.
2. Evolve registry + handler coordination in [app/core/model_registry.py](app/core/model_registry.py) and [app/handler/manager_protocol.py](app/handler/manager_protocol.py) to preserve the existing idle timestamp plumbing while separating manual load/unload, auto-unload timers, and group-driven eviction paths.
3. Update supervisor/daemon flows in [app/hub/daemon.py](app/hub/daemon.py), [app/hub/worker.py](app/hub/worker.py), and related controller glue so started-but-unloaded models remain visible on `/v1/models`, stopped ones never surface, and group capacity rules (429 vs longest-idle eviction) respect the 10 s availability cache.
4. Adjust `/v1/models` exposure and hub status reporting in [app/api/endpoints.py](app/api/endpoints.py) and [app/core/hub_status.py](app/core/hub_status.py) to honor the new visibility matrix without altering the confirmed idle timestamp source.
5. Ensure CLI + hub web routes ([app/cli.py](app/cli.py), [app/api/hub_routes.py](app/api/hub_routes.py)) allow manual load/unload regardless of JIT/auto-unload, surfacing validation errors when policies are violated.
6. Add a dedicated “VRAM lifecycle” subsection to [README.md](README.md) (linking to CLI flags and hub policies) and mirror key points in [docs/HUB_MODE.md](docs/HUB_MODE.md) plus [docs/HUB_LIFECYCLE_MAPPING.md](docs/HUB_LIFECYCLE_MAPPING.md); expand lifecycle/unit tests (e.g., [tests/test_hub_model_lifecycle.py](tests/test_hub_model_lifecycle.py), [tests/test_routes_by_mode.py](tests/test_routes_by_mode.py)) to cover visibility, eviction ordering, and error scenarios.

### Further Considerations
1. Documentation layout: confirm whether the README section should sit under “Key Features” or a new “VRAM lifecycle” heading near the existing JIT/auto-unload coverage, then cross-link HUB docs for deeper operator detail.
