Plan Part A: Streamline Hub Lifecycle

Refactor hub lifecycle into shared, modular utilities so daemon, CLI, and server modes stay in sync while reducing duplicated registry/worker logic. Focus on extracting reusable services from app/hub/daemon.py, aligning them with server.py abstractions, and tightening documentation/tests to reflect the single-daemon architecture.

Steps

1. ✅ Map overlapping lifecycle paths between daemon.py HubSupervisor and server.py bootstrap, documenting shared needs (registry updates, worker start/stop, idle controller wiring). Captured in docs/HUB_LIFECYCLE_MAPPING.md.

2. ✅ Extract registry/worker helper functions or classes (e.g., SidecarWorkerCoordinator, RegistrySyncService) into a shared module under core and replace call sites in both daemon/server code. RegistrySyncService/WorkerTelemetry now power app/hub/daemon.py and app/server.py.

3. ✅ Create a hub service interface (HubLifecycleService + HubServiceAdapter) and helper (get_hub_lifecycle_service) so hub_routes.py and endpoints.py resolve controllers via the shared surface; CLI continues to hit the daemon HTTP API.

4. ✅ Consolidate status snapshot building so both daemon UI and API reuse one formatter (new core/hub_status.py helper feeds HubSupervisor.get_status and /hub/status), keeping unload timestamps & group metadata consistent.

5. ✅ Reconcile documentation/tests: HUB_MODE.md now documents the shared helpers, hub daemon tests assert group summaries, and new hub_service/core_status unit tests cover the adapters and formatters.


Plan Part B: Retire Hub Service Shim

Remove the unused hub service HTTP shim so UI, CLI, and tests rely solely on the daemon/controller surface, simplifying maintenance and aligning docs with actual behavior.

Steps

1. ✅ Remove shim routes (/hub/service/...) and related helpers from hub_routes.py so the dashboard now calls the canonical daemon endpoints (reload via /hub/reload, shutdown via /hub/shutdown, per-model actions still under /hub/models/*).

2. ✅ Drop unused shim schemas (HubServiceResponse) from app/schemas/openai.py and clean up imports so only daemon response types remain referenced.

3. ✅ Update hub_status.html.jinja so dashboard buttons request the daemon endpoints directly (reload/shutdown + existing /hub/models actions) and rename all “service” hooks/messages to “daemon”.

4. ✅ Replaced test_hub_service.py with daemon-focused coverage: new tests/test_hub_daemon_routes.py drives /hub/models, /hub/reload, and /hub/shutdown (including background tasks) so the server API stays in sync with the daemon controller surface.

5. ✅ Updated HUB_MODE.md, HANDOFFS.md, HUB_LIFECYCLE_MAPPING.md, and the README hub section so they describe the daemon-only HTTP control plane (no `/hub/service/*`, no IPC shim) and link operators directly to `/hub/reload`, `/hub/shutdown`, and `/hub/models/*` endpoints.

Plan Part C: Remove HubService Shim

Delete /hub/service/* endpoints now, rewire UI/tests/docs to the daemon APIs, and ensure helper utilities (*_shutdown) plus schemas shift into their new homes so only the canonical lifecycle routes remain.

Steps

1. ✅ Shim routes/controllers already removed from hub_routes.py and HubServiceResponse was deleted from app/schemas/openai.py, leaving only the canonical daemon endpoints (/hub/models, /hub/reload, etc.).

2. ✅ hub_shutdown now schedules controller teardown and optional daemon exit inline, so the old helper is gone and background shutdown continues to work without the shim.

3. ✅ Dashboard calls (templates/hub_status.html.jinja) already target /hub/models/* and use "daemon" terminology per Part B step 3.

4. ✅ Legacy tests/test_hub_service.py was removed; daemon route coverage (including background shutdown scheduling) now lives entirely in tests/test_hub_daemon_routes.py.

5. ✅ HUB_MODE.md, HANDOFFS.md, HUB_LIFECYCLE_MAPPING.md, and the README hub section explain the daemon-only HTTP API (no HubService shim).
