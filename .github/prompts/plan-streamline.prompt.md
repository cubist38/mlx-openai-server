Plan Part A: Streamline Hub Lifecycle

Refactor hub lifecycle into shared, modular utilities so daemon, CLI, and server modes stay in sync while reducing duplicated registry/worker logic. Focus on extracting reusable services from app/hub/daemon.py, aligning them with server.py abstractions, and tightening documentation/tests to reflect the single-daemon architecture.

Steps

1. Map overlapping lifecycle paths between daemon.py HubSupervisor and server.py:1 bootstrap, documenting shared needs (registry updates, worker start/stop, idle controller wiring).

2. Extract registry/worker helper functions or classes (e.g., SidecarWorkerCoordinator, RegistrySyncService) into a shared module under core and replace call sites in both daemon/server code.

3. Create a hub service interface (e.g., HubLifecycleService) that encapsulates start/load/unload/stop/status; update hub routes in endpoints.py and CLI flows in cli.py to call it instead of bespoke logic.

4. Consolidate status snapshot building so both daemon UI and API reuse one formatter (new helper leveraged by daemon.py:1400 and server routes), ensuring unload timestamps & group data stay consistent.

5. Reconcile documentation/tests: update HUB_MODE.md plus hub-focused tests (test_hub_daemon.py, test_hub_service.py) to reflect the new shared services and verify idle controller/registry interactions.


Plan Part B: Retire Hub Service Shim

Remove the unused hub service HTTP shim so UI, CLI, and tests rely solely on the daemon/controller surface, simplifying maintenance and aligning docs with actual behavior.

Steps

1. Delete shim routes (/hub/service/...) and related helpers from hub_routes.py:1, ensuring dashboard buttons shift to existing daemon endpoints (/hub/models/..., /hub/runtime/...).

2. Drop unused shim schemas (e.g., HubServiceResponse) from openai.py:1 or whichever module defines them, updating imports where removed routes referenced them.

3. Rewrite dashboard JS in hub_status.html.jinja:1 to call the daemon routes (start/stop/load/unload/reload/status) and adjust any “service” terminology to “daemon”.

4. Replace test_hub_service.py:1 with daemon-focused coverage: ensure FastAPI hub app still exposes model control endpoints and background tasks fire as expected; migrate any surviving supervisor tests into a renamed module (e.g., test_hub_daemon_routes.py).

5. Update docs mentioning HubService (HUB_MODE.md:1, HANDOFFS.md:1, any README sections) to describe the daemon-only architecture and remove references to IPC sockets or service clients.

Plan Part C: Remove HubService Shim

Delete /hub/service/* endpoints now, rewire UI/tests/docs to the daemon APIs, and ensure helper utilities (*_shutdown) plus schemas shift into their new homes so only the canonical lifecycle routes remain.

Steps

1. Remove shim routes/controllers from hub_routes.py:1 and delete any HubServiceResponse schema usage in openai.py:1, keeping only daemon endpoints (/hub/models, /hub/runtime, etc.).

2. Inline or relocate schedule_shutdown() from hub_routes.py:350 into the remaining daemon route module so background shutdown still works without the shim.

3. Update dashboard calls in hub_status.html.jinja:1 to point buttons at /hub/models/{name}/start|stop|load|unload and /hub/runtime/reload, adjusting any “service” terminology.

4. Replace test_hub_service.py:1 with daemon-focused tests that hit the surviving routes (status, start/stop/load/unload, reload) and verify schedule_shutdown() still runs via FastAPI background tasks.

5. Clean up docs referencing HubService in HUB_MODE.md:1, HANDOFFS.md:1, and the README hub section, explaining that only the daemon HTTP API remains.
