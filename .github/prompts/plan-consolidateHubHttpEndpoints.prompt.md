TL;DR

Consolidate `/hub/...` handlers so there's a single canonical implementation (`app/api/hub_routes.py::hub_router`) and ensure process-mode controls which routes are exposed:
- `launch` mode: expose only OpenAI `/v1/...` endpoints.
- `hub` (daemon) mode: expose both `/v1/...` and `/hub/...` (mount `hub_router`).

Goals

- One handler per HTTP path (no duplicate route functions).
- Minimal, local changes: wire routing via `configure_fastapi_app` and update callers.
- No legacy flags; use which app factory is run to decide exposure.
- Run the full test suite after changes.

Planned Changes

1. `configure_fastapi_app(app, *, include_hub_routes: bool = False)`
   - Always register inference router(s) (routes in `app/api/endpoints.py`).
   - If `include_hub_routes` is True, include `hub_router` from `app/api/hub_routes.py`.
   - Update the function signature and callers.

2. `app/hub/daemon.py`
   - Remove local duplicate route handlers for `/hub/...` that were defined directly on the daemon `app` (delete the `@app.get` / `@app.post` duplicates).
   - In `create_app`, set up `app.state.supervisor` (as now) and call `configure_fastapi_app(app, include_hub_routes=True)` so the daemon exposes both `/v1` and the canonical `/hub` router.

3. Main/launch server
   - Ensure the main server startup calls `configure_fastapi_app(app, include_hub_routes=False)` so only `/v1/...` endpoints are mounted in launch mode.

4. Tests
   - Add `tests/test_routes_launch_mode.py`:
     - Create FastAPI app with `include_hub_routes=False` and assert `/v1/models` present and `/hub/status` absent (NotFound).
   - Add `tests/test_routes_hub_mode.py`:
     - Create daemon app with `include_hub_routes=True` and assert both `/v1/models` and `/hub/status` present and respond with expected types/schemas.
   - Run the whole test suite and fix any failing tests that assumed duplicate handlers.

5. Docs
   - Update `README.md` and `docs/HUB_MODE.md` to document the new behavior: launch exposes `/v1/...` only; hub daemon exposes `/v1/...` + `/hub/...`.

Test Plan

- Local quick checks (unit tests):
  - `tests/test_routes_launch_mode.py` should assert `404` or route-not-registered for `/hub/status` in launch mode.
  - `tests/test_routes_hub_mode.py` should assert `200` / valid schema for `/hub/status` in hub mode.
- Full test run:
  - `./.venv/bin/python -m pytest tests/`

Implementation Notes

- `hub_router` already contains logic to choose between an in-process controller (`app.state.hub_controller` / `supervisor`) and proxying to an external daemon via `_call_daemon_api_*`. Keep this logic unchanged; it becomes the single canonical implementation.
- Ensure `app.state.supervisor` is available to handlers when mounting `hub_router` in the daemon app. The router expects to find the supervisor on `request.app.state` (already satisfied by `create_app`).
- Removing duplicate handlers from `app/hub/daemon.py` will eliminate the 2-handlers-per-path problem; tests and CLI that previously assumed daemon-local functions should instead call the canonical `/hub/*` endpoints mounted on the daemon.

Patch Sketch (high-level diffs)

- app/server.py
  - def configure_fastapi_app(app: FastAPI, *, include_hub_routes: bool = False) -> None:
    - register inference routers
    - if include_hub_routes: app.include_router(hub_router)

- app/hub/daemon.py
  - remove duplicated `@app.get("/hub/status")`, `@app.post("/hub/...")` handlers
  - after setting `app.state.supervisor = supervisor`, call `configure_fastapi_app(app, include_hub_routes=True)`

- main server startup (where configure_fastapi_app is called for launch)
  - call `configure_fastapi_app(app, include_hub_routes=False)`

Files to Edit

- modify: `app/server.py` (configure_fastapi_app signature and router wiring)
- modify: `app/hub/daemon.py` (remove duplicate handlers; call configure_fastapi_app include_hub_routes=True)
- modify: main server startup entry (where configure_fastapi_app is invoked for launch mode)
- add: `tests/test_routes_launch_mode.py`
- add: `tests/test_routes_hub_mode.py`
- update docs: `README.md`, `docs/HUB_MODE.md`

Delivery & Next Steps

I created this plan file as requested. Next I can:
- Draft the exact patch (apply_patch diffs) implementing the wiring and removing duplicates, then run tests; or
- Draft the two test files first and run tests to see baseline failures.

Which step should I take next? (I recommend drafting the code patch, then running the full test suite.)
