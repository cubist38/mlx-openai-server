# Temp steps to resume Hub Daemon migration

This file captures the concrete steps required to migrate the existing multi-process hub work into a single-daemon FastAPI supervisor (the "hub daemon"). Use this as an implementation checklist for an agent applying the change set.

Summary
- Goal: Add a single FastAPI-based daemon that owns process supervision and runtime state, and update the CLI to control it via HTTP. No backwards compatibility shims will be kept.
- Entrypoint: `app/hub/daemon.py` (FastAPI app factory `create_app`).

Required edits (high-level)
- Create `app/hub/daemon.py` implementing `HubSupervisor` and FastAPI endpoints: `GET /health`, `GET /hub/status`, `POST /hub/reload`, `POST /hub/shutdown`, `POST /hub/models/{name}/start`, `POST /hub/models/{name}/stop`, `POST /hub/models/{name}/load`, `POST /hub/models/{name}/unload`.
- Replace `app/hub/service.py` usage: remove compatibility shims. All hub control flows must call the daemon HTTP API directly via `_call_daemon_api(...)` in `app/cli.py`.
- Edit `app/cli.py` to add `_call_daemon_api(...)` helper and make CLI hub commands call the daemon HTTP API exclusively (status, start/stop model, reload, memory load/unload, watch polling).
  - Note: the CLI intentionally fails fast if the daemon is unreachable and will suggest a local start command.
    Recommended local start command for development/testing:
    - `uvicorn app.hub.daemon:create_app --host 127.0.0.1 --port 8001`
 - Add `tests/test_hub_daemon_api.py` to exercise the new endpoints and update/remove tests that import internal hub runtime modules.
 - Update `docs/HANDOFFS.md` with a short paragraph describing the new single-daemon approach and the canonical API entrypoint.

Step-by-step plan
- [x] 1) Create the daemon scaffold
  - File: `app/hub/daemon.py` (see scaffold in the main task description).
  - Verify locally:
    - Quick import check: `python -c "import app.hub.daemon; print('ok')"`
    - Start with uvicorn: `uvicorn app.hub.daemon:create_app --host 127.0.0.1 --port 8001 --reload`
    - Hit health: `curl http://127.0.0.1:8001/health`

- [x] 2) Remove compatibility shim
  - Delete `app/hub/service.py` so it no longer provides a compatibility layer.
  - Verify: `python -c "import app.hub.service; print('ok')"` should either fail with a clear message during migration or the module should document the new requirement.

- [x] 3) CLI updates
  - Add `_call_daemon_api(config, method, path, json=None, timeout=5.0)` helper to `app/cli.py`.
  - Replace direct internal hub calls / IPC client usages with calls to the daemon using this helper. Do not implement a fallback path — fail fast if the daemon is unreachable.
  - Verify: run `pytest tests/test_cli_hub_actions.py tests/test_cli_hub_status.py` (or targeted CLI tests).

 - [x] 4) Tests
  - Add integration test `tests/test_hub_daemon_api.py` that:
    - Starts the FastAPI app via test client (or uses `httpx.AsyncClient(app=...)`) and exercises: `/health`, `/hub/status`, `/hub/reload` (mocked), `/hub/models/{name}/start|stop|load-model|unload-model` (mock supervisor methods). (ADDED)
  - Update/remove tests that import hub internals directly (list candidate files):
    - `tests/test_hub_controller.py` (may need to be adapted)
    - `tests/test_hub_runtime.py` (if it tests internal runtime behavior)
  - Converted `tests/test_cli_hub_actions.py` and ensured `tests/test_hub_service_api.py` use daemon stubs. (DONE)
  - Run the focused test subset, then `pytest` and iterate. Note: running tests requires installing dev deps (see `pyproject.toml` [dependency-groups.dev]).

- [x] 5) Docs & handoff
  - Update `docs/HANDOFFS.md` with this brief migration note and point to `app/hub/daemon.py` as the new entrypoint.
  - Keep a backup branch before large edits: `git switch -c backup/hub-migration-before-daemon`
  - Update `README.md` to document the CLI changes (hub commands now call the daemon HTTP API), include the uvicorn example for local testing, and note the daemon-only architecture.

Verification checklist
 - `uvicorn app.hub.daemon:create_app --host 127.0.0.1 --port 8001` runs and `GET /health` returns 200.
 - `GET /hub/status` returns a well-formed JSON snapshot (timestamp + models list) even when all models are unloaded.
 - CLI commands that previously used the hub IPC now call the daemon successfully (run specific CLI tests).
 - New `tests/test_hub_daemon_api.py` passes.

Current status
 - Daemon scaffold: implemented (`app/hub/daemon.py`).
 - CLI: updated to call daemon HTTP API via `_call_daemon_api` (`app/cli.py`).
 - Legacy shim: compatibility shim removed and tests updated to avoid importing it.
 - Tests: integration test added (`tests/test_hub_daemon_api.py`); CLI/api tests converted to use daemon stubs.
 - Dev deps: `pyproject.toml` updated to include `pytest-asyncio`, `pytest-httpx`, and `httpx` in the `dev` dependency group; run `pip install -e ".[dev]"` to install.

Remaining recommended steps
 - Run `.\.venv\Scripts\python.exe -m pip install -e ".[dev]"` and run the test suite locally to validate changes end-to-end.
 - Review `app/hub/` modules (`controller.py`, `manager.py`, `runtime.py`, `server.py`, `observability.py`) and decide which files to remove or refactor now that the daemon owns runtime state.
 - Update `docs/HUB_MODE.md` and `docs/HUB_OBSERVABILITY.md` to reference the daemon API, or consolidate observability docs into a single file.
 - Consider adding CI steps to install `.[dev]` and run the focused tests to prevent regressions.

Notes & considerations
 - Host binding: the daemon uses `config.host` (no enforced loopback). Operators are responsible for network security.
 - Long-running work: the daemon endpoints should schedule long-running tasks in background and return promptly. Use asyncio subprocesses for model processes.
 - No back-compat: calls that previously relied on in-process hub APIs must be ported to use the daemon HTTP API. `app/hub/service.py` will no longer be a compatibility shim.
 - When making process-level changes, ensure Windows compatibility (use `terminate()` on asyncio subprocesses and handle `SIGTERM` accordingly).

 Files to create/modify (summary)
 - Create: `app/hub/daemon.py`, `tests/test_hub_daemon_api.py`, `TEMP_PROJECT_STEPS.md`
 - Modify: `app/cli.py`, `docs/HANDOFFS.md`

Rollback guidance
 - If changes cause regressions, checkout the backup branch and open a new PR with incremental changes. Keep patches small and test after each edit.

Contact
 - Implementation agents should reference this file before making edits and add short PR descriptions explaining what changed and why.

Files safe to remove after migration
 - When the daemon is fully implemented and all callers updated to use the HTTP API, the following files in `app/hub/` are candidates for deletion. Before removing, run a global grep for their symbols and ensure no remaining imports reference them.
 - Candidate removals (paths):
   - `app/hub/controller.py` — previously drove multi-process orchestration; its responsibilities move to the daemon `HubSupervisor`.
   - `app/hub/manager.py` — legacy manager code for process/worker lifecycle now handled by the daemon.
   - `app/hub/runtime.py` — runtime reservation and slot accounting consolidated into the daemon; keep only if needed for separate unit tests.
   - `app/hub/server.py` — prior hub server helpers and lifespan wiring; replaced by the daemon app factory.
   - `app/hub/service.py` — will be removed or left as an explicit failure module per migration plan.
   - `app/hub/observability.py` — optionally move useful helpers into a shared `app/hub/observability.py` or central observability module; remove if fully ported.
   - `app/hub/errors.py` — if all error types are migrated into the daemon module or central `app/utils/errors.py`, this file can be removed.
   - `app/hub/__init__.py` — keep or simplify to expose only the new daemon factory; remove exports that reference deleted modules.
 - Evaluate existing hub docs
   - Files: `docs/HUB_MODE.md`, `docs/HUB_OBSERVABILITY.md`.
   - Action: review both files and decide per-file whether to:
     - Update: revise content to reference the daemon API, move actionable observability steps into a centralized observability doc, and adjust examples to use `/hub/*` endpoints.
     - Merge: combine overlapping observability guidance into a single `docs/HUB_OBSERVABILITY.md` or `docs/OBSERVABILITY.md` and remove duplicates.
     - Remove: delete files that are obsolete after migration (ensure no links reference them in README/docs). Keep a backup branch prior to deletion.
 - Verification:
   - Search for references: `git grep -n "HUB_MODE.md\|HUB_OBSERVABILITY.md"` and update links before deleting.
   - Run `pytest` and manual smoke tests to ensure documentation edits do not break automated doc checks (if any).

Verification before delete
 - Run: `git grep -n "app.hub.controller\|from app.hub.controller\|import app.hub.controller"` to find remaining references.
 - Run the full test suite: `pytest` and verify no import errors or missing symbols.
 - Optionally keep the removed files in a backup branch for at least one merge cycle before permanently deleting from `main`.
