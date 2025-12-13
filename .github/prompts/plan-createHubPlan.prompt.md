Plan

Split the "Create hub" PR into 7 sequential, backward-compatible PRs. Each PR is focused, includes targeted tests, and preserves existing single-model launch behavior if later PRs are not yet merged.

TL;DR: Add hub pieces incrementally (core → API → CLI → dashboard → policies → config/docs → tests). Keep hub disabled by default and register new runtime/CLI/API only behind guards so existing functionality continues to work if some PRs are pending.

Steps

1. PR 1 — Core Hub Infrastructure (branch: feature/hub-core)
   - Add hub internals only (non-default activation).
   - Files: `app/hub/__init__.py`, `app/hub/daemon.py`, `app/hub/worker.py`, `app/core/model_registry.py`, `app/core/manager_protocol.py`.
   - Tests: unit tests for registry and daemon.
   - Backward-compat: do not alter `app/main.py` startup path; expose `HubDaemon` but do not start unless explicit flag present.

2. PR 2 — API & Endpoints (readonly/status) (branch: feature/hub-api)
   - Add `/hub` endpoints and status routes that are safe to mount without changing existing routes.
   - Files: `app/api/hub_routes.py`, `app/api/endpoints.py`, `app/server.py`.
   - Tests: endpoint unit tests and mocked registry.
   - Backward-compat: register routes under `/hub/*` and guard handlers with `if hub_enabled():` so default API surface is unchanged.

3. PR 3 — CLI Enhancements (branch: feature/hub-cli)
   - Add hub CLI subcommands, keep old CLI commands unchanged.
   - Files: `app/cli.py`, `app/main.py`.
   - Tests: `tests/test_cli_hub_actions.py`, `tests/test_cli_hub_status.py`.
   - Backward-compat: new commands are additive; ensure help text and existing commands behave the same when hub disabled.

4. PR 4 — Web Dashboard (read-only) (branch: feature/hub-dashboard)
   - Add status UI and static assets that consume the status endpoints.
   - Files: `templates/hub_status.html.jinja`, `scripts/llm_health_dashboard.py`.
   - Tests: UI-render unit tests (server-side).
   - Backward-compat: dashboard links only shown if hub enabled; static files do not change default server behavior.

5. PR 5 — Availability, Scheduling & Policies (branch: feature/hub-policies)
   - Introduce group capacity, idle eviction, VRAM scheduling and associated logic.
   - Files: `app/core/model_registry.py` (policy extensions), `app/core/queue.py` or related scheduler files.
   - Tests: `tests/test_central_idle_controller.py`, `tests/test_registry_sync.py`.
   - Backward-compat: keep policies disabled by default; add policy hooks that are no-ops unless `hub.policy.enabled` true.

6. PR 6 — Configuration & Docs (branch: feature/hub-config-docs)
   - Add configuration keys, docs, and README updates.
   - Files: `app/config.py`, `README.md`, `docs/*`.
   - Tests: `tests/test_hub_config.py`.
   - Backward-compat: new config keys default to values preserving single-model behavior (`hub.enabled: false`).

7. PR 7 — Tests & Integration (branch: feature/hub-tests)
   - Add/enable broader integration and lifespan tests covering full hub flows.
   - Files: `tests/test_hub_runtime.py`, `tests/test_hub_daemon.py`, `tests/test_hub_model_lifecycle.py`, other new tests.
   - Backward-compat: tests should run but skip full-system hub integration unless `HUB_TESTS=true` or config enables hub.

Per-PR compatibility rules (apply to every PR)
- Feature gate: Add a single runtime check `hub_enabled()` that reads a config/env var; default false.
- Non-invasive registration: Only mount new routes under `/hub/*` and only register long-running background tasks when `hub_enabled()` returns true.
- API backward-compatibility: Do not change existing endpoint shapes/paths used by single-model mode.
- CLI safety: Make CLI additions additive; preserve argument parsing and exit codes for existing commands.
- Incremental tests: Each PR includes unit tests for new code and regression tests that assert single-model behavior remains unchanged.
- Documentation: Each PR’s description must document the compatibility approach and the feature gate.

Further Considerations
- Branch names & ordering: Use the sequential branch names above and open PRs in order (merge PR 1 → 2 → 3 → ...).
- Cherry-pick approach: Cherry-pick commits by logical group; if a file is tightly coupled across groups, include a small compatibility shim in the earlier PR and the full change later.
- CI gate for hub tests: Run hub integration tests only when hub config enabled or in a separate CI job to avoid failing main CI when hub is not merged.

Next steps
- Review the plan file and confirm grouping or any files you'd prefer moved between PRs.
- When ready, I can prepare the branch/commit split steps and provide exact cherry-pick commands.
