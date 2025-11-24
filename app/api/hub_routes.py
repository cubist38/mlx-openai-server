"""Dedicated FastAPI routes and helpers for hub-specific functionality."""

from __future__ import annotations

import asyncio
import contextlib
from http import HTTPStatus
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Literal, cast

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
import httpx
from loguru import logger

from ..const import (
    DEFAULT_API_HOST,
    DEFAULT_BIND_HOST,
    DEFAULT_HUB_CONFIG_PATH,
    DEFAULT_MODEL_STARTING_PORT,
)
from ..hub.config import PORT_MAX, HubConfigError, MLXHubConfig, _is_port_available, load_hub_config
from ..schemas.openai import (
    HubModelActionRequest,
    HubModelActionResponse,
    HubServiceActionResponse,
    HubStatusCounts,
    HubStatusResponse,
    Model,
)
from ..utils.errors import create_error_response


class HubServiceError(RuntimeError):
    """Raised when the daemon reports a service-level failure.

    Parameters
    ----------
    message : str
        Human-friendly error message.
    status_code : int | None
        Optional HTTP status code associated with the error.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def start_hub_service_process(
    config_path: str, *, host: str | None = None, port: int | None = None
) -> int:
    """Start the hub daemon process (development helper).

    This helper launches a uvicorn process in the background using the same
    Python interpreter. It returns the spawned PID. It is intended as a
    development convenience for the API's `/hub/service/start` endpoint.
    """
    host_val = host or DEFAULT_BIND_HOST
    starting_port = port or DEFAULT_MODEL_STARTING_PORT

    # Find an available port starting from the specified port
    port_val = starting_port
    while port_val <= PORT_MAX:
        if _is_port_available(host_val, port_val):
            break
        port_val += 1
    else:
        raise HubServiceError(
            f"Unable to find an available port for hub daemon starting at {starting_port}"
        )

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.hub.daemon:create_app",
        "--factory",
        "--host",
        host_val,
        "--port",
        str(port_val),
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc.pid


hub_router = APIRouter()

_HUB_STATUS_PAGE_HTML = """<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>MLX Hub Status · MLX OpenAI Server</title>
    <style>
        :root {
            color-scheme: dark;
            font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background-color: #0f1115;
            color: #f5f5f7;
        }
        body {
            margin: 0;
            padding: 24px;
            line-height: 1.5;
            background: linear-gradient(135deg, #0f1115 0%, #13161d 100%);
            min-height: 100vh;
        }
        header {
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-bottom: 24px;
        }
        header h1 {
            font-size: 1.75rem;
            margin: 0;
        }
        header p {
            margin: 0;
            color: #b0b5c0;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 999px;
            padding: 4px 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .pill--ok { color: #7af79a; }
        .pill--warn { color: #f7d67a; }
        .pill--error { color: #f77a7a; }
        .card {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 16px;
            box-shadow: 0 20px 45px rgba(0, 0, 0, 0.35);
            backdrop-filter: blur(8px);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 16px;
        }
        th, td {
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }
        th {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #9aa0af;
        }
        tbody tr:hover {
            background: rgba(255, 255, 255, 0.03);
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 16px;
        }
        .muted {
            color: #8b92a3;
        }
        .warnings {
            border-left: 3px solid #f7d67a;
            padding-left: 12px;
            margin-top: 12px;
        }
        .warnings ul {
            margin: 4px 0 0;
            padding-left: 20px;
        }
        .error-banner {
            background: rgba(247, 122, 122, 0.12);
            border: 1px solid rgba(247, 122, 122, 0.35);
            color: #f7b6b6;
            padding: 12px 16px;
            border-radius: 12px;
            margin-bottom: 16px;
        }
        button {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: inherit;
            border-radius: 12px;
            padding: 8px 16px;
            cursor: pointer;
            font-weight: 600;
        }
        button:hover {
            background: rgba(255, 255, 255, 0.12);
        }
        .actions {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        .action-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
            min-width: 140px;
        }
        .action-group__label {
            font-size: 0.7rem;
            text-transform: uppercase;
            color: #9aa0af;
            letter-spacing: 0.08em;
        }
        .action-group__buttons {
            display: flex;
            gap: 6px;
        }
        .action-btn--secondary {
            background: transparent;
            border-color: rgba(255, 255, 255, 0.25);
        }
        .action-btn[disabled] {
            opacity: 0.4;
            cursor: not-allowed;
        }
        .toast {
            position: fixed;
            bottom: 24px;
            right: 24px;
            padding: 12px 16px;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(12px);
            min-width: 220px;
            text-align: center;
        }
        .toast--success { color: #7af79a; border-color: rgba(122, 247, 154, 0.4); }
        .toast--error { color: #f77a7a; border-color: rgba(247, 122, 122, 0.4); }
        .flash {
            border-radius: 12px;
            padding: 12px 16px;
            font-weight: 600;
            border: 1px solid transparent;
            margin-bottom: 16px;
        }
        .flash--info {
            background: rgba(122, 207, 247, 0.12);
            border-color: rgba(122, 207, 247, 0.4);
            color: #a8e7ff;
        }
        .flash--success {
            background: rgba(122, 247, 154, 0.12);
            border-color: rgba(122, 247, 154, 0.35);
            color: #b2ffcb;
        }
        .flash--warn {
            background: rgba(247, 214, 122, 0.12);
            border-color: rgba(247, 214, 122, 0.35);
            color: #ffe3a6;
        }
        .flash--error {
            background: rgba(247, 122, 122, 0.12);
            border-color: rgba(247, 122, 122, 0.35);
            color: #ffc1c1;
        }
        .stat-block {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .stat-line {
            font-size: 1.25rem;
            font-weight: 600;
        }
        .stat-line--subtle {
            font-size: 1rem;
            color: #b0b5c0;
        }
    </style>
</head>
<body>
    <header>
        <h1>MLX Hub Status • MLX OpenAI Server</h1>
        <p>Live snapshot of registered models, groups, and controller health.</p>
    </header>

    <div id=\"error-banner\" class=\"error-banner\" hidden></div>
    <div id="flash-banner" class="flash flash--info" hidden></div>
    <div id=\"toast\" class=\"toast\" hidden></div>

    <section class="card">
        <div class="grid">
            <div class="stat-block">
                <div class="muted">Hub status</div>
                <div id="status-pill" class="pill pill--warn">Loading…</div>
            </div>
            <div class="stat-block">
                <div class="muted">Processes</div>
                <div id="started-counts" class="stat-line">—</div>
                <div id="counts" class="stat-line">—</div>
            </div>
            <div class="stat-block">
                <div class="muted">Last updated</div>
                <div id="updated-at" class="stat-line">—</div>
                <div class="muted" style="margin-top: 8px;">OpenAI URL</div>
                <div id="openai-url" class="stat-line">—</div>
            </div>
            <div class="stat-block">
                <div class="muted">Controls</div>
                <div class="actions">
                    <button class="refresh-btn" type="button">Refresh</button>
                    <button class="action-btn action-btn--secondary" type="button" data-service-action="reload">Reload</button>
                    <button class="action-btn action-btn--secondary" type="button" data-service-action="stop">Stop</button>
                </div>
            </div>
        </div>
        <div id="warnings" class="warnings" hidden>
            <strong>Warnings</strong>
            <ul id="warning-list"></ul>
        </div>
    </section>

    <section class=\"card\">
        <div class=\"muted\" style=\"margin-bottom: 8px;\">Registered Models</div>
        <table>
            <thead>
                <tr>
                    <th>Name</th>
                    <th>Process</th>
                    <th>Memory</th>
                    <th>Auto-Unload</th>
                    <th>Type</th>
                    <th>Group</th>
                    <th>Default</th>
                    <th>Model</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody id=\"models-body\">
                <tr>
                    <td colspan=\"8\" class=\"muted\">Fetching hub snapshot…</td>
                </tr>
            </tbody>
        </table>
    </section>

    <script>
        const REFRESH_INTERVAL_MS = 5000;
        const SERVICE_ENDPOINTS = {
            start: '/hub/service/start',
            stop: '/hub/service/stop',
            reload: '/hub/service/reload',
        };
        const CONTROLLER_WARNING_MESSAGE = 'Hub controller is not running on this server. Memory buttons stay disabled until the hub server is up.';
        let controllerAvailable = false;

        function escapeHtml(value) {
            return String(value ?? '')
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/\"/g, '&quot;')
                .replace(/'/g, '&#39;');
        }

        function formatTimestamp(ts) {
            if (!ts) {
                return '—';
            }
            try {
                return new Intl.DateTimeFormat(undefined, {
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                }).format(new Date(ts * 1000));
            } catch (err) {
                return new Date(ts * 1000).toLocaleTimeString();
            }
        }

        function setStatusPill(value) {
            const pill = document.getElementById('status-pill');
            const map = {
                ok: 'pill--ok',
                degraded: 'pill--warn',
                error: 'pill--error',
            };
            pill.textContent = value;
            pill.className = `pill ${map[value] ?? 'pill--warn'}`;
        }

        function formatOpenAiUrl(hostValue, portValue) {
            let host = typeof hostValue === 'string' ? hostValue.trim() : '';
            if (!host) {
                return '—';
            }
            if (host === '0.0.0.0' || host === '::' || host === '[::]') {
                host = 'localhost';
            }
            if (host.includes(':') && !(host.startsWith('[') && host.endsWith(']'))) {
                host = `[${host}]`;
            }
            let port = Number(portValue);
            if (!Number.isFinite(port) || port <= 0) {
                port = 8000;
            }
            return `http://${host}:${port}/v1`;
        }

        function renderWarnings(warnings) {
            const wrapper = document.getElementById('warnings');
            const list = document.getElementById('warning-list');
            if (!warnings || warnings.length === 0) {
                wrapper.hidden = true;
                list.innerHTML = '';
                return;
            }
            wrapper.hidden = false;
            list.innerHTML = warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join('');
        }

        function describeProcess(meta) {
            const state = String(meta.process_state ?? 'inactive').toLowerCase();
            const pid = meta.pid ? `PID ${meta.pid}` : null;
            const port = meta.port ? `PORT ${meta.port}` : null;
            const exitInfo = meta.exit_code && meta.exit_code !== 0 ? `exit ${meta.exit_code}` : null;
            const pieces = [state.toUpperCase()];
            if (state === 'running') {
                if (pid) {
                    pieces.push(pid);
                }
                if (port) {
                    pieces.push(port);
                }
            } else if (exitInfo) {
                pieces.push(exitInfo);
            }
            return pieces.join(' • ');
        }

        function describeMemory(meta) {
            const state = String(meta.memory_state ?? 'unloaded').toLowerCase();
            const transition = meta.memory_last_transition_at ? formatTimestamp(meta.memory_last_transition_at) : null;
            const error = meta.memory_last_error ? `Error: ${escapeHtml(meta.memory_last_error)}` : null;
            const pieces = [state.toUpperCase()];
            if (transition) {
                pieces.push(`${transition}`);
            }
            return {
                label: pieces.join(' • '),
                error,
            };
        }

        function renderModels(models) {
            const body = document.getElementById('models-body');
            if (!models || models.length === 0) {
                body.innerHTML = '<tr><td colspan="9">No models registered.</td></tr>';
                return;
            }
            body.innerHTML = models
                .map((model) => {
                    const meta = model.metadata ?? {};
                    const processState = String(meta.process_state ?? 'inactive').toLowerCase();
                    const memoryState = String(meta.memory_state ?? 'unloaded').toLowerCase();
                    const processRunning = processState === 'running';
                    const group = meta.group ?? '—';
                    const defaultFlag = meta.default ? '✅' : '—';
                    const modelType = meta.model_type ?? 'n/a';
                    const modelPath = meta.model_path ?? '';
                    const autoUnloadMinutes = meta.auto_unload_minutes;
                    const autoUnloadLabel = Number.isFinite(autoUnloadMinutes) ? `${autoUnloadMinutes} min` : '—';
                    const safeId = escapeHtml(model.id);
                    const processLabel = describeProcess(meta);
                    const memoryDescriptor = describeMemory(meta);
                    const memoryCell = processRunning
                        ? `<div>${escapeHtml(memoryDescriptor.label)}</div>${memoryDescriptor.error ? `<div class="muted">${memoryDescriptor.error}</div>` : ''}`
                        : '<div>—</div>';
                    const processStartVisible = !processRunning;
                    const processStopVisible = processRunning;
                    const processStartDisabled = ['starting'].includes(processState);
                    const processStopDisabled = ['stopping'].includes(processState);
                    const memoryGroupVisible = controllerAvailable && processRunning;
                    const memoryLoaded = memoryState === 'loaded';
                    const memoryButtonLabel = memoryLoaded ? 'Unload' : 'Load';
                    const memoryButtonAction = memoryLoaded ? 'unload-model' : 'load-model';
                    const memoryButtonDisabled = ['loading', 'unloading'].includes(memoryState);
                    return `<tr>
                        <td>${safeId}</td>
                        <td>${escapeHtml(processLabel)}</td>
                        <td>
                            ${memoryCell}
                        </td>
                        <td>${escapeHtml(autoUnloadLabel)}</td>
                        <td>${escapeHtml(modelType)}</td>
                        <td>${escapeHtml(group)}</td>
                        <td>${escapeHtml(defaultFlag)}</td>
                        <td>${escapeHtml(modelPath)}</td>
                        <td>
                            <div class=\"actions\">
                                <div class=\"action-group\">
                                    <div class=\"action-group__label\">Process</div>
                                    <div class=\"action-group__buttons\">
                                        ${processStartVisible ? `<button class=\"action-btn\" data-scope=\"process\" data-action=\"start-model\" data-model=\"${safeId}\" ${processStartDisabled ? 'disabled' : ''}>Start</button>` : ''}
                                        ${processStopVisible ? `<button class=\"action-btn action-btn--secondary\" data-scope=\"process\" data-action=\"stop-model\" data-model=\"${safeId}\" ${processStopDisabled ? 'disabled' : ''}>Stop</button>` : ''}
                                    </div>
                                </div>
                                ${memoryGroupVisible ? `
                                <div class=\"action-group\">
                                    <div class=\"action-group__label\">Memory</div>
                                    <div class=\"action-group__buttons\">
                                        <button class=\"action-btn ${memoryLoaded ? 'action-btn--secondary' : ''}\" data-scope=\"memory\" data-action=\"${memoryButtonAction}\" data-model=\"${safeId}\" ${memoryButtonDisabled ? 'disabled' : ''}>${memoryButtonLabel}</button>
                                    </div>
                                </div>` : ''}
                            </div>
                        </td>
                    </tr>`;
                })
                .join('');
            attachModelActionListeners();
        }

        function showError(message) {
            const banner = document.getElementById('error-banner');
            banner.textContent = message;
            banner.hidden = false;
        }

        function clearError() {
            const banner = document.getElementById('error-banner');
            banner.hidden = true;
            banner.textContent = '';
        }

        function showFlash(message, tone = 'info') {
            const flash = document.getElementById('flash-banner');
            flash.textContent = message;
            flash.className = `flash flash--${tone}`;
            flash.hidden = false;
        }

        function clearFlash() {
            const flash = document.getElementById('flash-banner');
            flash.hidden = true;
            flash.textContent = '';
        }

        let toastTimeoutId;
        function showToast(message, tone = 'success') {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.className = `toast toast--${tone}`;
            toast.hidden = false;
            if (toastTimeoutId) {
                clearTimeout(toastTimeoutId);
            }
            toastTimeoutId = setTimeout(() => {
                toast.hidden = true;
            }, 4000);
        }

        function attachModelActionListeners() {
            const buttons = document.querySelectorAll('[data-action]');
            buttons.forEach((button) => {
                button.addEventListener('click', onModelActionClick);
            });
        }

        function attachServiceActionListeners() {
            const buttons = document.querySelectorAll('[data-service-action]');
            buttons.forEach((button) => {
                button.addEventListener('click', onServiceActionClick);
            });
        }

        async function onModelActionClick(event) {
            const button = event.currentTarget;
            const model = button.getAttribute('data-model');
            const action = button.getAttribute('data-action');
            const scope = button.getAttribute('data-scope') ?? 'process';
            if (!model || !action) {
                return;
            }

            const row = button.closest('tr');
            const rowButtons = row ? row.querySelectorAll('[data-action]') : [button];
            rowButtons.forEach((btn) => {
                btn.disabled = true;
            });
            const originalLabel = button.textContent;
            const pendingLabel = {
                'start-model': 'Starting…',
                'stop-model': 'Stopping…',
                'load-model': 'Loading…',
                'unload-model': 'Unloading…',
            }[action] ?? 'Working…';
            button.textContent = pendingLabel;

            try {
                const endpoint = `/hub/models/${encodeURIComponent(model)}/${action}`;
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ reason: 'dashboard' }),
                });
                const payload = await response.json().catch(() => ({}));
                if (!response.ok) {
                    const friendlyAction = action.replace(/-/g, ' ');
                    const message = payload?.error?.message ?? payload?.message ?? `Failed to ${friendlyAction} ${model}`;
                    throw new Error(message);
                }
                const scopeLabel = scope === 'memory' ? 'memory' : 'process';
                const verb = {
                    'start-model': 'Start',
                    'stop-model': 'Stop',
                    'load-model': 'Load',
                    'unload-model': 'Unload',
                }[action] ?? 'Action';
                showToast(`${verb} ${scopeLabel} request sent for ${model}`, 'success');
            } catch (error) {
                showToast(error.message ?? 'Action failed', 'error');
            } finally {
                rowButtons.forEach((btn) => {
                    btn.disabled = false;
                });
                if (button.isConnected) {
                    button.textContent = originalLabel;
                }
                try {
                    await fetchSnapshot();
                } catch (err) {
                    console.error(err);
                }
            }
        }

        async function onServiceActionClick(event) {
            const button = event.currentTarget;
            const action = button.getAttribute('data-service-action');
            if (!action || !(action in SERVICE_ENDPOINTS)) {
                return;
            }

            const endpoint = SERVICE_ENDPOINTS[action];
            const originalLabel = button.textContent;
            button.disabled = true;
            button.textContent = `${action.charAt(0).toUpperCase() + action.slice(1)}…`;
            clearFlash();

            try {
                const response = await fetch(endpoint, { method: 'POST' });
                const payload = await response.json().catch(() => ({}));
                if (!response.ok) {
                    const message = payload?.error?.message ?? payload?.message ?? `Failed to ${action} hub manager`;
                    showFlash(message, 'error');
                    return;
                }
                const tone = action === 'stop' ? 'warn' : 'success';
                const message = payload?.message ?? `Hub ${action} request accepted`;
                showFlash(message, tone);
            } catch (error) {
                showFlash(error.message ?? `Failed to ${action} hub manager`, 'error');
            } finally {
                button.disabled = false;
                button.textContent = originalLabel;
                try {
                    await fetchSnapshot();
                } catch (err) {
                    console.error(err);
                }
            }
        }

        async function fetchSnapshot() {
            try {
                const response = await fetch('/hub/status');
                if (!response.ok) {
                    throw new Error(`Request failed with status ${response.status}`);
                }
                const data = await response.json();
                controllerAvailable = Boolean(data.controller_available);
                const warningList = Array.isArray(data.warnings) ? [...data.warnings] : [];
                if (!controllerAvailable && !warningList.includes(CONTROLLER_WARNING_MESSAGE)) {
                    warningList.push(CONTROLLER_WARNING_MESSAGE);
                }
                clearError();
                setStatusPill(data.status ?? 'unknown');
                const registered = data.counts?.registered ?? 0;
                const started = data.counts?.started ?? 0;
                const loaded = data.counts?.loaded ?? 0;
                document.getElementById('started-counts').textContent = `${started} / ${registered} started`;
                document.getElementById('counts').textContent = `${loaded} / ${registered} loaded`;
                document.getElementById('updated-at').textContent = formatTimestamp(data.timestamp);
                document.getElementById('openai-url').textContent = formatOpenAiUrl(data.host, data.port);
                renderWarnings(warningList);
                renderModels(data.models ?? []);
            } catch (error) {
                showError(error.message ?? 'Failed to fetch hub status');
                setStatusPill('error');
            }
        }

        document.querySelectorAll('.refresh-btn').forEach(btn => {
            btn.addEventListener('click', fetchSnapshot);
        });
        attachServiceActionListeners();
        fetchSnapshot();
        setInterval(fetchSnapshot, REFRESH_INTERVAL_MS);
    </script>
</body>
</html>"""


def _resolve_hub_config_path(raw_request: Request) -> Path:
    """Resolve the hub configuration file path from request state.

    Parameters
    ----------
    raw_request : Request
        The incoming FastAPI request.

    Returns
    -------
    Path
        Path to the hub configuration file.
    """
    override = getattr(raw_request.app.state, "hub_config_path", None)
    if override:
        return Path(str(override)).expanduser()

    server_config = getattr(raw_request.app.state, "server_config", None)
    if isinstance(server_config, MLXHubConfig) and server_config.source_path is not None:
        return server_config.source_path

    source_path = getattr(server_config, "source_path", None)
    if source_path:
        return Path(str(source_path)).expanduser()

    return DEFAULT_HUB_CONFIG_PATH


def _stop_controller_process(config: MLXHubConfig) -> bool:
    """Request the hub daemon to stop the controller and managed processes.

    This function proxies the shutdown request to the hub daemon HTTP API
    (POST /hub/shutdown) so the controller and supervised processes are
    stopped in the daemon process. Returns True on success and False when
    the daemon reports a service-level failure or is unreachable.

    Parameters
    ----------
    config : MLXHubConfig
        The hub configuration used to determine the daemon base URL.

    Returns
    -------
    bool
        True if the controller was stopped, False otherwise.
    """
    try:
        _call_daemon_api_sync(config, "POST", "/hub/shutdown", timeout=1.0)
    except HubServiceError:
        return False
    return True


def _load_hub_config_from_request(raw_request: Request) -> MLXHubConfig:
    """Load hub configuration from the resolved config path.

    Parameters
    ----------
    raw_request : Request
        The incoming FastAPI request.

    Returns
    -------
    MLXHubConfig
        Loaded hub configuration.
    """
    return load_hub_config(_resolve_hub_config_path(raw_request))


def _daemon_base_url(config: MLXHubConfig) -> str:
    """Return the base HTTP URL for the hub daemon for the given config."""
    host = (config.host or DEFAULT_BIND_HOST).strip()
    if host in {"0.0.0.0", "::", "[::]"}:
        host = DEFAULT_API_HOST
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = config.daemon_port
    return f"http://{host}:{port}"


async def _call_daemon_api_async(
    config: MLXHubConfig,
    method: str,
    path: str,
    *,
    json: object | None = None,
    timeout: float = 5.0,
) -> dict[str, object] | None:
    """Async call to the hub daemon HTTP API and return parsed JSON.

    Raises HubServiceError on non-2xx responses.
    """
    base = _daemon_base_url(config)
    url = f"{base.rstrip('/')}{path}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(method, url, json=json)
    except Exception as exc:  # pragma: no cover - network error handling
        raise HubServiceError(f"Failed to contact hub daemon at {base}: {exc}") from exc

    if resp.status_code >= 400:
        # Try to parse JSON error
        payload: object | None
        try:
            payload = resp.json()
        except Exception:
            payload = resp.text
        raise HubServiceError(
            f"Daemon responded {resp.status_code}: {payload}", status_code=resp.status_code
        )

    if resp.content:
        try:
            payload = resp.json()
        except Exception:
            return {"raw": resp.text}
        if isinstance(payload, dict):
            return payload
        # Wrap non-dict payloads in a dict to maintain the declared return type
        return {"raw": payload}
    return None


def _call_daemon_api_sync(
    config: MLXHubConfig,
    method: str,
    path: str,
    *,
    json: object | None = None,
    timeout: float = 1.0,
) -> dict[str, object] | None:
    """Synchronous HTTP call to the hub daemon used by sync code paths.

    Raises
    ------
    HubServiceError
        On connectivity failures or non-2xx responses.
    """
    base = _daemon_base_url(config)
    url = f"{base.rstrip('/')}{path}"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.request(method, url, json=json)
    except Exception as exc:  # pragma: no cover - network error handling
        raise HubServiceError(f"Failed to contact hub daemon at {base}: {exc}") from exc

    if resp.status_code >= 400:
        try:
            payload = resp.json()
        except Exception:
            payload = resp.text
        raise HubServiceError(
            f"Daemon responded {resp.status_code}: {payload}", status_code=resp.status_code
        )

    if resp.content:
        try:
            payload = resp.json()
        except Exception:
            return {"raw": resp.text}
        if isinstance(payload, dict):
            return payload
        # Wrap non-dict payloads in a dict to maintain the declared return type
        return {"raw": payload}
    return None


def _service_error_response(action: str, exc: HubServiceError) -> JSONResponse:
    """Create a JSON error response for hub service errors.

    Parameters
    ----------
    action : str
        The action that failed.
    exc : HubServiceError
        The service error exception.

    Returns
    -------
    JSONResponse
        Formatted error response.
    """
    status = exc.status_code or HTTPStatus.SERVICE_UNAVAILABLE
    error_type = (
        "rate_limit_error" if status == HTTPStatus.TOO_MANY_REQUESTS else "service_unavailable"
    )
    return JSONResponse(
        content=create_error_response(
            f"Failed to {action} via hub manager: {exc}", error_type, status
        ),
        status_code=status,
    )


def _hub_config_error_response(reason: str) -> JSONResponse:
    """Create a JSON error response for hub configuration errors.

    Parameters
    ----------
    reason : str
        The reason for the configuration error.

    Returns
    -------
    JSONResponse
        Formatted error response.
    """
    return JSONResponse(
        content=create_error_response(
            f"Hub configuration unavailable: {reason}",
            "invalid_request_error",
            HTTPStatus.BAD_REQUEST,
        ),
        status_code=HTTPStatus.BAD_REQUEST,
    )


def _manager_unavailable_response() -> JSONResponse:
    """Create a JSON error response for unavailable hub manager.

    Returns
    -------
    JSONResponse
        Formatted error response.
    """
    return JSONResponse(
        content=create_error_response(
            "Hub manager is not running. Start it via /hub/service/start or the CLI before issuing actions.",
            "service_unavailable",
            HTTPStatus.SERVICE_UNAVAILABLE,
        ),
        status_code=HTTPStatus.SERVICE_UNAVAILABLE,
    )


def _controller_unavailable_response() -> JSONResponse:
    """Return a standardized response when the hub controller is missing.

    Returns
    -------
    JSONResponse
        Error response indicating controller unavailability.
    """
    return JSONResponse(
        content=create_error_response(
            "Hub controller is not available. Ensure the hub server is running before issuing memory actions.",
            "service_unavailable",
            HTTPStatus.SERVICE_UNAVAILABLE,
        ),
        status_code=HTTPStatus.SERVICE_UNAVAILABLE,
    )


def _controller_error_response(exc: Exception) -> JSONResponse:
    """Convert a HubControllerError into a JSON API response.

    Parameters
    ----------
    exc : HubControllerError
        The controller error to convert.

    Returns
    -------
    JSONResponse
        JSON error response.
    """
    status = getattr(exc, "status_code", HTTPStatus.INTERNAL_SERVER_ERROR)
    error_type = "invalid_request_error"
    if status == HTTPStatus.TOO_MANY_REQUESTS:
        error_type = "rate_limit_error"
    elif status >= HTTPStatus.INTERNAL_SERVER_ERROR:
        error_type = "service_unavailable"
    return JSONResponse(
        content=create_error_response(str(exc), error_type, status),
        status_code=status,
    )


# Manager availability is determined by proxying to the hub daemon's
# `/health` endpoint using `_call_daemon_api_async` where appropriate.


def _normalize_model_name(model_name: str) -> str:
    """Sanitize a model target provided via the API.

    Parameters
    ----------
    model_name : str
        The model name to normalize.

    Returns
    -------
    str
        The normalized model name.
    """
    return model_name.strip()


def _model_created_timestamp(config: MLXHubConfig) -> int:
    """Get the creation timestamp for models in the config.

    Parameters
    ----------
    config : MLXHubConfig
        The hub configuration.

    Returns
    -------
    int
        Unix timestamp of model creation.
    """
    source_path = config.source_path
    if source_path is not None and source_path.exists():
        try:
            return int(source_path.stat().st_mtime)
        except OSError:  # pragma: no cover - filesystem race
            return int(time.time())
    return int(time.time())


def _build_models_from_config(
    config: MLXHubConfig,
    live_snapshot: dict[str, Any] | None,
) -> tuple[list[Model], HubStatusCounts]:
    """Build model list and status counts from hub config and live snapshot.

    Parameters
    ----------
    config : MLXHubConfig
        The hub configuration.
    live_snapshot : dict[str, Any] or None
        Live status snapshot from the service.
    runtime : HubRuntime, optional
        Runtime reference used to enrich metadata with memory lifecycle states.

    Returns
    -------
    tuple[list[Model], HubStatusCounts]
        Models list and status counts.
    """
    live_entries = {}
    if live_snapshot is not None:
        models = live_snapshot.get("models")
        if isinstance(models, list):
            for entry in models:
                name = entry.get("name")
                if isinstance(name, str):
                    live_entries[name] = entry

    created_ts = _model_created_timestamp(config)
    rendered: list[Model] = []
    process_running = 0
    memory_loaded = 0
    for server_cfg in config.models:
        name = server_cfg.name or server_cfg.model_identifier
        live = live_entries.get(name, {})
        state = str(live.get("state") or "inactive").lower()
        memory_state = None
        if state == "running":
            process_running += 1
        if memory_state == "loaded":
            memory_loaded += 1
        metadata = {
            "status": memory_state or state,
            "process_state": state,
            "memory_state": memory_state,
            "group": server_cfg.group,
            "default": server_cfg.is_default_model,
            "model_type": server_cfg.model_type,
            "model_path": server_cfg.model_path,
            "log_path": live.get("log_path") or server_cfg.log_file,
            "pid": live.get("pid"),
            "port": live.get("port") or server_cfg.port,
            "started_at": live.get("started_at"),
            "stopped_at": live.get("stopped_at"),
            "exit_code": live.get("exit_code"),
            "auto_unload_minutes": server_cfg.auto_unload_minutes,
        }
        rendered.append(
            Model(
                id=name,
                object="model",
                created=created_ts,
                owned_by="hub",
                metadata=metadata,
            )
        )

    loaded_count = process_running
    counts = HubStatusCounts(
        registered=len(rendered),
        started=process_running,
        loaded=loaded_count,
    )
    return rendered, counts


def get_running_hub_models(raw_request: Request) -> set[str] | None:
    """Return the set of model names whose processes are currently running.

    Parameters
    ----------
    raw_request : Request
        FastAPI request containing hub server state.

    Returns
    -------
    set[str] | None
        Names of running models, or ``None`` when the service is unavailable.
    """

    server_config = getattr(raw_request.app.state, "server_config", None)
    if not isinstance(server_config, MLXHubConfig):
        return None

    try:
        config = _load_hub_config_from_request(raw_request)
    except HubConfigError:
        return None

    try:
        snapshot = _call_daemon_api_sync(config, "GET", "/hub/status", timeout=1.0)
    except HubServiceError as exc:
        logger.debug(
            f"Hub manager status unavailable; skipping running model filter. {type(exc).__name__}: {exc}"
        )
        return None

    running: set[str] = set()
    models = snapshot.get("models") if isinstance(snapshot, dict) else None
    if not isinstance(models, list):
        return running

    for entry in models:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        state = str(entry.get("state") or "").lower()
        if isinstance(name, str) and state == "running":
            running.add(name)

    return running


def get_cached_model_metadata(raw_request: Request) -> dict[str, Any] | None:
    """Fetch cached model metadata from application state, if available.

    Parameters
    ----------
    raw_request : Request
        The incoming FastAPI request.

    Returns
    -------
    dict[str, Any] or None
        Cached model metadata or None.
    """
    metadata_cache = getattr(raw_request.app.state, "model_metadata", None)
    if isinstance(metadata_cache, list) and metadata_cache:
        entry = metadata_cache[0]
        if isinstance(entry, dict):
            return entry
    return None


def get_configured_model_id(raw_request: Request) -> str | None:
    """Return the configured model identifier from config or cache.

    Parameters
    ----------
    raw_request : Request
        The incoming FastAPI request.

    Returns
    -------
    str or None
        Model identifier or None.
    """
    config = getattr(raw_request.app.state, "server_config", None)
    if config is not None:
        identifier = cast("str | None", getattr(config, "model_identifier", None))
        if identifier:
            return identifier
        return getattr(config, "model_path", None)

    cached = get_cached_model_metadata(raw_request)
    if cached is not None:
        return cached.get("id")
    return None


@hub_router.get("/hub/status", response_model=HubStatusResponse)
async def hub_status(raw_request: Request) -> HubStatusResponse:
    """Return hub status derived from the hub manager service when available.

    Returns
    -------
    HubStatusResponse
        The hub status response.
    """

    try:
        config = _load_hub_config_from_request(raw_request)
    except HubConfigError as exc:
        controller_available = getattr(raw_request.app.state, "hub_controller", None) is not None
        return HubStatusResponse(
            status="degraded",
            timestamp=int(time.time()),
            host=None,
            port=None,
            models=[],
            counts=HubStatusCounts(registered=0, started=0, loaded=0),
            warnings=[f"Hub configuration unavailable: {exc}"],
            controller_available=controller_available,
        )

    warnings: list[str] = []
    snapshot: dict[str, Any] | None = None
    try:
        # Try to reconcile via daemon then fetch status snapshot
        with contextlib.suppress(HubServiceError):
            await _call_daemon_api_async(config, "POST", "/hub/reload")
        snapshot = await _call_daemon_api_async(config, "GET", "/hub/status")
    except HubServiceError as exc:
        warnings.append(f"Hub manager unavailable: {exc}")

    models, counts = _build_models_from_config(config, snapshot)
    response_timestamp = int(time.time())
    if snapshot is not None:
        timestamp_value = snapshot.get("timestamp")
        if isinstance(timestamp_value, (int, float)):
            response_timestamp = int(timestamp_value)

    controller_available = getattr(raw_request.app.state, "hub_controller", None) is not None

    return HubStatusResponse(
        status="ok" if snapshot is not None else "degraded",
        timestamp=response_timestamp,
        host=config.host,
        port=config.port,
        models=models,
        counts=counts,
        warnings=warnings,
        controller_available=controller_available,
    )


@hub_router.get("/hub", response_class=HTMLResponse)
async def hub_status_page(raw_request: Request) -> HTMLResponse:
    """Serve a lightweight HTML dashboard for hub operators.

    Parameters
    ----------
    raw_request : Request
        The incoming FastAPI request.

    Returns
    -------
    HTMLResponse
        HTML response with the dashboard.

    Raises
    ------
    HTTPException
        If the status page is disabled.
    """
    config = getattr(raw_request.app.state, "server_config", None)
    enabled = bool(getattr(config, "enable_status_page", False))
    if not enabled:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail="Hub status page is disabled in configuration.",
        )

    # Prefer the daemon's rendered status page when the daemon is available
    try:
        config = _load_hub_config_from_request(raw_request)
    except HubConfigError:
        # Fall back to the inline page when config missing
        return HTMLResponse(content=_HUB_STATUS_PAGE_HTML, media_type="text/html")

    try:
        base = _daemon_base_url(config)
        url = f"{base.rstrip('/')}/hub"
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(url)
            if resp.status_code == 200 and "text/html" in resp.headers.get("content-type", ""):
                return HTMLResponse(content=resp.text, media_type="text/html")
    except Exception:
        # If daemon is unreachable or returns non-HTML, fall back to inline page
        pass

    return HTMLResponse(content=_HUB_STATUS_PAGE_HTML, media_type="text/html")


@hub_router.post("/hub/service/start", response_model=HubServiceActionResponse)
async def hub_service_start(raw_request: Request) -> HubServiceActionResponse | JSONResponse:
    """Start the background hub manager service if it is not already running.

    Parameters
    ----------
    raw_request : Request
        The incoming FastAPI request.

    Returns
    -------
    HubServiceActionResponse or JSONResponse
        Response indicating the result of the start action.

    Raises
    ------
    HubConfigError
        If the hub configuration cannot be loaded.
    """
    try:
        config = _load_hub_config_from_request(raw_request)
    except HubConfigError as exc:
        return _hub_config_error_response(str(exc))

    # If daemon is already responding, return early
    try:
        await _call_daemon_api_async(config, "GET", "/health", timeout=1.0)
        return HubServiceActionResponse(
            status="ok", action="start", message="Hub manager already running."
        )
    except HubServiceError:
        pass

    if config.source_path is None:
        return _hub_config_error_response(
            "Hub configuration must be saved to disk before starting the manager."
        )

    pid = start_hub_service_process(
        str(config.source_path), host=config.host, port=config.daemon_port
    )

    # Wait for daemon to become available
    deadline = time.time() + 20.0
    available = False
    while time.time() < deadline:
        try:
            await _call_daemon_api_async(config, "GET", "/health", timeout=1.0)
            available = True
            break
        except HubServiceError:
            await asyncio.sleep(0.5)

    if not available:
        return JSONResponse(
            content=create_error_response(
                "Hub manager failed to start within 20 seconds.",
                "service_unavailable",
                HTTPStatus.SERVICE_UNAVAILABLE,
            ),
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )

    try:
        await _call_daemon_api_async(config, "POST", "/hub/reload")
        snapshot = await _call_daemon_api_async(config, "GET", "/hub/status")
    except HubServiceError as exc:
        snapshot = None
        logger.warning(f"Hub manager started (pid={pid}) but status fetch failed: {exc}")

    details: dict[str, Any] = {"pid": pid}
    if snapshot is not None:
        details["models"] = snapshot.get("models", [])
    return HubServiceActionResponse(
        status="ok",
        action="start",
        message="Hub manager started",
        details=details,
    )


@hub_router.post("/hub/service/stop", response_model=HubServiceActionResponse)
async def hub_service_stop(raw_request: Request) -> HubServiceActionResponse | JSONResponse:
    """Stop the hub controller and manager service when present.

    Parameters
    ----------
    raw_request : Request
        The incoming FastAPI request.

    Returns
    -------
    HubServiceActionResponse | JSONResponse
        Response indicating the result of the stop operation.

    Raises
    ------
    HubConfigError
        If the hub configuration cannot be loaded.
    HubServiceError
        If there is an error communicating with the hub service.
    """
    try:
        config = _load_hub_config_from_request(raw_request)
    except HubConfigError as exc:
        return _hub_config_error_response(str(exc))

    controller_stopped = _stop_controller_process(config)
    manager_shutdown = False

    try:
        # Ask daemon to reload before shutdown; if unavailable we still proceed
        with contextlib.suppress(HubServiceError):
            await _call_daemon_api_async(config, "POST", "/hub/reload")

        await _call_daemon_api_async(config, "POST", "/hub/shutdown")
        manager_shutdown = True
    except HubServiceError:
        # If daemon unreachable, treat as not running
        manager_shutdown = False

    message_parts = [
        "Hub controller stop requested" if controller_stopped else "Hub controller was not running",
        "Hub manager shutdown requested" if manager_shutdown else "Hub manager was not running",
    ]

    return HubServiceActionResponse(
        status="ok",
        action="stop",
        message=". ".join(message_parts),
        details={
            "controller_stopped": controller_stopped,
            "manager_shutdown": manager_shutdown,
        },
    )


@hub_router.post("/hub/service/reload", response_model=HubServiceActionResponse)
async def hub_service_reload(raw_request: Request) -> HubServiceActionResponse | JSONResponse:
    """Reload hub.yaml inside the running manager service and return the diff.

    Parameters
    ----------
    raw_request : Request
        The incoming FastAPI request.

    Returns
    -------
    HubServiceActionResponse or JSONResponse
        Response indicating the result of the reload action.

    Raises
    ------
    HubConfigError
        If the hub configuration cannot be loaded.
    HubServiceError
        If there is an error communicating with the hub service.
    """
    try:
        config = _load_hub_config_from_request(raw_request)
    except HubConfigError as exc:
        return _hub_config_error_response(str(exc))

    try:
        # Ensure daemon responds to health check
        await _call_daemon_api_async(config, "GET", "/health")
    except HubServiceError:
        return _manager_unavailable_response()

    try:
        diff = await _call_daemon_api_async(config, "POST", "/hub/reload") or {}
    except HubServiceError as exc:
        return _service_error_response("reload hub configuration", exc)

    return HubServiceActionResponse(
        status="ok",
        action="reload",
        message="Hub configuration reloaded",
        details=diff,
    )


@hub_router.post("/hub/models/{model_name}/start-model", response_model=HubModelActionResponse)
async def hub_start_model(
    model_name: str,
    raw_request: Request,
    payload: HubModelActionRequest | None = None,
) -> HubModelActionResponse | JSONResponse:
    """Request that the hub manager start ``model_name``.

    Parameters
    ----------
    model_name : str
        The name of the model to load.
    raw_request : Request
        The incoming request.
    payload : HubModelActionRequest, optional
        Additional payload for the request.

    Returns
    -------
    HubModelActionResponse or JSONResponse
        Response indicating the result of the load action.
    """

    _ = payload  # reserved for future compatibility
    return await _hub_model_service_action(raw_request, model_name, "start-model")


@hub_router.post("/hub/models/{model_name}/stop-model", response_model=HubModelActionResponse)
async def hub_stop_model(
    model_name: str,
    raw_request: Request,
    payload: HubModelActionRequest | None = None,
) -> HubModelActionResponse | JSONResponse:
    """Request that the hub manager stop ``model_name``.

    Parameters
    ----------
    model_name : str
        The name of the model to unload.
    raw_request : Request
        The incoming request.
    payload : HubModelActionRequest, optional
        Additional payload for the request.

    Returns
    -------
    HubModelActionResponse or JSONResponse
        Response indicating the result of the unload action.
    """

    _ = payload  # reserved for future compatibility
    return await _hub_model_service_action(raw_request, model_name, "stop-model")


@hub_router.post("/hub/models/{model_name}/load", response_model=HubModelActionResponse)
async def hub_load_model(
    model_name: str,
    raw_request: Request,
    payload: HubModelActionRequest | None = None,
) -> HubModelActionResponse | JSONResponse:
    """Request that the in-process controller load ``model_name`` into memory.

    Parameters
    ----------
    model_name : str
        The name of the model to load.
    raw_request : Request
        The incoming FastAPI request.
    payload : HubModelActionRequest, optional
        Additional payload for the request.

    Returns
    -------
    HubModelActionResponse or JSONResponse
        Response indicating the result of the load action.
    """
    return await _hub_memory_controller_action(raw_request, model_name, "load-model", payload)


@hub_router.post("/hub/models/{model_name}/unload", response_model=HubModelActionResponse)
async def hub_unload_model(
    model_name: str,
    raw_request: Request,
    payload: HubModelActionRequest | None = None,
) -> HubModelActionResponse | JSONResponse:
    """Request that the in-process controller unload ``model_name`` from memory.

    Parameters
    ----------
    model_name : str
        The name of the model to unload.
    raw_request : Request
        The incoming FastAPI request.
    payload : HubModelActionRequest, optional
        Additional payload for the request.

    Returns
    -------
    HubModelActionResponse or JSONResponse
        Response indicating the result of the unload action.
    """
    return await _hub_memory_controller_action(raw_request, model_name, "unload-model", payload)


async def _hub_model_service_action(
    raw_request: Request,
    model_name: str,
    action: Literal["start-model", "stop-model"],
) -> HubModelActionResponse | JSONResponse:
    """Execute a load or unload action on a model via the hub service.

    Parameters
    ----------
    raw_request : Request
        The incoming FastAPI request.
    model_name : str
        Name of the model to act on.
    action : Literal["start-model", "stop-model"]
        The action to perform.

    Returns
    -------
    HubModelActionResponse or JSONResponse
        Action response or error response.
    """
    target = _normalize_model_name(model_name)
    if not target:
        return JSONResponse(
            content=create_error_response(
                "Model name cannot be empty",
                "invalid_request_error",
                HTTPStatus.BAD_REQUEST,
            ),
            status_code=HTTPStatus.BAD_REQUEST,
        )

    try:
        config = _load_hub_config_from_request(raw_request)
    except HubConfigError as exc:
        return _hub_config_error_response(str(exc))

    try:
        await _call_daemon_api_async(config, "GET", "/health")
    except HubServiceError:
        return _manager_unavailable_response()

    try:
        await _call_daemon_api_async(config, "POST", "/hub/reload")
    except HubServiceError as exc:
        return _service_error_response("reload before executing the model action", exc)

    try:
        if action == "start-model":
            await _call_daemon_api_async(config, "POST", f"/hub/models/{target}/start")
            message = f"Model '{target}' start requested"
        else:
            await _call_daemon_api_async(config, "POST", f"/hub/models/{target}/stop")
            message = f"Model '{target}' stop requested"
    except HubServiceError as exc:
        friendly = action.replace("-", " ")
        return _service_error_response(f"{friendly} for model '{target}'", exc)

    return HubModelActionResponse(status="ok", action=action, model=target, message=message)


async def _hub_memory_controller_action(
    raw_request: Request,
    model_name: str,
    action: Literal["load-model", "unload-model"],
    payload: HubModelActionRequest | None,
) -> HubModelActionResponse | JSONResponse:
    """Execute a memory load/unload request using the in-process controller.

    Parameters
    ----------
    raw_request : Request
        The incoming FastAPI request.
    model_name : str
        The name of the model to act on.
    action : Literal["load-model", "unload-model"]
        The action to perform.
    payload : HubModelActionRequest or None
        Additional payload for the request.

    Returns
    -------
    HubModelActionResponse or JSONResponse
        Response indicating the result of the action.
    """
    target = _normalize_model_name(model_name)
    if not target:
        return JSONResponse(
            content=create_error_response(
                "Model name cannot be empty",
                "invalid_request_error",
                HTTPStatus.BAD_REQUEST,
            ),
            status_code=HTTPStatus.BAD_REQUEST,
        )

    controller = getattr(raw_request.app.state, "hub_controller", None)
    if controller is None:
        return _controller_unavailable_response()

    try:
        if action == "load-model":
            await controller.load_model(target)
            message = f"Model '{target}' memory load requested"
        else:
            await controller.unload_model(target)
            message = f"Model '{target}' memory unload requested"
    except Exception as exc:
        return _controller_error_response(exc)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception(
            f"Unexpected failure while executing {action} for {target}. {type(exc).__name__}: {exc}",
        )
        return JSONResponse(
            content=create_error_response(
                f"Unexpected failure while executing {action.replace('-', ' ')} for '{target}'",
                "internal_error",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            ),
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )

    return HubModelActionResponse(status="ok", action=action, model=target, message=message)


__all__ = [
    "hub_router",
    "get_cached_model_metadata",
    "get_configured_model_id",
]
