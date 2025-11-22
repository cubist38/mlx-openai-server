#!/usr/bin/env python3
"""Rich-powered health dashboard for the MLX OpenAI-compatible server."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
import json
import os
import time
from typing import Any, Literal

from loguru import logger

try:  # Dependency guards preserve useful error messages for local runs.
    import httpx
except ImportError as e:  # pragma: no cover - defensive
    logger.error("This dashboard requires httpx. Install via `pip install httpx`.")
    raise SystemExit(1) from e

try:
    from pydantic import BaseModel, ConfigDict
except ImportError as e:  # pragma: no cover
    logger.error("This dashboard requires pydantic. Install via `pip install pydantic`.")
    raise SystemExit(1) from e

try:
    from rich import box
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
except ImportError as e:  # pragma: no cover
    logger.error("This dashboard requires rich. Install via `pip install rich`.")
    raise SystemExit(1) from e


class ModelData(BaseModel):
    """Represents a model in the models list."""

    id: str
    object: Literal["model"]
    created: int | None = None
    owned_by: str | None = None

    model_config = ConfigDict(extra="allow")


class ModelList(BaseModel):
    """Represents the response from the models endpoint."""

    object: Literal["list"]
    data: list[ModelData]

    model_config = ConfigDict(extra="allow")


@dataclass
class DashboardSnapshot:
    """Snapshot of the dashboard state at a point in time."""

    timestamp: float
    reachable: bool
    latency_ms: float | None
    status_text: str
    models: list[ModelData]
    active_model: ModelData | None
    stream_ok: bool
    stream_message: str
    errors: list[str] = field(default_factory=list)


def build_headers() -> dict[str, str]:
    """Build authorization headers from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("MLX_API_KEY")
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def env_base_url() -> str:
    """Get the base URL from environment or default."""
    raw = os.getenv("MLX_URL", "http://127.0.0.1:8000")
    return raw.rstrip("/")


def gather_snapshot(client: httpx.Client) -> DashboardSnapshot:
    """Gather a snapshot of the server's health and models."""
    reachable = False
    latency_ms: float | None = None
    status_text = "unknown"
    errors: list[str] = []

    start = time.perf_counter()
    try:
        response = client.get("/health")
        response.raise_for_status()
        payload = response.json()
        status_text = str(payload.get("status", "unknown"))
        latency_ms = (time.perf_counter() - start) * 1000.0
        reachable = True
    except Exception as e:  # pragma: no cover - network dependent
        errors.append(f"health: {e}")

    models: list[ModelData] = []
    try:
        response = client.get("/v1/models")
        response.raise_for_status()
        models = ModelList.model_validate(response.json()).data
    except Exception as e:  # pragma: no cover
        errors.append(f"models: {e}")

    active_model = select_active_model(models)

    stream_ok = False
    stream_message = "stream skipped"
    if reachable and active_model:
        stream_ok, stream_message = streaming_sanity_check(client, active_model.id)
    elif not active_model:
        stream_message = "stream skipped (no models)"

    return DashboardSnapshot(
        timestamp=time.time(),
        reachable=reachable,
        latency_ms=latency_ms,
        status_text=status_text,
        models=models,
        active_model=active_model,
        stream_ok=stream_ok,
        stream_message=stream_message,
        errors=errors,
    )


def select_active_model(models: list[ModelData]) -> ModelData | None:
    """
    Select the active model from a list of available models.

    Uses environment variables MLX_ACTIVE_MODEL or MLX_MODEL_ID as preference,
    otherwise returns the first model in the list.

    Parameters
    ----------
    models : list[ModelData]
        List of available models.

    Returns
    -------
    ModelData | None
        The selected active model, or None if no models are available.
    """
    if not models:
        return None
    preferred = os.getenv("MLX_ACTIVE_MODEL") or os.getenv("MLX_MODEL_ID")
    if preferred:
        for model in models:
            if model.id == preferred:
                return model
    return models[0]


def streaming_sanity_check(client: httpx.Client, model_id: str) -> tuple[bool, str]:
    """
    Perform a streaming sanity check on a model.

    Tests streaming chat completion functionality by sending a simple request
    and validating the response stream.

    Parameters
    ----------
    client : httpx.Client
        HTTP client for making requests.
    model_id : str
        The model ID to test.

    Returns
    -------
    tuple[bool, str]
        A tuple of (success, message) indicating test result and details.
    """
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Stream 'hello dashboard'."}],
        "stream": True,
        "temperature": 0,
    }
    chunk_count = 0
    content_chars = 0
    try:
        with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            response.raise_for_status()
            for sse in iter_sse_payloads(response):
                if sse == "[DONE]":
                    break
                data = json.loads(sse)
                chunk = parse_chunk(data)
                chunk_count += 1
                for choice in chunk["choices"]:
                    delta = choice.get("delta", {})
                    text = delta.get("content")
                    if text:
                        content_chars += len(text)
    except Exception as e:  # pragma: no cover
        return False, f"stream error: {e}"

    if chunk_count == 0:
        return False, "stream incomplete (no chunks)"
    if content_chars == 0:
        return False, "stream returned no content"
    return True, f"chunks={chunk_count} chars~{content_chars}"


def parse_chunk(data: dict[str, Any]) -> dict[str, Any]:
    """
    Parse and validate a streaming response chunk.

    Ensures the chunk contains required OpenAI-compatible fields.

    Parameters
    ----------
    data : dict[str, Any]
        The parsed JSON data from a streaming chunk.

    Returns
    -------
    dict[str, Any]
        The validated chunk data.

    Raises
    ------
    ValueError
        If the chunk is missing required fields or has invalid structure.
    """
    # We only need to ensure essential OpenAI chunk keys exist.
    if not isinstance(data, dict):
        raise TypeError("Chunk payload must be a JSON object")
    required = {"id", "object", "created", "model", "choices"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Chunk missing fields: {missing}")
    if not isinstance(data["choices"], list) or not data["choices"]:
        raise ValueError("Chunk choices must be a non-empty list")
    return data


def iter_sse_payloads(response: httpx.Response) -> Generator[str, None, None]:
    """
    Iterate over Server-Sent Events payloads from a streaming response.

    Parses SSE-formatted lines and yields the data payloads.

    Parameters
    ----------
    response : httpx.Response
        The streaming HTTP response containing SSE data.

    Yields
    ------
    str
        The data payload from each SSE event.
    """
    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line.split("data:", 1)[1].lstrip()
        if payload:
            yield payload


def render_dashboard(snapshot: DashboardSnapshot, base_url: str) -> Layout:
    """
    Render the complete dashboard layout.

    Creates a rich terminal UI layout with status, models, and footer panels.

    Parameters
    ----------
    snapshot : DashboardSnapshot
        The current dashboard data snapshot.
    base_url : str
        The base URL of the MLX server.

    Returns
    -------
    Layout
        The rendered dashboard layout.
    """
    layout = Layout(name="root")
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(Layout(name="left"), Layout(name="right"))

    layout["header"].update(
        Panel(
            f"MLX Health Dashboard | Target: {base_url} | Updated: {time.strftime('%H:%M:%S', time.localtime(snapshot.timestamp))}",
            style="bold white on blue",
        )
    )

    layout["left"].update(render_status_panel(snapshot))
    layout["right"].update(render_models_panel(snapshot))
    layout["footer"].update(render_footer(snapshot))
    return layout


def render_status_panel(snapshot: DashboardSnapshot) -> Panel:
    """
    Render the status panel showing server health and active model info.

    Parameters
    ----------
    snapshot : DashboardSnapshot
        The current dashboard data snapshot.

    Returns
    -------
    Panel
        The rendered status panel.
    """
    table = Table.grid(padding=(0, 1))
    table.add_row("Reachable", "Yes" if snapshot.reachable else "No")
    latency = f"{snapshot.latency_ms:.1f} ms" if snapshot.latency_ms is not None else "--"
    table.add_row("Latency", latency)
    table.add_row("Health", snapshot.status_text)
    if snapshot.active_model:
        table.add_row("Active Model", snapshot.active_model.id)
        owner = snapshot.active_model.owned_by or "--"
        created = snapshot.active_model.created or "--"
        table.add_row("Owned By", str(owner))
        table.add_row("Created", str(created))
    table.add_row(
        "Streaming",
        f"{'OK' if snapshot.stream_ok else 'Fail'} ({snapshot.stream_message})",
    )
    return Panel(table, title="Server Status", box=box.ROUNDED)


def render_models_panel(snapshot: DashboardSnapshot) -> Panel:
    """
    Render the models panel showing available models.

    Parameters
    ----------
    snapshot : DashboardSnapshot
        The current dashboard data snapshot.

    Returns
    -------
    Panel
        The rendered models panel.
    """
    table = Table(title="Model Registry", box=box.MINIMAL_DOUBLE_HEAD)
    table.add_column("ID", ratio=2)
    table.add_column("Owner", ratio=1)
    table.add_column("Created", justify="right")

    if snapshot.models:
        for model in snapshot.models:
            created = str(model.created) if model.created is not None else "--"
            table.add_row(model.id, model.owned_by or "--", created)
    else:
        table.add_row("(no models detected)", "--", "--")

    return Panel(table, box=box.ROUNDED)


def render_footer(snapshot: DashboardSnapshot) -> Panel:
    """
    Render the footer panel showing errors or status notes.

    Parameters
    ----------
    snapshot : DashboardSnapshot
        The current dashboard data snapshot.

    Returns
    -------
    Panel
        The rendered footer panel.
    """
    if snapshot.errors:
        content = "\n".join(snapshot.errors)
    else:
        content = "All systems nominal." if snapshot.reachable else "Awaiting server response..."
    return Panel(content, title="Notes", box=box.ROUNDED)


def main() -> None:
    """
    Run the LLM health dashboard.

    Runs a live terminal dashboard displaying server health, models, and status.
    """
    base_url = env_base_url()
    console = Console()
    headers = build_headers()
    refresh_seconds = float(os.getenv("MLX_DASHBOARD_REFRESH", "2"))

    with (
        httpx.Client(base_url=base_url, headers=headers, timeout=30.0) as client,
        Live(console=console, screen=True, refresh_per_second=4) as live,
    ):
        try:
            while True:
                snapshot = gather_snapshot(client)
                live.update(render_dashboard(snapshot, base_url))
                time.sleep(refresh_seconds)
        except KeyboardInterrupt:
            console.print("\nExiting dashboard...")


if __name__ == "__main__":
    main()
