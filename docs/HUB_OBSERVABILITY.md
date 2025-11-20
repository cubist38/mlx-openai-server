# Hub Observability and Logging

The hub controller now treats observability as a first-class concern so each managed
model can be inspected, tailed, or measured independently.

## Per-model log layout

- Set `log_path` inside `hub.yaml` to pick a base directory (defaults to
  `~/mlx-openai-server/logs`).
- The controller writes every managed model's stdout/stderr to
  `log_path/<name>.log`. The base directory is created automatically by the hub.
- Because each model is isolated, rotating or truncating one log never impacts others.
- Follow activity for an individual model with `tail -F <log_path>/<name>.log`.

## Lifecycle hooks

`HubManager` emits lifecycle notifications through the `HubObservabilitySink` protocol
(`app/hub/observability.py`). The default `LoggingHubObservabilitySink` binds
`hub_model`, `hub_group`, and `hub_log_path` metadata to each Loguru record so log
collectors can filter or aggregate by model.

Implementations may forward the events to Prometheus, StatsD, OTLP, or any other
monitoring backend:

```python
from app.hub.observability import HubModelContext, HubObservabilitySink

class MetricsSink(HubObservabilitySink):
    def model_started(self, ctx: HubModelContext, *, pid: int | None) -> None:
        metrics.counter("hub_model_started", {"model": ctx.name, "group": ctx.group}).inc()

    def model_stopped(self, ctx: HubModelContext, *, exit_code: int | None) -> None:
        metrics.counter("hub_model_stopped", {"model": ctx.name}).inc()

    def model_failed(self, ctx: HubModelContext, *, exit_code: int | None) -> None:
        metrics.counter("hub_model_failed", {"model": ctx.name, "exit_code": exit_code}).inc()
```

Passing a custom sink to `HubManager(..., observability_sink=sink)` keeps the logging
bindings while enabling richer telemetry pipelines.

## Best practices

1. Keep the hub's `log_path` on a fast local disk to avoid blocking stdout/stderr writes.
2. Forward the per-model log directories to your existing log aggregation stack so
   each model's output can be searched independently.
3. Alert on `model_failed` events with non-zero exit codes—they mean the underlying
   handler crashed. Combine the alert with a `hub watch` session to confirm health.
4. Use `model_started`/`model_stopped` events to emit Prometheus counters so you can
   visualize churn and auto-restarts over time.
