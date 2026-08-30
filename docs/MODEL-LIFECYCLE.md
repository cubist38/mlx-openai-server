# Persistent multi-model service

The HTTP process stays resident while model workers are loaded only when
requested. Each model runs in an isolated subprocess. Idle workers expire after
their configured retention period, and active streams hold a lease that
prevents eviction until completion or disconnect.

## Install

Use Python 3.12 and install this checkout into a virtual environment:

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install -e .
```

Any environment that already contains MLX works as well. Whichever you choose,
the launchd service below must point at that same environment's
`mlx-openai-server` executable.

The project requires `mlx-vlm>=0.6.17,<0.7`, which includes the Qwen 3.5 model
implementation used by converted Qwen 3.8 checkpoints.

## Configure

Keep the daemon configuration outside this repository, since it holds absolute
paths specific to your machine. A two-model setup looks like this:

```yaml
server:
  host: "127.0.0.1"
  port: 8000
  max_loaded_models: 1
  model_load_timeout: 600

models:
  - model_path: /absolute/path/to/your/chat-model
    model_type: multimodal
    served_model_name: qwen-agentcoder
    on_demand: true
    on_demand_idle_timeout: 5m
    enable_auto_tool_choice: true
    tool_call_parser: qwen3_coder # must match the model, see below
    reasoning_parser: qwen3_5
    kv_bits: 4
    kv_group_size: 64
    quantized_kv_start: 1024

  - model_path: mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ
    model_type: embeddings
    served_model_name: qwen3-embedding
    on_demand: true
    on_demand_idle_timeout: 2m
```

`examples/config.yaml` is a tracked, machine-independent variant of the same
structure, and [CONFIGURATION.md](CONFIGURATION.md) documents every key with its
default and the model types it applies to.

With `max_loaded_models: 1` only one on-demand worker is resident at a time. If
the slot is occupied by an active request, the next model waits instead of
interrupting it. When the slot becomes idle, the least recently used worker is
replaced.

## Versioning and aliases

A model can answer to more than one name. `version` records which checkpoint is
being served and adds a `<name>:<version>` route; `aliases` adds arbitrary extra
names, typically a stable name that clients hardcode while the checkpoint behind
it changes:

```yaml
models:
  - model_path: /absolute/path/to/your/chat-model
    served_model_name: qwen-agentcoder-2025-08
    version: "2.1"
    aliases:
      - qwen-agentcoder # what applications ask for
      - qwen-agentcoder:stable
    on_demand: true
```

The three extra names above (`qwen-agentcoder-2025-08:2.1`, `qwen-agentcoder`
and `qwen-agentcoder:stable`) are accepted anywhere a model name is, including
`/v1/chat/completions`, `/v1/embeddings`, `/v1/models/load` and
`/v1/models/unload`. Resolution happens in the registry, so an alias and the
canonical name share one worker and one reference count: a request made through
an alias keeps the model from being unloaded or evicted just as the canonical
name does.

Names live in a single namespace. An alias that collides with another model's
name, version tag or alias is rejected at startup rather than silently making
one of the two unreachable. Aliases are routes, not models: `/v1/models` still
lists one entry per model, reporting the tag and the alternative names.

```console
$ curl -s http://127.0.0.1:8000/v1/models/status | jq '.[0] | {id, version, aliases}'
{
  "id": "qwen-agentcoder-2025-08",
  "version": "2.1",
  "aliases": [
    "qwen-agentcoder",
    "qwen-agentcoder-2025-08:2.1",
    "qwen-agentcoder:stable"
  ]
}
```

To promote a new checkpoint, point the alias at the new entry and restart the
service; applications that request `qwen-agentcoder` follow it without changing
their configuration, while `qwen-agentcoder-2025-08:2.1` keeps naming the exact
build.

## Start manually

```bash
mlx-openai-server launch --config /path/to/your/config.yaml
```

## Run as a macOS user service

Write a LaunchAgent that points at both the executable and the configuration by
absolute path:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.local.mlx-openai-server</string>
  <key>ProgramArguments</key>
  <array>
    <string>/absolute/path/to/.venv/bin/mlx-openai-server</string>
    <string>launch</string>
    <string>--config</string>
    <string>/absolute/path/to/your/config.yaml</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>ThrottleInterval</key>
  <integer>5</integer>
  <key>StandardOutPath</key>
  <string>/tmp/mlx-openai-server.log</string>
  <key>StandardErrorPath</key>
  <string>/tmp/mlx-openai-server.error.log</string>
</dict>
</plist>
```

Both paths must exist. `KeepAlive` restarts the process on exit, so a wrong
executable path turns into a restart loop every `ThrottleInterval` seconds.

Load and unload it with:

```bash
launchctl bootstrap "gui/$(id -u)" \
  "$HOME/Library/LaunchAgents/com.local.mlx-openai-server.plist"
launchctl bootout "gui/$(id -u)" \
  "$HOME/Library/LaunchAgents/com.local.mlx-openai-server.plist"
```

The service process stays available on port 8000. Model memory is independent
from process lifetime and is released by the configured idle timers.

## Lifecycle controls

Every inference endpoint accepts `keep_alive` as seconds or as a duration such
as `"30s"`, `"5m"`, or `"1h30m"`:

- omitted: use `on_demand_idle_timeout`;
- `0`: unload after the request or stream ends;
- negative number: keep loaded indefinitely.

Inspect all workers:

```bash
curl http://127.0.0.1:8000/v1/models/status
```

Preload a model:

```bash
curl http://127.0.0.1:8000/v1/models/load \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-agentcoder","keep_alive":"10m"}'
```

Unload it safely:

```bash
curl http://127.0.0.1:8000/v1/models/unload \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-agentcoder"}'
```

Safe unload returns HTTP 409 while the model has active requests. The optional
`force` flag interrupts active generation and should be reserved for recovery.

## Python streaming and reasoning

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")

stream = client.chat.completions.create(
    model="qwen-agentcoder",
    messages=[{"role": "user", "content": "Spiega questa funzione."}],
    stream=True,
    extra_body={
        "keep_alive": "5m",
        "chat_template_kwargs": {"enable_thinking": True},
    },
)

for chunk in stream:
    delta = chunk.choices[0].delta
    reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
    if reasoning:
        print(reasoning, end="", flush=True)
    if delta.content:
        print(delta.content, end="", flush=True)
```

Reasoning is emitted separately as both `reasoning_content` and `reasoning`.
Tool calls use the standard OpenAI-compatible `tool_calls` field.

### Matching the tool call parser to the model

`tool_call_parser` must match the format the model's own chat template emits,
otherwise the parser consumes the tool call and the response carries neither
`content` nor `tool_calls`. Check the checkpoint's `chat_template.jinja`:

```bash
grep -o '<tool_call>\|<function=\|<parameter=' MODEL_DIR/chat_template.jinja | sort -u
```

- `<function=` and `<parameter=` present: use `qwen3_coder`.
- only `<tool_call>` wrapping JSON: use `qwen3`.

Converted Qwen 3.8 agentcoder checkpoints emit the function/parameter form and
therefore need `qwen3_coder`, not `qwen3`.

## LangChain and LangGraph

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="local",
    model="qwen-agentcoder",
    streaming=True,
    extra_body={
        "keep_alive": "5m",
        "chat_template_kwargs": {"enable_thinking": True},
    },
)

agent = create_react_agent(model=llm, tools=[])

for event in agent.stream(
    {"messages": [("user", "Scrivi un piano e poi eseguilo.")]},
    stream_mode="values",
):
    print(event["messages"][-1])
```

For embeddings:

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    base_url="http://127.0.0.1:8000/v1",
    api_key="local",
    model="qwen3-embedding",
)
vector = embeddings.embed_query("testo da indicizzare")
```
