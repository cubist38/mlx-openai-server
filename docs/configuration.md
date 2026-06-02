# Configuration Reference

This document is the complete, source-of-truth reference for configuring
`mlx-openai-server`. It is generated from the actual definitions in
[`app/cli.py`](../app/cli.py) (the Click options) and
[`app/config.py`](../app/config.py) (the `MLXServerConfig`,
`ModelEntryConfig`, and `MultiModelServerConfig` dataclasses). If you change
those files, update this doc.

---

## The two configuration surfaces

There are **two ways** to configure the server, and they are mutually
exclusive:

| Mode | How it's selected | Where settings come from |
|------|-------------------|--------------------------|
| **Single-model** | `--model-path` given, **no** `--config` | CLI flags only |
| **Multi-handler** | `--config <file.yaml>` given | The YAML file only |

> ### ⚠️ The most important gotcha
>
> **When `--config` is supplied, every other CLI flag is silently ignored** —
> no warning, no error. This includes server-level flags you might expect to
> still apply, such as `--port`, `--host`, `--log-level`, `--log-file`, and
> `--queue-timeout`.
>
> In `--config` mode, *all* configuration must live in the YAML file. See
> [`app/cli.py`](../app/cli.py): the `if config_file is not None:` branch calls
> `start_multi(...)` and `return`s before any single-model flag is read.

Both modes ultimately run through the same subprocess-isolated handler path
(`HandlerProcessProxy`), so a single-model launch is internally converted to a
one-entry multi-model config.

---

## Single-model mode (CLI flags)

Launch:

```bash
mlx-openai-server launch --model-path mlx-community/SomeModel-4bit --model-type lm
```

### Server / process options

| Flag | Default | Type | Notes |
|------|---------|------|-------|
| `--model-path` | — (required) | str | Required unless `--config` is used. HF repo or local path. |
| `--model-type` | `lm` | choice | `lm`, `multimodal`, `image-generation`, `image-edit`, `embeddings`, `whisper` |
| `--served-model-name` | `model_path` | str | Name exposed via `/v1/models` and accepted in the request `model` field. |
| `--port` | `8000` | int | |
| `--host` | `0.0.0.0` | str | |
| `--queue-timeout` | `300` | int | **Request timeout in seconds.** Bounds the whole request (non-streaming) and is the per-chunk timeout (streaming). Also the queue-wait timeout. |
| `--queue-size` | `100` | int | Max pending requests in the queue. |
| `--log-file` | `logs/app.log` | str | Path to log file. |
| `--no-log-file` | off | flag | Disable file logging (console only). |
| `--log-level` | `INFO` | choice | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` (case-insensitive). |

### LM / multimodal options

| Flag | Default | Type | Applies to |
|------|---------|------|-----------|
| `--context-length` | model default | int | `lm`, `multimodal` |
| `--enable-auto-tool-choice` | off | flag | `lm`, `multimodal` |
| `--tool-call-parser` | auto-detect | choice | `lm`, `multimodal` |
| `--reasoning-parser` | auto-detect | choice | `lm`, `multimodal` |
| `--trust-remote-code` | off | flag | `lm`, `multimodal` |
| `--chat-template-file` | none | str | `lm`, `multimodal` |
| `--disable-auto-resize` | off | flag | `multimodal` (VLMs) |
| `--debug` | off | flag | `lm`, `multimodal` |

### Prompt KV cache (LM only)

| Flag | Default | Type |
|------|---------|------|
| `--prompt-cache-size` | `10` | int |
| `--max-bytes` | `2^63` | int (→ field `prompt_cache_max_bytes`) |
| `--prompt-cache-dir` | temp dir | path |

### KV cache quantization (`lm` / `multimodal` only)

| Flag | Default | Type |
|------|---------|------|
| `--kv-bits` | none | int (e.g. `4`, `8`) |
| `--kv-group-size` | `64` | int |
| `--quantized-kv-start` | `0` | int |

### Speculative decoding (LM only)

| Flag | Default | Type |
|------|---------|------|
| `--draft-model-path` | none | str |
| `--num-draft-tokens` | `2` | int |

### Continuous-batching concurrency (`lm` / `multimodal`)

These mirror mlx-lm's own server flags. Note the flag names differ from the
underlying field names.

| Flag | Field name | Default | Type |
|------|-----------|---------|------|
| `--decode-concurrency` | `batch_completion_size` | `32` | int |
| `--prompt-concurrency` | `batch_prefill_size` | `8` | int |
| `--prefill-step-size` | `batch_prefill_step_size` | `2048` | int |
| `--disable-batching` | `disable_batching` | off | flag |

### Image generation / edit options

| Flag | Default | Type | Notes |
|------|---------|------|-------|
| `--quantize` | none | int | Flux models only. |
| `--config-name` | type-dependent | choice | Required for image models. `image-generation` defaults to `flux-schnell`; `image-edit` to `flux-kontext-dev`. |
| `--lora-paths` | none | comma-separated str | Multiple paths separated by commas. |
| `--lora-scales` | none | comma-separated str | Multiple scales separated by commas. |

### Default sampling parameters

Used only when the API request omits the parameter. Flag names drop the
`default_` prefix that the config field uses.

| Flag | Field name | Type |
|------|-----------|------|
| `--max-tokens` | `default_max_tokens` | int |
| `--temperature` | `default_temperature` | float |
| `--top-p` | `default_top_p` | float |
| `--top-k` | `default_top_k` | int |
| `--min-p` | `default_min_p` | float |
| `--repetition-penalty` | `default_repetition_penalty` | float |
| `--presence-penalty` | `default_presence_penalty` | float |
| `--xtc-probability` | `default_xtc_probability` | float |
| `--xtc-threshold` | `default_xtc_threshold` | float |
| `--seed` | `default_seed` | int |
| `--repetition-context-size` | `default_repetition_context_size` | int |

---

## Multi-handler mode (YAML)

Launch:

```bash
mlx-openai-server launch --config examples/config.yaml
```

The YAML has two top-level sections: an optional `server:` mapping and a
required non-empty `models:` list. Each model entry is parsed directly into a
`ModelEntryConfig`, so **the valid YAML keys are exactly the field names of
`ModelEntryConfig`** (snake_case) — *not* the CLI flag spellings.

### `server:` section

All keys optional; defaults shown. (These replace the `--host`/`--port`/etc.
CLI flags, which are ignored in this mode.)

| Key | Default |
|-----|---------|
| `host` | `0.0.0.0` |
| `port` | `8000` |
| `log_level` | `INFO` |
| `log_file` | none |
| `no_log_file` | `false` |

### `models:` list — per-entry keys

`model_path` is required per entry; `served_model_name` must be unique across
entries (defaults to `model_path`). Every key below is a `ModelEntryConfig`
field.

| YAML key | Default | Notes |
|----------|---------|-------|
| `model_path` | — (required) | HF repo or local path. |
| `model_type` | `lm` | One of the six valid types. |
| `served_model_name` | `model_path` | Must be unique across entries. |
| `context_length` | model default | |
| `queue_timeout` | `300` | Request timeout (seconds). |
| `queue_size` | `100` | |
| `quantize` | none | Image models. |
| `config_name` | type-dependent | Image models. |
| `lora_paths` | none | **YAML list** of strings (not comma string). |
| `lora_scales` | none | **YAML list** of floats. |
| `on_demand` | `false` | **YAML-only.** Lazy-load + idle-unload this model. |
| `on_demand_idle_timeout` | `60` | **YAML-only.** Seconds idle before unloading. |
| `disable_auto_resize` | `false` | VLMs. |
| `enable_auto_tool_choice` | `false` | |
| `tool_call_parser` | none | |
| `reasoning_parser` | none | |
| `message_converter` | auto | **YAML-only.** Lowercased; auto-resolved for lm/multimodal. |
| `trust_remote_code` | `false` | |
| `chat_template_file` | none | |
| `debug` | `false` | |
| `prompt_cache_size` | `10` | |
| `prompt_cache_max_bytes` | `2^63` | (CLI: `--max-bytes`) |
| `prompt_cache_dir` | none | |
| `draft_model_path` | none | |
| `num_draft_tokens` | `2` | |
| `kv_bits` | none | |
| `kv_group_size` | `64` | |
| `quantized_kv_start` | `0` | |
| `batch_completion_size` | `32` | (CLI: `--decode-concurrency`) |
| `batch_prefill_size` | `8` | (CLI: `--prompt-concurrency`) |
| `batch_prefill_step_size` | `2048` | (CLI: `--prefill-step-size`) |
| `disable_batching` | `false` | |
| `default_max_tokens` | none | (CLI: `--max-tokens`) |
| `default_temperature` | none | (CLI: `--temperature`) |
| `default_top_p` | none | (CLI: `--top-p`) |
| `default_top_k` | none | (CLI: `--top-k`) |
| `default_min_p` | none | (CLI: `--min-p`) |
| `default_repetition_penalty` | none | (CLI: `--repetition-penalty`) |
| `default_presence_penalty` | none | (CLI: `--presence-penalty`) |
| `default_xtc_probability` | none | (CLI: `--xtc-probability`) |
| `default_xtc_threshold` | none | (CLI: `--xtc-threshold`) |
| `default_seed` | none | (CLI: `--seed`) |
| `default_repetition_context_size` | none | (CLI: `--repetition-context-size`) |

### Example

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  log_level: INFO

models:
  - model_path: mlx-community/MiniMax-M2.5-4bit
    model_type: lm
    served_model_name: Minimax-M2.5
    queue_timeout: 600          # request timeout, in seconds
    enable_auto_tool_choice: true
    tool_call_parser: minimax_m2
    reasoning_parser: minimax_m2

  - model_path: black-forest-labs/FLUX.2-klein-4B
    model_type: image-generation
    config_name: flux2-klein-4b
    quantize: 4
    served_model_name: flux2-klein-4b
    on_demand: true             # YAML-only: lazy load + idle unload
    on_demand_idle_timeout: 120
```

---

## What goes where — quick reference

| Concern | Single-model | Multi-handler (`--config`) |
|---------|--------------|----------------------------|
| Server host/port/logging | `--host`, `--port`, `--log-level`, … | `server:` section in YAML |
| Per-model settings | CLI flags | each entry under `models:` |
| Request timeout | `--queue-timeout` | per-model `queue_timeout` |
| Lazy / on-demand loading | *not available* | `on_demand`, `on_demand_idle_timeout` |
| Custom message converter | *not available* | `message_converter` |
| LoRA paths/scales | comma-separated strings | YAML lists |

### Naming traps

- **Flag ≠ field for several options.** `--max-bytes` → `prompt_cache_max_bytes`;
  `--decode-concurrency` → `batch_completion_size`;
  `--prompt-concurrency` → `batch_prefill_size`;
  `--prefill-step-size` → `batch_prefill_step_size`.
- **Sampling flags drop `default_`.** CLI `--temperature` is YAML
  `default_temperature`, and so on for the whole sampling group.
- **YAML-only keys** (no CLI equivalent): `on_demand`,
  `on_demand_idle_timeout`, `message_converter`.
- **Single-model-only**: there is no on-demand/idle-unload in single-model mode;
  use a one-entry `--config` YAML if you need it.

### Valid `model_type` values

`lm`, `multimodal`, `image-generation`, `image-edit`, `embeddings`, `whisper`.
