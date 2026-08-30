# Configuration file reference

A YAML file describes one server and the models it serves:

```bash
mlx-openai-server launch --config /path/to/config.yaml
```

The file has two sections: `server:` for host, logging and lifecycle limits, and
`models:` for the list of models. Only `models` is required, and every entry
needs at least a `model_path`.

```yaml
server:
  host: "127.0.0.1"
  port: 8000

models:
  - model_path: mlx-community/Qwen3-4B-4bit
    model_type: lm
    served_model_name: chat
```

`examples/config.yaml` is a tracked, machine-independent starting point. Keep
your real file outside the repository when it holds absolute paths specific to
one machine.

## `server:` section

| Key                  | Default     | Notes                                                                     |
| -------------------- | ----------- | ------------------------------------------------------------------------- |
| `host`               | `"0.0.0.0"` | Bind address. Use `127.0.0.1` to refuse connections from other machines.  |
| `port`               | `8000`      | TCP port.                                                                 |
| `log_level`          | `"INFO"`    | `DEBUG`, `INFO`, `WARNING`, `ERROR`.                                      |
| `log_file`           | unset       | Path to a log file, in addition to stdout.                                |
| `no_log_file`        | `false`     | Disable file logging entirely.                                            |
| `max_loaded_models`  | `1`         | Maximum on-demand models resident at once. `0` removes the limit.         |
| `model_load_timeout` | `300`       | Seconds a request waits for a free slot before failing. Must be positive. |

Unknown keys in this section are **ignored silently**, so a misspelled
`max_loaded_model` leaves the default in place with no warning. Unknown keys in
a model entry raise an error instead, which is also what happens if a
server-level key such as `max_loaded_models` is indented under a model by
mistake.

## `models:` section

### Identity and routing

| Key                 | Default      | Notes                                                                                 |
| ------------------- | ------------ | ------------------------------------------------------------------------------------- |
| `model_path`        | **required** | Hugging Face repository id or an absolute local path.                                 |
| `model_type`        | `lm`         | One of `lm`, `multimodal`, `image-generation`, `image-edit`, `embeddings`, `whisper`. |
| `served_model_name` | `model_path` | The name clients send as `"model"`. Must be unique across the file.                   |
| `version`           | unset        | Adds a `<served_model_name>:<version>` route and appears in `/v1/models`.             |
| `aliases`           | unset        | Extra names that route to this model.                                                 |

`version` and `aliases` share one namespace with `served_model_name`. A
collision anywhere in that namespace is rejected at startup instead of leaving
one name unreachable, a `version` may not contain `:` (it is joined with that
separator), and no name may be blank or contain whitespace. See
[MODEL-LIFECYCLE.md](MODEL-LIFECYCLE.md#versioning-and-aliases) for how aliases
behave at request time.

### Lifecycle

| Key                      | Default | Notes                                                                                                                            |
| ------------------------ | ------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `on_demand`              | `false` | `false` loads the model at startup and keeps it resident for the process lifetime. `true` loads it on the first request.         |
| `on_demand_idle_timeout` | `300`   | Seconds of inactivity before an on-demand model is unloaded. Accepts a number or a duration string such as `30s`, `5m`, `1h30m`. |

Requests may override the retention period per call with `keep_alive`; `0`
unloads once the request or stream ends and a negative value keeps the model
indefinitely. Active streams are never evicted.

### Memory and throughput

| Key                       | Default       | Applies to         | Notes                                                                                                   |
| ------------------------- | ------------- | ------------------ | ------------------------------------------------------------------------------------------------------- |
| `context_length`          | model default | `lm`, `multimodal` | Maximum prompt + generation length.                                                                     |
| `kv_bits`                 | unset         | `lm`, `multimodal` | Quantize the KV cache to this bit width, e.g. `4` or `8`. Ignored with a warning for other model types. |
| `kv_group_size`           | `64`          | `lm`, `multimodal` | Group size for KV quantization.                                                                         |
| `quantized_kv_start`      | `0`           | `lm`, `multimodal` | Token position after which quantization begins, so short prompts stay at full precision.                |
| `prompt_cache_size`       | `10`          | `lm` only          | Number of prompt prefixes retained for reuse across requests.                                           |
| `prompt_cache_max_bytes`  | unbounded     | `lm` only          | Memory ceiling for the prompt cache.                                                                    |
| `prompt_cache_dir`        | unset         | `lm` only          | Persist the prompt cache under this directory so reuse survives an unload.                              |
| `batch_completion_size`   | `32`          | `lm`, `multimodal` | Sequences generating concurrently in the continuous batcher.                                            |
| `batch_prefill_size`      | `8`           | `lm`, `multimodal` | Prompts prefilled concurrently.                                                                         |
| `batch_prefill_step_size` | `2048`        | `lm`, `multimodal` | Tokens per prefill step.                                                                                |
| `disable_batching`        | `false`       | `lm`, `multimodal` | Serve one request at a time. Required for per-request `seed`.                                           |
| `queue_timeout`           | `300`         | all                | Seconds a request may wait in this model's queue.                                                       |
| `queue_size`              | `100`         | all                | Maximum queued requests before new ones are rejected.                                                   |

Reusing a cached prefix skips prompt processing entirely, which is the largest
single win for multi-turn conversations; `kv_bits` instead trades a little
quality for a much smaller cache, which is what makes long contexts fit.

### Generation behaviour

| Key                       | Default | Applies to         | Notes                                                              |
| ------------------------- | ------- | ------------------ | ------------------------------------------------------------------ |
| `enable_auto_tool_choice` | `false` | `lm`, `multimodal` | Allow the model to emit tool calls without an explicit choice.     |
| `tool_call_parser`        | unset   | `lm`, `multimodal` | Must match the model family; see the parser table in the README.   |
| `reasoning_parser`        | unset   | `lm`, `multimodal` | Extracts reasoning into `reasoning_content`.                       |
| `message_converter`       | derived | `lm`, `multimodal` | Overrides the converter inferred from the parsers.                 |
| `chat_template_file`      | unset   | `lm`, `multimodal` | Replace the tokenizer's chat template.                             |
| `trust_remote_code`       | `false` | `lm`, `multimodal` | Allow custom model code from the repository.                       |
| `disable_auto_resize`     | `false` | `multimodal`       | Send images at their original resolution.                          |
| `draft_model_path`        | unset   | `lm` only          | Draft model for speculative decoding. Ignored elsewhere.           |
| `num_draft_tokens`        | `2`     | `lm` only          | Tokens drafted per step.                                           |
| `debug`                   | `false` | `lm`, `multimodal` | Verbose per-request logging, including prompt-processing progress. |

Sampling defaults are set per model with `default_max_tokens`,
`default_temperature`, `default_top_p`, `default_top_k`, `default_min_p`,
`default_repetition_penalty`, `default_presence_penalty`,
`default_xtc_probability`, `default_xtc_threshold`, `default_seed` and
`default_repetition_context_size`. Each applies when a request omits the
corresponding field.

### Image and adapter options

| Key           | Default        | Applies to                       | Notes                                                                           |
| ------------- | -------------- | -------------------------------- | ------------------------------------------------------------------------------- |
| `config_name` | family default | `image-generation`, `image-edit` | Defaults to `flux-schnell` and `flux-kontext-dev` respectively, with a warning. |
| `quantize`    | unset          | `image-generation`, `image-edit` | Weight quantization, e.g. `4` or `8`.                                           |
| `lora_paths`  | unset          | `image-generation`, `image-edit` | List of adapter paths.                                                          |
| `lora_scales` | unset          | `image-generation`, `image-edit` | List of scales matching `lora_paths`.                                           |

## Worked example

A chat model that is pinned by version, addressed by a stable alias, quantizes
its KV cache and persists its prompt cache, next to an embedding model that
shares the same single resident slot:

```yaml
server:
  host: "127.0.0.1"
  port: 8000
  max_loaded_models: 1 # one model resident at a time
  model_load_timeout: 600 # a large model may take minutes to load

models:
  - model_path: /absolute/path/to/your/chat-model
    model_type: multimodal
    served_model_name: assistant-2025-08
    version: "2.1" # also assistant-2025-08:2.1
    aliases: [assistant] # what applications hardcode
    on_demand: true
    on_demand_idle_timeout: 5m
    enable_auto_tool_choice: true
    tool_call_parser: qwen3_coder # must match the model
    reasoning_parser: qwen3_5
    kv_bits: 4
    kv_group_size: 64
    quantized_kv_start: 1024 # keep short prompts unquantized
    prompt_cache_dir: /absolute/path/to/cache

  - model_path: mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ
    model_type: embeddings
    served_model_name: embeddings
    aliases: [embed]
    on_demand: true
    on_demand_idle_timeout: 2m
```

With `max_loaded_models: 1`, whichever model is requested next waits if the slot
is busy with an active request, and replaces the least recently used idle worker
otherwise. Each model runs in its own subprocess, so a model crash cannot take
the server down with it.

## Startup errors

| Message                                                   | Cause                                                        |
| --------------------------------------------------------- | ------------------------------------------------------------ |
| `'models' section must be a non-empty list`               | No `models:` key, or it is empty.                            |
| `missing required key 'model_path'`                       | A model entry has no `model_path`.                           |
| `unexpected keyword argument '<key>'`                     | A misspelled key, or a `server:` key indented under a model. |
| `Invalid model_type '<type>'`                             | `model_type` is not one of the six valid values.             |
| `Duplicate served_model_name '<name>'`                    | Two models share a name.                                     |
| `Alias '<name>' ... collides`                             | An alias or version tag duplicates another name in the file. |
| `has a version containing ':'`                            | Use `aliases` for a full `name:tag` form.                    |
| `max_loaded_models must be greater than or equal to zero` | Negative limit.                                              |

## See also

- [MODEL-LIFECYCLE.md](MODEL-LIFECYCLE.md) — lifecycle endpoints, `keep_alive`,
  running as a macOS service, aliases at request time, LangChain and LangGraph.
- `examples/config.yaml` — a runnable starting point.
- The README's _Server Options_ section — the equivalent single-model CLI flags.
