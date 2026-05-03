# mlx-openai-server

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

OpenAI-compatible API server for local MLX models on Apple Silicon. It serves text, multimodal, image, embedding, and Whisper models through familiar OpenAI SDK endpoints.

> Requires macOS on Apple Silicon and Python 3.11+.

## Contents

- [Feature Launch](#feature-launch)
- [Install](#install)
- [Start a Server](#start-a-server)
- [API Usage](#api-usage)
- [Server Options](#server-options)
- [Multi-Model Config](#multi-model-config)
- [Long Context and Metal OOM](#long-context-and-metal-oom)
- [Advanced LM Options](#advanced-lm-options)
- [Troubleshooting](#troubleshooting)
- [Examples and Demos](#examples-and-demos)

## Feature Launch

Darwin 36B Opus is now available in MLX format for local text inference:

- Original model: [FINAL-Bench/Darwin-36B-Opus](https://huggingface.co/FINAL-Bench/Darwin-36B-Opus)
- MLX text-only 8-bit conversion: [GiaHuy/Darwin-36B-Opus-mlx-text-only-8bit](https://huggingface.co/GiaHuy/Darwin-36B-Opus-mlx-text-only-8bit)

Launch it with the required reasoning and tool-call parsers:

```bash
mlx-openai-server launch --model-path Darwin-36B-Opus-mlx-text-only-8bit --reasoning-parser qwen3_moe --tool-call-parser qwen3_coder --debug --served-model-name Darwin-36B-Opus
```

Darwin 36B Opus should be served with both `--reasoning-parser qwen3_moe` and `--tool-call-parser qwen3_coder`; without them, reasoning and tool-call output will not be parsed correctly.

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate

uv pip install mlx-openai-server
```

Install from GitHub instead:

```bash
uv pip install git+https://github.com/cubist38/mlx-openai-server.git
```

Whisper transcription also needs ffmpeg:

```bash
brew install ffmpeg
```

## Start a Server

Text model:

```bash
mlx-openai-server launch \
  --model-type lm \
  --model-path mlx-community/Qwen3-Coder-Next-4bit \
  --reasoning-parser qwen3_moe \
  --tool-call-parser qwen3_coder
```

Point OpenAI-compatible clients to:

```text
http://localhost:8000/v1
```

Use any non-empty API key, for example `not-needed`.

Common launch modes:

```bash
# Multimodal text/image/audio
mlx-openai-server launch \
  --model-type multimodal \
  --model-path <mlx-vlm-model>

# Image generation
mlx-openai-server launch \
  --model-type image-generation \
  --model-path <flux-or-qwen-image-model> \
  --config-name flux-dev \
  --quantize 8

# Image editing
mlx-openai-server launch \
  --model-type image-edit \
  --model-path <flux-or-qwen-image-edit-model> \
  --config-name flux-kontext-dev \
  --quantize 8

# Embeddings
mlx-openai-server launch \
  --model-type embeddings \
  --model-path <embedding-model>

# Whisper transcription
mlx-openai-server launch \
  --model-type whisper \
  --model-path mlx-community/whisper-large-v3-mlx
```

Supported model types:

| Type | Backend | Endpoint family |
|------|---------|-----------------|
| `lm` | `mlx-lm` | chat, responses |
| `multimodal` | `mlx-vlm` | chat, responses |
| `image-generation` | `mflux` | image generation |
| `image-edit` | `mflux` | image editing |
| `embeddings` | `mlx-embeddings` | embeddings |
| `whisper` | `mlx-whisper` | audio transcription |

Image `--config-name` values:

- Generation: `flux-schnell`, `flux-dev`, `flux-krea-dev`, `flux2-klein-4b`, `flux2-klein-9b`, `qwen-image`, `z-image-turbo`, `fibo`
- Editing: `flux-kontext-dev`, `flux2-klein-edit-4b`, `flux2-klein-edit-9b`, `qwen-image-edit`

## API Usage

### Chat

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="mlx-community/Qwen3-Coder-Next-4bit",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print(response.choices[0].message.content)
```

### Vision

```python
import base64
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

with open("image.jpg", "rb") as f:
    image = base64.b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model="local-multimodal",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
            ],
        }
    ],
)
print(response.choices[0].message.content)
```

### Images

```python
import base64
from io import BytesIO

import openai
from PIL import Image

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.images.generate(
    model="local-image-generation-model",
    prompt="A mountain lake at sunset",
    size="1024x1024",
)

image = Image.open(BytesIO(base64.b64decode(response.data[0].b64_json)))
image.show()
```

### Embeddings

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.embeddings.create(
    model="local-embedding-model",
    input=["The quick brown fox jumps over the lazy dog"],
)
print(len(response.data[0].embedding))
```

### Responses API

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.responses.create(
    model="local-model",
    input="Write a three sentence story.",
)

for item in response.output:
    if item.type == "message":
        for part in item.content:
            if getattr(part, "text", None):
                print(part.text)
```

Supported endpoints:

| Endpoint | Model types |
|----------|-------------|
| `GET /v1/models` | all |
| `POST /v1/chat/completions` | `lm`, `multimodal` |
| `POST /v1/responses` | `lm`, `multimodal` |
| `POST /v1/images/generations` | `image-generation` |
| `POST /v1/images/edits` | `image-edit` |
| `POST /v1/embeddings` | `embeddings` |
| `POST /v1/audio/transcriptions` | `whisper` |

The request `model` should be the model path, `--served-model-name`, or YAML `served_model_name`.

## Server Options

| Option | Default | Notes |
|--------|---------|-------|
| `--model-path` | required | Local path or Hugging Face repo |
| `--model-type` | `lm` | `lm`, `multimodal`, `image-generation`, `image-edit`, `embeddings`, `whisper` |
| `--served-model-name` | model path | Alias accepted in API requests |
| `--host` | `0.0.0.0` | Bind host |
| `--port` | `8000` | Bind port |
| `--context-length` | model default | LM and multimodal context/cache length |
| `--max-tokens` | `100000` | Default generated tokens when request omits `max_tokens` |
| `--temperature` | `1.0` | Default sampling temperature |
| `--top-p` | `1.0` | Default nucleus sampling |
| `--top-k` | `20` | Default top-k sampling |
| `--repetition-penalty` | `1.0` | Default repetition penalty |
| `--config-name` | model-dependent | Image model preset |
| `--quantize` | unset | Image model quantization: `4`, `8`, or `16` |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `--no-log-file` | `false` | Disable file logging |

LM-specific memory and batching options:

| Option | Default | Notes |
|--------|---------|-------|
| `--decode-concurrency` | `32` | Max concurrent batch decode sequences |
| `--prompt-concurrency` | `8` | Max prompts prefilled together |
| `--prefill-step-size` | `2048` | Tokens per prefill step |
| `--disable-batching` | `false` | Disable continuous batching; required if per-request positive `seed` values must be honored |
| `--prompt-cache-size` | `10` | Retained prompt KV cache entries |
| `--max-bytes` | unbounded | Prompt KV cache byte budget |
| `--prompt-cache-dir` | temp dir | Directory for disk-backed prompt KV cache payloads |
| `--kv-bits` | unset | KV cache quantization bits, usually `4` or `8` |
| `--kv-group-size` | `64` | KV quantization group size |
| `--quantized-kv-start` | `0` | Token step where KV quantization starts |
| `--draft-model-path` | unset | Smaller draft model for speculative decoding |
| `--num-draft-tokens` | `2` | Draft tokens proposed per step |

When continuous batching is enabled, LM and multimodal requests stay on the batched generation path whenever the backend supports it. Per-request positive `seed` values are ignored in this mode because the batch scheduler shares one RNG lane; launch with `--disable-batching` if request-level seed reproducibility is required.

Prompt KV caches are reused only by the generation path that created them. Batched and non-batched requests use separate cache entries because MLX cache and stream objects are thread-affine.

Multimodal (`mlx-vlm`) batching intentionally does not use prompt-prefix caching; current mlx-vlm prompt-cache support is limited, so VLM batching focuses on admitting parallel requests into the shared decode/prefill batch.

## Multi-Model Config

Use YAML when you want several models behind one server:

```bash
mlx-openai-server launch --config config.yaml
```

Example:

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  log_level: INFO

models:
  - model_path: mlx-community/MiniMax-M2.5-4bit
    model_type: lm
    served_model_name: minimax
    tool_call_parser: minimax_m2
    reasoning_parser: minimax_m2

  - model_path: black-forest-labs/FLUX.2-klein-4B
    model_type: image-generation
    served_model_name: flux2-klein-4b
    config_name: flux2-klein-4b
    quantize: 4
    on_demand: true
    on_demand_idle_timeout: 120
```

Important YAML keys:

| Key | Notes |
|-----|-------|
| `model_path`, `model_type`, `served_model_name` | Model identity and routing |
| `context_length` | LM/multimodal context length |
| `prompt_cache_size`, `prompt_cache_max_bytes`, `prompt_cache_dir` | Prompt KV cache limits and disk location |
| `batch_completion_size`, `batch_prefill_size`, `batch_prefill_step_size`, `disable_batching` | Continuous batching controls |
| `kv_bits`, `kv_group_size`, `quantized_kv_start` | KV cache quantization |
| `default_max_tokens` | Default generated tokens |
| `on_demand`, `on_demand_idle_timeout` | Load large models only when requested |

In multi-model mode, each model runs in a spawned subprocess. This isolates MLX/Metal runtime state and avoids process-fork semaphore issues on macOS.

## Long Context and Metal OOM

Large prompts, high concurrency, and long generations all increase KV-cache memory. If Metal runs out of memory, macOS may terminate the process with:

```text
libc++abi: terminating due to uncaught exception of type std::runtime_error: [METAL] Command buffer execution failed: Internal Error (0000000e:Internal Error)
```

Start with conservative settings:

```bash
mlx-openai-server launch \
  --model-type lm \
  --model-path <model-path> \
  --context-length 8192 \
  --decode-concurrency 4 \
  --prompt-concurrency 1 \
  --prefill-step-size 512 \
  --max-tokens 2048 \
  --prompt-cache-size 1 \
  --max-bytes 2147483648 \
  --kv-bits 4 \
  --kv-group-size 64 \
  --quantized-kv-start 0
```

Tune in this order:

1. Lower `--prompt-concurrency` and `--prefill-step-size` to reduce prefill spikes.
2. Lower `--decode-concurrency` to reduce active KV caches.
3. Lower `--max-tokens` to bound generation growth.
4. Lower `--prompt-cache-size` and set `--max-bytes` to limit retained caches.
5. Use `--kv-bits 4` for supported LM/multimodal models.
6. Reduce `--context-length` if the model still does not fit.

YAML equivalent:

```yaml
models:
  - model_path: <model-path>
    model_type: lm
    context_length: 8192
    batch_completion_size: 4
    batch_prefill_size: 1
    batch_prefill_step_size: 512
    default_max_tokens: 2048
    prompt_cache_size: 1
    prompt_cache_max_bytes: 2147483648
    kv_bits: 4
    kv_group_size: 64
    quantized_kv_start: 0
```

For large models on macOS 15+, you can also raise the wired memory limit:

```bash
bash configure_mlx.sh
```

## Advanced LM Options

### Tool and Reasoning Parsers

Some models need parser flags for tool calls or reasoning blocks:

```bash
mlx-openai-server launch \
  --model-type lm \
  --model-path <model> \
  --tool-call-parser qwen3 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice
```

Common parser names include `qwen3`, `qwen3_5`, `glm4_moe`, `qwen3_coder`, `qwen3_moe`, `qwen3_next`, `qwen3_vl`, `harmony`, and `minimax_m2`.

Message converters are auto-detected from parser selection when a compatible converter exists.

### Custom Chat Templates

```bash
mlx-openai-server launch \
  --model-type lm \
  --model-path <model> \
  --chat-template-file /path/to/template.jinja
```

### Speculative Decoding

```bash
mlx-openai-server launch \
  --model-type lm \
  --model-path <main-model> \
  --draft-model-path <smaller-draft-model> \
  --num-draft-tokens 4
```

Speculative decoding is only available for `lm`. It is not used by the continuous batch path.

### Structured Outputs

Chat completions accept OpenAI-style `response_format` JSON schema. The Responses API also supports `client.responses.parse()` with Pydantic models. See `examples/structured_outputs_examples.ipynb` and `examples/responses_api.ipynb`.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Model does not fit in memory | Use a smaller or pre-quantized model, lower `--context-length`, and see [Long Context and Metal OOM](#long-context-and-metal-oom). |
| Metal OOM during batching | Lower `--prompt-concurrency`, `--prefill-step-size`, `--decode-concurrency`, and `--max-tokens`. |
| `There is no Stream(gpu, N) in current thread` | Keep prompt-cache persistence on the worker thread and avoid sharing cache payloads across batch/non-batch paths; use `--disable-batching` when request seeds or single-request behavior are required. |
| Port already in use | Pass `--port 8001` or another free port. |
| Image model memory is too high | Use `--quantize 4` or `--quantize 8`. |
| Model loading says parameters are missing or unexpected | Upgrade the backend package from source. |
| Hugging Face download fails | Check network access and Hugging Face authentication for gated models. |

Upgrade backend packages for newly released model architectures:

```bash
uv pip install git+https://github.com/ml-explore/mlx-lm.git
uv pip install git+https://github.com/Blaizzy/mlx-vlm.git
uv pip install git+https://github.com/Blaizzy/mlx-embeddings.git
```

## Examples and Demos

Example notebooks live in `examples/`:

| Area | Notebooks |
|------|-----------|
| Text and Responses API | `responses_api.ipynb`, `simple_rag_demo.ipynb` |
| Vision | `vision_examples.ipynb` |
| Audio | `audio_examples.ipynb`, `transcription_examples.ipynb` |
| Embeddings | `embedding_examples.ipynb`, `lm_embeddings_examples.ipynb`, `vlm_embeddings_examples.ipynb` |
| Images | `image_generations.ipynb`, `image_edit.ipynb` |
| Structured outputs | `structured_outputs_examples.ipynb` |

Demos:

- [Darwin 36B Opus on MLX OpenAI Server](https://www.youtube.com/watch?v=iFngahmaJ3Y)
- [MLX OpenAI Server + Codex](https://youtu.be/CY5yVS8P5Vg)
- [OpenClaw AI Agent powered by Gemma 4](https://www.youtube.com/watch?v=5MSlDCH37Kc)
- [Serving Multiple Models at Once](https://www.youtube.com/watch?v=f7WXSOPZ5H4)

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Make changes with tests.
4. Submit a pull request.

Use conventional commit prefixes when possible, for example `fix:`, `feat:`, `docs:`, or `test:`.

## Support

- Issues: [GitHub Issues](https://github.com/cubist38/mlx-openai-server/issues)
- Discussions: [GitHub Discussions](https://github.com/cubist38/mlx-openai-server/discussions)
- License: [MIT](LICENSE)

## Acknowledgments

Built on [MLX](https://github.com/ml-explore/mlx), [mlx-lm](https://github.com/ml-explore/mlx-lm), [mlx-vlm](https://github.com/Blaizzy/mlx-vlm), [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings), [mflux](https://github.com/filipstrand/mflux), [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper), and [mlx-community](https://huggingface.co/mlx-community).
