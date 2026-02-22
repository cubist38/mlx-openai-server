# mlx-openai-server

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

A high-performance OpenAI-compatible API server for MLX models. Run text, vision, audio, and image generation models locally on Apple Silicon with a drop-in OpenAI replacement.

> **Note:** Requires **macOS with M-series chips** (MLX is optimized for Apple Silicon).

## Key Features

- 🚀 **OpenAI-compatible API** - Drop-in replacement for OpenAI services
- 🖼️ **Multimodal support** - Text, vision, audio, and image generation/editing
- 🎨 **Flux-series models** - Image generation (schnell, dev, krea-dev, flux-2-klein) and editing (kontext, qwen-image-edit)
- 🔌 **Easy integration** - Works with existing OpenAI client libraries
- 📦 **Multi-model mode** - Run multiple models in one server via a YAML config; route requests by model ID
- ⚡ **Performance** - Configurable quantization (4/8/16-bit), context length, and speculative decoding (lm)
- 🎛️ **LoRA adapters** - Fine-tuned image generation and editing
- 📈 **Queue management** - Built-in request queuing and monitoring

## Installation

### Prerequisites
- macOS with Apple Silicon (M-series)
- Python 3.11+

### Quick Install

```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install from PyPI
pip install mlx-openai-server

# Or install from GitHub
pip install git+https://github.com/cubist38/mlx-openai-server.git
```

### Optional: Whisper Support
For audio transcription models, install ffmpeg:
```bash
brew install ffmpeg
```

## Quick Start

### Start the Server

```bash
# Text-only or multimodal models
mlx-openai-server launch \
  --model-path <path-to-mlx-model> \
  --model-type <lm|multimodal>

# Text-only with speculative decoding (faster generation using a smaller draft model)
mlx-openai-server launch \
  --model-path <path-to-main-model> \
  --model-type lm \
  --draft-model-path <path-to-draft-model> \
  --num-draft-tokens 4

# Image generation (Flux-series)
mlx-openai-server launch \
  --model-type image-generation \
  --model-path <path-to-flux-model> \
  --config-name flux-dev \
  --quantize 8

# Image editing
mlx-openai-server launch \
  --model-type image-edit \
  --model-path <path-to-flux-model> \
  --config-name flux-kontext-dev \
  --quantize 8

# Embeddings
mlx-openai-server launch \
  --model-type embeddings \
  --model-path <embeddings-model-path>

# Whisper (audio transcription)
mlx-openai-server launch \
  --model-type whisper \
  --model-path mlx-community/whisper-large-v3-mlx
```

### Server Parameters

- `--model-path`: Path to MLX model (local or HuggingFace repo)
- `--model-type`: `lm`, `multimodal`, `image-generation`, `image-edit`, `embeddings`, or `whisper`
- `--config-name`: For image models - `flux-schnell`, `flux-dev`, `flux-krea-dev`, `flux-kontext-dev`, `flux2-klein-4b`, `flux2-klein-9b`, `qwen-image`, `qwen-image-edit`, `z-image-turbo`, `fibo`
- `--quantize`: Quantization level - `4`, `8`, or `16` (image models)
- `--context-length`: Max sequence length for memory optimization
- `--draft-model-path`: Path to draft model for speculative decoding (lm only)
- `--num-draft-tokens`: Draft tokens per step for speculative decoding (lm only, default: 2)
- **Sampling parameters** (defaults used when the API request omits them):
  - `--max-tokens`: Default maximum number of tokens to generate
  - `--temperature`: Default sampling temperature
  - `--top-p`: Default nucleus sampling (top-p) probability
  - `--top-k`: Default top-k sampling parameter
- `--max-concurrency`: Concurrent requests (default: 1)
- `--queue-timeout`: Request timeout in seconds (default: 300)
- `--lora-paths`: Comma-separated LoRA adapter paths (image models)
- `--lora-scales`: Comma-separated LoRA scales (must match paths)
- `--log-level`: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` (default: `INFO`)
- `--no-log-file`: Disable file logging (console only)

## Launching Multiple Models

You can run several models in one server using a YAML config file. Each model gets its own handler; requests are routed by the **model ID** you use in the API (the `model` field in the request).

**Video:** [Serving Multiple Models at Once? mlx-openai-server + OpenWebUI Test](https://www.youtube.com/watch?v=f7WXSOPZ5H4)

### Start with a config file

```bash
mlx-openai-server launch --config config.yaml
```

You must provide either `--config` (multi-handler) or `--model-path` (single model). You cannot mix them.

### YAML config format

Create a YAML file with a `server` section (host, port, logging) and a `models` list. Each entry in `models` defines one model and supports the same options as the CLI (model path, type, context length, queue settings, etc.).

| Key | Required | Description |
|-----|----------|-------------|
| `model_path` | Yes | Path or HuggingFace repo of the model |
| `model_type` | No | `lm`, `multimodal`, `image-generation`, `image-edit`, `embeddings`, `whisper` (default: `lm`) |
| `model_id` | No | ID used in API requests; defaults to `model_path` if omitted |
| `context_length` | No | Max context length (lm / multimodal) |
| `max_concurrency`, `queue_timeout`, `queue_size` | No | Per-model queue settings |

Example `config.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  log_level: INFO
  # log_file: logs/app.log     # uncomment to log to file
  # no_log_file: true           # uncomment to disable file logging

models:
  # Language model
  - model_path: mlx-community/GLM-4.7-Flash-8bit
    model_type: lm
    model_id: glm-4.7-flash    # optional alias (defaults to model_path)
    enable_auto_tool_choice: true
    tool_call_parser: glm4_moe
    reasoning_parser: glm47_flash
    message_converter: glm4_moe

  # Another language model
  - model_path: mlx-community/Qwen3-Coder-Next-4bit
    model_type: lm
    max_concurrency: 1
    tool_call_parser: qwen3_coder
    message_converter: qwen3_coder


  - model_path: mlx-community/Qwen3-VL-2B-Instruct-4bit
    model_type: multimodal
    tool_call_parser: qwen3_vl

  - model_path: black-forest-labs/FLUX.2-klein-4B
    model_type: image-generation
    config_name: flux2-klein-4b
    quantize: 4
    model_id: flux2-klein-4b
```

A full example is in `examples/config.yaml`.

### Multi-handler process isolation (HandlerProcessProxy)

In multi-handler mode, each model runs in a **dedicated subprocess** spawned via `multiprocessing.get_context("spawn")`. The main FastAPI process uses a `HandlerProcessProxy` to forward requests to the child process over multiprocessing queues.

This design prevents MLX Metal/GPU semaphore leaks on macOS. When MLX arrays or Metal runtime state are shared across forked processes, the resource tracker can report leaked semaphore objects at shutdown ([ml-explore/mlx#2457](https://github.com/ml-explore/mlx/issues/2457)). Using **spawn** instead of the default fork gives each model a clean Metal context, avoiding those warnings.

```
┌─────────────────────────────────────┐     ┌─────────────────────────────────────┐
│  Main Process (FastAPI)             │     │  Child Process (Handler)             │
│  ┌───────────────────────────────┐  │     │  ┌───────────────────────────────┐  │
│  │  HandlerProcessProxy          │  │     │  │  Concrete handler (e.g.       │  │
│  │  • request_queue ────────────┼──┼─────┼─>│    MLXLMHandler)              │  │
│  │  • response_queue <──────────┼──┼<────┼──│  • Model (MLX_LM)              │  │
│  │  • generate_*() forwards RPC  │  │     │  │  • InferenceWorker (thread)   │  │
│  └───────────────────────────────┘  │     │  └───────────────────────────────┘  │
└─────────────────────────────────────┘     └─────────────────────────────────────┘
```

The proxy exposes the same interface as the concrete handlers (`generate_text_stream`, `generate_embeddings_response`, etc.), so API endpoints work without changes. Requests and responses are serialized across the process boundary via queues; non-picklable objects (e.g. uploaded files) are pre-processed in the main process before being sent as file paths.

### Using the API with multiple models

Set the `model` field in your request to the **model ID** (the `model_id` from the config, or `model_path` if you did not set `model_id`). The server looks up the handler for that ID and runs the request on the correct model.

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Use the first model (qwen2.5-7b)
r1 = client.chat.completions.create(
    model="glm-4.7-flash",
    messages=[{"role": "user", "content": "Say hello in one word."}],
)
print(r1.choices[0].message.content)

# Use the second model (full path as model_id)
r2 = client.chat.completions.create(
    model="mlx-community/Qwen3-Coder-Next-4bit",
    messages=[{"role": "user", "content": "Say hello in one word."}],
)
print(r2.choices[0].message.content)
```

- **GET `/v1/models`** returns all loaded models (their IDs).
- If you send a `model` that is not in the config, the server returns **404** with an error listing available models.

## Supported Model Types

1. **Text-only** (`lm`) - Language models via `mlx-lm`
2. **Multimodal** (`multimodal`) - Text, images, audio via `mlx-vlm`
3. **Image generation** (`image-generation`) - Flux-series, Qwen Image, Z-Image Turbo, Fibo
4. **Image editing** (`image-edit`) - Flux kontext, Qwen Image Edit
5. **Embeddings** (`embeddings`) - Text embeddings via `mlx-embeddings`
6. **Whisper** (`whisper`) - Audio transcription (requires ffmpeg)

### Image Model Configurations

**Generation:**
- `flux-schnell` - Fast (4 steps, no guidance)
- `flux-dev` - Balanced (25 steps, 3.5 guidance)
- `flux-krea-dev` - High quality (28 steps, 4.5 guidance)
- `flux2-klein-4b` / `flux2-klein-9b` - Flux 2 Klein models
- `qwen-image` - Qwen image generation (50 steps, 4.0 guidance)
- `z-image-turbo` - Z-Image Turbo
- `fibo` - Fibo model

**Editing:**
- `flux-kontext-dev` - Context-aware editing (28 steps, 2.5 guidance)
- `flux2-klein-edit-4b` / `flux2-klein-edit-9b` - Flux 2 Klein editing
- `qwen-image-edit` - Qwen image editing (50 steps, 4.0 guidance)

## Using the API

The server provides OpenAI-compatible endpoints. Use standard OpenAI client libraries:

### Text Completion

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "What is the capital of France?"}]
)
print(response.choices[0].message.content)
```

### Vision (Multimodal)

```python
import openai
import base64

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

with open("image.jpg", "rb") as f:
    base64_image = base64.b64encode(f.read()).decode('utf-8')

response = client.chat.completions.create(
    model="local-multimodal",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    }]
)
print(response.choices[0].message.content)
```

### Image Generation

```python
import openai
import base64
from io import BytesIO
from PIL import Image

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.images.generate(
    prompt="A serene landscape with mountains and a lake at sunset",
    model="local-image-generation-model",
    size="1024x1024"
)

image_data = base64.b64decode(response.data[0].b64_json)
image = Image.open(BytesIO(image_data))
image.show()
```

### Image Editing

```python
import openai
import base64
from io import BytesIO
from PIL import Image

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

with open("image.png", "rb") as f:
    result = client.images.edit(
        image=f,
        prompt="make it like a photo in 1800s",
        model="flux-kontext-dev"
    )

image_data = base64.b64decode(result.data[0].b64_json)
image = Image.open(BytesIO(image_data))
image.show()
```

### Function Calling

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

messages = [{"role": "user", "content": "What is the weather in Tokyo?"}]
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather in a given city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name"}
            }
        }
    }
}]

completion = client.chat.completions.create(
    model="local-model",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

if completion.choices[0].message.tool_calls:
    tool_call = completion.choices[0].message.tool_calls[0]
    print(f"Function: {tool_call.function.name}")
    print(f"Arguments: {tool_call.function.arguments}")
```

### Embeddings

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.embeddings.create(
    model="local-model",
    input=["The quick brown fox jumps over the lazy dog"]
)

print(f"Embedding dimension: {len(response.data[0].embedding)}")
```

### Responses API

The server exposes the OpenAI [Responses API](https://platform.openai.com/docs/api-reference/responses) at `POST /v1/responses`. Use `client.responses.create()` with the OpenAI SDK for text and multimodal (lm/multimodal) models.

**Text input (non-streaming):**

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.responses.create(
    model="local-model",
    input="Tell me a three sentence bedtime story about a unicorn."
)
# response.output contains reasoning and message items
for item in response.output:
    if item.type == "message":
        for part in item.content:
            if getattr(part, "text", None):
                print(part.text)
```

**Text input (streaming):**

```python
response = client.responses.create(
    model="local-model",
    input="Tell me a three sentence bedtime story about a unicorn.",
    stream=True
)
for chunk in response:
    print(chunk)
```

**Image input (vision / multimodal):**

```python
response = client.responses.create(
    model="local-multimodal",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What is in this image?"},
                {
                    "type": "input_image",
                    "image_url": "path/to/image.jpg",
                    "detail": "low"
                }
            ]
        }
    ]
)
```

**Function calling:**

```python
tools = [{
    "type": "function",
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "The city and state"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["location", "unit"]
    }
}]

response = client.responses.create(
    model="local-model",
    tools=tools,
    input="What is the weather like in Boston today?",
    tool_choice="auto"
)
```

**Structured outputs (Pydantic):**

```python
from pydantic import BaseModel

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip: str

response = client.responses.parse(
    model="local-model",
    input=[{"role": "user", "content": "Format: 1 Hacker Wy Menlo Park CA 94025"}],
    text_format=Address
)
address = response.output_parsed  # Pydantic model instance
print(address)
```

See `examples/responses_api.ipynb` for full examples including streaming, image input, tool calls, and structured outputs.

### Structured Outputs (JSON Schema)

```python
import openai
import json

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "Address",
        "schema": {
            "type": "object",
            "properties": {
                "street": {"type": "string"},
                "city": {"type": "string"},
                "state": {"type": "string"},
                "zip": {"type": "string"}
            },
            "required": ["street", "city", "state", "zip"]
        }
    }
}

completion = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "Format: 1 Hacker Wy Menlo Park CA 94025"}],
    response_format=response_format
)

address = json.loads(completion.choices[0].message.content)
print(json.dumps(address, indent=2))
```

## Advanced Configuration

### Parser Configuration

For models requiring custom parsing (tool calls, reasoning):

```bash
mlx-openai-server launch \
  --model-path <path-to-model> \
  --model-type lm \
  --tool-call-parser qwen3 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice
```

Available parsers: `qwen3`, `glm4_moe`, `qwen3_coder`, `qwen3_moe`, `qwen3_next`, `qwen3_vl`, `harmony`, `minimax_m2`

### Message Converters

For models requiring message format conversion:

```bash
mlx-openai-server launch \
  --model-path <path-to-model> \
  --model-type lm \
  --message-converter glm4_moe
```

Available converters: `glm4_moe`, `minimax_m2`, `nemotron3_nano`, `qwen3_coder`

### Custom Chat Templates

```bash
mlx-openai-server launch \
  --model-path <path-to-model> \
  --model-type lm \
  --chat-template-file /path/to/template.jinja
```

### Speculative Decoding (lm)

Use a smaller draft model to propose tokens and verify them with the main model for faster text generation. Supported only for `--model-type lm`.

```bash
mlx-openai-server launch \
  --model-path mlx-community/MyModel-8B-4bit \
  --model-type lm \
  --draft-model-path mlx-community/MyModel-1B-4bit \
  --num-draft-tokens 4
```

- **`--draft-model-path`**: Path or HuggingFace repo of the draft model (smaller size model).
- **`--num-draft-tokens`**: Number of tokens the draft model generates per verification step (default: 2). Higher values can increase throughput at the cost of more draft compute.

## Request Queue System

The server includes a request queue system with monitoring:

```bash
# Check queue status
curl http://localhost:8000/v1/queue/stats
```

Response:
```json
{
  "status": "ok",
  "queue_stats": {
    "running": true,
    "queue_size": 3,
    "max_queue_size": 100,
    "active_requests": 1,
    "max_concurrency": 1
  }
}
```

## Example Notebooks

Check the `examples/` directory for comprehensive guides:
- `responses_api.ipynb` - Responses API (text, image, tools, streaming, structured outputs)
- `audio_examples.ipynb` - Audio processing
- `embedding_examples.ipynb` - Text embeddings
- `lm_embeddings_examples.ipynb` - Language model embeddings
- `vlm_embeddings_examples.ipynb` - Vision-language embeddings
- `vision_examples.ipynb` - Vision capabilities
- `image_generations.ipynb` - Image generation
- `image_edit.ipynb` - Image editing
- `structured_outputs_examples.ipynb` - JSON schema outputs
- `simple_rag_demo.ipynb` - RAG pipeline demo

## Large Models

For models that don't fit in RAM, improve performance on macOS 15.0+:

```bash
bash configure_mlx.sh
```

This raises the system's wired memory limit for better performance.

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

Follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.

## Support

- **Documentation**: This README and example notebooks
- **Issues**: [GitHub Issues](https://github.com/cubist38/mlx-openai-server/issues)
- **Discussions**: [GitHub Discussions](https://github.com/cubist38/mlx-openai-server/discussions)
- **Video Tutorials**: [Setup Demo](https://youtu.be/J1gkEMvmTSE), [RAG Demo](https://youtu.be/ANUEZkmR-0s), [Testing Qwen3-Coder-Next-4bit with Qwen-Code](https://youtu.be/X5Hsd3QR_E8), [Serving Multiple Models at Once? mlx-openai-server + OpenWebUI Test](https://www.youtube.com/watch?v=f7WXSOPZ5H4)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

Built on top of:
- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
- [mlx-lm](https://github.com/ml-explore/mlx-lm) - Language models
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) - Multimodal models
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) - Embeddings
- [mflux](https://github.com/filipstrand/mflux) - Flux image models
- [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) - Audio transcription
- [mlx-community](https://huggingface.co/mlx-community) - Model repository

---

[![GitHub stars](https://img.shields.io/github/stars/cubist38/mlx-openai-server?style=social&label=Star)](https://github.com/cubist38/mlx-openai-server)
