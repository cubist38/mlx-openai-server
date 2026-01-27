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
- ⚡ **Performance** - Configurable quantization (4/8/16-bit) and context length
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
- `--max-concurrency`: Concurrent requests (default: 1)
- `--queue-timeout`: Request timeout in seconds (default: 300)
- `--lora-paths`: Comma-separated LoRA adapter paths (image models)
- `--lora-scales`: Comma-separated LoRA scales (must match paths)
- `--log-level`: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` (default: `INFO`)
- `--no-log-file`: Disable file logging (console only)

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
- **Video Tutorials**: [Setup Demo](https://youtu.be/J1gkEMvmTSE), [RAG Demo](https://youtu.be/ANUEZkmR-0s)

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
