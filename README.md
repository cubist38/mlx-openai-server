# mlx-openai-server

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

## Description
This repository hosts a high-performance API server that provides OpenAI-compatible endpoints for MLX models. Developed using Python and powered by the FastAPI framework, it provides an efficient, scalable, and user-friendly solution for running MLX-based multimodal models locally with an OpenAI-compatible interface. The server supports text, vision, audio processing, and image generation capabilities with enhanced Flux-series model support.

> **Note:** This project currently supports **MacOS with M-series chips** only as it specifically leverages MLX, Apple's framework optimized for Apple Silicon.

## Table of Contents
- [Key Features](#key-features)
- [Demo](#demo)
- [OpenAI Compatibility](#openai-compatibility)
- [Supported Model Types](#supported-model-types)
- [Installation](#installation)
- [Usage](#usage)
  - [Starting the Server](#starting-the-server)
  - [CLI Usage](#cli-usage)
  - [Logging Configuration](#logging-configuration)
  - [Using the API](#using-the-api)
  - [Structured Outputs](#structured-outputs-with-json-schema)
- [Request Queue System](#request-queue-system)
- [API Response Schemas](#api-response-schemas)
- [Example Notebooks](#example-notebooks)
- [Community & Support](#community--support)
- [Large Models](#large-models)
- [Contributing](#contributing)
- [License](#license)
- [Support](#support)
- [Acknowledgments](#acknowledgments)

---

## Key Features
- 🚀 **Fast, local OpenAI-compatible API** for MLX models
- 🖼️ **Multimodal model support** with vision, audio, and text
- 🎨 **Advanced image generation and editing** with MLX Flux-series models (schnell, dev, Krea-dev, kontext)
- 🔌 **Drop-in replacement** for OpenAI API in your apps
- 📈 **Performance and queue monitoring endpoints**
- 🧑‍💻 **Easy Python and CLI usage**
- 🛡️ **Robust error handling and request management**
- 🎛️ **LoRA adapter support** for fine-tuned image generation and editing
- ⚡ **Configurable quantization** (4-bit, 8-bit, 16-bit) for optimal performance
- 🧠 **Customizable context length** for memory optimization and performance tuning

---

## Demo

### 🚀 See It In Action

Check out our [video demonstration](https://youtu.be/-h-AwPNvKiw) to see the server in action! The demo showcases:

- Deploy GLM4.5-Air with `mlx-openai-server` and test code abilities

<p align="center">
  <a href="https://youtu.be/-h-AwPNvKiw">
    <img src="https://img.youtube.com/vi/-h-AwPNvKiw/0.jpg" alt="MLX Server OAI-Compatible Demo" width="600">
  </a>
</p>

### 🧠 **NEW: GPT-OSS-20B (MXFP4-Q8) Integration with Opencode**

We're excited to announce our latest integration demo featuring **GPT-OSS-20B (MXFP4-Q8)** deployed with mlx-openai-server and integrated into **Opencode** to power advanced coding tasks! 

This demonstration showcases:
- **Large Language Model Deployment**: Running GPT-OSS-20B locally with MLX optimization
- **Advanced Coding Capabilities**: Leveraging the 20B parameter model for complex programming tasks
- **Seamless Integration**: Drop-in replacement for OpenAI API in Opencode workflows
- **Performance Optimization**: MXFP4-Q8 quantization for optimal speed and memory usage

<p align="center">
  <a href="https://youtu.be/MTmR_mPSs6k">
    <img src="https://img.youtube.com/vi/MTmR_mPSs6k/0.jpg" alt="GPT-OSS-20B Integration Demo" width="600">
  </a>
</p>

**Watch the full demo**: [GPT-OSS-20B + Opencode Integration](https://youtu.be/MTmR_mPSs6k)

---

## OpenAI Compatibility

This server implements the OpenAI API interface, allowing you to use it as a drop-in replacement for OpenAI's services in your applications. It supports:
- Chat completions (both streaming and non-streaming)
- Multimodal interactions (text, images, and audio)
- Advanced image generation and editing with Flux-series models
- Embeddings generation
- Function calling and tool use
- Standard OpenAI request/response formats
- Common OpenAI parameters (temperature, top_p, etc.)

## Supported Model Types

The server supports six types of MLX models:

1. **Text-only models** (`--model-type lm`) - Uses the `mlx-lm` library for pure language models
2. **Multimodal models** (`--model-type multimodal`) - Uses the `mlx-vlm` library for multimodal models that can process text, images, and audio
3. **Image generation models** (`--model-type image-generation`) - Uses the `mflux` library for Flux-series image generation models with enhanced configurations
4. **Image editing models** (`--model-type image-edit`) - Uses the `mflux` library for Flux-series image editing models
5. **Embeddings models** (`--model-type embeddings`) - Uses the `mlx-embeddings` library for text embeddings generation with optimized memory management
6. **Whisper models** (`--model-type whisper`) - Uses the `mlx-whisper` library for audio transcription and speech recognition ⚠️ *Requires ffmpeg installation*

### Whisper Models

> **⚠️ Note:** Whisper models require ffmpeg to be installed for audio processing: `brew install ffmpeg`

### Flux-Series Image Models

The server supports multiple Flux and Qwen model configurations for advanced image generation and editing:

#### Image Generation Models
- **`flux-schnell`** - Fast generation with 4 default steps, no guidance (best for quick iterations)
- **`flux-dev`** - High-quality generation with 25 default steps, 3.5 guidance (balanced quality/speed)
- **`flux-krea-dev`** - Premium quality with 28 default steps, 4.5 guidance (highest quality)
- **`qwen-image`** - Qwen image generation model with 50 default steps, 4.0 guidance (high-quality Qwen-based generation)
- **`z-image-turbo`** - Z-Image Turbo model for fast image generation
- **`fibo`** - Fibo model for image generation

#### Image Editing Models
- **`flux-kontext-dev`** - Context-aware editing with 28 default steps, 2.5 guidance (specialized for contextual image editing)
- **`qwen-image-edit`** - Qwen image editing model with 50 default steps, 4.0 guidance (high-quality Qwen-based editing)

Each configuration supports:
- **Quantization levels**: 4-bit, 8-bit, or 16-bit for memory/performance optimization
- **LoRA adapters**: Multiple LoRA paths with custom scaling for fine-tuned generation and editing (supported for all Flux and Qwen image models).
- **Custom parameters**: Steps, guidance, negative prompts, and more

### Context Length Configuration

The server supports customizable context length for language models to optimize memory usage and performance:

- **Default behavior**: When `--context-length` is not specified, the server uses the model's default context length
- **Memory optimization**: Setting a smaller context length can significantly reduce memory usage, especially for large models
- **Performance tuning**: Adjust context length based on your specific use case and available system resources
- **Supported models**: Context length configuration works with both text-only (`lm`) and multimodal (`multimodal`) model types
- **Prompt caching**: The server uses prompt caching to optimize memory usage when context length is specified

**Example use cases:**
- **Short conversations**: Use smaller context lengths (e.g., 2048, 4096) for chat applications
- **Document processing**: Use larger context lengths (e.g., 8192, 16384) for long document analysis
- **Memory-constrained systems**: Reduce context length to fit larger models in limited RAM

### Custom Chat Templates

The server supports custom chat templates for language models (`lm`) and multimodal models (`multimodal`). Chat templates define how conversation messages are formatted before being sent to the model.

**Features:**
- **Custom formatting**: Override the model's default chat template with your own Jinja2 template
- **Model compatibility**: Works with both text-only and multimodal models
- **File-based configuration**: Specify a path to a `.jinja` template file when launching the server

**Usage:**
```bash
# Launch server with custom chat template
python -m app.main \
  --model-path <path-to-model> \
  --model-type lm \
  --chat-template-file /path/to/custom_template.jinja
```

**Template file format:**
Chat templates use Jinja2 syntax and should follow the standard format expected by the tokenizer/processor. The template receives a `messages` variable containing the conversation history.

**Example use cases:**
- **Custom prompt formatting**: Modify how system prompts, user messages, and assistant responses are formatted
- **Model-specific requirements**: Adapt templates for models that require specific formatting
- **Fine-tuning compatibility**: Use templates that match your fine-tuning data format

> **Note:** If the chat template file does not exist, the server will raise an error during startup. Make sure the file path is correct and the file is accessible.

## Installation

Follow these steps to set up the MLX-powered server:

### Prerequisites
- MacOS with Apple Silicon (M-series) chip
- Python 3.11 (native ARM version)
- pip package manager

### Setup Steps
1. Create a virtual environment for the project:
    ```bash
    python3.11 -m venv oai-compat-server
    ```

2. Activate the virtual environment:
    ```bash
    source oai-compat-server/bin/activate
    ```

3. Install the package:
    ```bash
    # Option 1: Install from PyPI
    pip install mlx-openai-server

    # Option 2: Install directly from GitHub
    pip install git+https://github.com/cubist38/mlx-openai-server.git
    
    # Option 3: Clone and install in development mode
    git clone https://github.com/cubist38/mlx-openai-server.git
    cd mlx-openai-server
    pip install -e .
    ```

### Using Conda (Recommended)

For better environment management and to avoid architecture issues, we recommend using conda:

1. **Install conda** (if not already installed):
    ```bash
    mkdir -p ~/miniconda3
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
    source ~/miniconda3/bin/activate
    conda init --all
    ```

2. **Create a new conda environment** with Python 3.11:
    ```bash
    conda create -n mlx-server python=3.11
    conda activate mlx-server
    ```

3. **Install the package**:
    ```bash
    # Option 1: Install from PyPI
    pip install mlx-openai-server

    # Option 2: Install directly from GitHub
    pip install git+https://github.com/cubist38/mlx-openai-server.git
    
    # Option 3: Clone and install in development mode
    git clone https://github.com/cubist38/mlx-openai-server.git
    cd mlx-openai-server
    pip install -e .
    ```

### Optional Dependencies

The server supports optional dependencies for enhanced functionality:

#### Base Installation
```bash
pip install mlx-openai-server
```
**Includes:**
- Text-only language models (`--model-type lm`)
- Multimodal models (`--model-type multimodal`) 
- Embeddings models (`--model-type embeddings`)
- All core API endpoints and functionality

#### Image Generation & Editing Support
The server includes support for image generation and editing capabilities:

**Additional features:**
- Image generation models (`--model-type image-generation`)
- Image editing models (`--model-type image-edit`)
- MLX Flux-series model support
- Qwen Image model support
- LoRA adapter support for fine-tuned generation and editing

#### Whisper Models Support
For whisper models to work properly, you need to install ffmpeg:

```bash
# Install ffmpeg using Homebrew
brew install ffmpeg
```

**Features with ffmpeg:**
- Audio transcription models (`--model-type whisper`)
- Speech recognition capabilities
- Support for various audio formats (WAV, MP3, M4A, etc.)

> **Note:** Whisper models require ffmpeg for audio processing. Make sure to install it before using whisper model types.

### Troubleshooting
**Issue:** My OS and Python versions meet the requirements, but `pip` cannot find a matching distribution.

**Cause:** You might be using a non-native Python version. Run the following command to check:
```bash
python -c "import platform; print(platform.processor())"
```
If the output is `i386` (on an M-series machine), you are using a non-native Python. Switch to a native Python version. A good approach is to use [Conda](https://stackoverflow.com/questions/65415996/how-to-specify-the-architecture-or-platform-for-a-new-conda-environment-apple).

## Usage

### Starting the Server

You can start the MLX server using either the Python module or the CLI command. Both methods support the same parameters, including logging configuration options.

#### Method 1: Python Module
```bash
# For text-only or multimodal models
python -m app.main \
  --model-path <path-to-mlx-model> \
  --model-type <lm|multimodal> \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --queue-size 100

# For image generation models (Flux-series, Qwen, Z-Image Turbo, or Fibo)
python -m app.main \
  --model-type image-generation \
  --model-path <path-to-local-model> \
  --config-name <flux-schnell|flux-dev|flux-krea-dev|qwen-image|z-image-turbo|fibo> \
  --quantize <4|8|16> \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --queue-size 100

# For image editing models (Flux-series or Qwen)
python -m app.main \
  --model-type image-edit \
  --model-path <path-to-local-model> \
  --config-name <flux-kontext-dev|qwen-image-edit> \
  --quantize <4|8|16> \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --queue-size 100

# For embeddings models
python -m app.main \
  --model-type embeddings \
  --model-path <embeddings-model-path> \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --queue-size 100

# For whisper models
python -m app.main \
  --model-type whisper \
  --model-path <whisper-model-path> \
  --max-concurrency 1 \
  --queue-timeout 600 \
  --queue-size 50

# With logging configuration options
python -m app.main \
  --model-path <path-to-mlx-model> \
  --model-type lm \
  --no-log-file \
  --log-level INFO

python -m app.main \
  --model-path <path-to-mlx-model> \
  --model-type lm \
  --log-file /tmp/custom.log \
  --log-level DEBUG
```

#### Method 2: CLI Command
```bash
# For text-only or multimodal models
mlx-openai-server launch \
  --model-path <path-to-mlx-model> \
  --model-type <lm|multimodal> \


# For image generation models (Flux-series, Qwen, Z-Image Turbo, or Fibo)
mlx-openai-server launch \
  --model-type image-generation \
  --model-path <path-to-local-model> \
  --config-name <flux-schnell|flux-dev|flux-krea-dev|qwen-image|z-image-turbo|fibo> \
  --quantize 8 \


# For image editing models (Flux-series or Qwen)
mlx-openai-server launch \
  --model-type image-edit \
  --model-path <path-to-local-model> \
  --config-name <flux-kontext-dev|qwen-image-edit> \
  --quantize 8 \


# For whisper models
mlx-openai-server launch \
  --model-path mlx-community/whisper-large-v3-mlx \
  --model-type whisper \
  --max-concurrency 1 \
  --queue-timeout 600 \
  --queue-size 50 \


# With LoRA adapters
mlx-openai-server launch \
  --model-type image-generation \
  --model-path <path-to-local-flux-model> \
  --config-name flux-dev \
  --lora-paths "/path/to/lora1.safetensors,/path/to/lora2.safetensors" \
  --lora-scales "0.8,0.6" \

```

#### Server Parameters
- `--model-path`: Path to the MLX model directory (local path or Hugging Face model repository). Required for `lm`, `multimodal`, `embeddings`, `image-generation`, `image-edit`, and `whisper` model types.
- `--model-type`: Type of model to run:
  - `lm` for text-only models
  - `multimodal` for multimodal models (text, vision, audio)
  - `image-generation` for image generation models
  - `image-edit` for image editing models
  - `embeddings` for embeddings models
  - `whisper` for whisper models (audio transcription)
  - Default: `lm`
- `--context-length`: Context length for language models. Controls the maximum sequence length for text processing and memory usage optimization. Default: `None` (uses model's default context length).
- `--config-name`: Model configuration to use. Only used for `image-generation` and `image-edit` model types:
  - For `image-generation`: `flux-schnell`, `flux-dev`, `flux-krea-dev`, `qwen-image`, `z-image-turbo`, `fibo`
  - For `image-edit`: `flux-kontext-dev`, `qwen-image-edit`
  - Default: `flux-schnell` for image-generation, `flux-kontext-dev` for image-edit
- `--quantize`: Quantization level for Flux models. Available options: `4`, `8`, `16`. Default: `8`
- `--lora-paths`: Comma-separated paths to LoRA adapter files.
- `--lora-scales`: Comma-separated scale factors for LoRA adapters. Must match the number of LoRA paths.
- `--max-concurrency`: Maximum number of concurrent requests (default: 1)
- `--queue-timeout`: Request timeout in seconds (default: 300)
- `--queue-size`: Maximum queue size for pending requests (default: 100)
- `--port`: Port to run the server on (default: 8000)
- `--host`: Host to run the server on (default: 0.0.0.0)
- `--disable-auto-resize`: Disable automatic model resizing. Only works for Vision Language Models.
- `--enable-auto-tool-choice`: Enable automatic tool choice. Only works with language models (`lm` or `multimodal` model types).
- `--tool-call-parser`: Specify tool call parser to use instead of auto-detection. Only works with language models (`lm` or `multimodal` model types). Available options: `qwen3`, `glm4_moe`, `qwen3_moe`, `qwen3_next`, `qwen3_vl`, `harmony`, `minimax`.
- `--reasoning-parser`: Specify reasoning parser to use instead of auto-detection. Only works with language models (`lm` or `multimodal` model types). Available options: `qwen3`, `glm4_moe`, `qwen3_moe`, `qwen3_next`, `qwen3_vl`, `harmony`, `minimax`.
- `--trust-remote-code`: Enable `trust_remote_code` when loading models. This allows loading custom code from model repositories. Default: `False` (disabled). Only works with `lm` or `multimodal` model types.
- `--chat-template-file`: Path to a custom chat template file. Only works with language models (`lm`) and multimodal models (`multimodal`). Default: `None` (uses model's default chat template).
- `--log-file`: Path to log file. If not specified, logs will be written to 'logs/app.log' by default.
- `--no-log-file`: Disable file logging entirely. Only console output will be shown.
- `--log-level`: Set the logging level. Choices: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Default: `INFO`.

### Parser Configuration

The server supports manual configuration of parsers for tool calls and reasoning/reasoning content extraction. When parsers are not explicitly specified, they will be `None` by default, meaning no parsing will be performed.

#### Available Parsers

The following parsers are available for both tool call and reasoning parsing:

- **`qwen3`**: Parser for Qwen3 model formats
- **`glm4_moe`**: Parser for GLM4 MoE model formats
- **`qwen3_moe`**: Parser for Qwen3 MoE model formats
- **`qwen3_next`**: Parser for Qwen3 Next model formats
- **`qwen3_vl`**: Parser for Qwen3 Vision-Language model formats
- **`harmony`**: Unified parser for Harmony/GPT-OSS models (handles both thinking and tools)
- **`minimax`**: Parser for MiniMax model formats

#### Parser Parameters

- **`--tool-call-parser`**: Specify which parser to use for extracting tool calls from model responses
- **`--reasoning-parser`**: Specify which parser to use for extracting reasoning/thinking content from model responses
- **`--enable-auto-tool-choice`**: Enable automatic tool choice when using tool calling

#### Usage Examples

**Basic usage without parsers (default):**
```bash
mlx-openai-server launch \
  --model-path /path/to/model \
  --model-type lm
```

**With tool call parser only:**
```bash
mlx-openai-server launch \
  --model-path /path/to/model \
  --model-type lm \
  --tool-call-parser qwen3
```

**With both parsers:**
```bash
mlx-openai-server launch \
  --model-path /path/to/model \
  --model-type lm \
  --tool-call-parser qwen3 \
  --reasoning-parser qwen3
```

**With auto tool choice enabled:**
```bash
mlx-openai-server launch \
  --model-path /path/to/model \
  --model-type lm \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3 \
  --reasoning-parser qwen3
```

**Using Harmony parser (unified parser):**
```bash
mlx-openai-server launch \
  --model-path /path/to/model \
  --model-type lm \
  --reasoning-parser harmony \
  --tool-call-parser harmony
```

> **Note:** Parser configuration is only applicable to language models (`lm` or `multimodal` model types). If parsers are not specified, the server will not perform any parsing, and raw model responses will be returned.

#### Example Configurations

**Text-only model:**
```bash
python -m app.main \
  --model-path mlx-community/gemma-3-4b-it-4bit \
  --model-type lm \
  --context-length 8192 \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --queue-size 100
```

**Multimodal model:**
```bash
python -m app.main \
  --model-path mlx-community/llava-phi-3-vision-4bit \
  --model-type multimodal \
  --context-length 4096 \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --queue-size 100
```

**Model with trust_remote_code enabled:**
```bash
python -m app.main \
  --model-path <path-to-model-requiring-custom-code> \
  --model-type lm \
  --trust-remote-code \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --queue-size 100
```

**Model with custom chat template:**
```bash
python -m app.main \
  --model-path <path-to-model> \
  --model-type lm \
  --chat-template-file /path/to/custom_template.jinja \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --queue-size 100
```

**Image generation models:**

*Fast generation with Schnell:*
```bash
python -m app.main \
  --model-type image-generation \
  --model-path <path-to-local-flux-model> \
  --config-name flux-schnell \
  --quantize 8 \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --queue-size 100
```

*High-quality generation with Dev:*
```bash
python -m app.main \
  --model-type image-generation \
  --model-path <path-to-local-flux-model> \
  --config-name flux-dev \
  --quantize 8 \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --queue-size 100
```

*Premium quality with Krea-Dev:*
```bash
python -m app.main \
  --model-type image-generation \
  --model-path <path-to-local-flux-model> \
  --config-name flux-krea-dev \
  --quantize 8 \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --queue-size 100
```

*High-quality generation with Qwen Image:*
```bash
python -m app.main \
  --model-type image-generation \
  --model-path <path-to-local-qwen-model> \
  --config-name qwen-image \
  --quantize 8 \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --queue-size 100
```

*Fast generation with Z-Image Turbo:*
```bash
mlx-openai-server launch --model-path z-image-turbo --model-type image-generation --config-name z-image-turbo
```

*Generation with Fibo:*
```bash
mlx-openai-server launch --model-path fibo --model-type image-generation --config-name fibo
```

*Image editing with Kontext:*
```bash
python -m app.main \
  --model-type image-edit \
  --model-path <path-to-local-flux-model> \
  --config-name flux-kontext-dev \
  --quantize 8 \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --queue-size 100
```

*Image editing with Qwen Image Edit:*
```bash
python -m app.main \
  --model-type image-edit \
  --model-path <path-to-local-qwen-model> \
  --config-name qwen-image-edit \
  --quantize 8 \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --queue-size 100
```

*With LoRA adapters (image generation):*
```bash
python -m app.main \
  --model-type image-generation \
  --model-path <path-to-local-flux-model> \
  --config-name flux-dev \
  --quantize 8 \
  --lora-paths "/path/to/lora1.safetensors,/path/to/lora2.safetensors" \
  --lora-scales "0.8,0.6" \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --queue-size 100
```

*With LoRA adapters (image editing):*
```bash
python -m app.main \
  --model-type image-edit \
  --model-path <path-to-local-flux-model> \
  --config-name flux-kontext-dev \
  --quantize 8 \
  --lora-paths "/path/to/lora1.safetensors,/path/to/lora2.safetensors" \
  --lora-scales "0.8,0.6" \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --queue-size 100
```

**Whisper models:**

*Audio transcription with Whisper:*
```bash
python -m app.main \
  --model-type whisper \
  --model-path mlx-community/whisper-large-v3-mlx \
  --max-concurrency 1 \
  --queue-timeout 600 \
  --queue-size 50
```

### CLI Usage

The server provides a convenient CLI interface for easy startup and management:

**Check version and help:**
```bash
mlx-openai-server --version
mlx-openai-server --help
mlx-openai-server launch --help
```

**Launch the server:**
```bash
# For text-only or multimodal models
mlx-openai-server launch --model-path <path-to-mlx-model> --model-type <lm|multimodal> --context-length 8192

# For image generation models (Flux-series, Qwen, Z-Image Turbo, or Fibo)
mlx-openai-server launch --model-type image-generation --model-path <path-to-local-model> --config-name <flux-schnell|flux-dev|flux-krea-dev|qwen-image|z-image-turbo|fibo>

# For image editing models (Flux-series or Qwen)
mlx-openai-server launch --model-type image-edit --model-path <path-to-local-model> --config-name <flux-kontext-dev|qwen-image-edit>

# For whisper models
mlx-openai-server launch --model-path mlx-community/whisper-large-v3-mlx --model-type whisper

# With LoRA adapters (image generation)
mlx-openai-server launch --model-type image-generation --model-path <path-to-local-flux-model> --config-name flux-dev --lora-paths "/path/to/lora1.safetensors,/path/to/lora2.safetensors" --lora-scales "0.8,0.6"

# With LoRA adapters (image editing)
mlx-openai-server launch --model-type image-edit --model-path <path-to-local-flux-model> --config-name flux-kontext-dev --lora-paths "/path/to/lora1.safetensors,/path/to/lora2.safetensors" --lora-scales "0.8,0.6"

# With custom logging configuration
mlx-openai-server launch --model-path <path-to-mlx-model> --model-type lm --log-file /tmp/server.log --log-level DEBUG

# Disable file logging (console only)
mlx-openai-server launch --model-path <path-to-mlx-model> --model-type lm --no-log-file

# Use default logging (logs/app.log, INFO level)
mlx-openai-server launch --model-path <path-to-mlx-model> --model-type lm

# With parser configuration for tool calls and reasoning
mlx-openai-server launch \
  --model-path <path-to-mlx-model> \
  --model-type lm \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3 \
  --reasoning-parser qwen3

# With trust_remote_code enabled (for models requiring custom code)
mlx-openai-server launch \
  --model-path <path-to-mlx-model> \
  --model-type lm \
  --trust-remote-code

# With custom chat template file
mlx-openai-server launch \
  --model-path <path-to-mlx-model> \
  --model-type lm \
  --chat-template-file /path/to/custom_template.jinja

# Using python -m app.main (alternative method)
python -m app.main --model-path <path-to-mlx-model> --model-type lm --no-log-file
python -m app.main --model-path <path-to-mlx-model> --model-type lm --log-file /tmp/custom.log
python -m app.main --model-path <path-to-mlx-model> --model-type lm --trust-remote-code
python -m app.main --model-path <path-to-mlx-model> --model-type lm --chat-template-file /path/to/custom_template.jinja
```

> **Note:** Text embeddings via the `/v1/embeddings` endpoint are now available with both text-only models (`--model-type lm`) and multimodal models (`--model-type multimodal`).

### Logging Configuration

The server provides flexible logging options to help you monitor and debug your MLX server:

#### Logging Options

- **`--log-file`**: Specify a custom path for log files
  - Default: `logs/app.log`
  - Example: `--log-file /tmp/my-server.log`

- **`--no-log-file`**: Disable file logging entirely
  - Only console output will be shown
  - Useful for development or when you don't need persistent logs

- **`--log-level`**: Control the verbosity of logging
  - Choices: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
  - Default: `INFO`
  - `DEBUG`: Most verbose, includes detailed debugging information
  - `INFO`: Standard operational messages (default)
  - `WARNING`: Important notices about potential issues
  - `ERROR`: Error messages only
  - `CRITICAL`: Only critical system errors

#### Logging Examples

```bash
# Use default logging (logs/app.log, INFO level)
mlx-openai-server launch --model-path <path-to-model> --model-type lm

# Custom log file with debug level
mlx-openai-server launch --model-path <path-to-model> --model-type lm --log-file /tmp/debug.log --log-level DEBUG

# Console-only logging (no file output)
mlx-openai-server launch --model-path <path-to-model> --model-type lm --no-log-file

# High-level logging (errors only)
mlx-openai-server launch --model-path <path-to-model> --model-type lm --log-level ERROR

# Using python -m app.main with logging options
python -m app.main --model-path <path-to-model> --model-type lm --no-log-file
python -m app.main --model-path <path-to-model> --model-type lm --log-file /tmp/custom.log --log-level DEBUG
```

#### Log File Features

- **Automatic rotation**: Log files are automatically rotated when they reach 500 MB
- **Retention**: Log files are kept for 10 days by default
- **Formatted output**: Both console and file logs include timestamps, log levels, and structured formatting
- **Colorized console**: Console output includes color coding for better readability

### Using the API

The server provides OpenAI-compatible endpoints that you can use with standard OpenAI client libraries. Here are some examples:

#### Text Completion
```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # API key is not required for local server
)

response = client.chat.completions.create(
    model="local-model",  # Model name doesn't matter for local server
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    temperature=0.7
)
print(response.choices[0].message.content)
```

#### Multimodal Model (Vision + Audio)
```python
import openai
import base64

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Load and encode image
with open("image.jpg", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

response = client.chat.completions.create(
    model="local-multimodal",  # Model name doesn't matter for local server
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
)
print(response.choices[0].message.content)
```

#### Audio Input Support
```python
import openai
import base64

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Load and encode audio file
with open("audio.wav", "rb") as audio_file:
    audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

response = client.chat.completions.create(
    model="local-multimodal",  # Model name doesn't matter for local server
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this audio?"
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_base64,
                        "format": "wav"
                    },
                },
            ],
        }
    ],
    max_tokens=64,
)
print(response.choices[0].message.content)
```

#### Advanced Image Generation with Flux-Series Models

```python
import openai
import base64
from io import BytesIO
from PIL import Image

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Basic image generation
response = client.images.generate(
    prompt="A serene landscape with mountains and a lake at sunset",
    model="local-image-generation-model",
    size="1024x1024",
    n=1
)

# Display the generated image
image_data = base64.b64decode(response.data[0].b64_json)
image = Image.open(BytesIO(image_data))
image.show()
```

#### Advanced Image Generation with Custom Parameters
```python
import requests

# For more control, use direct API calls
payload = {
    "prompt": "A beautiful cyberpunk city at night with neon lights",
    "model": "local-image-generation-model",
    "size": "1024x1024",
    "negative_prompt": "blurry, low quality, distorted",
    "steps": 8,
    "seed": 42,
    "priority": "normal"
}

response = requests.post(
    "http://localhost:8000/v1/images/generations",
    json=payload,
    headers={"Authorization": "Bearer fake-api-key"}
)

if response.status_code == 200:
    result = response.json()
    # Handle the base64 image data
    image_data = base64.b64decode(result['data'][0]['b64_json'])
    image = Image.open(BytesIO(image_data))
    image.show()
```

**Image Generation Parameters:**
- `prompt`: Text description of the desired image (required, max 1000 characters)
- `model`: Model identifier (defaults to "local-image-generation-model")
- `size`: Image dimensions - "256x256", "512x512", or "1024x1024" (default: "1024x1024")
- `negative_prompt`: What to avoid in the generated image (optional)
- `steps`: Number of inference steps, 1-50 (default varies by config: 4 for Schnell, 25 for Dev, 28 for Krea-Dev, 50 for Qwen Image)
- `seed`: Random seed for reproducible generation (optional)
- `priority`: Task priority - "low", "normal", "high" (default: "normal")
- `async_mode`: Whether to process asynchronously (default: false)

> **Note:** Image generation requires running the server with `--model-type image-generation`. The server supports MLX Flux-series models (flux-schnell, flux-dev, flux-krea-dev), Qwen Image models (qwen-image), Z-Image Turbo (z-image-turbo), and Fibo (fibo) models for high-quality image generation with configurable quality/speed trade-offs.

#### Image Editing with Flux-Series Models

```python
import openai
import base64
from io import BytesIO
from PIL import Image

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Edit an existing image
with open("images/china.png", "rb") as image_file:
    result = client.images.edit(
        image=image_file,
        prompt="make it like a photo in 1800s",
        model="flux-kontext-dev"
    )

# Display the edited image
image_data = base64.b64decode(result.data[0].b64_json)
image = Image.open(BytesIO(image_data))
image.show()
```

#### Advanced Image Editing with Custom Parameters
```python
import requests

# For more control, use direct API calls with form data
with open("images/china.png", "rb") as image_file:
    files = {"image": image_file}
    data = {
        "prompt": "make it like a photo in 1800s",
        "model": "flux-kontext-dev",
        "negative_prompt": "modern, digital, high contrast",
        "guidance_scale": 2.5,
        "steps": 4,
        "seed": 42,
        "size": "1024x1024",
        "response_format": "b64_json"
    }
    
    response = requests.post(
        "http://localhost:8000/v1/images/edits",
        files=files,
        data=data,
        headers={"Authorization": "Bearer fake-api-key"}
    )

if response.status_code == 200:
    result = response.json()
    # Handle the base64 image data
    image_data = base64.b64decode(result['data'][0]['b64_json'])
    image = Image.open(BytesIO(image_data))
    image.show()
```

**Image Edit Parameters:**
- `image`: The image file to edit (required, PNG, JPEG, or JPG format, max 10MB)
- `prompt`: Text description of the desired edit (required, max 1000 characters)
- `model`: Model identifier (defaults to "flux-kontext-dev")
- `negative_prompt`: What to avoid in the edited image (optional)
- `guidance_scale`: Controls how closely the model follows the prompt (default: 2.5 for flux-kontext-dev, 4.0 for qwen-image-edit)
- `steps`: Number of inference steps, 1-50 (default: 4 for flux-kontext-dev, 50 for qwen-image-edit)
- `seed`: Random seed for reproducible editing (default: 42)
- `size`: Output image dimensions - "256x256", "512x512", or "1024x1024" (optional)
- `response_format`: Response format - "b64_json" (default: "b64_json")

> **Note:** Image editing requires running the server with `--model-type image-edit`. The server supports MLX Flux-series models (flux-kontext-dev) and Qwen Image Edit models (qwen-image-edit) for high-quality image editing with configurable quality/speed trade-offs.

#### Function Calling
```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Define the messages and tools
messages = [
    {
        "role": "user",
        "content": "What is the weather in Tokyo?"
    }
]

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather in a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city to get the weather for"}
                }
            }
        }
    }
]

# Make the API call
completion = client.chat.completions.create(
    model="local-model",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

# Handle the tool call response
if completion.choices[0].message.tool_calls:
    tool_call = completion.choices[0].message.tool_calls[0]
    print(f"Function called: {tool_call.function.name}")
    print(f"Arguments: {tool_call.function.arguments}")
    
    # Process the tool call - typically you would call your actual function here
    # For this example, we'll just hardcode a weather response
    weather_info = {"temperature": "22°C", "conditions": "Sunny", "humidity": "65%"}
    
    # Add the tool call and function response to the conversation
    messages.append(completion.choices[0].message)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_call.function.name,
        "content": str(weather_info)
    })
    
    # Continue the conversation with the function result
    final_response = client.chat.completions.create(
        model="local-model",
        messages=messages
    )
    print("\nFinal response:")
    print(final_response.choices[0].message.content)
```

#### Structured Outputs with JSON Schema

The server supports structured outputs using JSON schema, allowing you to get responses in specific JSON formats:

```python
import openai
import json

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Define the messages and response format
messages = [
    {
        "role": "system",
        "content": "Extract the address from the user input into the specified JSON format."
    },
    {
        "role": "user",
        "content": "Please format this address: 1 Hacker Wy Menlo Park CA 94025"
    }
]

response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "Address",
        "schema": {
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                        "state": {
                            "type": "string", 
                            "description": "2 letter abbreviation of the state"
                        },
                        "zip": {
                            "type": "string", 
                            "description": "5 digit zip code"
                        }
                    },
                    "required": ["street", "city", "state", "zip"]
                }
            },
            "required": ["address"],
            "type": "object"
        }
    }
}

# Make the API call with structured output
completion = client.chat.completions.create(
    model="local-model",
    messages=messages,
    response_format=response_format
)

# Parse the structured response
response_content = completion.choices[0].message.content
parsed_address = json.loads(response_content)
print("Structured Address:")
print(json.dumps(parsed_address, indent=2))
```

**Response Format Parameters:**
- `type`: Must be set to `"json_schema"` for structured outputs
- `json_schema`: A JSON schema object defining the expected response structure
  - `name`: Optional name for the schema
  - `schema`: The actual JSON schema definition with properties, types, and requirements

**Example Response:**
```json
{
  "address": {
    "street": "1 Hacker Wy",
    "city": "Menlo Park",
    "state": "CA",
    "zip": "94025"
  }
}
```

> **Note:** Structured outputs work with text-only models (`--model-type lm`). The model will attempt to format its response according to the provided JSON schema.

#### Embeddings

1. Text-only model embeddings:
```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Generate embeddings for a single text
embedding_response = client.embeddings.create(
    model="mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-MLX-Q8",
    input=["The quick brown fox jumps over the lazy dog"]
)
print(f"Embedding dimension: {len(embedding_response.data[0].embedding)}")

# Generate embeddings for multiple texts
batch_response = client.embeddings.create(
    model="mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-MLX-Q8",
    input=[
        "Machine learning algorithms improve with more data",
        "Natural language processing helps computers understand human language",
        "Computer vision allows machines to interpret visual information"
    ]
)
print(f"Number of embeddings: {len(batch_response.data)}")
```

2. Multimodal model embeddings:
```python
import openai
import base64
from PIL import Image
from io import BytesIO

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Helper function to encode images as base64
def image_to_base64(image_path):
    image = Image.open(image_path)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image_data = buffer.getvalue()
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"

# Encode the image
image_uri = image_to_base64("images/attention.png")

# Generate embeddings for text+image
multimodal_embedding = client.embeddings.create(
    model="mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
    input=["Describe the image in detail"],
    extra_body={"image_url": image_uri}
)
print(f"Multimodal embedding dimension: {len(multimodal_embedding.data[0].embedding)}")
```

> **Note:** Replace the model name and image path as needed. The `extra_body` parameter is used to pass the image data URI to the API.

> **Warning:** Make sure you're running the server with `--model-type vlm` when making multimodal requests (with images or audio). If you send a multimodal request to a server running with `--model-type lm` (text-only model), you'll receive a 400 error with a message that multimodal requests are not supported with text-only models.

## Request Queue System

The server implements a robust request queue system to manage and optimize MLX model inference requests. This system ensures efficient resource utilization and fair request processing.

### Key Features

- **Concurrency Control**: Limits the number of simultaneous model inferences to prevent resource exhaustion
- **Request Queuing**: Implements a fair, first-come-first-served queue for pending requests
- **Timeout Management**: Automatically handles requests that exceed the configured timeout
- **Real-time Monitoring**: Provides endpoints to monitor queue status and performance metrics

### Architecture

The queue system consists of two main components:

1. **RequestQueue**: An asynchronous queue implementation that:
   - Manages pending requests with configurable queue size
   - Controls concurrent execution using semaphores
   - Handles timeouts and errors gracefully
   - Provides real-time queue statistics

2. **Model Handlers**: Specialized handlers for different model types:
   - `MLXLMHandler`: Manages text-only model requests
   - `MLXVLMHandler`: Manages multimodal model requests
   - `MLXFluxHandler`: Manages Flux-series image generation requests

### Queue Monitoring

Monitor queue statistics using the `/v1/queue/stats` endpoint:

```bash
curl http://localhost:8000/v1/queue/stats
```

Example response:
```json
{
  "status": "ok",
  "queue_stats": {
    "running": true,
    "queue_size": 3,
    "max_queue_size": 100,
    "active_requests": 5,
    "max_concurrency": 2
  }
}
```

### Error Handling

The queue system handles various error conditions:

1. **Queue Full (429)**: When the queue reaches its maximum size
```json
{
  "detail": "Too many requests. Service is at capacity."
}
```

2. **Request Timeout**: When a request exceeds the configured timeout
```json
{
  "detail": "Request processing timed out after 300 seconds"
}
```

3. **Model Errors**: When the model encounters an error during inference
```json
{
  "detail": "Failed to generate response: <error message>"
}
```

### Streaming Responses

The server supports streaming responses with proper chunk formatting:
```python
{
    "id": "chatcmpl-1234567890",
    "object": "chat.completion.chunk",
    "created": 1234567890,
    "model": "local-model",
    "choices": [{
        "index": 0,
        "delta": {"content": "chunk of text"},
        "finish_reason": null
    }]
}
```

## API Response Schemas

The server implements OpenAI-compatible API response schemas to ensure seamless integration with existing applications. Below are the key response formats:

### Chat Completions Response

```json
{
  "id": "chatcmpl-123456789",
  "object": "chat.completion",
  "created": 1677858242,
  "model": "local-model",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "This is the response content from the model."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

### Embeddings Response

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.001, 0.002, ..., 0.999],
      "index": 0
    }
  ],
  "model": "local-model",
  "usage": {
    "prompt_tokens": 8,
    "total_tokens": 8
  }
}
```

### Function/Tool Calling Response

```json
{
  "id": "chatcmpl-123456789",
  "object": "chat.completion",
  "created": 1677858242,
  "model": "local-model",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"city\":\"Tokyo\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 25,
    "total_tokens": 40
  }
}
```

### Image Generation Response

```json
{
  "created": 1677858242,
  "data": [
    {
      "b64_json": "iVBORw0KGgoAAAANSUhEUgAA...",
      "url": null
    }
  ]
}
```

### Error Response

```json
{
  "error": {
    "message": "Error message describing what went wrong",
    "type": "invalid_request_error",
    "param": null,
    "code": null
  }
}
```

## Example Notebooks

The repository includes example notebooks to help you get started with different aspects of the API:

- **function_calling_examples.ipynb**: A practical guide to implementing and using function calling with local models, including:
  - Setting up function definitions
  - Making function calling requests
  - Handling function call responses
  - Working with streaming function calls
  - Building multi-turn conversations with tool use

- **structured_outputs_examples.ipynb**: A comprehensive guide to using structured outputs with JSON schema, including:
  - Setting up JSON schema definitions
  - Making requests with response format specifications
  - Parsing structured responses
  - Working with complex nested schemas
  - Building data extraction pipelines with structured outputs

- **vision_examples.ipynb**: A comprehensive guide to using the vision capabilities of the API, including:
  - Processing image inputs in various formats
  - Vision analysis and object detection
  - Multi-turn conversations with images
  - Using vision models for detailed image description and analysis

- **lm_embeddings_examples.ipynb**: A comprehensive guide to using the embeddings API for text-only models, including:
  - Generating embeddings for single and batch inputs
  - Computing semantic similarity between texts
  - Building a simple vector-based search system
  - Comparing semantic relationships between concepts

- **vlm_embeddings_examples.ipynb**: A detailed guide to working with Vision-Language Model embeddings, including:
  - Generating embeddings for images with text prompts
  - Creating text-only embeddings with VLMs
  - Calculating similarity between text and image representations
  - Understanding the shared embedding space of multimodal models
  - Practical applications of VLM embeddings

- **simple_rag_demo.ipynb**: A practical guide to building a lightweight Retrieval-Augmented Generation (RAG) pipeline over PDF documents using local MLX Server, including:
  - Reading and chunking PDF documents  
  - Generating text embeddings via MLX Server  
  - Creating a simple vector store for retrieval  
  - Performing question answering based on relevant chunks
  - End-to-end demonstration of document QA using Qwen3 local model
  <p align="center">
    <a href="https://youtu.be/ANUEZkmR-0s">
      <img src="https://img.youtube.com/vi/ANUEZkmR-0s/0.jpg" alt="RAG Demo" width="600">
    </a>
  </p>

- **audio_examples.ipynb**: A comprehensive guide to audio processing capabilities with MLX Server, including:
  - Setting up connection to MLX Server for audio processing
  - Loading and encoding audio files for API transmission
  - Sending audio input to multimodal models for analysis
  - Combining audio with text prompts for rich, context-aware responses
  - Exploring different types of audio analysis prompts
  - Understanding audio transcription and content analysis capabilities

- **image_generations.ipynb**: A comprehensive guide to image generation using MLX Flux-series and Qwen Image models, including:
  - Setting up connection to MLX Server for image generation
  - Basic image generation with default parameters
  - Advanced image generation with custom parameters (negative prompts, steps, seed)
  - Working with different Flux configurations (schnell, dev, Krea-dev) and Qwen Image models
  - Using LoRA adapters for fine-tuned generation
  - Optimizing performance with quantization settings

- **image_edit.ipynb**: A comprehensive guide to image editing using MLX Flux-series and Qwen Image Edit models, including:
  - Setting up connection to MLX Server for image editing
  - Basic image editing with default parameters
  - Advanced image editing with custom parameters (guidance scale, steps, seed)
  - Working with the flux-kontext-dev and qwen-image-edit configurations for contextual editing
  - Using LoRA adapters for fine-tuned editing
  - Understanding the differences between generation and editing workflows
  - Best practices for effective image editing prompts

## Community & Support

We're thrilled by the incredible community that has grown around this project! Join thousands of developers, researchers, and AI enthusiasts who are building the future of local AI with MLX.

### 🌟 Show Your Support

**Star this repository** if you find it useful! Your stars help:
- 📈 **Increase visibility** and help other developers discover this tool
- 🚀 **Motivate continued development** and new features
- 🤝 **Build community recognition** for the MLX ecosystem

[![GitHub stars](https://img.shields.io/github/stars/cubist38/mlx-openai-server?style=social&label=Star)](https://github.com/cubist38/mlx-openai-server)
[![GitHub forks](https://img.shields.io/github/forks/cubist38/mlx-openai-server?style=social&label=Fork)](https://github.com/cubist38/mlx-openai-server/fork)
[![GitHub issues](https://img.shields.io/github/issues/cubist38/mlx-openai-server?color=blue&label=Issues)](https://github.com/cubist38/mlx-openai-server/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/cubist38/mlx-openai-server?color=green&label=PRs)](https://github.com/cubist38/mlx-openai-server/pulls)
[![GitHub discussions](https://img.shields.io/github/discussions/cubist38/mlx-openai-server?color=purple&label=Discussions)](https://github.com/cubist38/mlx-openai-server/discussions)

### 🆘 Get Help & Connect

#### 📚 Learning Resources
- **📖 Documentation**: This README and comprehensive API docs
- **🎥 Video Tutorials**: 
  - [Setup & Installation Demo](https://youtu.be/-h-AwPNvKiw)
  - [RAG Implementation Demo](https://youtu.be/ANUEZkmR-0s)
- **📓 Example Notebooks**: Check out our `examples/` directory for practical use cases
- **🔍 Search Issues**: Your question might already be answered!

#### 💬 Community Channels
- **🗨️ GitHub Discussions**: [Ask questions, share ideas, and connect with users](https://github.com/cubist38/mlx-openai-server/discussions)
- **🐛 GitHub Issues**: [Report bugs, request features, or get technical help](https://github.com/cubist38/mlx-openai-server/issues)
- **📢 Community Showcase**: Share your projects and use cases

#### 🚨 Before Asking for Help
1. **Search existing issues** and discussions
2. **Check the documentation** and examples
3. **Watch the video tutorials** for common setup issues
4. **Provide system details** (macOS version, Python version, model used)
5. **Include error messages** and logs when reporting issues

### 🤝 Contribute & Give Back

We welcome contributions of all kinds! Here's how you can help:

#### 💻 Code Contributions
- **🐛 Bug Fixes**: Help squash bugs and improve stability
- **✨ New Features**: Add capabilities that benefit the community
- **🔧 Performance**: Optimize code for better speed and efficiency
- **🧪 Testing**: Improve test coverage and reliability

#### 📚 Documentation & Examples
- **📖 Documentation**: Improve guides, tutorials, and API docs
- **📓 Examples**: Share your use cases and example notebooks
- **🎥 Tutorials**: Create video guides or blog posts
- **🌐 Translations**: Help translate documentation to other languages

#### 🎯 Community Support
- **💬 Help Others**: Answer questions in discussions and issues
- **🔍 Bug Reports**: Report issues with detailed reproduction steps
- **💡 Feature Requests**: Suggest improvements and new capabilities
- **⭐ Code Reviews**: Review pull requests and provide feedback

### 📢 Share Your Success

**We love hearing about your projects!** Share how you're using MLX OpenAI Server:

#### 🏆 Success Stories
- **Research Projects**: Academic papers, experiments, and discoveries
- **Production Apps**: Real-world applications and services
- **Open Source Tools**: Libraries and frameworks built on top
- **Educational Content**: Courses, workshops, and tutorials

#### 📈 Community Impact
- **Mention us** in your projects, papers, or presentations
- **Write blog posts** about your experience and use cases
- **Share on social media** with #MLX #LocalAI #OpenAI
- **Present at conferences** or meetups about local AI

### 🏗️ Community Highlights

#### 🚀 Active Development
- **Regular Updates**: New features and improvements every week
- **Rapid Response**: Quick fixes for critical issues
- **Community-Driven**: Features prioritized based on user feedback
- **Open Roadmap**: Transparent development planning

#### 🌍 Growing Ecosystem
- **MLX Integration**: Deep integration with Apple's MLX framework
- **Model Support**: Expanding support for new model types
- **Tool Ecosystem**: Compatible with popular AI tools and frameworks
- **Community Models**: Support for community-contributed models

#### 💼 Real-World Impact
- **Research**: Used in academic research and experiments
- **Startups**: Powering production AI applications
- **Education**: Teaching local AI development
- **Open Source**: Contributing to the broader AI ecosystem

#### 🎯 Community Values
- **Open Source**: MIT licensed and community-driven
- **Inclusive**: Welcoming to developers of all skill levels
- **Helpful**: Supportive community focused on helping others succeed
- **Innovative**: Pushing the boundaries of local AI capabilities

### 🎉 Join the Movement

**You're part of something special!** This project represents the future of local AI development:

- **🔒 Privacy-First**: Run AI models locally without sending data to the cloud
- **⚡ Performance**: Optimized for Apple Silicon with incredible speed
- **🌱 Sustainable**: Reduce carbon footprint with local processing
- **🎯 Accessible**: Make AI development available to everyone

**Your support and contributions make this project better for everyone in the MLX and local AI community!** 🚀

---

## Large Models
When using models that are large relative to your system's available RAM, performance may suffer. mlx-lm tries to improve speed by wiring the memory used by the model and its cache—this optimization is only available on macOS 15.0 or newer.
If you see the following warning message:
> [WARNING] Generating with a model that requires ...
it means the model may run slowly on your machine. If the model fits in RAM, you can often improve performance by raising the system's wired memory limit. To do this, run:
```bash
bash configure_mlx.sh
```

## Contributing

We welcome contributions to improve this project! Whether you're fixing bugs, adding features, improving documentation, or sharing examples, your contributions are valuable to the community.

### 🚀 Quick Start
1. **Fork** the repository to your GitHub account
2. **Clone** your fork locally:
    ```bash
    git clone https://github.com/YOUR_USERNAME/mlx-openai-server.git
    cd mlx-openai-server
    ```
3. **Create** a new branch for your changes:
    ```bash
    git checkout -b feature/your-feature-name
    ```
4. **Make** your changes and test them thoroughly
5. **Commit** with clear, descriptive messages:
    ```bash
    git commit -m "feat: add new model support"
    git commit -m "fix: resolve audio processing issue"
    git commit -m "docs: update installation guide"
    ```
6. **Push** to your fork:
    ```bash
    git push origin feature/your-feature-name
    ```
7. **Open** a pull request with a detailed description

### 📋 Contribution Guidelines

#### Code Contributions
- Follow existing code style and patterns
- Add tests for new features when possible
- Update documentation for API changes
- Ensure all tests pass before submitting

#### Documentation Improvements
- Fix typos, grammar, or unclear explanations
- Add missing examples or use cases
- Improve code comments and docstrings
- Update README sections as needed

#### Bug Reports
- Use the [GitHub issue template](https://github.com/cubist38/mlx-openai-server/issues/new)
- Include detailed steps to reproduce
- Provide system information (macOS version, Python version)
- Share error messages and logs

#### Feature Requests
- Describe the use case and expected behavior
- Explain why this feature would be valuable
- Consider implementation complexity
- Check if similar features already exist

### 🏷️ Commit Message Convention
We follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

### 🤝 Review Process
- All contributions require review before merging
- We aim to review PRs within 48 hours
- Maintainers may request changes or improvements
- Once approved, your contribution will be merged

Thank you for contributing to the MLX ecosystem!

## License
This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute it under the terms of the license.

## Support

Need help with MLX OpenAI Server? We're here to support you! Here are the best ways to get assistance:

### 🆘 Getting Help

#### 📝 Before Asking for Help
1. **Check the documentation** - This README and example notebooks
2. **Search existing issues** - Your question might already be answered
3. **Review the examples** - Check the `examples/` directory for use cases
4. **Watch the demos** - Our [setup video](https://youtu.be/-h-AwPNvKiw), [RAG demo](https://youtu.be/ANUEZkmR-0s), and [GPT-OSS-20B + Opencode integration](https://youtu.be/MTmR_mPSs6k)

#### 🐛 Reporting Issues
- **GitHub Issues**: [Create a new issue](https://github.com/cubist38/mlx-openai-server/issues/new) with:
  - Clear description of the problem
  - Steps to reproduce
  - System information (macOS version, Python version)
  - Error messages and logs
  - Expected vs actual behavior

#### 💬 Community Support
- **GitHub Discussions**: [Ask questions and share experiences](https://github.com/cubist38/mlx-openai-server/discussions)
- **GitHub Issues**: For bug reports and feature requests
- **Example Notebooks**: Learn from community examples

### 🔧 Common Issues & Solutions

#### Installation Problems
- **MLX Installation**: Ensure you're on macOS with M-series chip
- **Python Version**: Use Python 3.11+ for best compatibility
- **Dependencies**: Run `pip install -r requirements.txt` if needed

#### Model Loading Issues
- **Memory**: Large models may require more RAM
- **Model Path**: Verify the model path is correct
- **Quantization**: Try different quantization levels (4-bit, 8-bit, 16-bit)

#### Performance Issues
- **Context Length**: Adjust context length for memory optimization
- **Quantization**: Use lower precision for faster inference
- **System Resources**: Close other applications to free up memory

### 📚 Additional Resources
- **MLX Documentation**: [ml-explore/mlx](https://github.com/ml-explore/mlx)
- **MLX Models**: [mlx-community](https://huggingface.co/mlx-community)
- **Example Notebooks**: Check the `examples/` directory
- **Video Tutorials**: Watch our setup, demo, and integration videos

### ⏱️ Response Times
- **Bug Reports**: We aim to respond within 24-48 hours
- **Feature Requests**: Reviewed weekly
- **Community Questions**: Usually answered within a few days
- **Pull Requests**: Reviewed within 48 hours

Stay tuned for updates and enhancements!

## Acknowledgments

We extend our heartfelt gratitude to the following individuals and organizations whose contributions have been instrumental in making this project possible:

### 🏗️ Core Technologies
- **[MLX team](https://github.com/ml-explore/mlx)** - For developing the groundbreaking MLX framework, which provides the foundation for efficient machine learning on Apple Silicon
- **[mlx-lm](https://github.com/ml-explore/mlx-lm)** - For efficient large language models support and optimization
- **[mlx-vlm](https://github.com/Blaizzy/mlx-vlm/tree/main)** - For pioneering multimodal model support within the MLX ecosystem
- **[mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings)** - For text embeddings generation with optimized memory management
- **[mflux](https://github.com/filipstrand/mflux)** - For Flux-series image generation models with advanced configurations
- **[mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)** - For audio transcription and speech recognition with optimized inference on Apple Silicon 
- **[mlx-community](https://huggingface.co/mlx-community)** - For curating and maintaining a diverse collection of high-quality MLX models

### 🤝 Open Source Community
We deeply appreciate the broader open-source community for their invaluable contributions:

- **Model Developers** - For creating and sharing the models that power this server
- **Framework Contributors** - For building the tools and libraries that make this possible
- **Documentation Writers** - For creating comprehensive guides and tutorials
- **Community Members** - For testing, feedback, and continuous improvement

### 🌟 Community Contributors
A special acknowledgment to all contributors, users, and supporters who have helped shape this project through their:
- **Feedback and suggestions** that drive improvements
- **Bug reports** that help maintain quality
- **Code contributions** that add new features
- **Documentation improvements** that help others
- **Community support** that helps users succeed
- **Example sharing** that demonstrates real-world applications

### 📈 Growing Together
Your engagement and contributions help make this project better for everyone in the MLX and local AI community. Together, we're building a more accessible and powerful ecosystem for local AI development.

Thank you for being part of this journey! 🚀