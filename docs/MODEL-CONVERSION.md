# Converting models to MLX

Each model type is served by a different MLX package, and each package ships
its own converter. This page collects the command per type, what the output
looks like, and how to point the server at it.

Conversion is not what makes a checkpoint loadable. `mlx-lm`, `mlx-vlm` and
`mlx-embeddings` read Hugging Face `safetensors` directly for every
architecture they implement, so an unquantized repo can be served as it comes.
Converting is how you get a _quantized_ copy: less memory, a faster load, and a
local directory that no longer depends on the Hub. A repo that publishes
weights only as `pytorch_model.bin` is readable by neither path — the loader
globs `model*.safetensors` and fails with `No safetensors found`.

| `model_type`                     | Loaded by        | Converter                          |
| -------------------------------- | ---------------- | ---------------------------------- |
| `lm`                             | `mlx-lm`         | `mlx_lm.convert`                   |
| `multimodal`                     | `mlx-vlm`        | `mlx_vlm.convert`                  |
| `embeddings`                     | `mlx-embeddings` | `python -m mlx_embeddings.convert` |
| `whisper`                        | `mlx-whisper`    | not shipped, see below             |
| `image-generation`, `image-edit` | `mflux`          | not needed, see below              |

Every converter is installed with this project, so use the environment the
server itself runs in and the versions match by construction:

```bash
./.venv/bin/mlx_lm.convert --help
```

A `uv tool install` keeps them in the tool environment without linking them
into `~/.local/bin`, so call them by path:

```bash
~/.local/share/uv/tools/mlx-openai-server/bin/mlx_lm.convert --help
```

## Check the Hub first

Most popular checkpoints are already converted and quantized under the
[mlx-community](https://huggingface.co/mlx-community) organization, and every
`model_path` accepts a repo id, so nothing needs to be downloaded by hand:

```bash
mlx-openai-server launch --model-type lm \
  --model-path mlx-community/Qwen3-Coder-Next-4bit
```

Convert yourself when the model is not there, when it is there at the wrong
precision, or when you want the weights on local disk rather than in the shared
cache. To see what the cache already holds, and to reclaim space:

```bash
mlx_lm.manage --scan
mlx_lm.manage --delete --pattern <substring-of-the-repo-id>
```

## Text models (`lm`)

```bash
mlx_lm.convert \
  --hf-path Qwen/Qwen3-4B-Instruct-2507 \
  --mlx-path ~/models/qwen3-4b-4bit \
  -q --q-bits 4 --q-group-size 64
```

`-q` on its own quantizes to 4 bits with group size 64. The defaults follow
`--q-mode`: `affine` 4 bits / group 64, `mxfp4` 4 / 32, `nvfp4` 4 / 16, `mxfp8`
8 / 32.

| Flag                         | Effect                                                                               |
| ---------------------------- | ------------------------------------------------------------------------------------ |
| `-q`, `--quantize`           | Quantize. Without it the weights are only re-saved as `--dtype`.                     |
| `--q-bits`, `--q-group-size` | Bits per weight and group size. Smaller groups keep more accuracy.                   |
| `--q-mode`                   | `affine` (default), `mxfp4`, `nvfp4`, `mxfp8`.                                       |
| `--quant-predicate`          | Mixed-bit recipe: `mixed_2_6`, `mixed_3_4`, `mixed_3_6`, `mixed_4_6`. `affine` only. |
| `--dtype`                    | Precision of what stays unquantized; defaults to the checkpoint's `torch_dtype`.     |
| `-d`, `--dequantize`         | Back to full precision. Mutually exclusive with `-q`.                                |
| `--trust-remote-code`        | Needed when the tokenizer ships custom Python.                                       |
| `--upload-repo`              | Push the result to the Hub after saving.                                             |

A mixed recipe spends the higher bit width where rounding hurts most — the
`v_proj` and `down_proj` projections, and the first and last eighth of the layer
stack — and the lower one everywhere else. It requires `affine`; the other modes
raise.

The converter reports the average bits per weight it achieved, which is the
number to sanity-check the run against:

```console
$ mlx_lm.convert --hf-path HuggingFaceTB/SmolLM2-135M-Instruct \
    --mlx-path ~/models/smollm2-135m-4bit -q --q-bits 4 --q-group-size 64
[INFO] Loading
[INFO] Using dtype: bfloat16
[INFO] Quantizing
[INFO] Quantized model with 4.503 bits per weight.
```

That 135M checkpoint's weights go from 257 MB to 72 MB. Serve the result by
path:

```yaml
models:
  - model_path: /Users/you/models/smollm2-135m-4bit
    model_type: lm
    served_model_name: smollm2
```

## Multimodal models (`multimodal`)

Same flags, plus a few of its own:

```bash
mlx_vlm.convert \
  --hf-path Qwen/Qwen3-VL-4B-Instruct \
  --mlx-path ~/models/qwen3-vl-4b-4bit \
  -q --q-bits 4 --q-group-size 64
```

| Extra flag                 | Effect                                                                          |
| -------------------------- | ------------------------------------------------------------------------------- |
| `--revision`               | Convert a specific branch or tag from the Hub.                                  |
| `--quant-method awq`       | Calibrate before rounding: slower, more accurate at low bit widths.             |
| `--calibration multimodal` | Calibrate on image/audio inputs, optionally yours via `--calibration-data DIR`. |
| `--mtp`, `--mtp-output`    | Extract the checkpoint's native multi-token-prediction tensors as a drafter.    |

**The vision and audio towers are never quantized.** `mlx_vlm.convert` skips
any module whose path contains `vision_model`, `vision_tower`, `audio_model`,
`audio_tower`, `multi_modal_projector`, `vl_connector` and a handful of other
projector names — with the default predicate and with a `--quant-predicate`
recipe alike. Only the language tower shrinks, so the reported average lands
well above what you asked for whenever the encoder is a large share of the
model:

```console
$ mlx_vlm.convert --hf-path HuggingFaceTB/SmolVLM-256M-Instruct \
    --mlx-path ~/models/smolvlm-256m-4bit -q --q-bits 4 --q-group-size 64
[INFO] Quantizing
[INFO] Quantized model with 8.377 bits per weight.
```

The `language_model.*` weights in that output carry `scales`; every
`vision_model.*` weight is still `bfloat16`. That is deliberate — image quality
degrades quickly when the encoder is quantized — but it means a small VLM
barely shrinks, and the number to compare against a plain `lm` conversion is
the size of the language tower, not the whole directory.

## Embedding models (`embeddings`)

The `mlx_embeddings` console script is broken in 0.0.5
(`ModuleNotFoundError: mlx_embeddings.cli`), so invoke the module:

```bash
./.venv/bin/python -m mlx_embeddings.convert \
  --hf-path sentence-transformers/all-MiniLM-L6-v2 \
  --mlx-path ~/models/minilm-l6-4bit \
  -q --q-bits 4 --q-group-size 64
```

The flag set is narrower — no `--q-mode`, no mixed recipes, no
`--trust-remote-code` — and the defaults differ from `mlx-lm`: quantization is
4 bits at group size 64 and `--dtype` is `float16`.

Embedding models are small to begin with (MiniLM-L6: 87 MB down to 12 MB at 4
bits), so the memory saved is rarely the point. Retrieval quality is sensitive
to quantization in a way that generation is not: prefer 8 bits, or an existing
DWQ build such as `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ`, and check
recall on your own data before adopting a 4-bit conversion.

## Whisper models (`whisper`)

The `mlx-whisper` wheel ships no converter — `python -m mlx_whisper.convert`
reports no such module. In practice the pre-converted repos cover every size,
and `model_path` takes the repo id straight:

```yaml
models:
  - model_path: mlx-community/whisper-large-v3-mlx
    model_type: whisper
    served_model_name: whisper
```

To convert a fine-tune of your own, take `whisper/convert.py` from
[ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples/tree/main/whisper).
Its flags do not follow the others:

```bash
python convert.py \
  --torch-name-or-path ./my-whisper-finetune \
  --mlx-path ~/models/my-whisper-mlx \
  -q --q-bits 4 --q-group-size 64 \
  --dtype float16
```

## Image models (`image-generation`, `image-edit`)

These need no conversion step. `mflux` resolves `model_path` — a built-in name,
a Hub repo or a local directory — and applies `quantize` while loading, so a
config is enough to serve a quantized FLUX:

```yaml
models:
  - model_path: black-forest-labs/FLUX.1-dev
    model_type: image-generation
    config_name: flux-dev
    quantize: 8
```

The cost is paying for that quantization on every load. `mflux-save` writes the
quantized weights out once so later loads read them as they are:

```bash
mflux-save --model dev --quantize 8 --path ~/models/flux-dev-8bit
```

Then point the entry at the directory and drop `quantize` — the saved weights
already carry it:

```yaml
models:
  - model_path: /Users/you/models/flux-dev-8bit
    model_type: image-generation
    config_name: flux-dev
```

`config_name` selects the architecture on this side; `--model` names the same
model in `mflux`'s own vocabulary. They are not spelled the same:

| `config_name`                           | Upstream weights                       | `mflux-save --model` |
| --------------------------------------- | -------------------------------------- | -------------------- |
| `flux-schnell`                          | `black-forest-labs/FLUX.1-schnell`     | `schnell`            |
| `flux-dev`                              | `black-forest-labs/FLUX.1-dev`         | `dev`                |
| `flux-krea-dev`                         | `black-forest-labs/FLUX.1-Krea-dev`    | `krea-dev`           |
| `flux-kontext-dev`                      | `black-forest-labs/FLUX.1-Kontext-dev` | `dev-kontext`        |
| `qwen-image`                            | `Qwen/Qwen-Image-2512`                 | `qwen-image`         |
| `qwen-image-edit`                       | `Qwen/Qwen-Image-Edit-2509`            | `qwen-image-edit`    |
| `fibo`                                  | `briaai/FIBO`                          | `fibo`               |
| `z-image-turbo`                         | `Tongyi-MAI/Z-Image-Turbo`             | `z-image-turbo`      |
| `flux2-klein-4b`, `flux2-klein-edit-4b` | `black-forest-labs/FLUX.2-klein-4B`    | `flux2-klein-4b`     |
| `flux2-klein-9b`, `flux2-klein-edit-9b` | `black-forest-labs/FLUX.2-klein-9B`    | `flux2-klein-9b`     |

The edit and generation variants of Klein share one set of weights, so a single
saved directory serves both. `mflux-save` accepts 3, 4, 5, 6 or 8 bits, and a
config's `quantize` reaches `mflux` unchanged. When saving from a third-party
repo or a local path rather than a built-in name, add `--base-model` to say
which architecture it is:

```bash
mflux-save --model some-org/flux-finetune --base-model dev \
  --quantize 8 --path ~/models/flux-finetune-8bit
```

`--lora` with `--bake-lora` folds adapters into the saved weights, which is the
way to serve a LoRA without listing `lora_paths` on every entry.

## Choosing a bit width

4 bits is the usual default for `lm` and `multimodal`: roughly a quarter of the
memory, with a quality loss most chat and coding work does not notice. Go to 8
bits when output quality matters more than residency, or when the model is
small enough that the difference is a few hundred megabytes.
`--quant-predicate mixed_4_6` sits between the two, spending the extra bits on
the layers that are most sensitive to rounding.

Read the reported average as an average. A model whose parameters sit mostly in
tensors the quantizer skips — embeddings that do not divide by the group size,
a vision encoder, anything without a quantizable weight — reports a number far
above `--q-bits`, and shrinks correspondingly less.

## Better quantization at the same bit width

Both of these take an existing model and spend compute to recover accuracy the
plain round-to-nearest conversion above loses. They are worth it for a model
you will serve for months.

```bash
# Distilled weight quantization: trains the quantized student against the
# full-precision teacher on a calibration set.
mlx_lm.dwq --model Qwen/Qwen3-4B-Instruct-2507 \
  --mlx-path ~/models/qwen3-4b-4bit-dwq --bits 4 --group-size 64

# Activation-aware quantization: rescales channels by observed activation
# magnitude before rounding.
mlx_lm.awq --model Qwen/Qwen3-4B-Instruct-2507 \
  --mlx-path ~/models/qwen3-4b-4bit-awq --bits 4 --group-size 64
```

`mlx_lm.dwq` also improves a model that is already quantized, via
`--quantized-model`, and takes `--num-samples`, `--batch-size`,
`--learning-rate` and `--data-path` for the calibration run. This is how the
`-DWQ` builds on the Hub are made.

## When the converter fails on the config

An architecture mlx-lm fully implements can still refuse to load, because the
config the checkpoint ships and the config mlx-lm expects were written by
different versions of `transformers`. Three failures worth recognizing, all
seen on the hybrid Mamba/attention/MoE families:

- `TypeError: ModelArgs.__init__() missing 1 required positional argument:
  'num_hidden_layers'` — the checkpoint describes its layer stack as a
  `layers_block_type` list and leaves the count implicit. mlx-lm does derive
  the count from that list, but in `__post_init__`, which never runs because
  the dataclass field is required and `from_dict` raises first.
- `KeyError: 'linear_attention'` — the same list, in the newer spelling
  (`linear_attention`, `full_attention`); the map mlx-lm translates it with
  knows `mamba` and `attention`.
- `ValueError: Invalid type dict received in array initialization`, on the
  first generated token rather than during conversion — `transformers` 5.x
  keeps `config.json` strictly valid JSON by writing infinity as
  `{"__float__": "Infinity"}`, so `time_step_limit` reaches `mx.clip` as a
  dict.

None of these are weight problems, and none of them need the checkpoint edited.
`mlx_lm.convert` takes no config override, but the `load` underneath it does,
and the patch is merged into the config the converter writes out — so the
converted model is self-contained and later loads with a plain `mlx_lm.load`:

```python
import json

from mlx_lm.utils import load, quantize_model, save

SRC, DST = "/path/to/checkpoint", "/path/to/output-4bit"
BLOCKS = {"linear_attention": "M", "full_attention": "*", "moe": "E", "mlp": "-"}

with open(f"{SRC}/config.json") as f:
    pattern = "".join(BLOCKS[t] for t in json.load(f)["layers_block_type"])

patch = {
    "hybrid_override_pattern": pattern,
    "num_hidden_layers": len(pattern),
    "time_step_limit": [0.0, float("inf")],
}

model, tokenizer, config = load(SRC, model_config=patch, lazy=True, return_config=True)
model, config = quantize_model(model, config, 64, 4)
save(DST, SRC, model, tokenizer, config)
```

`load` with `lazy=True` matches every weight name and shape against the model
it just built without reading the tensors, so running only that first call is a
seconds-long check that the rest will work — worth doing before committing to a
conversion that takes minutes and tens of gigabytes.

## Check the result before wiring it in

A conversion that finished is not necessarily a conversion the server can load.
Each loader below is the one the matching handler uses, so a clean run here
means the model will come up:

```bash
PY=/path/to/mlx-openai-server/.venv/bin/python
cd /tmp        # not the checkout, see below

# lm
$PY -c "
from mlx_lm.utils import load
from mlx_lm.generate import generate
m, t = load('$HOME/models/smollm2-135m-4bit')
print(generate(m, t, prompt='2+2=', max_tokens=8, verbose=False))"

# multimodal
$PY -c "
from mlx_vlm import load
m, p = load('$HOME/models/smolvlm-256m-4bit')
print('ok', type(m).__name__)"

# embeddings
$PY -c "
from mlx_embeddings.utils import load
m, t = load('$HOME/models/minilm-l6-4bit')
print('ok', type(m).__name__)"
```

The `cd` matters: `python -c` puts the working directory first on `sys.path`, so
run from the checkout the local `app` package shadows the installed one — a
different snapshot whenever the server was installed as a uv tool.

Once the model loads, add it to the config as in
[CONFIGURATION.md](CONFIGURATION.md), or launch it directly to try it out:

```bash
mlx-openai-server launch --model-type lm --model-path ~/models/smollm2-135m-4bit
```
