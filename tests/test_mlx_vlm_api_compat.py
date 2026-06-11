"""Compatibility tests against the *installed* ``mlx_vlm`` API.

The scheduler unit tests run against ``FakeVLMBatchGenerator``, which accepts
``**kwargs`` and therefore cannot catch a mismatch between the kwargs
``VLMBatchScheduler`` passes and what the real ``mlx_vlm.generate.BatchGenerator``
accepts (e.g. ``kv_bits`` raising ``TypeError`` on mlx-vlm 0.4.x at runtime).
These tests pin the real, installed surface.
"""

from __future__ import annotations

import inspect

import pytest

mlx_vlm_generate = pytest.importorskip("mlx_vlm.generate")


def test_batch_generator_accepts_scheduler_constructor_kwargs():
    """Every kwarg ``VLMBatchScheduler._create_batch_generator`` passes must exist.

    mlx-vlm < 0.6 has no ``kv_bits``/``kv_group_size``/``kv_quant_scheme``/
    ``quantized_kv_start`` on ``BatchGenerator.__init__``; passing them makes the
    scheduler thread die on the first request and the request hang forever.
    """
    sig = inspect.signature(mlx_vlm_generate.BatchGenerator.__init__)
    passed_kwargs = {
        "stop_tokens",
        "sampler",
        "completion_batch_size",
        "prefill_batch_size",
        "prefill_step_size",
        "kv_bits",
        "kv_group_size",
        "kv_quant_scheme",
        "quantized_kv_start",
        "stream",
    }
    params = set(sig.parameters)
    missing = passed_kwargs - params
    assert not missing, (
        f"Installed mlx_vlm BatchGenerator.__init__ does not accept {sorted(missing)}; "
        "VLMBatchScheduler would crash on first request"
    )


def test_batch_generator_insert_accepts_scheduler_kwargs():
    """``_admit_pending`` calls ``insert(prompts, max_tokens=, prompt_kwargs=, logits_processors=)``."""
    sig = inspect.signature(mlx_vlm_generate.BatchGenerator.insert)
    passed_kwargs = {"max_tokens", "prompt_kwargs", "logits_processors"}
    params = set(sig.parameters)
    missing = passed_kwargs - params
    assert not missing, (
        f"Installed mlx_vlm BatchGenerator.insert does not accept {sorted(missing)}"
    )


def test_batch_generator_has_methods_used_by_scheduler():
    """Scheduler also relies on next/stats/remove/close."""
    for name in ("next", "stats", "remove", "close"):
        assert hasattr(mlx_vlm_generate.BatchGenerator, name), (
            f"Installed mlx_vlm BatchGenerator lacks .{name}()"
        )


def test_generate_module_exports_kv_defaults():
    """vlm_batch_scheduler imports these alongside BatchGenerator; if any is
    missing the whole import fails and batching silently disables itself."""
    for name in (
        "DEFAULT_KV_GROUP_SIZE",
        "DEFAULT_KV_QUANT_SCHEME",
        "DEFAULT_QUANTIZED_KV_START",
    ):
        assert hasattr(mlx_vlm_generate, name), f"mlx_vlm.generate lacks {name}"


def test_model_wrapper_imports_resolve():
    """``app.models.mlx_vlm`` imports these mlx_vlm symbols at import/call time.

    mlx-vlm 0.6.x restructured ``generate.py`` into a ``generate/`` package, so
    each import path the wrapper uses must still resolve.
    """
    from mlx_vlm import load, stream_generate  # noqa: F401
    from mlx_vlm.sample_utils import top_p_sampling  # noqa: F401
    from mlx_vlm.utils import load_image, process_inputs_with_fallback  # noqa: F401
    from mlx_vlm.video_generate import process_vision_info  # noqa: F401


def test_stream_generate_accepts_wrapper_kwargs():
    """``MLX_VLM.__call__`` forwards these kwargs to ``stream_generate``."""
    sig = inspect.signature(mlx_vlm_generate.stream_generate)
    params = set(sig.parameters)
    if "kwargs" in params and any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    ):
        return  # accepts **kwargs: every forwarded key is accepted
    passed_kwargs = {
        "max_tokens",
        "temperature",
        "repetition_penalty",
        "repetition_context_size",
        "top_p",
        "logits_processors",
    }
    missing = passed_kwargs - params
    assert not missing, (
        f"Installed mlx_vlm stream_generate does not accept {sorted(missing)}"
    )
