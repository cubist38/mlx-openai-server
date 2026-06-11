"""Dtype handling for torch->MLX multimodal input conversion.

The HF processor path runs with ``return_tensors="pt"``, so token ids and
masks arrive as torch int64. mlx-vlm's Qwen3-VL decode path silently
produces corrupted logits (every sampled token is id 0, rendered as
``!!!!...``) when ``input_ids`` are int64, while the mlx-native processor
path emits int32 and generates correctly. The conversion boundary must
therefore normalize integer tensors to int32; float tensors (pixel_values)
must keep their dtype.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")


def test_to_mlx_inputs_casts_torch_integer_tensors_to_int32() -> None:
    from app.models.mlx_vlm import MLX_VLM

    model = object.__new__(MLX_VLM)
    out = model._to_mlx_inputs(
        {
            "input_ids": torch.ones((1, 5), dtype=torch.long),
            "mask": torch.ones((1, 5), dtype=torch.long),
            "pixel_values": torch.zeros((2, 3), dtype=torch.float32),
        }
    )

    assert out["input_ids"].dtype == mx.int32
    assert out["mask"].dtype == mx.int32
    assert out["pixel_values"].dtype == mx.float32


def test_to_mlx_inputs_output_is_usable_from_another_thread() -> None:
    """Inputs are built on the event-loop thread but consumed on the
    inference-worker / batch-scheduler thread. MLX lazy arrays cannot be
    evaluated on a different thread than the one that recorded their ops
    ("There is no Stream(gpu, N) in current thread"), so the conversion
    boundary must return materialized arrays.
    """
    import threading

    from app.models.mlx_vlm import MLX_VLM

    model = object.__new__(MLX_VLM)
    out = model._to_mlx_inputs({"input_ids": mx.ones((1, 5), dtype=mx.int64)})

    errors: list[BaseException] = []

    def consume() -> None:
        try:
            stream = mx.new_thread_local_stream(mx.default_device())
            with mx.stream(stream):
                out["input_ids"].tolist()
        except BaseException as exc:  # noqa: BLE001 - assert below
            errors.append(exc)

    thread = threading.Thread(target=consume)
    thread.start()
    thread.join()
    assert not errors, f"worker thread could not evaluate converted inputs: {errors[0]}"


def test_to_mlx_inputs_casts_int64_mlx_arrays_to_int32() -> None:
    """mlx-vlm 0.6.x processors return MLX arrays directly (ignoring
    ``return_tensors="pt"``), still with int64 ids — so the normalization
    must apply to MLX arrays too, not only torch tensors.
    """
    from app.models.mlx_vlm import MLX_VLM

    model = object.__new__(MLX_VLM)
    out = model._to_mlx_inputs(
        {
            "input_ids": mx.ones((1, 5), dtype=mx.int64),
            "mask": mx.ones((1, 5), dtype=mx.int64),
            "pixel_values": mx.zeros((2, 3), dtype=mx.float32),
        }
    )

    assert out["input_ids"].dtype == mx.int32
    assert out["mask"].dtype == mx.int32
    assert out["pixel_values"].dtype == mx.float32
