"""Tests for disk-backed prompt KV cache retention."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types
from typing import Any


class _FakeCacheLayer:
    """Serializable fake cache layer with MLX-like byte accounting."""

    def __init__(self, value: str, nbytes: int = 10) -> None:
        self.value = value
        self.nbytes = nbytes


def _load_prompt_cache_module(monkeypatch: Any, *, trimmable: bool) -> Any:
    """Import ``app.utils.prompt_cache`` with a fake MLX cache module."""
    fake_mlx_lm = types.ModuleType("mlx_lm")
    fake_models = types.ModuleType("mlx_lm.models")
    fake_cache = types.ModuleType("mlx_lm.models.cache")

    fake_cache.can_trim_prompt_cache = lambda _cache: trimmable

    def trim_prompt_cache(cache: list[Any], n: int) -> int:
        del cache[-n:]
        return n

    fake_cache.trim_prompt_cache = trim_prompt_cache
    fake_models.cache = fake_cache
    fake_mlx_lm.models = fake_models

    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", fake_models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", fake_cache)
    sys.modules.pop("app.utils.prompt_cache", None)
    module = importlib.import_module("app.utils.prompt_cache")
    return importlib.reload(module)


def test_prompt_cache_payload_is_written_to_disk(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """LRU entries should retain metadata in memory and payloads on disk."""
    prompt_cache_module = _load_prompt_cache_module(monkeypatch, trimmable=False)
    cache = prompt_cache_module.LRUPromptCache(max_size=10, cache_dir=tmp_path)

    cache.insert_cache([1, 2, 3], [_FakeCacheLayer("a")])

    entry = cache._trie.get([1, 2, 3])
    assert not hasattr(entry, "prompt_cache")
    assert entry.file_path.exists()

    result, rest = cache.fetch_nearest_cache([1, 2, 3])

    assert rest == []
    assert result is not None
    assert result[0].value == "a"


def test_prompt_cache_eviction_deletes_disk_payload(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Evicting an LRU entry should remove its serialized cache file."""
    prompt_cache_module = _load_prompt_cache_module(monkeypatch, trimmable=False)
    cache = prompt_cache_module.LRUPromptCache(max_size=1, cache_dir=tmp_path)

    cache.insert_cache([1], [_FakeCacheLayer("old")])
    old_path = cache._trie.get([1]).file_path

    cache.insert_cache([2], [_FakeCacheLayer("new")])

    assert not old_path.exists()
    assert cache.fetch_nearest_cache([1]) == (None, [1])
    result, rest = cache.fetch_nearest_cache([2])
    assert rest == []
    assert result is not None
    assert result[0].value == "new"


def test_prompt_cache_longer_hit_loads_and_trims_from_disk(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Longer trimmable hits should load the payload only for the hit path."""
    prompt_cache_module = _load_prompt_cache_module(monkeypatch, trimmable=True)
    cache = prompt_cache_module.LRUPromptCache(max_size=10, cache_dir=tmp_path)

    cache.insert_cache(
        [1, 2, 3, 4],
        [
            _FakeCacheLayer("one"),
            _FakeCacheLayer("two"),
            _FakeCacheLayer("three"),
            _FakeCacheLayer("four"),
        ],
    )

    result, rest = cache.fetch_nearest_cache([1, 2, 9])

    assert rest == [9]
    assert result is not None
    assert [layer.value for layer in result] == ["one", "two"]
