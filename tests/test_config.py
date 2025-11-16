import pytest

from app.config import MLXServerConfig


def test_auto_unload_requires_jit_when_configured():
    with pytest.raises(ValueError):
        MLXServerConfig(model_path="dummy", auto_unload_minutes=5)


def test_auto_unload_minutes_must_be_positive():
    with pytest.raises(ValueError):
        MLXServerConfig(model_path="dummy", jit_enabled=True, auto_unload_minutes=0)


def test_auto_unload_valid_when_jit_enabled():
    config = MLXServerConfig(model_path="dummy", jit_enabled=True, auto_unload_minutes=2)
    assert config.auto_unload_minutes == 2
