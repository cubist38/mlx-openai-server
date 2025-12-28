"""Tests for MLX server configuration validation."""

import pytest

from app.config import MLXServerConfig


def test_auto_unload_requires_jit_when_configured() -> None:
    """Test that auto-unload requires JIT to be enabled."""
    with pytest.raises(ValueError, match="Auto-unload requires JIT loading to be enabled"):
        MLXServerConfig(model_path="dummy", auto_unload_minutes=5)


def test_auto_unload_minutes_must_be_positive() -> None:
    """Test that auto-unload minutes must be positive."""
    with pytest.raises(ValueError, match="Auto-unload minutes must be a positive integer"):
        MLXServerConfig(model_path="dummy", jit_enabled=True, auto_unload_minutes=0)


def test_auto_unload_valid_when_jit_enabled() -> None:
    """Test that auto-unload is valid when JIT is enabled."""
    config = MLXServerConfig(model_path="dummy", jit_enabled=True, auto_unload_minutes=2)
    assert config.auto_unload_minutes == 2


def test_jit_enabled_accepts_text_literals() -> None:
    """JIT flag should accept common truthy strings from YAML/CLI."""
    config = MLXServerConfig(model_path="dummy", jit_enabled="TRUE")
    assert config.jit_enabled is True


def test_invalid_jit_literal_raises_value_error() -> None:
    """Unrecognized JIT literals should fail fast with a clear message."""
    with pytest.raises(ValueError, match="jit_enabled"):
        MLXServerConfig(model_path="dummy", jit_enabled="maybe")


def test_auto_unload_accepts_string_minutes() -> None:
    """auto_unload_minutes should coerce numeric strings into integers."""
    config = MLXServerConfig(model_path="dummy", jit_enabled=True, auto_unload_minutes="15")
    assert config.auto_unload_minutes == 15


def test_auto_unload_rejects_non_numeric_string() -> None:
    """Non-numeric auto-unload values raise a descriptive error."""
    with pytest.raises(ValueError, match="auto_unload_minutes"):
        MLXServerConfig(model_path="dummy", jit_enabled=True, auto_unload_minutes="soon")
