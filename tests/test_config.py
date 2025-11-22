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
