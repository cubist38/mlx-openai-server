"""Every YAML config example in the documentation must actually load.

The examples are meant to be copy-pasted, so a stale key or a typo in a doc is
a startup crash for whoever copies it. Prose cannot be tested, but the fenced
YAML blocks can: each one is fed to the real
:func:`app.config.load_config_from_yaml`, which is the same code path the server
uses at launch. Renaming or removing a config key therefore fails here until the
docs are updated with it.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest
import yaml

from app.config import load_config_from_yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Docs that contain configuration examples users are expected to copy.
DOCUMENTED_FILES = [
    "README.md",
    "docs/CONFIGURATION.md",
    "docs/MODEL-LIFECYCLE.md",
]

YAML_BLOCK = re.compile(r"```ya?ml\n(.*?)```", re.DOTALL)


def _yaml_blocks() -> list[tuple[str, str]]:
    """Return ``(id, body)`` for every fenced YAML block in the documented files.

    Returns
    -------
    list[tuple[str, str]]
        Test id (``path:line``) and the block body, in document order.
    """
    blocks: list[tuple[str, str]] = []
    for rel in DOCUMENTED_FILES:
        text = (REPO_ROOT / rel).read_text()
        for match in YAML_BLOCK.finditer(text):
            line = text[: match.start()].count("\n") + 1
            blocks.append((f"{rel}:{line}", match.group(1)))
    return blocks


DOC_BLOCKS = _yaml_blocks()


def test_the_documentation_actually_contains_config_examples() -> None:
    """Guard the extraction itself, so a regex that matches nothing cannot pass."""
    assert len(DOC_BLOCKS) >= 5, f"expected several documented YAML examples, found {DOC_BLOCKS}"


@pytest.mark.parametrize(("block_id", "body"), DOC_BLOCKS, ids=[b[0] for b in DOC_BLOCKS])
def test_documented_yaml_block_loads(block_id: str, body: str, tmp_path: Path) -> None:
    """Each documented block parses and validates like a real config file."""
    parsed = yaml.safe_load(body)
    assert isinstance(parsed, dict), f"{block_id} is not a YAML mapping"

    if "models" not in parsed:
        pytest.skip(f"{block_id} is a fragment, not a full config")

    config_file = tmp_path / "config.yaml"
    config_file.write_text(body)

    # Unknown keys in a model entry raise, so this also catches invented options.
    config = load_config_from_yaml(str(config_file))
    assert config.models, f"{block_id} produced no models"


def test_shipped_example_config_loads() -> None:
    """``examples/config.yaml`` is the file the docs tell users to start from."""
    config = load_config_from_yaml(str(REPO_ROOT / "examples/config.yaml"))
    assert config.models


def test_documented_alias_example_resolves_to_the_names_the_docs_claim() -> None:
    """The alias example must produce the tag the surrounding prose promises.

    ``examples/config.yaml`` documents ``version`` plus ``aliases``; the derived
    ``name:version`` tag is what makes the documented promotion workflow work,
    and it is computed rather than written down, so it is asserted explicitly.
    """
    config = load_config_from_yaml(str(REPO_ROOT / "examples/config.yaml"))
    aliased = [entry for entry in config.models if entry.version or entry.aliases]
    assert aliased, "examples/config.yaml no longer demonstrates version/aliases"

    for entry in aliased:
        names = entry.alias_names()
        if entry.version:
            assert f"{entry.served_model_name}:{entry.version}" in names
        for alias in entry.aliases or []:
            assert alias in names
        assert entry.served_model_name not in names
        assert len(names) == len(set(names))
