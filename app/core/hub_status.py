"""Shared hub status formatting helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


def build_group_state(
    group_configs: Iterable[Any],
    snapshot_models: list[Mapping[str, Any]] | None,
    *,
    fallback_members: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """Return normalized group summaries for hub status surfaces.

    Parameters
    ----------
    group_configs : Iterable[Any]
        Sequence of group configuration objects with ``name`` attributes.
    snapshot_models : list[Mapping[str, Any]] | None
        Model entries produced by ``HubLifecycleService.get_status``.
    fallback_members : dict[str, list[str]] | None, optional
        Static membership hints (typically derived from configuration) used
        when a snapshot lacks live membership data.

    Returns
    -------
    list[dict[str, Any]]
        Group summary dictionaries containing ``name``, ``max_loaded``,
        ``idle_unload_trigger_min``, ``loaded``, and ``models`` keys.
    """

    members: dict[str, set[str]] = defaultdict(set)
    loaded_counts: dict[str, int] = defaultdict(int)

    if fallback_members:
        for group_name, entries in fallback_members.items():
            normalized = [str(item).strip() for item in entries if str(item).strip()]
            if normalized:
                members[group_name].update(normalized)

    if snapshot_models:
        for entry in snapshot_models:
            if not isinstance(entry, Mapping):
                continue
            group_from_entry = entry.get("group")
            if not isinstance(group_from_entry, str) or not group_from_entry:
                continue
            model_name = entry.get("name")
            if isinstance(model_name, str) and model_name:
                members[group_from_entry].add(model_name)
            if bool(entry.get("memory_loaded")):
                loaded_counts[group_from_entry] += 1

    summaries: list[dict[str, Any]] = []
    for cfg in group_configs:
        name = getattr(cfg, "name", None)
        if not isinstance(name, str) or not name:
            continue
        entry = {
            "name": name,
            "max_loaded": getattr(cfg, "max_loaded", None),
            "idle_unload_trigger_min": getattr(cfg, "idle_unload_trigger_min", None),
            "loaded": loaded_counts.get(name, 0),
            "models": sorted(members.get(name, set())),
        }
        summaries.append(entry)

    summaries.sort(key=lambda item: item["name"])
    return summaries
