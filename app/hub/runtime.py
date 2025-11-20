"""Hub runtime scaffolding for multi-model orchestration."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
import time
from typing import Literal

from loguru import logger

from ..config import MLXServerConfig
from .config import HubConfigError, MLXHubConfig

HubModelStatus = Literal["unloaded", "loading", "loaded", "failed"]


@dataclass(slots=True)
class HubModelState:
    """Mutable runtime snapshot for a configured model entry."""

    config: MLXServerConfig
    status: HubModelStatus = "unloaded"
    last_error: str | None = None
    last_transition_at: float = field(default_factory=lambda: time.time())

    def as_summary(self) -> dict[str, str | bool | float | None]:
        """Return a CLI-friendly summary of the model state."""

        return {
            "name": self.config.name,
            "model_path": self.config.model_path,
            "model_type": self.config.model_type,
            "group": self.config.group,
            "default": self.config.is_default_model,
            "log_file": self.config.log_file,
            "status": self.status,
            "last_error": self.last_error,
            "last_transition_at": self.last_transition_at,
        }


class HubRuntime:
    """Lightweight runtime planner for hub-managed deployments."""

    def __init__(self, config: MLXHubConfig) -> None:
        """Initialize the hub runtime.

        Parameters
        ----------
        config : MLXHubConfig
            The hub configuration.
        """

        if not config.models:
            raise HubConfigError("Hub configuration must define at least one model entry")

        self.config = config
        self._models = self._build_model_states(config.models)
        self._groups = {group.name: group for group in config.groups}
        self._group_usage: dict[str, int] = {group.name: 0 for group in config.groups}
        logger.debug(
            "Prepared hub runtime with %d model(s) across %d group(s)",
            len(self._models),
            len(self._groups),
        )

    @staticmethod
    def _build_model_states(models: list[MLXServerConfig]) -> dict[str, HubModelState]:
        """Build model states from configurations.

        Parameters
        ----------
        models : list[MLXServerConfig]
            List of model configurations.

        Returns
        -------
        dict[str, HubModelState]
            Mapping of model names to states.
        """

        states: dict[str, HubModelState] = {}
        for model in models:
            if not model.name:
                raise HubConfigError("Each hub model requires a unique name")
            if model.name in states:
                raise HubConfigError(f"Duplicate model name '{model.name}' detected")
            states[model.name] = HubModelState(config=model)
        return states

    def describe_models(
        self, selection: Iterable[str] | None = None
    ) -> list[dict[str, str | bool | float | None]]:
        """Return serialized summaries for selected models.

        Parameters
        ----------
        selection : Iterable[str] | None, optional
            Model names to include, or None for all.

        Returns
        -------
        list[dict[str, str | bool | float | None]]
            List of model summaries.
        """

        states = self._select_models(selection)
        return [state.as_summary() for state in states]

    def bootstrap_targets(self) -> list[str]:
        """Return models that should be memory-loaded automatically.

        Auto-loading is now opt-in via explicit controller calls, so this returns an empty
        list by default.

        Returns
        -------
        list[str]
            Names of models slated for automatic memory loading.
        """

        return []

    def can_load(self, name: str) -> bool:
        """Return True when the model can enter the loading state.

        Parameters
        ----------
        name : str
            Model name.

        Returns
        -------
        bool
            Whether the model can be loaded.
        """

        state = self._require_state(name)
        if state.status in {"loading", "loaded"}:
            return False
        group_name = state.config.group
        if group_name is None:
            return True
        return self._has_available_slot(group_name)

    def mark_loading(self, name: str) -> None:
        """Transition a model into the loading state with group accounting.

        Parameters
        ----------
        name : str
            Model name.
        """

        state = self._require_state(name)
        if state.status not in {"unloaded", "failed"}:
            raise HubConfigError(f"Model '{name}' cannot enter loading from state '{state.status}'")
        self._reserve_group_slot(state)
        self._change_status(state, "loading")

    def mark_loaded(self, name: str) -> None:
        """Transition a model into the loaded state.

        Parameters
        ----------
        name : str
            Model name.
        """

        state = self._require_state(name)
        if state.status != "loading":
            raise HubConfigError(
                f"Model '{name}' cannot be marked loaded from state '{state.status}'"
            )
        self._change_status(state, "loaded")
        state.last_error = None

    def mark_failed(self, name: str, error: str) -> None:
        """Mark a loading model as failed and free its group slot.

        Parameters
        ----------
        name : str
            Model name.
        error : str
            Error message.
        """

        state = self._require_state(name)
        if state.status != "loading":
            raise HubConfigError(f"Model '{name}' cannot fail from state '{state.status}'")
        self._release_group_slot(state)
        state.last_error = error
        self._change_status(state, "failed")

    def mark_unloaded(self, name: str) -> None:
        """Transition a model back to unloaded, releasing any group slots.

        Parameters
        ----------
        name : str
            Model name.
        """

        state = self._require_state(name)
        if state.status == "unloaded":
            return
        if state.status in {"loading", "loaded"}:
            self._release_group_slot(state)
        self._change_status(state, "unloaded")
        state.last_error = None

    def _select_models(self, selection: Iterable[str] | None) -> list[HubModelState]:
        """Select models based on names or return all.

        Parameters
        ----------
        selection : Iterable[str] | None
            Model names to select, or None for all.

        Returns
        -------
        list[HubModelState]
            Selected model states.
        """

        if selection is None:
            return list(self._models.values())

        normalized: list[str] = []
        for name in selection:
            normalized_name = name.strip()
            if normalized_name:
                normalized.append(normalized_name)
        if not normalized:
            return list(self._models.values())

        missing = [name for name in normalized if name not in self._models]
        if missing:
            joined = ", ".join(sorted(missing))
            raise HubConfigError(
                f"Unknown model name(s): {joined}. Define them inside hub.yaml before referencing them."
            )
        return [self._models[name] for name in normalized]

    def _require_state(self, name: str) -> HubModelState:
        """Get the state for a model, raising if not found.

        Parameters
        ----------
        name : str
            Model name.

        Returns
        -------
        HubModelState
            The model state.
        """

        if name not in self._models:
            raise HubConfigError(f"Unknown model '{name}'")
        return self._models[name]

    def model_names(self) -> list[str]:
        """Return all configured model names.

        Returns
        -------
        list[str]
            List of model names.
        """

        return list(self._models.keys())

    def get_config(self, name: str) -> MLXServerConfig:
        """Return the original server config for ``name``.

        Parameters
        ----------
        name : str
            Model name.

        Returns
        -------
        MLXServerConfig
            The server configuration.
        """

        return self._require_state(name).config

    def get_status(self, name: str) -> HubModelStatus:
        """Return the current lifecycle status for ``name``.

        Parameters
        ----------
        name : str
            Identifier of the model inside the hub configuration.

        Returns
        -------
        HubModelStatus
            One of ``"unloaded"``, ``"loading"``, ``"loaded"``, or ``"failed"``.
        """

        return self._require_state(name).status

    def _change_status(self, state: HubModelState, new_status: HubModelStatus) -> None:
        """Change the status of a model state.

        Parameters
        ----------
        state : HubModelState
            The model state.
        new_status : HubModelStatus
            The new status.
        """

        state.status = new_status
        state.last_transition_at = time.time()

    def _has_available_slot(self, group_name: str) -> bool:
        """Check if a group has an available slot.

        Parameters
        ----------
        group_name : str
            The group name.

        Returns
        -------
        bool
            Whether a slot is available.
        """

        group_cfg = self._groups.get(group_name)
        if group_cfg is None or group_cfg.max_loaded is None:
            return True
        current = self._group_usage.get(group_name, 0)
        return current < group_cfg.max_loaded

    def _reserve_group_slot(self, state: HubModelState) -> None:
        """Reserve a group slot for the state.

        Parameters
        ----------
        state : HubModelState
            The model state.
        """

        group_name = state.config.group
        if group_name is None:
            return
        if not self._has_available_slot(group_name):
            raise HubConfigError(
                f"Group '{group_name}' has no free slots (max_loaded={self._groups[group_name].max_loaded})"
            )
        self._group_usage[group_name] = self._group_usage.get(group_name, 0) + 1

    def _release_group_slot(self, state: HubModelState) -> None:
        """Release a slot for the model's group in the usage tracking.

        Parameters
        ----------
        state : HubModelState
            The model state for which to release the slot.

        Returns
        -------
        None
        """
        group_name = state.config.group
        if group_name is None:
            return
        current = self._group_usage.get(group_name, 0)
        if current > 0:
            self._group_usage[group_name] = current - 1
