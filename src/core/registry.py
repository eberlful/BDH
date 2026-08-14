"""Named component registries used by YAML configuration."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any, TypeVar


T = TypeVar("T")


class Registry:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, type[Any] | Callable[..., Any]] = {}

    def register(self, name: str, component: type[Any] | Callable[..., Any] | None = None):
        """Register a component directly or return a decorator."""
        if component is None:
            return lambda value: self.register(name, value)
        if name in self._items and self._items[name] is not component:
            raise ValueError(f"A {self.kind} named {name!r} is already registered.")
        self._items[name] = component
        return component

    def get(self, name: str) -> type[Any] | Callable[..., Any]:
        try:
            return self._items[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "none"
            raise ValueError(f"Unknown {self.kind} {name!r}. Available: {available}.") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def instantiate(self, spec: str | Mapping[str, Any], **context: Any) -> Any:
        if isinstance(spec, str):
            name, params = spec, {}
        elif isinstance(spec, Mapping):
            name = spec.get("name")
            params = dict(spec.get("params", {}))
            if not isinstance(name, str) or not name:
                raise ValueError(f"A {self.kind} specification requires a non-empty 'name'.")
            if not isinstance(spec.get("params", {}), Mapping):
                raise TypeError(f"Parameters for {self.kind} {name!r} must be a mapping.")
        else:
            raise TypeError(f"A {self.kind} specification must be a string or mapping.")

        component = self.get(name)
        try:
            signature = inspect.signature(component)
            accepts_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            for key, value in context.items():
                if accepts_kwargs or key in signature.parameters:
                    params.setdefault(key, value)
        except (TypeError, ValueError):
            params.update({key: value for key, value in context.items() if key not in params})
        return component(**params)


MODEL_REGISTRY = Registry("model")
DATA_REGISTRY = Registry("data module")
CALLBACK_REGISTRY = Registry("callback")
LOGGER_REGISTRY = Registry("logger")
TRAINER_REGISTRY = Registry("trainer")

_BUILTINS_LOADED = False


def load_builtin_components() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    # Imports intentionally happen here so users can import the base APIs without
    # importing torch/tiktoken-backed concrete implementations.
    from ..callbacks import checkpoint as _checkpoint  # noqa: F401
    from ..data import data as _data  # noqa: F401
    from ..data import hf as _hf  # noqa: F401
    from ..data import sudoku_cot as _sudoku_cot  # noqa: F401
    from ..logging import loggers as _loggers  # noqa: F401
    from ..model import bdh as _model  # noqa: F401
    from ..model import bdh_cq as _model_cq  # noqa: F401
    from ..training import trainer as _trainer  # noqa: F401

    _BUILTINS_LOADED = True

