"""Centralized registry for models, losses, augmentations and trainers."""

from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar, Dict

T = TypeVar("T")


class Registry(Generic[T]):
    """Generic registry for mapping names to callables or classes."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._entries: Dict[str, T] = {}

    def register(self, name: str | None = None) -> Callable[[T], T]:
        """Decorator to register an entry under a specific name.

        Args:
            name: Entry name. If None, the object's __name__ is used.
        """

        def decorator(entry: T) -> T:
            entry_name = name or getattr(entry, "__name__", str(entry))
            if entry_name in self._entries:
                raise KeyError(f"Entry {entry_name!r} already registered in {self._name}")
            self._entries[entry_name] = entry
            return entry

        return decorator

    def get(self, name: str) -> T:
        """Return a registered entry by name.

        Raises:
            KeyError: If the entry is not registered.
        """

        try:
            return self._entries[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._entries))
            raise KeyError(f"Unknown entry {name!r} in {self._name}. Registered: {known}") from exc

    def list(self) -> list[str]:
        """Return a list of registered entry names."""
        return sorted(self._entries.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._entries


# Global registries
MODEL_REGISTRY = Registry[Callable[..., Any]]("Models")
LOSS_REGISTRY = Registry[Callable[..., Any]]("Losses")
AUGMENTATION_REGISTRY = Registry[Callable[..., Any]]("Augmentations")
TRAINER_REGISTRY = Registry[Callable[..., Any]]("Trainers")
