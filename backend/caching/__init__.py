"""Caching service managing a simple in-memory cache."""

from typing import Any, Dict

__all__ = ["CachingService"]


class CachingService:
    """Entry point for managing cached artifacts."""

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def get(self, key: str) -> Any:
        """Returns the cached value for the key if available."""
        return self._store.get(key)

    def set(self, key: str, value: Any, config: Dict[str, Any]) -> None:
        """Stores a value in the cache with optional TTL handling."""
        self._store[key] = {
            "value": value,
            "config": config,
        }
