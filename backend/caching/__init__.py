"""Caching service managing a simple in-memory cache."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

__all__ = ["CachingService"]


@dataclass
class CachePayload:
    value: Any
    config_signature: str
    dataset_id: Optional[str]
    file_hash: Optional[str]
    config_payload: Dict[str, Any]
    cached_at: str


class CachingService:
    """Entry point for managing cached artifacts."""

    def __init__(self) -> None:
        self._data_cache: Dict[str, CachePayload] = {}
        self._resource_cache: Dict[str, CachePayload] = {}

    def get_data(self, key: str, config: Dict[str, Any]) -> Optional[Any]:
        """Returns cached dataset/intermediate data if still valid for the provided config."""
        payload = self._data_cache.get(key)
        if payload and payload.config_signature == self._hash_config(config):
            return payload.value
        return None

    def set_data(
        self,
        key: str,
        value: Any,
        config: Dict[str, Any],
        dataset_id: Optional[str] = None,
        file_hash: Optional[str] = None,
    ) -> None:
        """Caches dataset/intermediate results keyed off dataset/config/file state."""
        self._data_cache[key] = CachePayload(
            value=value,
            config_signature=self._hash_config(config),
            dataset_id=dataset_id,
            file_hash=file_hash,
            config_payload=config,
            cached_at=self._current_timestamp(),
        )

    def get_resource(self, key: str, config: Dict[str, Any]) -> Optional[Any]:
        """Returns cached model/resource if the configuration still matches."""
        payload = self._resource_cache.get(key)
        if payload and payload.config_signature == self._hash_config(config):
            return payload.value
        return None

    def set_resource(
        self,
        key: str,
        value: Any,
        config: Dict[str, Any],
        dataset_id: Optional[str] = None,
        file_hash: Optional[str] = None,
    ) -> None:
        """Caches model/resource outputs such as trained learners or forecasts."""
        self._resource_cache[key] = CachePayload(
            value=value,
            config_signature=self._hash_config(config),
            dataset_id=dataset_id,
            file_hash=file_hash,
            config_payload=config,
            cached_at=self._current_timestamp(),
        )

    def invalidate(self, key: str) -> None:
        """Purges a cached entry regardless of its type."""
        self._data_cache.pop(key, None)
        self._resource_cache.pop(key, None)

    def build_key(
        self,
        namespace: str,
        dataset_id: Optional[str],
        component: str,
        config: Dict[str, Any],
        extra_signature: Optional[str] = None,
    ) -> str:
        """Builds a deterministic cache key driven by dataset and configuration context."""
        digest_source = {
            "namespace": namespace,
            "dataset_id": dataset_id,
            "component": component,
            "config": config,
            "extra": extra_signature,
        }
        return hashlib.sha256(json.dumps(digest_source, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def _hash_config(self, config: Dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def _current_timestamp(self) -> str:
        from datetime import datetime

        return datetime.utcnow().isoformat()
