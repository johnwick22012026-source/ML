"""Caching service managing a simple in-memory cache."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from backend.utils.run_context import compute_config_signature, normalize_config

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

    def get_data(
        self,
        key: str,
        config: Dict[str, Any],
        dataset_id: Optional[str] = None,
        file_hash: Optional[str] = None,
    ) -> Optional[Any]:
        """Returns cached dataset/intermediate data if still valid for the provided config and state."""
        payload = self._data_cache.get(key)
        if self._is_payload_valid(payload, config, dataset_id, file_hash):
            return payload.value
        return None

    def get_data_payload(
        self,
        key: str,
        config: Dict[str, Any],
        dataset_id: Optional[str] = None,
        file_hash: Optional[str] = None,
    ) -> Optional[CachePayload]:
        """Returns the cached payload including metadata when the config and state still matches."""
        payload = self._data_cache.get(key)
        if self._is_payload_valid(payload, config, dataset_id, file_hash):
            return payload
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
        config_payload, signature = self._prepare_config_payload(config)
        self._data_cache[key] = CachePayload(
            value=value,
            config_signature=signature,
            dataset_id=dataset_id,
            file_hash=file_hash,
            config_payload=config_payload,
            cached_at=self._current_timestamp(),
        )

    def get_resource(
        self,
        key: str,
        config: Dict[str, Any],
        dataset_id: Optional[str] = None,
        file_hash: Optional[str] = None,
    ) -> Optional[Any]:
        """Returns cached model/resource if the configuration and state still matches."""
        payload = self._resource_cache.get(key)
        if self._is_payload_valid(payload, config, dataset_id, file_hash):
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
        config_payload, signature = self._prepare_config_payload(config)
        self._resource_cache[key] = CachePayload(
            value=value,
            config_signature=signature,
            dataset_id=dataset_id,
            file_hash=file_hash,
            config_payload=config_payload,
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
        file_hash: Optional[str] = None,
    ) -> str:
        """Builds a deterministic cache key driven by dataset and configuration context."""
        _, config_signature = self._prepare_config_payload(config)
        digest_source = {
            "namespace": namespace,
            "dataset_id": dataset_id,
            "component": component,
            "config_signature": config_signature,
            "file_hash": file_hash,
            "extra": extra_signature,
        }
        return hashlib.sha256(json.dumps(digest_source, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def _is_payload_valid(
        self,
        payload: Optional[CachePayload],
        config: Dict[str, Any],
        dataset_id: Optional[str],
        file_hash: Optional[str],
    ) -> bool:
        if not payload:
            return False
        if payload.config_signature != self._hash_config(config):
            return False
        if payload.dataset_id != dataset_id:
            return False
        if payload.file_hash != file_hash:
            return False
        return True

    def _prepare_config_payload(self, config: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        normalized = normalize_config(config)
        signature = compute_config_signature(normalized)
        return normalized, signature

    def _hash_config(self, config: Dict[str, Any]) -> str:
        return compute_config_signature(normalize_config(config))

    def _current_timestamp(self) -> str:
        from datetime import datetime

        return datetime.utcnow().isoformat()
