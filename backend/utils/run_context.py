"""Run-time helpers for cache-aware pipelines and reproducible metadata."""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "normalize_config",
    "compute_config_signature",
    "compute_file_signature",
    "build_run_id",
    "CacheVersionTracker",
    "RunMetadata",
    "RunMetadataRegistry",
    "run_metadata_registry",
]

_CACHE_VERSION_REGISTRY: Dict[str, Dict[str, str]] = {}

def _now() -> str:
    return datetime.utcnow().isoformat()

def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(',', ':'))

def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()

def normalize_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    def _normalize_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: _normalize_value(value[key]) for key in sorted(value)}
        if isinstance(value, (list, tuple)):
            return [_normalize_value(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    return _normalize_value(config or {})  # type: ignore[arg-type]

def compute_config_signature(config: Optional[Dict[str, Any]]) -> str:
    normalized = normalize_config(config)
    return _stable_hash(normalized)

def compute_file_signature(file_bytes: Optional[bytes]) -> Optional[str]:
    if file_bytes is None:
        return None
    return hashlib.sha256(file_bytes).hexdigest()

def build_run_id(file_signature: Optional[str], config_signature: str) -> str:
    digest_source = {"file_signature": file_signature, "config_signature": config_signature}
    return _stable_hash(digest_source)

class CacheVersionTracker:
    """Tracks stage-level cache versions for a given dataset context."""

    def __init__(self, dataset_id: Optional[str]) -> None:
        self._dataset_key = dataset_id or "__global__"
        self._global_versions = _CACHE_VERSION_REGISTRY.setdefault(self._dataset_key, {})
        self._local_versions: Dict[str, str] = {}

    def track_stage(
        self, stage: str, config_fragment: Optional[Dict[str, Any]]
    ) -> Tuple[str, bool]:
        version = _stable_hash(normalize_config(config_fragment))
        previous = self._global_versions.get(stage)
        self._global_versions[stage] = version
        self._local_versions[stage] = version
        invalidated = previous is not None and previous != version
        return version, invalidated

    def current_versions(self) -> Dict[str, str]:
        return dict(self._local_versions)

@dataclass
class RunMetadata:
    """Captures lightweight metadata about the current pipeline run."""

    run_id: str
    config_signature: str
    file_signature: Optional[str] = None
    dataset_id: Optional[str] = None
    data_signature: Optional[str] = None
    dataset_file_hash: Optional[str] = None
    cache_context: Optional[str] = None
    feature_signature: Optional[str] = None
    created_at: str = field(default_factory=_now)
    last_updated_at: str = field(default_factory=_now)
    stage_statuses: List[Dict[str, Any]] = field(default_factory=list)

    def record_stage(
        self,
        stage: str,
        version: str,
        invalidated: bool,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = {
            "stage": stage,
            "version": version,
            "invalidated": invalidated,
            "details": details or {},
            "timestamp": _now(),
        }
        self.stage_statuses.append(entry)
        self.last_updated_at = entry["timestamp"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "file_signature": self.file_signature,
            "config_signature": self.config_signature,
            "data_signature": self.data_signature,
            "dataset_file_hash": self.dataset_file_hash,
            "cache_context": self.cache_context,
            "feature_signature": self.feature_signature,
            "created_at": self.created_at,
            "last_updated_at": self.last_updated_at,
            "stage_statuses": self.stage_statuses,
        }

class RunMetadataRegistry:
    """Registry for holding the most recent metadata objects per run."""

    def __init__(self) -> None:
        self._runs: Dict[str, RunMetadata] = {}

    def register_run(
        self,
        file_signature: Optional[str],
        config_signature: str,
        dataset_id: Optional[str] = None,
        data_signature: Optional[str] = None,
        dataset_file_hash: Optional[str] = None,
        cache_context: Optional[str] = None,
    ) -> RunMetadata:
        run_id = build_run_id(file_signature, config_signature)
        metadata = RunMetadata(
            run_id=run_id,
            file_signature=file_signature,
            config_signature=config_signature,
            dataset_id=dataset_id,
            data_signature=data_signature,
            dataset_file_hash=dataset_file_hash,
            cache_context=cache_context,
        )
        self._runs[run_id] = metadata
        return metadata

    def get_run(self, run_id: str) -> Optional[RunMetadata]:
        return self._runs.get(run_id)

run_metadata_registry = RunMetadataRegistry()
