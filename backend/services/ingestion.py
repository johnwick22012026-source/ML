import io
import json
import hashlib
import threading
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st


@dataclass
class DatasetState:
    dataset_id: str
    name: str
    dataframe: pd.DataFrame
    config_snapshot: Dict[str, Any]
    config_hash: str
    version: int = 0
    file_name: str = ""
    file_bytes: bytes = b""


@dataclass
class DatasetContext:
    dataset_id: str
    name: str
    dataframe: pd.DataFrame
    column_roles: Dict[str, List[str]]
    available_columns: List[str]
    required_roles: List[str]
    role_config_snapshot: Dict[str, Any]
    read_config_snapshot: Dict[str, Any]

    def get_columns_for_role(self, role: str) -> List[str]:
        return self.column_roles.get(role, [])


_DATASETS_LOCK = threading.Lock()


@st.cache_resource(show_spinner=False)
def _dataset_store() -> Dict[str, DatasetState]:
    """Provides a reusable dataset registry across Streamlit reruns."""
    return {}


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_dataframe_raw(file_bytes: bytes, file_name: str, config: Dict[str, Any]) -> pd.DataFrame:
    lower = file_name.lower()
    buffer = io.BytesIO(file_bytes)
    read_params = config.get("read_params", {})
    if lower.endswith(".csv"):
        return pd.read_csv(buffer, **read_params)
    if lower.endswith(".xls") or lower.endswith(".xlsx") or lower.endswith(".xlsm"):
        return pd.read_excel(buffer, **read_params)
    raise ValueError(f"Unsupported file type for ingestion: {file_name}")


@st.cache_data(
    show_spinner=False,
    hash_funcs={bytes: _hash_bytes},
)
def _load_dataframe_cached(file_bytes: bytes, file_name: str, config: Dict[str, Any]) -> pd.DataFrame:
    """Caches Base DataFrame reads so they only rerun when file or config changes."""
    return _read_dataframe_raw(file_bytes, file_name, config)


def _read_dataframe(file_bytes: bytes, file_name: str, config: Dict[str, Any]) -> pd.DataFrame:
    return _load_dataframe_cached(file_bytes=file_bytes, file_name=file_name, config=config)


def _compute_hash(obj: Any) -> str:
    serialized = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _ensure_column_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    raise ValueError("Column role definitions must be a string or a sequence of strings.")


def _normalize_column_roles(columns: Sequence[str], roles_config: Dict[str, Any]) -> Dict[str, List[str]]:
    normalized: Dict[str, List[str]] = {}
    available_columns = set(columns)
    for role, raw in roles_config.items():
        assigned = [item.strip() for item in _ensure_column_list(raw) if str(item).strip()]
        if not assigned:
            raise ValueError(f"Role '{role}' must reference at least one column.")
        missing = [col for col in assigned if col not in available_columns]
        if missing:
            raise ValueError(
                f"Role '{role}' references columns that do not exist in the dataset: {missing}"
            )
        seen: List[str] = []
        for col in assigned:
            if col not in seen:
                seen.append(col)
        normalized[role] = seen
    return normalized


def _ensure_required_roles(normalized_roles: Dict[str, List[str]], required_roles: Sequence[str]) -> None:
    if not required_roles:
        return
    missing = [role for role in required_roles if not normalized_roles.get(role)]
    if missing:
        raise ValueError(
            f"The following required roles have not been assigned columns in the configuration: {missing}"
        )


def _ensure_mapping(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"'{name}' must be a mapping, but got {type(value).__name__}.")
    return value


def _ensure_sequence(value: Any, name: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    raise ValueError(f"'{name}' must be a sequence of strings, but got {type(value).__name__}.")


def ingest_dataset(
    dataset_id: str,
    name: str,
    file_name: str,
    file_bytes: bytes,
    config: Dict[str, Any],
) -> DatasetState:
    """Reads an uploaded file (CSV/Excel) and registers it under dataset_id.

    Returns the in-memory DatasetState which includes the dataframe and metadata.
    """
    df = _read_dataframe(file_bytes, file_name, config)
    config_snapshot = deepcopy(config)
    config_hash = _compute_hash(config_snapshot)
    registry = _dataset_store()
    with _DATASETS_LOCK:
        state = DatasetState(
            dataset_id=dataset_id,
            name=name,
            dataframe=df,
            config_snapshot=config_snapshot,
            config_hash=config_hash,
            version=1,
            file_name=file_name,
            file_bytes=file_bytes,
        )
        registry[dataset_id] = state
    return state


def replace_dataset(
    dataset_id: str,
    file_name: str,
    file_bytes: bytes,
    config: Dict[str, Any],
) -> DatasetState:
    """Replaces an existing dataset with a new upload.

    If no dataset exists yet, behaves like ingest_dataset.
    """
    registry = _dataset_store()
    with _DATASETS_LOCK:
        existing = registry.get(dataset_id)
    name = existing.name if existing else dataset_id
    return ingest_dataset(dataset_id, name, file_name, file_bytes, config)


def reload_dataset(dataset_id: str, config: Optional[Dict[str, Any]] = None) -> DatasetState:
    """Re-reads the stored upload bytes optionally with a new configuration.

    Useful for rerunning ingestion when downstream config changes. This triggers cache invalidation
    via a new config hash. The file bytes are kept from the last upload.
    """
    registry = _dataset_store()
    with _DATASETS_LOCK:
        state = registry.get(dataset_id)
        if state is None:
            raise KeyError(f"Dataset '{dataset_id}' not registered")
        effective_config = config if config is not None else state.config_snapshot
        df = _read_dataframe(state.file_bytes, state.file_name, effective_config)
        config_snapshot = deepcopy(effective_config)
        config_hash = _compute_hash(config_snapshot)
        state.dataframe = df
        state.config_snapshot = config_snapshot
        state.config_hash = config_hash
        state.version += 1
    return state


def remove_dataset(dataset_id: str) -> None:
    """Deletes dataset metadata and frees the in-memory reference."""
    registry = _dataset_store()
    with _DATASETS_LOCK:
        registry.pop(dataset_id, None)


def sample_dataset(dataset_id: str, sample_size: int, random_state: Optional[int] = None) -> pd.DataFrame:
    """Returns a sampled DataFrame without modifying the original state."""
    registry = _dataset_store()
    with _DATASETS_LOCK:
        state = registry.get(dataset_id)
        if state is None:
            raise KeyError(f"Dataset '{dataset_id}' not registered")
        df = state.dataframe
    if sample_size <= 0 or sample_size >= len(df):
        return df.copy()
    return df.sample(n=sample_size, random_state=random_state)


def get_dataset_state(dataset_id: str) -> DatasetState:
    registry = _dataset_store()
    with _DATASETS_LOCK:
        state = registry.get(dataset_id)
        if state is None:
            raise KeyError(f"Dataset '{dataset_id}' not registered")
        return state


def list_datasets() -> Dict[str, Tuple[str, int, str]]:
    """Returns a lightweight view of datasets for orchestration use."""
    registry = _dataset_store()
    with _DATASETS_LOCK:
        return {
            dataset_id: (state.name, state.version, state.config_hash)
            for dataset_id, state in registry.items()
        }


def build_dataset_context(dataset_id: str, config: Dict[str, Any]) -> DatasetContext:
    """Constructs a DatasetContext that normalizes column roles for downstream services."""
    if not isinstance(config, dict):
        raise ValueError("'config' must be a mapping of configuration values.")
    registry = _dataset_store()
    with _DATASETS_LOCK:
        state = registry.get(dataset_id)
        if state is None:
            raise KeyError(f"Dataset '{dataset_id}' not registered")
        dataframe = state.dataframe
    columns = list(dataframe.columns)
    role_definitions = _ensure_mapping(config.get("column_roles", {}), "column_roles")
    required_roles = _ensure_sequence(config.get("required_roles", []), "required_roles")
    normalized_roles = _normalize_column_roles(columns, role_definitions)
    _ensure_required_roles(normalized_roles, required_roles)
    read_params = _ensure_mapping(config.get("read_params", {}), "read_params")
    context = DatasetContext(
        dataset_id=dataset_id,
        name=state.name,
        dataframe=dataframe,
        column_roles=normalized_roles,
        available_columns=columns,
        required_roles=required_roles,
        role_config_snapshot=role_definitions,
        read_config_snapshot=read_params,
    )
    return context
