import io
import json
import hashlib
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import pandas as pd


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


# Global in-memory registry. In a real Streamlit app this would be backed by st.session_state
# or st.cache_data reading from functions that depend on load_dataset() to keep data in memory.
_DATASETS: Dict[str, DatasetState] = {}
_DATASETS_LOCK = threading.Lock()


def _compute_hash(obj: Any) -> str:
    serialized = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _read_dataframe(file_bytes: bytes, file_name: str, config: Dict[str, Any]) -> pd.DataFrame:
    lower = file_name.lower()
    buffer = io.BytesIO(file_bytes)
    read_params = config.get("read_params", {})
    if lower.endswith(".csv"):
        return pd.read_csv(buffer, **read_params)
    if lower.endswith( ".xls") or lower.endswith( ".xlsx") or lower.endswith(".xlsm"):
        return pd.read_excel(buffer, **read_params)
    raise ValueError(f"Unsupported file type for ingestion: {file_name}")


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
    config_hash = _compute_hash(config)
    with _DATASETS_LOCK:
        state = DatasetState(
            dataset_id=dataset_id,
            name=name,
            dataframe=df,
            config_snapshot=config,
            config_hash=config_hash,
            version=1,
            file_name=file_name,
            file_bytes=file_bytes,
        )
        _DATASETS[dataset_id] = state
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
    with _DATASETS_LOCK:
        existing = _DATASETS.get(dataset_id)
    name = existing.name if existing else dataset_id
    return ingest_dataset(dataset_id, name, file_name, file_bytes, config)


def reload_dataset(dataset_id: str, config: Optional[Dict[str, Any]] = None) -> DatasetState:
    """Re-reads the stored upload bytes optionally with a new configuration.

    Useful for rerunning ingestion when downstream config changes. This triggers cache invalidation
    via a new config hash. The file bytes are kept from the last upload.
    """
    with _DATASETS_LOCK:
        state = _DATASETS.get(dataset_id)
        if state is None:
            raise KeyError(f"Dataset '{dataset_id}' not registered")
        effective_config = config if config is not None else state.config_snapshot
        df = _read_dataframe(state.file_bytes, state.file_name, effective_config)
        config_hash = _compute_hash(effective_config)
        state.dataframe = df
        state.config_snapshot = effective_config
        state.config_hash = config_hash
        state.version += 1
    return state


def remove_dataset(dataset_id: str) -> None:
    """Deletes dataset metadata and frees the in-memory reference."""
    with _DATASETS_LOCK:
        if dataset_id in _DATASETS:
            del _DATASETS[dataset_id]


def sample_dataset(dataset_id: str, sample_size: int, random_state: Optional[int] = None) -> pd.DataFrame:
    """Returns a sampled DataFrame without modifying the original state."""
    with _DATASETS_LOCK:
        state = _DATASETS.get(dataset_id)
        if state is None:
            raise KeyError(f"Dataset '{dataset_id}' not registered")
        df = state.dataframe
    if sample_size <= 0 or sample_size >= len(df):
        return df.copy()
    return df.sample(n=sample_size, random_state=random_state)


def get_dataset_state(dataset_id: str) -> DatasetState:
    with _DATASETS_LOCK:
        state = _DATASETS.get(dataset_id)
        if state is None:
            raise KeyError(f"Dataset '{dataset_id}' not registered")
        return state


def list_datasets() -> Dict[str, Tuple[str, int, str]]:
    """Returns a lightweight view of datasets for orchestration use."""
    with _DATASETS_LOCK:
        return {
            dataset_id: (state.name, state.version, state.config_hash)
            for dataset_id, state in _DATASETS.items()
        }
