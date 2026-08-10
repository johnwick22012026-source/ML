import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from backend.services.ingestion import DatasetState, get_dataset_state


@dataclass
class DatasetInspectionResult:
    dataset_id: str
    dataset_version: int
    row_count: int
    column_count: int
    preview: Dict[str, Any]
    schema: List[Dict[str, Any]]
    dtype_summary: Dict[str, int]
    memory_usage_bytes: int
    missing_value_counts: Dict[str, int]
    total_missing_values: int
    duplicate_row_count: int
    quality_score: float


def inspect_dataset(
    dataset_id: str, config: Optional[Dict[str, Any]] = None
) -> DatasetInspectionResult:
    """Returns dataset inspection results with stable cache invalidation."""
    state = get_dataset_state(dataset_id)
    file_hash = _hash_bytes(state.file_bytes)
    config_snapshot = config if config is not None else {}
    config_serialized = json.dumps(config_snapshot, sort_keys=True, default=str)
    return _inspect_dataset_cached(
        dataset_id=dataset_id,
        dataset_version=state.version,
        config_hash=state.config_hash,
        file_hash=file_hash,
        inspection_config_serialized=config_serialized,
    )


@st.cache_data(show_spinner=False)
def _inspect_dataset_cached(
    dataset_id: str,
    dataset_version: int,
    config_hash: str,
    file_hash: str,
    inspection_config_serialized: str,
) -> DatasetInspectionResult:
    state = get_dataset_state(dataset_id)
    return _build_inspection(state, dataset_version)


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _build_inspection(state: DatasetState, dataset_version: int) -> DatasetInspectionResult:
    df = state.dataframe
    row_count = len(df)
    column_count = df.shape[1]
    preview_records = df.head(5).to_dict(orient="records")
    schema_summary = []
    for column in df.columns:
        column_series = df[column]
        schema_summary.append(
            {
                "column": column,
                "dtype": str(column_series.dtype),
                "non_null_count": int(column_series.count()),
                "unique_values": int(column_series.nunique(dropna=True)),
            }
        )
    dtype_summary = {
        str(dtype): int(count)
        for dtype, count in df.dtypes.astype(str).value_counts().items()
    }
    memory_usage = int(df.memory_usage(deep=True).sum())
    missing_counts = {
        column: int(count)
        for column, count in df.isna().sum().items()
    }
    total_missing = sum(missing_counts.values())
    duplicate_rows = int(df.duplicated().sum())
    quality_score = _compute_quality_score(
        row_count, column_count, total_missing, duplicate_rows
    )
    return DatasetInspectionResult(
        dataset_id=state.dataset_id,
        dataset_version=dataset_version,
        row_count=row_count,
        column_count=column_count,
        preview={
            "columns": list(df.columns),
            "records": preview_records,
        },
        schema=schema_summary,
        dtype_summary=dtype_summary,
        memory_usage_bytes=memory_usage,
        missing_value_counts=missing_counts,
        total_missing_values=total_missing,
        duplicate_row_count=duplicate_rows,
        quality_score=quality_score,
    )


def _compute_quality_score(
    rows: int,
    columns: int,
    missing_values: int,
    duplicate_rows: int,
) -> float:
    if rows == 0 or columns == 0:
        return 1.0
    missing_ratio = missing_values / (rows * columns)
    duplicate_ratio = duplicate_rows / rows
    raw_score = 1.0 - (0.65 * missing_ratio + 0.35 * duplicate_ratio)
    return float(max(0.0, min(1.0, raw_score)))
