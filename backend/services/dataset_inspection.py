"""Backend contract for dataset inspection payloads consumed by Streamlit.

The DatasetInspectionResult holds the inspection summary, a reproducible quality score breakdown,
and the metadata that surfaces identity tokens required for cache invalidation or reuse. This
contract can be extended safely by future pipeline steps without requiring UI changes because
new fields are additive and grouped into clearly defined substructures.
"""
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from backend.services.ingestion import DatasetState, get_dataset_state


DATASET_INSPECTION_PAYLOAD_VERSION = "1.0"


@dataclass
class QualityScoreInputs:
    missing_ratio: float
    duplicate_ratio: float
    rows: int
    columns: int
    missing_values: int
    duplicate_rows: int


@dataclass
class InspectionMetadata:
    payload_version: str
    dataset_id: str
    dataset_version: int
    ingestion_config_hash: str
    dataset_file_hash: str
    inspection_config_signature: str
    cache_key: str
    inspected_at: str


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
    quality_score_inputs: QualityScoreInputs
    metadata: InspectionMetadata


def inspect_dataset(
    dataset_id: str, config: Optional[Dict[str, Any]] = None
) -> DatasetInspectionResult:
    """Returns dataset inspection results with stable cache invalidation."""
    state = get_dataset_state(dataset_id)
    file_hash = _hash_bytes(state.file_bytes)
    config_snapshot = config if config is not None else {}
    config_serialized = json.dumps(config_snapshot, sort_keys=True, default=str)
    return _inspect_dataset_cached(
        state=state,
        dataset_id=dataset_id,
        dataset_version=state.version,
        ingestion_config_hash=state.config_hash,
        dataset_file_hash=file_hash,
        inspection_config_serialized=config_serialized,
    )


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_dataset_state(state: DatasetState) -> Tuple[str, int, str, str]:
    return (
        state.dataset_id,
        state.version,
        state.config_hash,
        _hash_bytes(state.file_bytes),
    )


@st.cache_data(
    show_spinner=False,
    hash_funcs={DatasetState: _hash_dataset_state},
)
def _inspect_dataset_cached(
    state: DatasetState,
    dataset_id: str,
    dataset_version: int,
    ingestion_config_hash: str,
    dataset_file_hash: str,
    inspection_config_serialized: str,
) -> DatasetInspectionResult:
    return _build_inspection(
        state=state,
        dataset_version=dataset_version,
        ingestion_config_hash=ingestion_config_hash,
        dataset_file_hash=dataset_file_hash,
        inspection_config_serialized=inspection_config_serialized,
    )


def _build_inspection(
    state: DatasetState,
    dataset_version: int,
    ingestion_config_hash: str,
    dataset_file_hash: str,
    inspection_config_serialized: str,
) -> DatasetInspectionResult:
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
    quality_score, quality_inputs = _compute_quality_score(
        rows=row_count,
        columns=column_count,
        missing_values=total_missing,
        duplicate_rows=duplicate_rows,
    )
    inspection_config_signature = _hash_bytes(
        inspection_config_serialized.encode("utf-8")
    )
    cache_key = _build_cache_key(
        dataset_id=state.dataset_id,
        dataset_version=dataset_version,
        ingestion_config_hash=ingestion_config_hash,
        dataset_file_hash=dataset_file_hash,
        inspection_signature=inspection_config_signature,
    )
    metadata = InspectionMetadata(
        payload_version=DATASET_INSPECTION_PAYLOAD_VERSION,
        dataset_id=state.dataset_id,
        dataset_version=dataset_version,
        ingestion_config_hash=ingestion_config_hash,
        dataset_file_hash=dataset_file_hash,
        inspection_config_signature=inspection_config_signature,
        cache_key=cache_key,
        inspected_at=pd.Timestamp.utcnow().isoformat(),
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
        quality_score_inputs=quality_inputs,
        metadata=metadata,
    )


def _build_cache_key(
    dataset_id: str,
    dataset_version: int,
    ingestion_config_hash: str,
    dataset_file_hash: str,
    inspection_signature: str,
) -> str:
    digest_source = (
        f"{dataset_id}:{dataset_version}:{ingestion_config_hash}:{dataset_file_hash}:{inspection_signature}"
    )
    return hashlib.sha256(digest_source.encode("utf-8")).hexdigest()


def _compute_quality_score(
    rows: int,
    columns: int,
    missing_values: int,
    duplicate_rows: int,
) -> (float, QualityScoreInputs):
    if rows == 0 or columns == 0:
        inputs = QualityScoreInputs(
            missing_ratio=0.0,
            duplicate_ratio=0.0,
            rows=rows,
            columns=columns,
            missing_values=missing_values,
            duplicate_rows=duplicate_rows,
        )
        return 1.0, inputs
    missing_ratio = missing_values / (rows * columns)
    duplicate_ratio = duplicate_rows / (rows or 1)
    raw_score = 1.0 - (0.65 * missing_ratio + 0.35 * duplicate_ratio)
    bounded_score = float(max(0.0, min(1.0, raw_score)))
    inputs = QualityScoreInputs(
        missing_ratio=missing_ratio,
        duplicate_ratio=duplicate_ratio,
        rows=rows,
        columns=columns,
        missing_values=missing_values,
        duplicate_rows=duplicate_rows,
    )
    return bounded_score, inputs
