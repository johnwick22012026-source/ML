"""Helper utilities to build dataset validation summaries for Streamlit."""
from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.services.dataset_inspection import QualityScoreInputs

VALIDATION_PAYLOAD_VERSION = "1.0"


@dataclass
class DatasetValidationMetadata:
    payload_version: str
    file_hash: str
    config_signature: str
    generated_at: str


@dataclass
class DatasetValidationResult:
    preview: Dict[str, Any]
    schema: List[Dict[str, Any]]
    dimensions: Dict[str, int]
    dtype_summary: Dict[str, int]
    memory_usage_bytes: int
    missing_value_counts: Dict[str, int]
    total_missing_values: int
    duplicate_row_count: int
    quality_score: float
    quality_score_inputs: QualityScoreInputs
    metadata: DatasetValidationMetadata


class DatasetValidationService:
    """Run validation on an uploaded dataset according to configuration."""

    def validate(
        self,
        *,
        file_name: str,
        file_bytes: bytes,
        config: Optional[Dict[str, Any]] = None,
    ) -> DatasetValidationResult:
        """Inspect the dataset and compute metrics defined by the validation config."""
        if not file_bytes:
            raise ValueError("An uploaded file is required for validation.")

        normalized_config = config or {}
        validation_config = self._extract_validation_config(normalized_config)
        df = self._load_dataframe(
            file_name=file_name,
            file_bytes=file_bytes,
            validation_config=validation_config,
        )

        row_count = len(df)
        column_count = df.shape[1]
        bound_preview_rows = self._resolve_preview_rows(validation_config, row_count)
        preview_mode = validation_config.get("preview_mode", "head")
        preview = self._build_preview(
            df=df,
            rows=bound_preview_rows,
            mode=preview_mode,
            seed=self._resolve_seed(validation_config),
        )
        schema = self._build_schema(df)
        dtype_summary = self._build_dtype_summary(df)
        memory_usage = int(df.memory_usage(deep=True).sum())
        missing_counts = {col: int(cnt) for col, cnt in df.isna().sum().items()}
        total_missing = sum(missing_counts.values())
        duplicate_rows = int(df.duplicated().sum())
        quality_score, quality_inputs = _compute_quality_score(
            rows=row_count,
            columns=column_count,
            missing_values=total_missing,
            duplicate_rows=duplicate_rows,
        )
        metadata = DatasetValidationMetadata(
            payload_version=VALIDATION_PAYLOAD_VERSION,
            file_hash=_hash_bytes(file_bytes),
            config_signature=_hash_config(normalized_config),
            generated_at=datetime.utcnow().isoformat() + "Z",
        )
        return DatasetValidationResult(
            preview=preview,
            schema=schema,
            dimensions={"rows": row_count, "columns": column_count},
            dtype_summary=dtype_summary,
            memory_usage_bytes=memory_usage,
            missing_value_counts=missing_counts,
            total_missing_values=total_missing,
            duplicate_row_count=duplicate_rows,
            quality_score=quality_score,
            quality_score_inputs=quality_inputs,
            metadata=metadata,
        )

    @staticmethod
    def _extract_validation_config(config: Dict[str, Any]) -> Dict[str, Any]:
        validation_config = config.get("validation")
        if isinstance(validation_config, dict):
            return validation_config
        return {}

    @staticmethod
    def _resolve_seed(config: Dict[str, Any]) -> Optional[int]:
        seed_value = config.get("seed")
        if seed_value is None:
            return None
        try:
            return int(seed_value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _resolve_preview_rows(config: Dict[str, Any], row_count: int) -> int:
        candidate = config.get("preview_rows", 5)
        try:
            preview_rows = int(candidate)
        except (TypeError, ValueError):
            preview_rows = 5
        if preview_rows < 1:
            preview_rows = 1
        return min(preview_rows, max(1, row_count))

    @staticmethod
    def _build_preview(
        df: pd.DataFrame,
        rows: int,
        mode: str,
        seed: Optional[int],
    ) -> Dict[str, Any]:
        if rows <= 0:
            rows = 1
        if mode == "sample" and seed is not None and len(df) > 0:
            sampled = df.sample(n=min(rows, len(df)), random_state=seed)
        else:
            sampled = df.head(rows)
        return {"columns": list(sampled.columns), "records": sampled.to_dict(orient="records")}

    @staticmethod
    def _build_schema(df: pd.DataFrame) -> List[Dict[str, Any]]:
        schema = []
        for column in df.columns:
            column_series = df[column]
            schema.append(
                {
                    "column": column,
                    "dtype": str(column_series.dtype),
                    "non_null_count": int(column_series.count()),
                    "unique_values": int(column_series.nunique(dropna=True)),
                }
            )
        return schema

    @staticmethod
    def _build_dtype_summary(df: pd.DataFrame) -> Dict[str, int]:
        return {str(dtype): int(count) for dtype, count in df.dtypes.astype(str).value_counts().items()}

    @staticmethod
    def _load_dataframe(
        file_name: str,
        file_bytes: bytes,
        validation_config: Dict[str, Any],
    ) -> pd.DataFrame:
        buffer = io.BytesIO(file_bytes)
        lower_name = file_name.lower()
        data_settings = validation_config.get("data_settings", {})
        csv_options = data_settings.get("csv_options", {})
        excel_options = data_settings.get("excel_options", {})
        try:
            if lower_name.endswith(".csv"):
                return pd.read_csv(buffer, **csv_options)
            if lower_name.endswith(('.xlsx', '.xlsm', '.xls')):
                excel_kwargs = {**excel_options}
                extension = os.path.splitext(lower_name)[1]
                if extension == ".xls":
                    excel_kwargs.setdefault("engine", "xlrd")
                else:
                    excel_kwargs.setdefault("engine", "openpyxl")
                return pd.read_excel(buffer, **excel_kwargs)
        except Exception as exc:
            raise ValueError(f"Unable to parse the uploaded dataset: {exc}")
        raise ValueError("Unsupported file type for validation. Only CSV and Excel are supported.")


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_config(config: Dict[str, Any]) -> str:
    config_serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_serialized.encode("utf-8")).hexdigest()


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
