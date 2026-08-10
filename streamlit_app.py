import hashlib
import io
import json
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backend.eda import EDAService
from backend.services.dataset_inspection import DatasetInspectionResult, inspect_dataset
from backend.services.ingestion import (
    DatasetState,
    ingest_dataset,
    reload_dataset,
    remove_dataset,
    replace_dataset,
)
from backend.validation.dataset_validation import DatasetValidationResult, DatasetValidationService


st.set_page_config(page_title="Dataset Workspace", layout="wide")
DATASET_ID = "workspace_dataset"
DEFAULT_INGEST_CONFIG: Dict[str, Any] = {"read_params": {"low_memory": False}}

ROLE_DEFINITIONS = [
    {"role": "target", "label": "Target column", "multi": False},
    {"role": "datetime", "label": "Datetime column", "multi": False},
    {"role": "id", "label": "ID column", "multi": False},
    {"role": "grouping", "label": "Grouping columns", "multi": True},
    {"role": "ignore", "label": "Ignore columns", "multi": True},
    {"role": "categorical", "label": "Categorical columns", "multi": True},
    {"role": "numeric", "label": "Numeric columns", "multi": True},
    {"role": "forecasting", "label": "Forecasting columns", "multi": True},
]

MISSING_VALUE_STRATEGIES = [
    "drop rows",
    "drop columns",
    "mean",
    "median",
    "mode",
    "constant",
    "KNN Imputer",
    "Iterative Imputer",
    "forward fill",
    "backward fill",
]

OUTLIER_METHODS = [
    "none",
    "IQR",
    "Z-score",
    "Isolation Forest",
    "Local Outlier Factor",
    "Winsorization",
]

ENCODING_STRATEGIES = [
    "none",
    "one-hot",
    "ordinal",
    "frequency",
    "label",
]

SCALING_STRATEGIES = [
    "none",
    "standard",
    "min-max",
    "robust",
    "quantile",
    "power",
]

DATE_FEATURE_OPTIONS = ["year", "month", "day", "weekday", "hour", "minute", "quarter"]
ROLLING_STATS_OPTIONS = ["mean", "median", "sum", "std", "min", "max"]
HOLIDAY_CALENDARS = ["US Federal", "US Observed", "UK", "Canada"]
FEATURE_SELECTION_METHODS = ["variance", "select_k_best", "recursive_feature_elimination"]

PREPROCESSING_LABEL_DEFAULTS = {
    "missing": "Missing value strategy",
    "outlier": "Outlier handling",
    "encoding": "Encoding strategy",
    "scaling": "Scaling strategy",
}

PREPROCESSING_DEFAULT_STRATEGIES = {
    "missing": "drop rows",
    "outlier": "none",
    "encoding": "none",
    "scaling": "none",
}

PREPROCESSING_SESSION_KEYS = {
    "missing": "preprocessing_missing_strategy",
    "outlier": "preprocessing_outlier_method",
    "encoding": "preprocessing_encoding_strategy",
    "scaling": "preprocessing_scaling_strategy",
}

CACHE_STATUS_COPY: Dict[str, str] = {
    "not_run": "Preprocessing has not been run yet.",
    "dataset_changed": "Dataset change invalidates cached preprocessing results.",
    "reused": "Cache hit: preprocessing results reused without recomputation.",
    "recomputed": "Cache miss: preprocessing recomputed due to configuration changes.",
}


FEATURE_ENGINEERING_DEFAULTS = {
    "feature_engineering_enabled": True,
    "fe_polynomial_enabled": False,
    "fe_polynomial_columns": [],
    "fe_polynomial_degree": 2,
    "fe_polynomial_include_bias": False,
    "fe_polynomial_interaction_only": False,
    "fe_interaction_enabled": False,
    "fe_interaction_columns": [],
    "fe_lag_enabled": False,
    "fe_lag_columns": [],
    "fe_lag_max_lag": 1,
    "fe_lag_fill_value": "",
    "fe_rolling_enabled": False,
    "fe_rolling_columns": [],
    "fe_rolling_windows": "3,7,14",
    "fe_rolling_stats": ["mean"],
    "fe_date_enabled": False,
    "fe_date_column": "",
    "fe_date_features": ["year", "month", "day"],
    "fe_date_drop_original": False,
    "fe_holiday_enabled": False,
    "fe_holiday_column": "",
    "fe_holiday_calendars": [],
    "fe_holiday_custom": "",
    "fe_cyclical_enabled": False,
    "fe_cyclical_columns": [],
    "fe_cyclical_max_value": 24,
    "fe_pca_enabled": False,
    "fe_pca_columns": [],
    "fe_pca_n_components": 1,
    "fe_pca_whiten": False,
    "fe_feature_selection_enabled": False,
    "fe_feature_selection_columns": [],
    "fe_feature_selection_method": "variance",
    "fe_feature_selection_top_k": 0,
    "fe_feature_selection_threshold": 0.0,
}


PREPROCESSING_OPTIONS = {
    "missing": MISSING_VALUE_STRATEGIES,
    "outlier": OUTLIER_METHODS,
    "encoding": ENCODING_STRATEGIES,
    "scaling": SCALING_STRATEGIES,
}

EDA_VIEW_CHOICES = ["Original dataset", "Validation preview"]

ML_TASK_TYPES = [
    "regression",
    "classification",
    "clustering",
    "forecasting",
    "anomaly_detection",
]

ML_TASK_DISPLAY_NAMES: Dict[str, str] = {
    "regression": "Regression",
    "classification": "Classification",
    "clustering": "Clustering",
    "forecasting": "Forecasting",
    "anomaly_detection": "Anomaly Detection",
}


def _format_ml_task_label(value: str) -> str:
    return ML_TASK_DISPLAY_NAMES.get(value, value.replace("_", " ").title())


def _ensure_task_session_state() -> None:
    if "active_task_type" not in st.session_state:
        st.session_state["active_task_type"] = ML_TASK_TYPES[0]


def _build_runtime_ml_engine_configuration(payload: Dict[str, Any]) -> Dict[str, Any]:
    task_type = st.session_state.get("active_task_type", ML_TASK_TYPES[0])
    runtime_config = {
        "task_type": task_type,
        "preprocessing": payload,
        "generated_at": datetime.utcnow().isoformat(),
    }
    st.session_state["ml_engine_configuration"] = runtime_config
    return runtime_config


def _handle_task_selection_change() -> None:
    columns = _extract_dataframe_columns(st.session_state.get("dataset_state"))
    payload = st.session_state.get("preprocessing_payload_snapshot")
    if not payload:
        payload = _build_preprocessing_payload(columns)
    _build_runtime_ml_engine_configuration(payload)


def _generate_sample_frame() -> pd.DataFrame:
    dates = pd.date_range(end=pd.Timestamp.today(), periods=30, freq="D")
    values = np.sin(np.linspace(0, np.pi * 3, len(dates))) * 10 + np.linspace(0, 5, len(dates))
    categories = ["Baseline" if i % 3 == 0 else "Variant" if i % 3 == 1 else "Control" for i in range(len(dates))]
    return pd.DataFrame({"measurement_date": dates, "value": values.round(2), "category": categories})


def _generate_clustering_frame() -> pd.DataFrame:
    centers = np.array([[1, 1], [5, 5], [9, 2]])
    points = np.vstack([np.random.randn(30, 2) + center for center in centers])
    ids = [f"point-{i+1}" for i in range(len(points))]
    categories = ["A" if i < 30 else "B" if i < 60 else "C" for i in range(len(points))]
    return pd.DataFrame(
        {
            "x_coord": points[:, 0].round(2),
            "y_coord": points[:, 1].round(2),
            "cluster": categories,
            "record_id": ids,
            "measured_at": pd.date_range(end=pd.Timestamp.today(), periods=len(points), freq="H"),
        }
    )


SAMPLE_DATASETS = {
    "Sine wave trajectory": _generate_sample_frame,
    "Clustered experiment": _generate_clustering_frame,
}


def _serialize_dataframe_to_bytes(df: pd.DataFrame, file_name: str) -> bytes:
    if file_name.lower().endswith(".xlsx"):
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        return buffer.getvalue()
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


def _compute_file_hash(file_bytes: Optional[bytes]) -> Optional[str]:
    if not file_bytes:
        return None
    return hashlib.sha256(file_bytes).hexdigest()


def _derive_dataset_hash_from_state(state: Optional[DatasetState]) -> Optional[str]:
    if not state:
        return None
    for attr in ("file_hash", "file_signature", "signature", "dataset_hash"):
        candidate = getattr(state, attr, None)
        if candidate:
            return str(candidate)
    file_bytes = getattr(state, "file_bytes", None)
    if isinstance(file_bytes, (bytes, bytearray)):
        return _compute_file_hash(bytes(file_bytes))
    return None


def _extract_dataframe_columns(state: Optional[DatasetState]) -> list[str]:
    if not state:
        return []
    df = getattr(state, "dataframe", None)
    if isinstance(df, pd.DataFrame):
        return df.columns.tolist()
    if hasattr(df, "columns"):
        try:
            return list(df.columns)
        except Exception:
            return []
    return st.session_state.get("dataset_config", {}).get("available_columns", [])


def _build_preprocessing_payload(columns: list[str]) -> Dict[str, Any]:
    column_config = st.session_state.get("dataset_config", {}).get("column_roles", {})
    numeric_columns = column_config.get("numeric", [])
    categorical_columns = column_config.get("categorical", [])

    missing_selection = st.session_state.get(
        PREPROCESSING_SESSION_KEYS["missing"], PREPROCESSING_DEFAULT_STRATEGIES["missing"]
    )
    missing_cfg = _build_missing_config(missing_selection)
    if columns:
        missing_cfg["columns"] = columns

    outlier_selection = st.session_state.get(
        PREPROCESSING_SESSION_KEYS["outlier"], PREPROCESSING_DEFAULT_STRATEGIES["outlier"]
    )
    outlier_cfg = _build_outlier_config(outlier_selection)
    if numeric_columns:
        outlier_cfg["columns"] = numeric_columns

    encoding_selection = st.session_state.get(
        PREPROCESSING_SESSION_KEYS["encoding"], PREPROCESSING_DEFAULT_STRATEGIES["encoding"]
    )
    encoding_cfg = _build_encoding_config(encoding_selection, categorical_columns)

    scaling_selection = st.session_state.get(
        PREPROCESSING_SESSION_KEYS["scaling"], PREPROCESSING_DEFAULT_STRATEGIES["scaling"]
    )
    scaling_cfg = _build_scaling_config(scaling_selection, numeric_columns)

    label_overrides = {
        section: st.session_state.get(
            f"preprocessing_label_override_{section}", PREPROCESSING_LABEL_DEFAULTS[section]
        )
        for section in PREPROCESSING_LABEL_DEFAULTS
    }
    default_strategies = {
        section: st.session_state.get(
            f"preprocessing_default_{section}", PREPROCESSING_DEFAULT_STRATEGIES[section]
        )
        for section in PREPROCESSING_DEFAULT_STRATEGIES
    }
    strategy_overrides = {
        section: st.session_state.get(PREPROCESSING_SESSION_KEYS[section], default_strategies[section])
        for section in PREPROCESSING_DEFAULT_STRATEGIES
    }

    payload: Dict[str, Any] = {
        "data_settings": {
            "dataset_id": DATASET_ID,
            "dataset_display_name": st.session_state.get("dataset_display_name", ""),
            "dataset_file_hash": st.session_state.get("dataset_file_hash"),
        },
        "column_selection": column_config,
        "labels": label_overrides,
        "default_strategies": default_strategies,
        "active_strategies": strategy_overrides,
        "missing_value": missing_cfg,
        "outlier": outlier_cfg,
        "encoding": encoding_cfg,
        "scaling": scaling_cfg,
        "feature_engineering": _build_feature_engineering_payload(columns),
    }
    return payload


def _normalize_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: _normalize_payload(payload[key]) for key in sorted(payload)}
    if isinstance(payload, list):
        return [_normalize_payload(val) for val in payload]
    if isinstance(payload, tuple):
        return tuple(_normalize_payload(val) for val in payload)
    if isinstance(payload, (str, int, float, bool)) or payload is None:
        return payload
    return str(payload)


def _build_cache_signature(payload: Any) -> str:
    normalized = json.dumps(_normalize_payload(payload), sort_keys=True)
    return hashlib.sha256(normalized.encode()).hexdigest()


def _parse_int_list(raw: str) -> list[int]:
    if not raw:
        return []
    values = []
    for part in raw.split(","):
        candidate = part.strip()
        if not candidate:
            continue
        try:
            parsed = int(candidate)
            if parsed > 0:
                values.append(parsed)
        except ValueError:
            continue
    return sorted(set(values))


def _coerce_scalar_value(raw: str) -> Optional[Any]:
    if raw is None:
        return None
    normalized = raw.strip()
    if not normalized:
        return None
    lowered = normalized.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(normalized)
    except ValueError:
        pass
    try:
        return float(normalized)
    except ValueError:
        pass
    return normalized


def _ensure_preprocessing_session_state() -> None:
    for section, default_strategy in PREPROCESSING_DEFAULT_STRATEGIES.items():
        session_key = PREPROCESSING_SESSION_KEYS[section]
        if session_key not in st.session_state:
            st.session_state[session_key] = default_strategy
        label_key = f"preprocessing_label_override_{section}"
        if label_key not in st.session_state:
            st.session_state[label_key] = PREPROCESSING_LABEL_DEFAULTS[section]
        default_key = f"preprocessing_default_{section}"
        if default_key not in st.session_state:
            st.session_state[default_key] = default_strategy


def _ensure_feature_engineering_session_state() -> None:
    for key, default in FEATURE_ENGINEERING_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _ensure_validation_session_state() -> None:
    if "validation_last_signature" not in st.session_state:
        st.session_state["validation_last_signature"] = ""
    if "validation_last_run_time" not in st.session_state:
        st.session_state["validation_last_run_time"] = ""