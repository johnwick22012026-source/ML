import hashlib
import io
import json
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

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


def _ensure_feature_engineering_session_state() -> None:
    for key, default in FEATURE_ENGINEERING_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _build_feature_engineering_payload(columns: list[str]) -> Dict[str, Any]:
    aggregated_holidays = list(st.session_state.get("fe_holiday_calendars", []))
    custom_holidays = st.session_state.get("fe_holiday_custom", "")
    if custom_holidays:
        for item in [part.strip() for part in custom_holidays.split(",") if part.strip()]:
            if item and item not in aggregated_holidays:
                aggregated_holidays.append(item)

    rolling_windows = _parse_int_list(st.session_state.get("fe_rolling_windows", ""))
    if not rolling_windows:
        rolling_windows = [3]

    cyclical_columns = st.session_state.get("fe_cyclical_columns", []) or []
    cyclical_max = st.session_state.get("fe_cyclical_max_value", 24)

    pca_columns = st.session_state.get("fe_pca_columns", []) or []
    requested_components = st.session_state.get("fe_pca_n_components", 1)
    component_limit = max(len(pca_columns), 1)
    if requested_components > component_limit:
        requested_components = component_limit

    feature_selection_columns = st.session_state.get("fe_feature_selection_columns", []) or []
    selection_top_k = st.session_state.get("fe_feature_selection_top_k", 0)
    if selection_top_k <= 0:
        selection_top_k = None

    payload = {
        "enabled": st.session_state.get("feature_engineering_enabled", True),
        "polynomial": {
            "enabled": st.session_state.get("fe_polynomial_enabled", False),
            "columns": st.session_state.get("fe_polynomial_columns", []),
            "degree": st.session_state.get("fe_polynomial_degree", 2),
            "include_bias": st.session_state.get("fe_polynomial_include_bias", False),
            "interaction_only": st.session_state.get("fe_polynomial_interaction_only", False),
        },
        "interaction": {
            "enabled": st.session_state.get("fe_interaction_enabled", False),
            "columns": st.session_state.get("fe_interaction_columns", []),
        },
        "lag": {
            "enabled": st.session_state.get("fe_lag_enabled", False),
            "columns": st.session_state.get("fe_lag_columns", []),
            "max_lag": max(1, st.session_state.get("fe_lag_max_lag", 1)),
            "fill_value": _coerce_scalar_value(st.session_state.get("fe_lag_fill_value", "")),
        },
        "rolling": {
            "enabled": st.session_state.get("fe_rolling_enabled", False),
            "columns": st.session_state.get("fe_rolling_columns", []),
            "windows": rolling_windows,
            "stats": st.session_state.get("fe_rolling_stats", ["mean"]),
        },
        "date": {
            "enabled": st.session_state.get("fe_date_enabled", False),
            "column": st.session_state.get("fe_date_column") or None,
            "features": st.session_state.get("fe_date_features", ["year", "month", "day"]),
            "drop_original": st.session_state.get("fe_date_drop_original", False),
        },
        "holiday": {
            "enabled": st.session_state.get("fe_holiday_enabled", False),
            "column": st.session_state.get("fe_holiday_column") or None,
            "holidays": aggregated_holidays,
        },
        "cyclical": {
            "enabled": st.session_state.get("fe_cyclical_enabled", False),
            "columns": cyclical_columns,
            "max_values": {col: cyclical_max for col in cyclical_columns},
        },
        "pca": {
            "enabled": st.session_state.get("fe_pca_enabled", False),
            "columns": pca_columns,
            "n_components": requested_components,
            "whiten": st.session_state.get("fe_pca_whiten", False),
        },
        "feature_selection": {
            "enabled": st.session_state.get("fe_feature_selection_enabled", False),
            "method": st.session_state.get("fe_feature_selection_method", "variance"),
            "top_k": selection_top_k,
            "threshold": st.session_state.get("fe_feature_selection_threshold", 0.0),
            "columns": feature_selection_columns,
        },
    }
    return payload


def _prepare_cache_payload(columns: list[str]) -> Dict[str, Any]:
    payload = _build_preprocessing_payload(columns)
    normalized = json.dumps(_normalize_payload(payload), sort_keys=True)
    signature = hashlib.sha256(normalized.encode()).hexdigest()
    dataset_hash = st.session_state.get("dataset_file_hash")
    previous_signature = st.session_state.get("preprocessing_last_signature")
    previous_dataset_hash = st.session_state.get("preprocessing_last_dataset_hash")

    reused = (
        previous_signature
        and previous_dataset_hash
        and previous_signature == signature
        and previous_dataset_hash == dataset_hash
    )
    cache_status = "reused" if reused else "recomputed"

    st.session_state["preprocessing_last_signature"] = signature
    st.session_state["preprocessing_last_dataset_hash"] = dataset_hash
    st.session_state["preprocessing_payload_snapshot"] = payload
    st.session_state["preprocessing_payload_json"] = normalized
    st.session_state["preprocessing_cache_status"] = cache_status
    st.session_state["preprocessing_last_run_time"] = datetime.utcnow().isoformat()

    return {
        "payload": payload,
        "signature": signature,
        "status": cache_status,
        "dataset_hash": dataset_hash,
    }


def _mark_dataset_change(file_bytes: Optional[bytes] = None, dataset_state: Optional[DatasetState] = None) -> None:
    file_hash = _compute_file_hash(file_bytes)
    if not file_hash:
        file_hash = _derive_dataset_hash_from_state(dataset_state)
    st.session_state["dataset_file_hash"] = file_hash
    st.session_state["preprocessing_cache_status"] = "dataset_changed"


def _reset_preprocessing_tracking() -> None:
    st.session_state["dataset_file_hash"] = None
    st.session_state["preprocessing_cache_status"] = "not_run"
    st.session_state["preprocessing_last_signature"] = ""
    st.session_state["preprocessing_last_dataset_hash"] = None
    st.session_state["preprocessing_payload_snapshot"] = None
    st.session_state["preprocessing_payload_json"] = ""
    st.session_state["preprocessing_last_run_time"] = ""


def _reset_validation_state() -> None:
    st.session_state["dataset_validation_result"] = None
    st.session_state["validation_error"] = ""


def _run_validation_flow(state: DatasetState) -> None:
    service = DatasetValidationService()
    try:
        validation_result = service.validate(
            file_name=state.file_name,
            file_bytes=state.file_bytes,
            config=DEFAULT_INGEST_CONFIG,
        )
        st.session_state["dataset_validation_result"] = validation_result
        st.session_state["validation_error"] = ""
    except Exception as exc:
        st.session_state["dataset_validation_result"] = None
        st.session_state["validation_error"] = f"Validation failed: {exc}"


def _refresh_inspection(state: Optional[DatasetState]) -> None:
    if not state:
        st.session_state["dataset_inspection"] = None
        st.session_state["inspection_error"] = ""
        _reset_validation_state()
        return
    try:
        st.session_state["dataset_inspection"] = inspect_dataset(DATASET_ID, config=DEFAULT_INGEST_CONFIG)
        st.session_state["inspection_error"] = ""
    except Exception as exc:
        st.session_state["dataset_inspection"] = None
        st.session_state["inspection_error"] = f"Failed to inspect dataset: {exc}"
        st.session_state["dataset_validation_result"] = None
        st.session_state["validation_error"] = ""


def _clear_role_configuration() -> None:
    for role_def in ROLE_DEFINITIONS:
        key = f"dataset_role_{role_def['role']}"
        if key in st.session_state:
            del st.session_state[key]
    st.session_state["dataset_config"] = {"column_roles": {}, "available_columns": []}


def _update_session_state(state: Optional[DatasetState], reset_roles: bool = False) -> None:
    st.session_state["dataset_state"] = state
    _refresh_inspection(state)
    if reset_roles or state is None:
        _clear_role_configuration()
    _reset_validation_state()


def _ensure_preprocessing_session_state() -> None:
    for section, label in PREPROCESSING_LABEL_DEFAULTS.items():
        label_key = f"preprocessing_label_override_{section}"
        if label_key not in st.session_state:
            st.session_state[label_key] = label
    for section, default_value in PREPROCESSING_DEFAULT_STRATEGIES.items():
        default_key = f"preprocessing_default_{section}"
        if default_key not in st.session_state:
            st.session_state[default_key] = default_value
        selector_key = PREPROCESSING_SESSION_KEYS[section]
        if selector_key not in st.session_state:
            st.session_state[selector_key] = default_value


if "dataset_state" not in st.session_state:
    st.session_state["dataset_state"] = None
    st.session_state["dataset_inspection"] = None
    st.session_state["inspection_error"] = ""
    st.session_state["dataset_config"] = {"column_roles": {}, "available_columns": []}
    st.session_state["dataset_display_name"] = ""
    _reset_preprocessing_tracking()
    _reset_validation_state()

_ensure_preprocessing_session_state()
_ensure_feature_engineering_session_state()

st.title("📊 Dataset Workspace")
dataset_columns_for_payload = []

with st.container():
    st.markdown("### Dataset Management")
    management_columns = st.columns([1, 1])
    with management_columns[0]:
        st.subheader("Upload & Repository")
        uploaded_file = st.file_uploader(
            label="Upload CSV or Excel",
            type=["csv", "xls", "xlsx", "xlsm"],
            key="dataset_upload",
        )
        dataset_name_input = st.text_input(
            "Dataset display name",
            value="workspace_dataset",
            key="dataset_name",
        )
        ingestion_col1, ingestion_col2 = st.columns(2)
        with ingestion_col1:
            if st.button("Upload dataset"):
                if not uploaded_file:
                    st.warning("Select a file before uploading.")
                else:
                    file_name = uploaded_file.name
                    file_bytes = uploaded_file.getvalue()
                    try:
                        state = ingest_dataset(
                            dataset_id=DATASET_ID,
                            name=dataset_name_input,
                            file_name=file_name,
                            file_bytes=file_bytes,
                            config=DEFAULT_INGEST_CONFIG,
                        )
                        _update_session_state(state, reset_roles=True)
                        st.session_state["dataset_display_name"] = dataset_name_input
                        _mark_dataset_change(file_bytes=file_bytes, dataset_state=state)
                        st.success("Dataset uploaded successfully.")
                    except Exception as exc:
                        st.error(f"Failed to ingest dataset: {exc}")
        with ingestion_col2:
            if st.button("Replace dataset"):
                if not uploaded_file:
                    st.warning("Upload a file before replacing the dataset.")
                elif not st.session_state["dataset_state"]:
                    st.warning("No dataset exists yet. Use Upload instead.")
                else:
                    try:
                        file_bytes = uploaded_file.getvalue()
                        state = replace_dataset(
                            dataset_id=DATASET_ID,
                            file_name=uploaded_file.name,
                            file_bytes=file_bytes,
                            config=DEFAULT_INGEST_CONFIG,
                        )
                        _update_session_state(state, reset_roles=True)
                        st.session_state["dataset_display_name"] = dataset_name_input
                        _mark_dataset_change(file_bytes=file_bytes, dataset_state=state)
                        st.success("Dataset replaced successfully.")
                    except Exception as exc:
                        st.error(f"Failed to replace dataset: {exc}")
    with management_columns[1]:
        st.subheader("Lifecycle Controls")
        lifecycle_col1, lifecycle_col2 = st.columns(2)
        with lifecycle_col1:
            if st.button("Reload dataset"):
                if not st.session_state["dataset_state"]:
                    st.warning("No dataset loaded yet.")
                else:
                    try:
                        state = reload_dataset(
                            dataset_id=DATASET_ID, config=DEFAULT_INGEST_CONFIG
                        )
                        _update_session_state(state)
                        _mark_dataset_change(state=state)
                        st.success("Dataset reloaded with latest configuration.")
                    except Exception as exc:
                        st.error(f"Reload failed: {exc}")
        with lifecycle_col2:
            if st.button("Remove dataset"):
                if not st.session_state["dataset_state"]:
                    st.warning("No dataset loaded yet.")
                else:
                    try:
                        remove_dataset(dataset_id=DATASET_ID)
                        _update_session_state(None, reset_roles=True)
                        _reset_preprocessing_tracking()
                        st.info("Dataset removed from memory.")
                    except Exception as exc:
                        st.error(f"Failed to remove dataset: {exc}")
        sample_choice = st.selectbox(
            "Select a sample dataset",
            options=list(SAMPLE_DATASETS.keys()),
            key="preferred_sample_dataset",
        )
        if st.button("Load sample dataset"):
            generator = SAMPLE_DATASETS.get(sample_choice)
            if generator:
                sample_df = generator()
                sample_bytes = _serialize_dataframe_to_bytes(sample_df, "sample_dataset.csv")
                try:
                    state = ingest_dataset(
                        dataset_id=DATASET_ID,
                        name=f"Sample - {sample_choice}",
                        file_name="sample_dataset.csv",
                        file_bytes=sample_bytes,
                        config=DEFAULT_INGEST_CONFIG,
                    )
                    _update_session_state(state, reset_roles=True)
                    st.session_state["dataset_display_name"] = f"Sample - {sample_choice}"
                    _mark_dataset_change(file_bytes=sample_bytes, dataset_state=state)
                    st.success("Sample dataset loaded for quick experimentation.")
                except Exception as exc:
                    st.error(f"Sample dataset ingestion failed: {exc}")
            else:
                st.warning("Selected sample dataset generator is unavailable.")

st.markdown("---")

state = st.session_state["dataset_state"]
inspection = st.session_state["dataset_inspection"]
inspection_error = st.session_state["inspection_error"]
validation_result = st.session_state.get("dataset_validation_result")
validation_error = st.session_state.get("validation_error", "")

dataset_columns = _extract_dataframe_columns(state)

with st.container():
    st.markdown("### Dataset validation")
    st.caption("Trigger validation to surface preview, schema, quality metrics, and actionable scores before modeling.")
    validation_col1, validation_col2 = st.columns([2, 1])
    with validation_col1:
        if st.button("Validate Dataset", disabled=not state):
            if not state:
                st.warning("Load or upload a dataset before validating.")
            else:
                _run_validation_flow(state)
                if st.session_state.get("dataset_validation_result"):
                    st.success("Dataset validation completed and cached for this run.")
    with validation_col2:
        if validation_error:
            st.error(validation_error)
        elif validation_result:
            st.success("Latest validation run available below.")
        else:
            st.info("Run validation to inspect dataset health indicators.")

    if validation_result:
        st.markdown("#### Preview")
        preview_columns = validation_result.preview.get("columns", [])
        preview_records = validation_result.preview.get("records", [])
        if preview_records:
            preview_df = pd.DataFrame(preview_records)
            st.dataframe(preview_df[preview_columns], use_container_width=True)
        else:
            st.info("No rows available in the preview.")

        st.markdown("#### Schema & Dtypes")
        schema_df = pd.DataFrame(validation_result.schema)
        st.dataframe(schema_df, use_container_width=True)
        dtype_summary = validation_result.dtype_summary
        st.markdown("#### Dimension & Memory Signature")
        metrics_row = st.columns(4)
        metrics_row[0].metric("Rows", validation_result.dimensions.get("rows", 0))
        metrics_row[1].metric("Columns", validation_result.dimensions.get("columns", 0))
        metrics_row[2].metric("Memory (MB)", f"{validation_result.memory_usage_bytes / (1024**2):.2f}")
        metrics_row[3].metric("Quality score", f"{validation_result.quality_score * 100:.1f}%")

        st.markdown("#### Type distribution")
        dtype_items = [{"dtype": key, "count": value} for key, value in dtype_summary.items()]
        dtype_df = pd.DataFrame(dtype_items)
        st.bar_chart(dtype_df.set_index("dtype"))

        st.markdown("#### Missing & Duplicate insights")
        insights_col1, insights_col2 = st.columns(2)
        total_missing = validation_result.total_missing_values
        duplicate_count = validation_result.duplicate_row_count
        insights_col1.metric("Total missing values", total_missing)
        insights_col1.dataframe(
            pd.DataFrame.from_dict(validation_result.missing_value_counts, orient="index", columns=["missing"])
            .sort_values("missing", ascending=False)
            .head(10),
            use_container_width=True,
        )
        insights_col2.metric("Duplicate rows", duplicate_count)
        quality_inputs = validation_result.quality_score_inputs
        insights_col2.write(
            {
                "missing ratio": f"{quality_inputs.missing_ratio:.3f}",
                "duplicate ratio": f"{quality_inputs.duplicate_ratio:.3f}",
                "rows": quality_inputs.rows,
                "columns": quality_inputs.columns,
            }
        )

        metadata = validation_result.metadata
        with st.expander("Validation metadata", expanded=False):
            st.write({
                "payload version": metadata.payload_version,
                "file hash": metadata.file_hash,
                "config signature": metadata.config_signature,
                "cache key": metadata.cache_key,
                "generated at": metadata.generated_at,
            })
    elif inspection_error:
        st.warning("Dataset inspection is lagging due to the following error. Load or validate a dataset once the issue is resolved.")

st.markdown("---")

with st.container():
    st.markdown("### Preprocessing configuration panel")
    st.caption("Drive missing value, outlier, encoding, and scaling choices directly from the browser without touching backend code.")

    with st.expander("Customize panel labels & defaults", expanded=False):
        st.write("Use these controls to rename sections or change the defaults that feed into the preprocessing panel on every rerun.")
        for section_key in PREPROCESSING_SESSION_KEYS:
            label_key = f"preprocessing_label_override_{section_key}"
            default_key = f"preprocessing_default_{section_key}"
            label_override = st.session_state.get(label_key, PREPROCESSING_LABEL_DEFAULTS[section_key])
            default_strategy = st.session_state.get(default_key, PREPROCESSING_DEFAULT_STRATEGIES[section_key])

            label_col, default_col = st.columns([3, 2])
            with label_col:
                st.text_input(
                    f"{section_key.title()} label",
                    value=label_override,
                    key=label_key,
                    help="Rename the section label that appears in the preprocessing payload.",
                )
            with default_col:
                st.selectbox(
                    "Default strategy",
                    options=PREPROCESSING_OPTIONS.get(section_key, []),
                    index=PREPROCESSING_OPTIONS.get(section_key, []).index(default_strategy)
                    if default_strategy in PREPROCESSING_OPTIONS.get(section_key, [])
                    else 0,
                    key=default_key,
                    help="Select the strategy that populates this step unless overridden in the panel.",
                )

    st.markdown("---")
    st.markdown("#### Preprocessing run & cache insights")
    st.caption("All preprocessing decisions flow through the payload below and can be hashed for caching without needing to restart the application.")
    cache_status = st.session_state.get("preprocessing_cache_status", "not_run")
    dataset_hash = st.session_state.get("dataset_file_hash")
    last_signature = st.session_state.get("preprocessing_last_signature", "")
    last_run_time = st.session_state.get("preprocessing_last_run_time", "")

    status_col1, status_col2, status_col3 = st.columns([2, 1, 1])
    status_message = CACHE_STATUS_COPY.get(cache_status, "Cache status pending.")
    emoji = {
        "not_run": "⚪️",
        "dataset_changed": "⚠️",
        "reused": "✅",
        "recomputed": "🔄",
    }.get(cache_status, "ℹ️")
    with status_col1:
        st.markdown(f"**{emoji} {status_message}**")
        if not dataset_columns:
            st.caption("Dataset not available yet. Upload, reload, or sample to unlock preprocessing runs.")
        else:
            st.caption("Use the button below to rerun preprocessing once the configuration looks right.")
    with status_col2:
        st.markdown("**Dataset fingerprint**")
        st.code(dataset_hash or "No dataset fingerprinting information yet")
    with status_col3:
        st.markdown("**Last preprocessing run**")
        st.write(last_run_time or "Not run yet")

    run_disabled = not dataset_columns
    if st.button("Run preprocessing with current configuration", disabled=run_disabled, key="preprocessing_run"):
        run_result = _prepare_cache_payload(dataset_columns)
        if run_result["status"] == "reused":
            st.success("Cached preprocessing results reused — no recomputation required.")
        else:
            st.info("Preprocessing recomputed; cache context has been refreshed.")

    payload_snapshot = st.session_state.get("preprocessing_payload_snapshot")
    payload_json = st.session_state.get("preprocessing_payload_json")
    if payload_snapshot and payload_json:
        with st.expander("Payload sent to backend cache layer", expanded=False):
            st.json(payload_snapshot)
            st.caption("This JSON blob is hashed on the backend to key preprocessing cache entries.")
            st.code(payload_json)
    else:
        st.info("Preprocessing payload will appear here once you run a cache-aware preprocessing job.")

    st.markdown("---")
    st.markdown("#### Feature Engineering Controls")
    st.caption("Toggle and parameterize the feature engineering steps that feed into the backend configuration object.")
    with st.expander("Configure feature engineering", expanded=True):
        st.checkbox("Enable feature engineering", key="feature_engineering_enabled")
        st.markdown("---")
        st.markdown("##### Polynomial Features")
        st.checkbox("Apply polynomial feature expansion", key="fe_polynomial_enabled")
        if st.session_state.get("fe_polynomial_enabled"):
            st.multiselect(
                "Select columns for polynomial features",
                options=dataset_columns,
                key="fe_polynomial_columns",
                label_visibility="collapsed",
            )
            st.number_input(
                "Degree",
                min_value=2,
                max_value=10,
                step=1,
                key="fe_polynomial_degree",
            )
            st.checkbox("Include bias term", key="fe_polynomial_include_bias")
            st.checkbox("Only interactions (exclude duplicate powers)", key="fe_polynomial_interaction_only")
        st.markdown("##### Interaction Features")
        st.checkbox("Create explicit interactions", key="fe_interaction_enabled")
        if st.session_state.get("fe_interaction_enabled"):
            st.multiselect(
                "Select columns for interaction features",
                options=dataset_columns,
                key="fe_interaction_columns",
                label_visibility="collapsed",
            )
        st.markdown("##### Lag Features")
        st.checkbox("Generate lagged features", key="fe_lag_enabled")
        if st.session_state.get("fe_lag_enabled"):
            st.multiselect(
                "Select columns to lag",
                options=dataset_columns,
                key="fe_lag_columns",
                label_visibility="collapsed",
            )
            st.number_input(
                "Max lag (periods)",
                min_value=1,
                max_value=52,
                step=1,
                key="fe_lag_max_lag",
            )
            st.text_input(
                "Fill value for missing lagged rows",
                key="fe_lag_fill_value",
                placeholder="e.g., 0 or mean",
            )
        st.markdown("##### Rolling Statistics")
        st.checkbox("Compute rolling statistics", key="fe_rolling_enabled")
        if st.session_state.get("fe_rolling_enabled"):
            st.multiselect(
                "Select columns for rolling stats",
                options=dataset_columns,
                key="fe_rolling_columns",
                label_visibility="collapsed",
            )
            st.text_input(
                "Rolling window sizes (comma-separated)",
                key="fe_rolling_windows",
                placeholder="3,7,14",
            )
            st.multiselect(
                "Statistics to compute",
                options=ROLLING_STATS_OPTIONS,
                key="fe_rolling_stats",
                default=st.session_state.get("fe_rolling_stats", ["mean"]),
            )
        st.markdown("##### Date Features")
        st.checkbox("Derive date/time features", key="fe_date_enabled")
        if st.session_state.get("fe_date_enabled"):
            st.selectbox(
                "Date/time column",
                options=["", *dataset_columns],
                key="fe_date_column",
                format_func=lambda val: val if val else "Select a column",
            )
            st.multiselect(
                "Extracted date features",
                options=DATE_FEATURE_OPTIONS,
                key="fe_date_features",
                default=st.session_state.get("fe_date_features", ["year", "month", "day"]),
            )
            st.checkbox("Drop original date column after extraction", key="fe_date_drop_original")
        st.markdown("##### Holiday Features")
        st.checkbox("Add holiday calendars", key="fe_holiday_enabled")
        if st.session_state.get("fe_holiday_enabled"):
            st.selectbox(
                "Holiday reference column",
                options=["", *dataset_columns],
                key="fe_holiday_column",
                format_func=lambda val: val if val else "Select a column",
            )
            st.multiselect(
                "Holiday calendars in use",
                options=HOLIDAY_CALENDARS,
                key="fe_holiday_calendars",
                default=st.session_state.get("fe_holiday_calendars", []),
            )
            st.text_input(
                "Custom holidays (comma-separated)",
                key="fe_holiday_custom",
                placeholder="e.g., Independence Day, Labor Day",
            )
        st.markdown("##### Cyclical Encoding")
        st.checkbox("Encode cyclical columns", key="fe_cyclical_enabled")
        if st.session_state.get("fe_cyclical_enabled"):
            st.multiselect(
                "Columns for cyclical encoding",
                options=dataset_columns,
                key="fe_cyclical_columns",
                label_visibility="collapsed",
            )
            st.number_input(
                "Max value range",
                min_value=1,
                max_value=1000,
                step=1,
                key="fe_cyclical_max_value",
            )
        st.markdown("##### PCA")
        st.checkbox("Apply PCA", key="fe_pca_enabled")
        if st.session_state.get("fe_pca_enabled"):
            st.multiselect(
                "Columns entering PCA",
                options=dataset_columns,
                key="fe_pca_columns",
                label_visibility="collapsed",
            )
            max_pca_components = max(len(st.session_state.get("fe_pca_columns", []) or []), 1)
            st.number_input(
                "Number of components",
                min_value=1,
                max_value=max_pca_components,
                step=1,
                key="fe_pca_n_components",
            )
            st.checkbox("Whiten components", key="fe_pca_whiten")
        st.markdown("##### Feature Selection")
        st.checkbox("Enable feature selection", key="fe_feature_selection_enabled")
        if st.session_state.get("fe_feature_selection_enabled"):
            st.multiselect(
                "Columns to evaluate",
                options=dataset_columns,
                key="fe_feature_selection_columns",
                label_visibility="collapsed",
            )
            st.selectbox(
                "Selection method",
                options=FEATURE_SELECTION_METHODS,
                key="fe_feature_selection_method",
            )
            st.number_input(
                "Top K features (0 = unset)",
                min_value=0,
                max_value=len(dataset_columns) if dataset_columns else 10,
                step=1,
                key="fe_feature_selection_top_k",
            )
            st.number_input(
                "Threshold (0.0 if unset)",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                format="%.2f",
                key="fe_feature_selection_threshold",
            )

analysis_section = st.container()
with analysis_section:
    st.markdown("---")
    st.markdown("### Future analysis controls")
    st.caption("This area will separate dataset management from later modeling, forecasting, or reporting controls.")
