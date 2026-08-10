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


def _refresh_inspection(state: Optional[DatasetState]) -> None:
    if not state:
        st.session_state["dataset_inspection"] = None
        st.session_state["inspection_error"] = ""
        return
    try:
        st.session_state["dataset_inspection"] = inspect_dataset(DATASET_ID, config=DEFAULT_INGEST_CONFIG)
        st.session_state["inspection_error"] = ""
    except Exception as exc:
        st.session_state["dataset_inspection"] = None
        st.session_state["inspection_error"] = f"Failed to inspect dataset: {exc}"


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

_ensure_preprocessing_session_state()

st.title("📊 Dataset Workspace")

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

st.markdown("---")

with st.container():
    st.markdown("### Preprocessing configuration panel")
    st.caption("Drive missing value, outlier, encoding, and scaling choices directly from the browser without touching backend code.")

    with st.expander("Customize panel labels & defaults", expanded=False):
        st.write("Use these controls to rename sections or change the defaults that feed into the preprocessing panel on every rerun.")
        label_cols = st.columns(2)
        for idx, key in enumerate(PREPROCESSING_SESSION_KEYS):
            pass

    st.markdown("---")
    st.markdown("#### Preprocessing run & cache insights")
    cache_status = st.session_state.get("preprocessing_cache_status", "not_run")
    dataset_columns = _extract_dataframe_columns(state)
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

analysis_section = st.container()
with analysis_section:
    st.markdown("---")
    st.markdown("### Future analysis controls")
    st.caption("This area will separate dataset management from later modeling, forecasting, or reporting controls.")
