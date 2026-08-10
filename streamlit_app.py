import io
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


def _ensure_dataset_config_columns(columns: list[str]) -> None:
    snapshot = st.session_state.get("dataset_config", {})
    snapshot["available_columns"] = columns
    column_roles = snapshot.get("column_roles", {})
    filtered_roles = {
        role: [col for col in selections if col in columns]
        for role, selections in column_roles.items()
        if selections
    }
    snapshot["column_roles"] = filtered_roles
    st.session_state["dataset_config"] = snapshot


def _render_inspection(inspect_result: DatasetInspectionResult) -> None:
    st.subheader("Dataset Preview & Inspection")
    preview_df = pd.DataFrame(inspect_result.preview.get("records", []))
    if not preview_df.empty:
        st.dataframe(preview_df, use_container_width=True)
    else:
        st.info("No preview records available")

    dims = f"{inspect_result.row_count:,} rows × {inspect_result.column_count:,} columns"
    st.markdown(f"**Dimensions:** {dims}")
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    metrics_col1.metric("Memory usage", f"{inspect_result.memory_usage_bytes:,} bytes")
    metrics_col2.metric("Duplicate rows", f"{inspect_result.duplicate_row_count:,}")
    metrics_col3.metric("Quality score", f"{inspect_result.quality_score * 100:.2f} / 100")

    st.markdown("#### Schema")
    st.table(pd.DataFrame(inspect_result.schema))

    st.markdown("#### Datatype summary")
    dtype_df = (
        pd.DataFrame(
            inspect_result.dtype_summary.items(), columns=["dtype", "count"]
        )
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    st.table(dtype_df)

    st.markdown("#### Missing values")
    missing_df = pd.DataFrame(
        list(inspect_result.missing_value_counts.items()),
        columns=["column", "missing_count"],
    ).sort_values("missing_count", ascending=False)
    st.table(missing_df)

    st.markdown("#### Quality score breakdown")
    inputs = inspect_result.quality_score_inputs
    breakdown_df = pd.DataFrame(
        {
            "metric": [
                "Missing ratio",
                "Duplicate ratio",
                "Rows",
                "Columns",
                "Missing values",
                "Duplicate rows",
            ],
            "value": [
                f"{inputs.missing_ratio:.4f}",
                f"{inputs.duplicate_ratio:.4f}",
                inputs.rows,
                inputs.columns,
                inputs.missing_values,
                inputs.duplicate_rows,
            ],
        }
    )
    st.table(breakdown_df)


def _render_role_configuration(columns: list[str]) -> None:
    if not columns:
        st.warning("No columns available for role configuration.")
        return

    st.markdown("### Column role configuration")
    st.caption("Define how the dataset columns should be interpreted downstream without touching backend code.")
    role_layouts = st.columns(2)

    column_roles: Dict[str, list[str]] = {}
    for idx, role_def in enumerate(ROLE_DEFINITIONS):
        column_slot = role_layouts[idx % 2]
        with column_slot:
            role = role_def["role"]
            label = role_def["label"]
            key = f"dataset_role_{role}"
            if role_def["multi"]:
                previous = st.session_state.get(key, [])
                default = [col for col in previous if col in columns]
                selection = st.multiselect(
                    label,
                    options=columns,
                    default=default,
                    key=key,
                )
                if selection:
                    column_roles[role] = selection
            else:
                options = [""] + columns
                prev_value = st.session_state.get(key, "")
                if prev_value in options:
                    default_index = options.index(prev_value)
                else:
                    default_index = 0
                value = st.selectbox(
                    label,
                    options,
                    index=default_index,
                    key=key,
                )
                if value:
                    column_roles[role] = [value]

    st.session_state["dataset_config"] = {
        "column_roles": column_roles,
        "available_columns": columns,
    }
    st.markdown("#### Current configuration")
    st.json(st.session_state["dataset_config"], expanded=False)

PREPROCESSING_SESSION_KEYS = {
    "missing": "preprocessing_missing_strategy",
    "outlier": "preprocessing_outlier_method",
    "encoding": "preprocessing_encoding_strategy",
    "scaling": "preprocessing_scaling_strategy",
}


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


def _build_missing_config(selection: str) -> Dict[str, Any]:
    mapping: Dict[str, Any] = {}
    if selection == "drop rows":
        mapping["strategy"] = "drop"
        mapping["how"] = "any"
    elif selection == "drop columns":
        mapping["strategy"] = "drop"
        mapping["how"] = "any"
        mapping["axis"] = 1
    elif selection == "mean":
        mapping["strategy"] = "mean"
    elif selection == "median":
        mapping["strategy"] = "median"
    elif selection == "mode":
        mapping["strategy"] = "most_frequent"
    elif selection == "constant":
        mapping["strategy"] = "constant"
    elif selection == "KNN Imputer":
        mapping["strategy"] = "knn"
    elif selection == "Iterative Imputer":
        mapping["strategy"] = "iterative"
    elif selection == "forward fill":
        mapping["strategy"] = "ffill"
    elif selection == "backward fill":
        mapping["strategy"] = "bfill"
    return mapping


def _build_outlier_config(selection: str) -> Dict[str, Any]:
    method_map = {
        "none": "none",
        "IQR": "iqr",
        "Z-score": "zscore",
        "Isolation Forest": "isolation_forest",
        "Local Outlier Factor": "local_outlier_factor",
        "Winsorization": "winsorization",
    }
    return {"method": method_map.get(selection, "none")}


def _build_encoding_config(selection: str, columns: list[str]) -> Dict[str, Any]:
    config: Dict[str, Any] = {"strategy": selection if selection != "none" else "none"}
    if columns:
        config["columns"] = columns
    return config


def _build_scaling_config(selection: str, columns: list[str]) -> Dict[str, Any]:
    config: Dict[str, Any] = {"strategy": selection if selection != "none" else "none"}
    if columns:
        config["columns"] = columns
    return config


if "dataset_state" not in st.session_state:
    st.session_state["dataset_state"] = None
    st.session_state["dataset_inspection"] = None
    st.session_state["inspection_error"] = ""
    st.session_state["dataset_config"] = {"column_roles": {}, "available_columns": []}

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
                        state = replace_dataset(
                            dataset_id=DATASET_ID,
                            file_name=uploaded_file.name,
                            file_bytes=uploaded_file.getvalue(),
                            config=DEFAULT_INGEST_CONFIG,
                        )
                        _update_session_state(state, reset_roles=True)
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
                    st.success("Sample dataset loaded for quick experimentation.")
                except Exception as exc:
                    st.error(f"Sample dataset ingestion failed: {exc}")
            else:
                st.warning("Selected sample dataset generator is unavailable.")

st.markdown("---")

state = st.session_state["dataset_state"]
inspection = st.session_state["dataset_inspection"]
inspection_error = st.session_state["inspection_error"]

two_column_layout = st.columns([3, 1])
with two_column_layout[0]:
    if not state:
        st.info("No dataset is available yet. Upload, replace, or load a sample dataset to continue.")
    elif inspection_error:
        st.error(inspection_error)
    elif inspection:
        _render_inspection(inspection)
    else:
        st.warning("Dataset is loaded but inspection is pending.")

with two_column_layout[1]:
    st.subheader("Quick Sampling & Stats")
    if state and inspection:
        sample_size = st.slider("Sample rows", min_value=5, max_value=100, value=10)
        if st.button("Generate random preview sample"):
            try:
                preview_df = pd.DataFrame(inspection.preview.get("records", []))
                if preview_df.empty:
                    st.warning("No data available for sampling yet.")
                else:
                    sampled = preview_df.sample(n=min(sample_size, len(preview_df)), replace=False, random_state=42)
                    st.dataframe(sampled)
            except ValueError as exc:
                st.warning(f"Unable to sample dataset: {exc}")
    else:
        st.info("Sampling will become available once a dataset is inspected.")

if state and inspection:
    columns = state.dataframe.columns.tolist()
    _ensure_dataset_config_columns(columns)
    _render_role_configuration(columns)
else:
    st.markdown("---")
    st.markdown("### Dataset role configuration")
    st.info("Configure column roles once a dataset is available.")

preprocessing_section = st.container()
with preprocessing_section:
    st.markdown("---")
    st.markdown("### Preprocessing configuration panel")
    st.caption("Drive missing value, outlier, encoding, and scaling choices directly from the browser without touching backend code.")

    with st.expander("Customize panel labels & defaults", expanded=False):
        st.write("Use these controls to rename sections or change the defaults that feed into the preprocessing panel on every rerun.")
        label_cols = st.columns(2)
        for idx, key in enumerate(PREPROCESSI... (truncated for brevity)

analysis_section = st.container()
with analysis_section:
    st.markdown("---")
    st.markdown("### Future analysis controls")
    st.caption("This area will separate dataset management from later modeling, forecasting, or reporting controls.")
