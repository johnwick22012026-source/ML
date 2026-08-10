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


def _serialize_dataframe_to_bytes(df: pd.DataFrame, file_name: str) -> bytes:
    if file_name.lower().endswith(".xlsx"):
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        return buffer.getvalue()
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


def _generate_sample_frame() -> pd.DataFrame:
    dates = pd.date_range(end=pd.Timestamp.today(), periods=30, freq="D")
    values = np.sin(np.linspace(0, np.pi * 3, len(dates))) * 10 + np.linspace(0, 5, len(dates))
    categories = ["Baseline" if i % 3 == 0 else "Variant" if i % 3 == 1 else "Control" for i in range(len(dates))]
    return pd.DataFrame({"measurement_date": dates, "value": values.round(2), "category": categories})


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


def _update_session_state(state: Optional[DatasetState]) -> None:
    st.session_state["dataset_state"] = state
    _refresh_inspection(state)


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


if "dataset_state" not in st.session_state:
    st.session_state["dataset_state"] = None
    st.session_state["dataset_inspection"] = None
    st.session_state["inspection_error"] = ""

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
                        _update_session_state(state)
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
                        _update_session_state(state)
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
                remove_dataset(dataset_id=DATASET_ID)
                _update_session_state(None)
                st.info("Dataset removed from memory.")
        if st.button("Load sample dataset"):
            sample_df = _generate_sample_frame()
            sample_bytes = _serialize_dataframe_to_bytes(sample_df, "sample_dataset.csv")
            try:
                state = ingest_dataset(
                    dataset_id=DATASET_ID,
                    name="Sample experimental dataset",
                    file_name="sample_dataset.csv",
                    file_bytes=sample_bytes,
                    config=DEFAULT_INGEST_CONFIG,
                )
                _update_session_state(state)
                st.success("Sample dataset loaded for quick experimentation.")
            except Exception as exc:
                st.error(f"Sample dataset ingestion failed: {exc}")

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

analysis_section = st.container()
with analysis_section:
    st.markdown("---")
    st.markdown("### Future analysis controls")
    st.caption("This area will separate dataset management from later modeling, forecasting, or reporting controls.")
