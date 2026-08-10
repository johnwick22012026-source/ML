import io
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

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


def _compute_quality_score(df: pd.DataFrame) -> float:
    total_cells = df.size or 1
    missing_cells = int(df.isna().sum().sum())
    completeness_ratio = (total_cells - missing_cells) / total_cells
    dup_rows = int(df.duplicated().sum())
    uniqueness_ratio = 1.0 - (dup_rows / (len(df) or 1))
    score = (completeness_ratio * 0.6 + uniqueness_ratio * 0.4) * 100
    return max(0.0, min(100.0, round(score, 2)))


def _summarize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    dims = (len(df), len(df.columns))
    memory_usage_bytes = int(df.memory_usage(deep=True).sum())
    missing = df.isna().sum()
    duplicates = int(df.duplicated().sum())
    quality_score = _compute_quality_score(df)
    dtype_summary = (
        df.dtypes.reset_index()
        .rename(columns={"index": "column", 0: "dtype"})
        .groupby("dtype")
        .size()
        .reset_index(name="count")
    )
    schema = df.dtypes.reset_index().rename(columns={"index": "column", 0: "dtype"})
    return {
        "dimensions": dims,
        "memory": memory_usage_bytes,
        "missing_values": missing.to_dict(),
        "duplicate_rows": duplicates,
        "quality_score": quality_score,
        "dtype_summary": dtype_summary,
        "schema": schema,
    }


def _update_session_state(state: Optional[DatasetState]) -> None:
    st.session_state["dataset_state"] = state


if "dataset_state" not in st.session_state:
    st.session_state["dataset_state"] = None
    st.session_state["status_message"] = ""

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

# Make dataset state and dataframe available for both panels
state = st.session_state["dataset_state"]
df = state.dataframe if state else None

inspection_columns = st.columns([2, 1])
with inspection_columns[0]:
    st.markdown("### Dataset Preview & Insights")
    if not state:
        st.info("No dataset is available yet. Upload, replace, or load a sample dataset to continue.")
    else:
        st.dataframe(df.head(10), use_container_width=True)
        summary = _summarize_dataframe(df)
        dims = summary["dimensions"]
        st.markdown(f"**Name:** {state.name}")
        st.markdown(f"**Dimensions:** {dims[0]:,} rows × {dims[1]:,} columns")
        dims_col1, dims_col2, dims_col3 = st.columns(3)
        with dims_col1:
            st.metric("Duplicate rows", f"{summary['duplicate_rows']}")
        with dims_col2:
            st.metric("Memory usage", f"{summary['memory']:,} bytes")
        with dims_col3:
            st.metric("Quality score", f"{summary['quality_score']} / 100")
        st.markdown("#### Schema")
        st.table(summary["schema"])
        st.markdown("#### Datatype summary")
        st.table(summary["dtype_summary"])
        st.markdown("#### Missing values")
        missing_df = pd.DataFrame(
            list(summary["missing_values"].items()),
            columns=["column", "missing_count"],
        )
        st.table(missing_df)
with inspection_columns[1]:
    st.markdown("### Sampling & Quick Stats")
    if state:
        sample_size = st.slider("Sample rows", min_value=5, max_value=100, value=10)
        if st.button("Generate random preview sample"):
            try:
                sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
                st.write(sample_df)
            except ValueError as exc:
                st.warning(f"Unable to sample dataset: {exc}")
    else:
        st.info("Sampling will become available once a dataset is loaded.")

analysis_section = st.container()
with analysis_section:
    st.markdown("---")
    st.markdown("### Future analysis controls")
    st.caption("This area will separate dataset management from later modeling, forecasting, or reporting controls.")
