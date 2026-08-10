import hashlib
import io
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backend.eda import EDAService
from backend.ml import ModelRegistry
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

SUPPORTED_MODEL_TASKS = ["regression", "classification"]

MODEL_HYPERPARAMETER_TEMPLATES: Dict[str, List[Dict[str, Any]]] = {
    "linear_regression": [
        {"name": "fit_intercept", "type": "bool", "default": True},
    ],
    "random_forest_regressor": [
        {"name": "n_estimators", "type": "int", "default": 100, "min": 10, "max": 1000, "step": 10},
        {"name": "max_depth", "type": "int", "default": 5, "min": 1, "max": 50, "step": 1},
        {"name": "random_state", "type": "int", "default": 42, "min": 0, "max": 9999, "step": 1},
    ],
    "svr": [
        {"name": "kernel", "type": "select", "default": "rbf", "options": ["rbf", "linear", "poly", "sigmoid"]},
        {"name": "C", "type": "float", "default": 1.0, "min": 0.01, "max": 100.0, "step": 0.1},
        {"name": "epsilon", "type": "float", "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01},
    ],
    "logistic_regression": [
        {"name": "C", "type": "float", "default": 1.0, "min": 0.01, "max": 100.0, "step": 0.1},
        {"name": "max_iter", "type": "int", "default": 100, "min": 50, "max": 1000, "step": 50},
        {"name": "solver", "type": "select", "default": "lbfgs", "options": ["lbfgs", "liblinear", "sag"]},
    ],
    "random_forest_classifier": [
        {"name": "n_estimators", "type": "int", "default": 100, "min": 10, "max": 1000, "step": 10},
        {"name": "max_depth", "type": "int", "default": 5, "min": 1, "max": 50, "step": 1},
        {"name": "random_state", "type": "int", "default": 42, "min": 0, "max": 9999, "step": 1},
    ],
    "svc": [
        {"name": "kernel", "type": "select", "default": "rbf", "options": ["rbf", "linear", "poly", "sigmoid"]},
        {"name": "C", "type": "float", "default": 1.0, "min": 0.01, "max": 100.0, "step": 0.1},
        {"name": "gamma", "type": "select", "default": "scale", "options": ["scale", "auto"]},
    ],
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
        "models": _collect_selected_model_configs(),
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


def _build_feature_engineering_payload(columns: list[str]) -> Dict[str, Any]:
    _ensure_feature_engineering_session_state()
    payload: Dict[str, Any] = {
        "columns": columns,
        "feature_engineering_enabled": st.session_state["feature_engineering_enabled"],
        "polynomial": {
            "enabled": st.session_state["fe_polynomial_enabled"],
            "columns": st.session_state["fe_polynomial_columns"],
            "degree": st.session_state["fe_polynomial_degree"],
            "include_bias": st.session_state["fe_polynomial_include_bias"],
            "interaction_only": st.session_state["fe_polynomial_interaction_only"],
        },
        "interaction": {
            "enabled": st.session_state["fe_interaction_enabled"],
            "columns": st.session_state["fe_interaction_columns"],
        },
        "lags": {
            "enabled": st.session_state["fe_lag_enabled"],
            "columns": st.session_state["fe_lag_columns"],
            "max_lag": st.session_state["fe_lag_max_lag"],
            "fill_value": st.session_state["fe_lag_fill_value"],
        },
        "rolling": {
            "enabled": st.session_state["fe_rolling_enabled"],
            "columns": st.session_state["fe_rolling_columns"],
            "windows": _parse_int_list(st.session_state.get("fe_rolling_windows", "")),
            "stats": st.session_state["fe_rolling_stats"],
        },
        "date": {
            "enabled": st.session_state["fe_date_enabled"],
            "column": st.session_state["fe_date_column"],
            "features": st.session_state["fe_date_features"],
            "drop_original": st.session_state["fe_date_drop_original"],
        },
        "holiday": {
            "enabled": st.session_state["fe_holiday_enabled"],
            "column": st.session_state["fe_holiday_column"],
            "calendars": st.session_state["fe_holiday_calendars"],
            "custom": st.session_state["fe_holiday_custom"],
        },
        "cyclical": {
            "enabled": st.session_state["fe_cyclical_enabled"],
            "columns": st.session_state["fe_cyclical_columns"],
            "max_value": st.session_state["fe_cyclical_max_value"],
        },
        "pca": {
            "enabled": st.session_state["fe_pca_enabled"],
            "columns": st.session_state["fe_pca_columns"],
            "n_components": st.session_state["fe_pca_n_components"],
            "whiten": st.session_state["fe_pca_whiten"],
        },
        "feature_selection": {
            "enabled": st.session_state["fe_feature_selection_enabled"],
            "columns": st.session_state["fe_feature_selection_columns"],
            "method": st.session_state["fe_feature_selection_method"],
            "top_k": st.session_state["fe_feature_selection_top_k"],
            "threshold": st.session_state["fe_feature_selection_threshold"],
        },
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


def _ensure_model_selection_session_state() -> None:
    if "active_task_type" not in st.session_state:
        st.session_state["active_task_type"] = SUPPORTED_MODEL_TASKS[0]
    for task in SUPPORTED_MODEL_TASKS:
        key = f"selected_models_{task}"
        if key not in st.session_state:
            options = ModelRegistry.available_models(task)
            st.session_state[key] = options[:1] if options else []


def _collect_selected_model_configs() -> List[Dict[str, Any]]:
    task_type = st.session_state.get("active_task_type", SUPPORTED_MODEL_TASKS[0])
    selection_key = f"selected_models_{task_type}"
    selected_models = st.session_state.get(selection_key, [])
    model_configs: List[Dict[str, Any]] = []
    for model_name in selected_models:
        hyperparameters: Dict[str, Any] = {}
        for param in MODEL_HYPERPARAMETER_TEMPLATES.get(model_name, []):
            state_key = f"model_hyper_{model_name}_{param['name']}"
            if state_key not in st.session_state:
                st.session_state[state_key] = param.get("default")
            hyperparameters[param["name"]] = st.session_state[state_key]
        model_configs.append({"name": model_name, "hyperparameters": hyperparameters})
    return model_configs


def _render_model_hyperparameter_inputs(model_name: str) -> None:
    parameters = MODEL_HYPERPARAMETER_TEMPLATES.get(model_name, [])
    if not parameters:
        st.caption("This model only uses default hyperparameters.")
        return

    for param in parameters:
        state_key = f"model_hyper_{model_name}_{param['name']}"
        if state_key not in st.session_state:
            st.session_state[state_key] = param.get("default")
        label = param.get("label", param["name"].replace("_", " ").title())
        input_kwargs: Dict[str, Any] = {
            "label": label,
            "key": state_key,
            "on_change": _handle_task_selection_change,
        }
        if param["type"] == "bool":
            st.checkbox(
                value=bool(st.session_state[state_key]),
                **input_kwargs,
            )
        elif param["type"] in {"int", "float"}:
            min_value = param.get("min")
            max_value = param.get("max")
            step = param.get("step")
            if min_value is not None:
                input_kwargs["min_value"] = min_value
            if max_value is not None:
                input_kwargs["max_value"] = max_value
            if step is not None:
                input_kwargs["step"] = step
            if param["type"] == "int":
                input_kwargs["value"] = int(st.session_state[state_key])
                st.number_input(**input_kwargs)
            else:
                input_kwargs["value"] = float(st.session_state[state_key])
                st.number_input(**input_kwargs)
        elif param["type"] == "select":
            options = param.get("options", [])
            default = st.session_state[state_key]
            if default not in options and options:
                st.session_state[state_key] = options[0]
                default = options[0]
            st.selectbox(
                label,
                options=options,
                index=options.index(default) if options else 0,
                key=state_key,
                on_change=_handle_task_selection_change,
            )
        else:
            st.text_input(
                label,
                value=str(st.session_state[state_key]),
                key=state_key,
                on_change=_handle_task_selection_change,
            )


def _render_model_selection_panel() -> None:
    _ensure_model_selection_session_state()
    with st.sidebar.expander("Model selection & hyperparameters", expanded=True):
        current_task = st.session_state.get("active_task_type", SUPPORTED_MODEL_TASKS[0])
        task_index = SUPPORTED_MODEL_TASKS.index(current_task) if current_task in SUPPORTED_MODEL_TASKS else 0
        st.radio(
            "Select task type",
            options=SUPPORTED_MODEL_TASKS,
            format_func=_format_ml_task_label,
            index=task_index,
            key="active_task_type",
            on_change=_handle_task_selection_change,
        )
        available_models = ModelRegistry.available_models(st.session_state["active_task_type"])
        selection_key = f"selected_models_{st.session_state['active_task_type']}"
        default_models = st.session_state.get(selection_key, available_models[:1])
        st.multiselect(
            "Algorithms",
            options=available_models,
            default=default_models,
            key=selection_key,
            help="Pick one or more algorithms to train for the selected task.",
            on_change=_handle_task_selection_change,
        )
        selected = st.session_state.get(selection_key, [])
        if not selected:
            st.warning("Select at least one model before running training or prediction.")
        for model_name in selected:
            with st.expander(f"{model_name.replace('_', ' ').title()} hyperparameters", expanded=False):
                _render_model_hyperparameter_inputs(model_name)


# Additional UI initialization for model configuration
_render_model_selection_panel()
