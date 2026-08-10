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

from backend.evaluation import EvaluationService
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

WORKFLOW_METRIC_OPTIONS = ["rmse", "mae", "mse", "r2", "accuracy"]
WORKFLOW_EVALUATION_SCOPES = ["validation", "holdout", "production"]

SIMULATION_NOTICE = (
    "The optimization, comparison, and diagnostics sections currently show simulated data. "
    "No real training is executed in this demo workspace. Use the dataset inputs and workflow cues "
    "to validate config decisions before integrating with an actual ML pipeline."
)


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


def _ensure_workflow_session_state() -> None:
    if "workflow_selected_metrics" not in st.session_state:
        st.session_state["workflow_selected_metrics"] = WORKFLOW_METRIC_OPTIONS[:3]
    if "workflow_evaluation_scope" not in st.session_state:
        st.session_state["workflow_evaluation_scope"] = WORKFLOW_EVALUATION_SCOPES[0]
    if "workflow_comparison" not in st.session_state:
        st.session_state["workflow_comparison"] = None
    if "workflow_evaluation_artifact" not in st.session_state:
        st.session_state["workflow_evaluation_artifact"] = None
    if "workflow_last_run" not in st.session_state:
        st.session_state["workflow_last_run"] = ""
    if "workflow_last_evaluation" not in st.session_state:
        st.session_state["workflow_last_evaluation"] = ""


def _render_simulation_notice() -> None:
    st.info(SIMULATION_NOTICE)


def _simulate_workflow_response(
    selected_models: List[str],
    metrics: List[str],
    evaluation_scope: str,
) -> Dict[str, Any]:
    rng = np.random.default_rng(int(datetime.utcnow().timestamp()))
    models: List[Dict[str, Any]] = []
    for model_name in selected_models:
        metric_map: Dict[str, float] = {}
        for metric in metrics:
            if metric in {"rmse", "mae", "mse"}:
                metric_map[metric] = round(max(0.01, rng.normal(0.7, 0.3)), 3)
            elif metric == "accuracy":
                metric_map[metric] = round(min(1.0, max(0.0, rng.normal(0.65, 0.15))), 3)
            else:
                metric_map[metric] = round(min(1.0, max(0.0, rng.normal(0.5, 0.2))), 3)
        diagnostics = {
            "training_time_sec": round(rng.uniform(12, 90), 1),
            "convergence_pct": round(rng.uniform(82, 99), 1),
            "validation_loss": round(metric_map.get("rmse", metric_map.get("mse", 0.0)), 3),
        }
        models.append(
            {
                "model_name": model_name,
                "metrics": metric_map,
                "diagnostics": diagnostics,
            }
        )
    primary_metric = metrics[0] if metrics else "accuracy"
    goal = "min" if primary_metric in {"rmse", "mae", "mse"} else "max"
    best_model = sorted(
        models,
        key=lambda entry: entry["metrics"].get(primary_metric, 0.0),
        reverse=goal == "max",
    )[0]
    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "selected_models": selected_models,
        "metrics": metrics,
        "evaluation_scope": evaluation_scope,
        "models": models,
        "best_model": best_model,
    }
    payload["artifacts"] = {
        "comparison_payload": json.dumps(payload, indent=2),
    }
    return payload


def _simulate_evaluation_artifact(
    evaluation_scope: str,
    model_payloads: List[Dict[str, Any]],
) -> Dict[str, Any]:
    service = EvaluationService()
    predictions: List[int] = []
    targets: List[int] = []
    for entry in model_payloads:
        accuracy = entry["metrics"].get("accuracy", 0.5)
        sample_size = 60
        matches = min(sample_size, max(0, int(round(accuracy * sample_size))))
        predictions.extend([1] * matches + [0] * (sample_size - matches))
        targets.extend([1] * sample_size)
    evaluation = service.evaluate(predictions, targets, {"scope": evaluation_scope})
    artifact = {
        "scope": evaluation_scope,
        "generated_at": datetime.utcnow().isoformat(),
        "evaluation": evaluation,
        "artifacts": {
            "evaluation_report": json.dumps(evaluation, indent=2),
        },
    }
    return artifact


def _render_workflow_results_panel() -> None:
    comparison = st.session_state.get("workflow_comparison")
    evaluation = st.session_state.get("workflow_evaluation_artifact")
    if not comparison and not evaluation:
        return
    st.header("Optimization & evaluation outputs")
    _render_simulation_notice()
    if comparison:
        with st.expander("Model comparison summary", expanded=True):
            st.subheader("Metrics overview")
            metric_records: List[Dict[str, Any]] = []
            for entry in comparison["models"]:
                row = {"Model": entry["model_name"]}
                for metric_name, value in entry["metrics"].items():
                    row[metric_name.upper()] = value
                metric_records.append(row)
            if metric_records:
                st.dataframe(pd.DataFrame(metric_records))
            st.caption(f"Primary metric: {comparison['metrics'][0] if comparison['metrics'] else 'accuracy'}")
            st.metric(
                "Best performing model",
                comparison["best_model"]["model_name"],
                delta=None,
            )
            if comparison["best_model"].get("metrics"):
                best_metrics = comparison["best_model"]["metrics"]
                st.json(best_metrics)
            st.download_button(
                "Download comparison payload",
                data=comparison["artifacts"]["comparison_payload"].encode("utf-8"),
                file_name="model_comparison_payload.json",
                mime="application/json",
            )
            for entry in comparison["models"]:
                with st.expander(f"Diagnostics: {entry['model_name']}", expanded=False):
                    st.table(pd.DataFrame([entry["diagnostics"]]))
    if evaluation:
        with st.expander("Evaluation diagnostics", expanded=True):
            st.subheader(f"Scope: {evaluation['scope'].title()}")
            st.caption("Evaluations are generated from synthetic predictions for this demo workspace.")
            st.json(evaluation["evaluation"])
            st.download_button(
                "Download evaluation artifact",
                data=evaluation["artifacts"]["evaluation_report"].encode("utf-8"),
                file_name="evaluation_artifact.json",
                mime="application/json",
            )


def _handle_workflow_actions() -> None:
    current_task = st.session_state.get("active_task_type", SUPPORTED_MODEL_TASKS[0])
    selection_key = f"selected_models_{current_task}"
    selected_models = st.session_state.get(selection_key, [])
    if not selected_models:
        st.warning("Optimization requires at least one selected algorithm.")
        return
    metrics = st.session_state.get("workflow_selected_metrics", WORKFLOW_METRIC_OPTIONS[:3])
    if not metrics:
        st.warning("Pick one or more metrics to surface in the comparison summary.")
        return
    scope = st.session_state.get("workflow_evaluation_scope", WORKFLOW_EVALUATION_SCOPES[0])
    comparison_payload = _simulate_workflow_response(selected_models, metrics, scope)
    st.session_state["workflow_comparison"] = comparison_payload
    st.session_state["workflow_evaluation_artifact"] = _simulate_evaluation_artifact(
        scope, comparison_payload["models"]
    )
    st.session_state["workflow_last_run"] = datetime.utcnow().isoformat()
    st.success("Optimization and comparison completed with simulated diagnostics. Interpret the summaries as previews, not as actual training outcomes.")


def _refresh_evaluation_outputs() -> None:
    comparison = st.session_state.get("workflow_comparison")
    if not comparison:
        st.warning("Run optimization before generating evaluation artifacts.")
        return
    scope = st.session_state.get("workflow_evaluation_scope", WORKFLOW_EVALUATION_SCOPES[0])
    st.session_state["workflow_evaluation_artifact"] = _simulate_evaluation_artifact(scope, comparison["models"])
    st.session_state["workflow_last_evaluation"] = datetime.utcnow().isoformat()
    st.success("Evaluation diagnostics refreshed (simulated).")


# Diagnostics helpers -----------------------------------------------------------------------------

def _compute_curve_points(actual: np.ndarray, probs: np.ndarray) -> Dict[str, List[float]]:
    thresholds = np.linspace(0.0, 1.0, 41)
    fprs: List[float] = []
    tprs: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []

    positives = actual == 1
    negatives = actual == 0

    for threshold in thresholds:
        predicted = (probs >= threshold).astype(int)
        tp = np.sum(predicted[positives] == 1)
        fp = np.sum(predicted[negatives] == 1)
        tn = np.sum(predicted[negatives] == 0)
        fn = np.sum(predicted[positives] == 0)

        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tpr

        tprs.append(tpr)
        fprs.append(fpr)
        precisions.append(precision)
        recalls.append(recall)

    return {
        "thresholds": thresholds.tolist(),
        "fpr": fprs,
        "tpr": tprs,
        "precision": precisions,
        "recall": recalls,
    }


@st.cache_resource
def _get_model_diagnostics(model_name: str, task_type: str) -> Dict[str, Any]:
    seed_value = abs(hash((model_name, task_type))) % (2 ** 32)
    rng = np.random.default_rng(seed_value)
    features = [f"feature_{i+1}" for i in range(6)]
    raw_importances = np.abs(rng.normal(0.4, 0.2, len(features)))
    normalized_importances = raw_importances / (raw_importances.sum() or 1)
    feature_importance = pd.DataFrame(
        {"feature": features, "importance": normalized_importances}
    ).sort_values("importance", ascending=False)

    sample_size = 200
    if task_type == "classification":
        actual = rng.integers(0, 2, sample_size)
        predicted_probs = np.clip(rng.beta(2.2, 2.0, sample_size), 0, 1)
        predicted_values = predicted_probs
        thresholded_predictions = (predicted_probs >= 0.5).astype(int)
    else:
        actual = rng.normal(60, 12, sample_size)
        predicted_values = actual + rng.normal(0, 5, sample_size)
        predicted_probs = None
        thresholded_predictions = (predicted_values >= actual.mean()).astype(int)

    residuals = pd.DataFrame(
        {
            "index": np.arange(sample_size),
            "actual": actual,
            "predicted": predicted_values,
            "residual": predicted_values - actual,
        }
    )

    learning_sizes = np.linspace(0.1, 1.0, 8)
    learning_curve = pd.DataFrame(
        {
            "training_size": (learning_sizes * 100).astype(int),
            "training_score": np.clip(
                np.linspace(0.62, 0.95, len(learning_sizes)) + rng.normal(0, 0.02, len(learning_sizes)),
                0,
                1,
            ),
            "validation_score": np.clip(
                np.linspace(0.58, 0.9, len(learning_sizes)) + rng.normal(0, 0.02, len(learning_sizes)),
                0,
                1,
            ),
        }
    )

    param_range = np.linspace(1, 8, 8)
    validation_curve = pd.DataFrame(
        {
            "param_value": np.round(param_range, 2),
            "training_score": np.clip(
                np.linspace(0.65, 0.96, len(param_range)) + rng.normal(0, 0.015, len(param_range)),
                0,
                1,
            ),
            "validation_score": np.clip(
                np.linspace(0.6, 0.91, len(param_range)) + rng.normal(0, 0.015, len(param_range)),
                0,
                1,
            ),
        }
    )

    confusion_matrix = None
    roc_data = None
    precision_recall_data = None
    if task_type == "classification":
        diag = _compute_curve_points(actual, predicted_probs)
        roc_data = {"fpr": diag["fpr"], "tpr": diag["tpr"]}
        precision_recall_data = {"precision": diag["precision"], "recall": diag["recall"]}
        tp = np.sum((thresholded_predictions == 1) & (actual == 1))
        fn = np.sum((thresholded_predictions == 0) & (actual == 1))
        fp = np.sum((thresholded_predictions == 1) & (actual == 0))
        tn = np.sum((thresholded_predictions == 0) & (actual == 0))
        confusion_matrix = pd.DataFrame(
            {
                "Pred=0": [tn, fn],
                "Pred=1": [fp, tp],
            },
            index=["Actual=0", "Actual=1"],
        )

    return {
        "feature_importance": feature_importance,
        "residuals": residuals,
        "learning_curve": learning_curve,
        "validation_curve": validation_curve,
        "roc": roc_data,
        "precision_recall": precision_recall_data,
        "confusion_matrix": confusion_matrix,
        "task_type": task_type,
    }


def _render_feature_importance_chart(df: pd.DataFrame) -> None:
    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        title="Feature importance",
        text_auto=".2f",
    )
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, use_container_width=True)


def _render_residual_chart(df: pd.DataFrame) -> None:
    fig = px.scatter(
        df,
        x="predicted",
        y="residual",
        size_max=6,
        hover_data={"index": True, "actual": True},
        title="Residual diagnostics",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)


def _render_roc_chart(roc_data: Dict[str, List[float]]) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=roc_data["fpr"],
            y=roc_data["tpr"],
            mode="lines+markers",
            name="ROC",
        )
    )
    fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        y0=0,
        y1=1,
        line=dict(dash="dash", color="gray"),
        name="Chance",
    )
    fig.update_layout(
        title="ROC curve",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        legend=dict(orientation="h"),
        hovermode="closest",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_precision_recall_chart(data: Dict[str, List[float]]) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["recall"],
            y=data["precision"],
            mode="lines+markers",
            name="Precision-Recall",
        )
    )
    fig.update_layout(
        title="Precision-Recall curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        hovermode="closest",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_confusion_matrix(confusion_matrix: pd.DataFrame) -> None:
    fig = go.Figure(
        data=go.Heatmap(
            z=confusion_matrix.values,
            x=confusion_matrix.columns.tolist(),
            y=confusion_matrix.index.tolist(),
            colorscale="Blues",
            hoverongaps=False,
            showscale=True,
        )
    )
    fig.update_layout(title="Aggregated confusion matrix")
    st.plotly_chart(fig, use_container_width=True)


def _render_learning_curve(chart_data: pd.DataFrame) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_data["training_size"],
            y=chart_data["training_score"],
            mode="lines+markers",
            name="Training",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_data["training_size"],
            y=chart_data["validation_score"],
            mode="lines+markers",
            name="Validation",
        )
    )
    fig.update_layout(
        xaxis_title="Training set size (%)",
        yaxis_title="Score",
        title="Learning curve",
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_validation_curve(chart_data: pd.DataFrame) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_data["param_value"],
            y=chart_data["training_score"],
            mode="lines+markers",
            name="Training",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_data["param_value"],
            y=chart_data["validation_score"],
            mode="lines+markers",
            name="Validation",
        )
    )
    fig.update_layout(
        xaxis_title="Hyperparameter value",
        yaxis_title="Score",
        title="Validation curve",
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_model_diagnostics_panel() -> None:
    current_task = st.session_state.get("active_task_type", SUPPORTED_MODEL_TASKS[0])
    selection_key = f"selected_models_{current_task}"
    selected_models = st.session_state.get(selection_key, [])
    st.header("Trained model diagnostics")
    _render_simulation_notice()
    if not selected_models:
        st.info("Select and train at least one model to inspect diagnostics.")
        return
    for model_name in selected_models:
        diag_data = _get_model_diagnostics(model_name, current_task)
        st.subheader(f"{model_name.replace('_', ' ').title()} diagnostics")
        tabs = st.tabs(
            [
                "Feature importance",
                "Residual diagnostics",
                "ROC curve",
                "Precision-Recall",
                "Confusion matrix",
                "Learning curve",
                "Validation curve",
            ]
        )
        with tabs[0]:
            _render_feature_importance_chart(diag_data["feature_importance"])
        with tabs[1]:
            _render_residual_chart(diag_data["residuals"])
        with tabs[2]:
            if diag_data["roc"]:
                _render_roc_chart(diag_data["roc"])
            else:
                st.info("ROC curve is available for classification tasks only.")
        with tabs[3]:
            if diag_data["precision_recall"]:
                _render_precision_recall_chart(diag_data["precision_recall"])
            else:
                st.info("Precision-recall curve is only available for classification tasks.")
        with tabs[4]:
            if diag_data["confusion_matrix"] is not None:
                _render_confusion_matrix(diag_data["confusion_matrix"])
            else:
                st.info("Confusion matrix requires discrete prediction outputs, e.g., classification models.")
        with tabs[5]:
            _render_learning_curve(diag_data["learning_curve"])
        with tabs[6]:
            _render_validation_curve(diag_data["validation_curve"])


# Additional UI initialization for model configuration
_ensure_preprocessing_session_state()
_ensure_feature_engineering_session_state()
_render_model_selection_panel()
_ensure_workflow_session_state()

with st.sidebar.expander("Optimization & evaluation workflow", expanded=True):
    st.caption("Select metrics and artifacts to carry into the optimization pipeline.")
    st.multiselect(
        "Comparison metrics",
        options=WORKFLOW_METRIC_OPTIONS,
        default=st.session_state["workflow_selected_metrics"],
        key="workflow_selected_metrics",
        help="These metrics appear in the summary table post-comparison.",
    )
    st.radio(
        "Evaluation scope",
        options=WORKFLOW_EVALUATION_SCOPES,
        index=WORKFLOW_EVALUATION_SCOPES.index(
            st.session_state["workflow_evaluation_scope"]
        ),
        key="workflow_evaluation_scope",
    )
    if st.button("Optimize & compare models", key="optimize_models"):
        _handle_workflow_actions()
    if st.button("Refresh evaluation diagnostics", key="refresh_evaluation"):
        _refresh_evaluation_outputs()

_render_workflow_results_panel()
_render_model_diagnostics_panel()
