"""Configuration driven preprocessing service implementation."""

from __future__ import annotations

import hashlib
import inspect
import json
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import IsolationForest
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

__all__ = ["PreprocessingService"]


def _has_signature_parameter(func: Any, parameter: str) -> bool:
    try:
        return parameter in inspect.signature(func.__init__).parameters
    except (ValueError, TypeError):
        return False


def _to_column_sequence(value: Optional[Any]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _resolve_single_column(value: Optional[Any]) -> Optional[str]:
    columns = _to_column_sequence(value)
    return columns[0] if columns else None


def _collect_column_groups(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    return numeric_cols, categorical_cols


def _filter_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    return [col for col in columns if col in df.columns]


def _select_columns(
    df: pd.DataFrame, selection_cfg: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    cfg = selection_cfg or {}
    roles = {
        "target": _to_column_sequence(cfg.get("target")),
        "datetime": _to_column_sequence(cfg.get("datetime")),
        "id": _to_column_sequence(cfg.get("id")),
        "grouping": _to_column_sequence(cfg.get("grouping")),
        "ignore": _to_column_sequence(cfg.get("ignore")),
        "categorical": _to_column_sequence(cfg.get("categorical")),
        "numeric": _to_column_sequence(cfg.get("numeric")),
    }

    df = df.drop(columns=[col for col in roles["ignore"] if col in df.columns], errors="ignore")
    numeric_auto, categorical_auto = _collect_column_groups(df)

    if not roles["numeric"]:
        roles["numeric"] = numeric_auto
    else:
        roles["numeric"] = _filter_columns(df, roles["numeric"])
    if not roles["categorical"]:
        roles["categorical"] = categorical_auto
    else:
        roles["categorical"] = _filter_columns(df, roles["categorical"])

    selected = {
        "target": _filter_columns(df, roles["target"]),
        "datetime": _filter_columns(df, roles["datetime"]),
        "id": _filter_columns(df, roles["id"]),
        "grouping": _filter_columns(df, roles["grouping"]),
        "ignore": roles["ignore"],
        "numeric": roles["numeric"],
        "categorical": roles["categorical"],
    }
    return df, selected


def _apply_simple_imputer(
    df: pd.DataFrame,
    columns: List[str],
    strategy: str,
    fill_value: Optional[Any] = None,
) -> pd.DataFrame:
    imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
    if not columns:
        return df
    df.loc[:, columns] = imputer.fit_transform(df[columns])
    return df


def _apply_knn_imputer(
    df: pd.DataFrame, columns: List[str], n_neighbors: int = 5
) -> pd.DataFrame:
    if not columns:
        return df
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df.loc[:, columns] = imputer.fit_transform(df[columns])
    return df


def _apply_iterative_imputer(
    df: pd.DataFrame,
    columns: List[str],
    estimator: Optional[Any] = None,
    max_iter: int = 10,
    tol: float = 1e-3,
) -> pd.DataFrame:
    if not columns:
        return df
    estimator = estimator or BayesianRidge()
    imputer = IterativeImputer(estimator=estimator, max_iter=max_iter, tol=tol)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        df.loc[:, columns] = imputer.fit_transform(df[columns])
    return df


def _apply_outlier_limits(
    df: pd.DataFrame, columns: List[str], method: str, params: Dict[str, Any]
) -> Tuple[pd.DataFrame, Set[str]]:
    removed: Set[str] = set()
    if method == "iqr":
        k = params.get("k", 1.5)
        for col in columns:
            if df[col].dtype.kind not in "if":
                continue
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - k * iqr
            upper = q3 + k * iqr
            mask = df[col].between(lower, upper) | df[col].isna()
            if not mask.all():
                df = df[mask]
                removed.add(col)
    elif method == "zscore":
        threshold = params.get("threshold", 3.0)
        for col in columns:
            if df[col].dtype.kind not in "if":
                continue
            std = df[col].std()
            mean = df[col].mean()
            zscore = (df[col] - mean) / std if std else pd.Series(0, index=df.index)
            mask = zscore.abs() <= threshold
            mask |= df[col].isna()
            if not mask.all():
                df = df[mask]
                removed.add(col)
    elif method == "isolation_forest":
        detector = IsolationForest(random_state=params.get("seed", 42))
        numeric = df[columns].select_dtypes(include=[np.number])
        if not numeric.empty:
            preds = detector.fit_predict(numeric.fillna(numeric.mean()))
            mask = preds == 1
            if not mask.all():
                df = df[mask]
                removed.update(columns)
    elif method == "local_outlier_factor":
        detector = LocalOutlierFactor(n_neighbors=params.get("n_neighbors", 20))
        numeric = df[columns].select_dtypes(include=[np.number])
        if not numeric.empty:
            preds = detector.fit_predict(numeric.fillna(numeric.mean()))
            mask = preds == 1
            if not mask.all():
                df = df[mask]
                removed.update(columns)
    elif method == "winsorization":
        lower = params.get("lower", 0.01)
        upper = params.get("upper", 0.99)
        df = df.clip(lower=df.quantile(lower), upper=df.quantile(upper), axis=1)
    return df, removed


def _apply_encoding(
    df: pd.DataFrame,
    strategy: Optional[str],
    categorical: List[str],
    metadata: Dict[str, Any],
    config: Dict[str, Any],
) -> pd.DataFrame:
    if not categorical or not strategy or strategy == "none":
        return df

    if strategy == "one_hot":
        encoder_kwargs: Dict[str, Any] = {"handle_unknown": "ignore"}
        if _has_signature_parameter(OneHotEncoder, "sparse_output"):
            encoder_kwargs["sparse_output"] = False
        else:
            encoder_kwargs["sparse"] = False
        encoder = OneHotEncoder(**encoder_kwargs)
        encoded = encoder.fit_transform(df[categorical].astype(str))
        columns = encoder.get_feature_names_out(categorical)
        encoded_df = pd.DataFrame(encoded, columns=columns, index=df.index)
        df = df.drop(columns=categorical, errors="ignore")
        df = pd.concat([df, encoded_df], axis=1)
    elif strategy == "ordinal":
        encoder = OrdinalEncoder()
        df.loc[:, categorical] = encoder.fit_transform(df[categorical].astype(str))
    elif strategy == "frequency":
        for col in categorical:
            freq = df[col].value_counts(normalize=True)
            df.loc[:, col] = df[col].map(freq).fillna(0)
    elif strategy == "label":
        for col in categorical:
            le = LabelEncoder()
            df.loc[:, col] = le.fit_transform(df[col].astype(str))
    elif strategy == "target":
        target_col = _resolve_single_column(config.get("target_column"))
        if not target_col:
            target_col = _resolve_single_column(config.get("target"))
        if not target_col or target_col not in df.columns:
            metadata.setdefault("encoding", {}).setdefault("warnings", []).append(
                "Missing target column for target encoding"
            )
            return df
        for col in categorical:
            aggregated = df.groupby(col)[target_col].mean()
            df.loc[:, col] = df[col].map(aggregated).fillna(df[target_col].mean())
    return df


def _apply_scaling(
    df: pd.DataFrame, strategy: Optional[str], numeric: List[str], params: Dict[str, Any]
) -> Tuple[pd.DataFrame, Optional[Any]]:
    if not numeric or not strategy or strategy == "none":
        return df, None

    scaler: Optional[Any] = None
    estimator_params = params.get("params", {}) if isinstance(params, dict) else {}

    if strategy == "standard":
        scaler = StandardScaler(**estimator_params)
    elif strategy == "minmax":
        feature_range = params.get("feature_range", (0, 1)) if isinstance(params, dict) else (0, 1)
        scaler = MinMaxScaler(feature_range=feature_range, **(params.get("params", {}) if isinstance(params, dict) else {}))
    elif strategy == "robust":
        scaler = RobustScaler(**estimator_params)
    elif strategy == "quantile":
        scaler = QuantileTransformer(**estimator_params)
    elif strategy == "power":
        scaler = PowerTransformer(**estimator_params)
    if not scaler:
        return df, None

    df.loc[:, numeric] = scaler.fit_transform(df[numeric])
    return df, scaler


def _normalize_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    def _normalize_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: _normalize_value(value[key]) for key in sorted(value)}
        if isinstance(value, (list, tuple)):
            return [_normalize_value(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    return _normalize_value(config or {})  # type: ignore[arg-type]


def _hash_dataframe(df: pd.DataFrame) -> str:
    try:
        hashed = pd.util.hash_pandas_object(df, index=True).values
        digest = hashlib.sha256(hashed.tobytes()).hexdigest()
    except Exception:
        digest = hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()
    return digest


def _config_signature(config_json: str) -> str:
    return hashlib.sha256(config_json.encode()).hexdigest()


def _record_pipeline_state(
    metadata: Dict[str, Any],
    intermediates: List[Dict[str, Any]],
    stage: str,
    df: pd.DataFrame,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    state = {
        "step": stage,
        "columns": list(df.columns),
        "hash": _hash_dataframe(df),
        "details": details or {},
    }
    metadata.setdefault("pipeline", []).append(state)
    intermediates.append({"step": stage, "data": df.copy(), "details": details or {}})


@dataclass(frozen=True)
class PreprocessingBlueprint:
    signature: str
    normalized_config: Dict[str, Any]


@st.cache_resource
def _build_preprocessing_blueprint(config_json: str) -> PreprocessingBlueprint:
    normalized = json.loads(config_json)
    signature = _config_signature(config_json)
    return PreprocessingBlueprint(signature=signature, normalized_config=normalized)


def _apply_preprocessing_pipeline(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {"steps": []}
    intermediates: List[Dict[str, Any]] = []

    # Column selection
    column_cfg = config.get("column_selection", {})
    df, column_roles = _select_columns(df, column_cfg)
    metadata["column_selection"] = column_roles
    _record_pipeline_state(metadata, intermediates, "column_selection", df, {"roles": column_roles})

    numeric_cols = column_roles.get("numeric", [])
    categorical_cols = column_roles.get("categorical", [])

    # Missing value handling
    missing_cfg = config.get("missing_value", {})
    if missing_cfg:
        strategy = missing_cfg.get("strategy", "none")
        metadata.setdefault("missing_value", {}).update({"strategy": strategy})
        targeted_columns = _filter_columns(
            df, _to_column_sequence(missing_cfg.get("columns", numeric_cols + categorical_cols))
        )
        if strategy in {"mean", "median", "most_frequent", "constant"}:
            fill_value = missing_cfg.get("fill_value")
            df = _apply_simple_imputer(
                df,
                targeted_columns,
                strategy if strategy != "mode" else "most_frequent",
                fill_value,
            )
        elif strategy == "knn":
            df = _apply_knn_imputer(
                df,
                targeted_columns,
                n_neighbors=missing_cfg.get("n_neighbors", 5),
            )
        elif strategy == "iterative":
            df = _apply_iterative_imputer(
                df,
                targeted_columns,
                max_iter=missing_cfg.get("max_iter", 10),
                tol=missing_cfg.get("tol", 1e-3),
            )
        elif strategy == "drop_rows":
            how = missing_cfg.get("how", "any")
            df = df.dropna(axis=0, how=how)
        elif strategy == "drop_columns":
            df = df.drop(columns=targeted_columns, errors="ignore")
        elif strategy == "forward_fill":
            df.loc[:, targeted_columns] = df[targeted_columns].fillna(method="ffill")
        elif strategy == "backward_fill":
            df.loc[:, targeted_columns] = df[targeted_columns].fillna(method="bfill")
        metadata["missing_value"].update({"handled_columns": targeted_columns})
        _record_pipeline_state(
            metadata,
            intermediates,
            "missing_value",
            df,
            {"strategy": strategy, "columns": targeted_columns},
        )

    # Outlier handling
    outlier_cfg = config.get("outlier", {})
    method = outlier_cfg.get("method", "none")
    if method and method != "none":
        metadata.setdefault("outlier", {}).update({"method": method})
        df, removed = _apply_outlier_limits(df, numeric_cols, method, outlier_cfg)
        metadata["outlier"]["columns_affected"] = list(removed)
        _record_pipeline_state(
            metadata,
            intermediates,
            "outlier",
            df,
            {"method": method, "removed_columns": list(removed)},
        )

    # Encoding
    encoding_cfg = config.get("encoding", {})
    encoding_strategy = encoding_cfg.get("strategy", "none")
    if encoding_cfg and encoding_strategy != "none":
        metadata.setdefault("encoding", {}).update(
            {
                "strategy": encoding_strategy,
                "columns": categorical_cols,
                "target_column": _resolve_single_column(encoding_cfg.get("target_column")),
            }
        )
        df = _apply_encoding(df, encoding_strategy, categorical_cols, metadata, encoding_cfg)
        numeric_cols, categorical_cols = _collect_column_groups(df)
        _record_pipeline_state(
            metadata,
            intermediates,
            "encoding",
            df,
            {"strategy": encoding_strategy, "columns": categorical_cols},
        )

    # Scaling
    scaling_cfg = config.get("scaling", {})
    scaling_strategy = scaling_cfg.get("strategy", "none")
    if scaling_cfg and scaling_strategy != "none":
        metadata.setdefault("scaling", {}).update({"strategy": scaling_strategy})
        df, scaler = _apply_scaling(df, scaling_strategy, numeric_cols, scaling_cfg)
        if scaler is not None:
            metadata["scaling"]["scaler"] = scaler.__class__.__name__
        _record_pipeline_state(
            metadata,
            intermediates,
            "scaling",
            df,
            {"strategy": scaling_strategy, "columns": numeric_cols},
        )

    metadata["columns"] = list(df.columns)
    return {"data": df, "metadata": metadata, "intermediates": intermediates}


@st.cache_data(show_spinner=False)
def _cached_preprocessing(
    dataframe: pd.DataFrame,
    data_signature: str,
    dataset_id: str,
    config_json: str,
) -> Dict[str, Any]:
    df = dataframe.copy()
    blueprint = _build_preprocessing_blueprint(config_json)
    result = _apply_preprocessing_pipeline(df, blueprint.normalized_config)
    result["metadata"]["config_signature"] = blueprint.signature
    result["metadata"]["dataset_id"] = dataset_id
    result["metadata"]["data_signature"] = data_signature
    return result


@dataclass
class PreprocessingService:
    """Entry point for configuration driven preprocessing operations."""

    def run(self, data: Any, config: Dict[str, Any], dataset_id: str = "") -> Dict[str, Any]:
        """Applies configured preprocessing steps to the provided dataset."""
        if data is None:
            return {"data": data, "metadata": {"steps": []}, "intermediates": []}

        df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data.copy()
        normalized_config = _normalize_config(config)
        config_json = json.dumps(normalized_config, sort_keys=True)
        data_signature = _hash_dataframe(df)

        return _cached_preprocessing(
            dataframe=df,
            data_signature=data_signature,
            dataset_id=dataset_id,
            config_json=config_json,
        )
