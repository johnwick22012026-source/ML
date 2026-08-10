"""Configuration driven preprocessing service implementation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.ensemble import IsolationForest

__all__ = ["PreprocessingService"]


def _collect_column_groups(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    return numeric_cols, categorical_cols


def _apply_simple_imputer(
    df: pd.DataFrame,
    columns: List[str],
    strategy: str,
    fill_value: Optional[Any] = None,
) -> pd.DataFrame:
    imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
    df.loc[:, columns] = imputer.fit_transform(df[columns])
    return df


def _apply_knn_imputer(df: pd.DataFrame, columns: List[str], n_neighbors: int = 5) -> pd.DataFrame:
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df.loc[:, columns] = imputer.fit_transform(df[columns])
    return df


def _apply_iterative_imputer(
    df: pd.DataFrame, columns: List[str], estimator: Optional[Any] = None, max_iter: int = 10, tol: float = 1e-3
) -> pd.DataFrame:
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
            zscore = (df[col] - df[col].mean()) / df[col].std() if df[col].std() else pd.Series(0, index=df.index)
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
    df: pd.DataFrame, strategy: Optional[str], categorical: List[str], params: Dict[str, Any]
) -> pd.DataFrame:
    if not categorical or not strategy or strategy == "none":
        return df

    if strategy == "one_hot":
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = encoder.fit_transform(df[categorical].astype(str))
        columns = encoder.get_feature_names_out(categorical)
        encoded_df = pd.DataFrame(encoded, columns=columns, index=df.index)
        df = df.drop(columns=categorical)
        df = pd.concat([df, encoded_df], axis=1)
    elif strategy == "ordinal":
        encoder = OrdinalEncoder()
        df.loc[:, categorical] = encoder.fit_transform(df[categorical])
    elif strategy == "frequency":
        for col in categorical:
            freq = df[col].value_counts(normalize=True)
            df.loc[:, col] = df[col].map(freq).fillna(0)
    elif strategy == "label":
        for col in categorical:
            le = LabelEncoder()
            df.loc[:, col] = le.fit_transform(df[col].astype(str))
    return df


def _apply_scaling(
    df: pd.DataFrame, strategy: Optional[str], numeric: List[str], params: Dict[str, Any]
) -> Tuple[pd.DataFrame, Optional[Any]]:
    if not numeric or not strategy or strategy == "none":
        return df, None

    scaler_map = {
        "standard": StandardScaler,
        "minmax": lambda: StandardScaler(with_mean=True, with_std=True),
        "robust": RobustScaler,
        "quantile": QuantileTransformer,
        "power": PowerTransformer,
    }
    scaler_cls = scaler_map.get(strategy)
    if not scaler_cls:
        return df, None

    scaler = scaler_cls()
    df.loc[:, numeric] = scaler.fit_transform(df[numeric])
    return df, scaler


@dataclass
class PreprocessingService:
    """Entry point for configuration driven preprocessing operations."""

    def run(self, data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Applies configured preprocessing steps to the provided dataset."""
        if data is None:
            return {"data": data, "metadata": {"steps": []}}

        df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data.copy()

        metadata: Dict[str, Any] = {"steps": []}

        numeric_cols, categorical_cols = _collect_column_groups(df)

        missing_cfg = config.get("missing_value", {})
        if missing_cfg:
            strategy = missing_cfg.get("strategy", "none")
            metadata["missing_value"] = {"strategy": strategy}
            if strategy in {"mean", "median", "most_frequent", "constant"}:
                fill = missing_cfg.get("fill_value")
                df = _apply_simple_imputer(df, numeric_cols, strategy, fill)
            elif strategy == "knn":
                n_neighbors = missing_cfg.get("n_neighbors", 5)
                df = _apply_knn_imputer(df, numeric_cols, n_neighbors)
            elif strategy == "iterative":
                max_iter = missing_cfg.get("max_iter", 10)
                tol = missing_cfg.get("tol", 1e-3)
                df = _apply_iterative_imputer(df, numeric_cols, max_iter=max_iter, tol=tol)
            elif strategy == "drop":
                df = df.dropna(axis=0, how=missing_cfg.get("how", "any"))
            metadata["missing_value"].update({"filled_columns": numeric_cols})

        outlier_cfg = config.get("outlier", {})
        if outlier_cfg:
            method = outlier_cfg.get("method", "none")
            if method != "none":
                metadata.setdefault("outlier", {}).update({"method": method})
                df, removed = _apply_outlier_limits(df, numeric_cols, method, outlier_cfg)
                metadata["outlier"]["columns_affected"] = list(removed)

        encoding_cfg = config.get("encoding", {})
        encoding_strategy = encoding_cfg.get("strategy", "none")
        if encoding_cfg and encoding_strategy != "none":
            metadata.setdefault("encoding", {}).update(
                {"strategy": encoding_strategy, "columns": categorical_cols}
            )
            df = _apply_encoding(df, encoding_strategy, categorical_cols, encoding_cfg)
            numeric_cols, categorical_cols = _collect_column_groups(df)

        scaling_cfg = config.get("scaling", {})
        scaling_strategy = scaling_cfg.get("strategy", "none")
        if scaling_cfg and scaling_strategy != "none":
            metadata.setdefault("scaling", {}).update({"strategy": scaling_strategy})
            df, scaler = _apply_scaling(df, scaling_strategy, numeric_cols, scaling_cfg)
            if scaler is not None:
                metadata["scaling"]["scaler"] = scaler.__class__.__name__ if hasattr(scaler, "__class__") else None

        metadata["columns"] = list(df.columns)
        return {"data": df, "metadata": metadata}
