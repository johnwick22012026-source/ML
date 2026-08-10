"""Forecasting service that executes configured forecasting jobs entirely in memory."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

from backend.caching import CachingService
from backend.utils.run_context import (
    build_run_id,
    compute_config_signature,
    compute_file_signature,
    normalize_config,
)


__all__ = ["ForecastingService"]


ForecastHorizon = int
TimeSeries = Union[pd.Series, List[Union[int, float]], List[Dict[str, Any]]]


@dataclass(frozen=True)
class ForecastResult:
    model: str
    horizon: ForecastHorizon
    forecast: List[float]
    metadata: Dict[str, Any]
    model_details: Dict[str, Any]


class ForecastingService:
    """Entry point for forecast execution driven completely by configuration."""

    SUPPORTED_METHODS = {
        "arima",
        "sarima",
        "sarimax",
        "ets",
        "prophet",
        "holt_winters",
        "auto_arima",
        "xgboost",
        "random_forest",
    }

    def __init__(self, caching_service: Optional[CachingService] = None) -> None:
        self._caching_service = caching_service or CachingService()

    def forecast(
        self,
        history: TimeSeries,
        config: Dict[str, Any],
        dataset_id: Optional[str] = None,
        dataset_file: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """Produce forecasts driven entirely by the supplied configuration with caching."""
        runtime_config = self._normalize_config(config)
        file_hash = compute_file_signature(dataset_file)
        cache_key = self._build_cache_key(
            namespace="forecasting",
            dataset_id=dataset_id,
            component="forecast_run",
            config=runtime_config,
            file_hash=file_hash,
        )
        cached = self._caching_service.get_data(cache_key, runtime_config, dataset_id, file_hash)
        if cached:
            return cached

        series = self._prepare_series(history, runtime_config, dataset_id, file_hash)
        forecast_values, details = self._dispatch_method(
            series, runtime_config, self._build_cache_context(dataset_id, file_hash)
        )
        runtime_config["runtime_generated_at"] = datetime.utcnow().isoformat() + "Z"
        metadata: Dict[str, Any] = {
            "requested_at": datetime.utcnow().isoformat() + "Z",
            "method": runtime_config["method"],
            "horizon": runtime_config["horizon"],
            "seed": runtime_config.get("seed"),
            "trained_samples": len(series),
        }

        result = {
            "model": runtime_config["method"],
            "horizon": runtime_config["horizon"],
            "forecast": [float(v) for v in forecast_values],
            "metadata": metadata,
            "model_details": details,
        }
        artifacts = self._build_session_artifacts(result, runtime_config, dataset_id, file_hash)
        result_with_artifacts = {**result, "artifacts": artifacts}
        self._caching_service.set_data(cache_key, result_with_artifacts, runtime_config, dataset_id, file_hash)
        return result_with_artifacts

    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        if not config or "method" not in config:
            raise ValueError("Forecast configuration must provide a 'method'.")
        method = str(config["method"]).strip().lower()
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Forecasting method '{method}' is not supported. Supported: {sorted(self.SUPPORTED_METHODS)}"
            )
        horizon = int(config.get("horizon", 1))
        if horizon <= 0:
            raise ValueError("Forecast horizon must be greater than zero.")
        seed = config.get("seed")
        if seed is not None:
            seed = int(seed)
            np.random.seed(seed)
        return {
            "method": method,
            "horizon": horizon,
            "seasonal_periods": config.get("seasonal_periods"),
            "order": config.get("order"),
            "seasonal_order": config.get("seasonal_order"),
            "exogenous": config.get("exogenous"),
            "params": config.get("params", {}),
            "seed": seed,
            "ml_lookback": int(config.get("ml_lookback", 6)),
        }

    def _prepare_series(
        self,
        history: TimeSeries,
        runtime_config: Dict[str, Any],
        dataset_id: Optional[str],
        file_hash: Optional[str],
    ) -> pd.Series:
        cache_key = self._build_cache_key(
            namespace="forecasting",
            dataset_id=dataset_id,
            component="prepared_series",
            config=runtime_config,
            file_hash=file_hash,
        )
        cached_series = self._caching_service.get_data(
            cache_key, runtime_config, dataset_id, file_hash
        )
        if cached_series is not None:
            return cached_series.copy()
        if isinstance(history, pd.Series):
            series = history.copy()
        elif isinstance(history, list):
            if history and isinstance(history[0], dict) and "ds" in history[0]:
                series = pd.Series({entry["ds"]: entry["y"] for entry in history})
            else:
                series = pd.Series(history)
        else:
            raise ValueError("History data must be a pandas Series or list of values.")
        if series.dropna().empty:
            raise ValueError("History must contain at least one non-null observation.")
        if runtime_config.get("seasonal_periods"):
            series = series.asfreq("D")
        self._caching_service.set_data(
            cache_key, series.copy(), runtime_config, dataset_id, file_hash
        )
        return series

    def _dispatch_method(
        self,
        series: pd.Series,
        runtime_config: Dict[str, Any],
        cache_context: Dict[str, Optional[str]],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        method = runtime_config["method"]
        dispatcher = {
            "arima": self._run_arima,
            "sarima": self._run_sarima,
            "sarimax": self._run_sarimax,
            "ets": self._run_ets,
            "prophet": self._run_prophet,
            "holt_winters": self._run_holt_winters,
            "auto_arima": self._run_auto_arima,
            "xgboost": self._run_ml_forecast,
            "random_forest": self._run_ml_forecast,
        }
        handler = dispatcher[method]
        forecast_series, details = handler(series, runtime_config, cache_context)
        flatten = np.asarray(forecast_series).ravel()
        if flatten.shape[0] < runtime_config["horizon"]:
            flatten = np.resize(flatten, runtime_config["horizon"])
        return flatten[-runtime_config["horizon"] :], details

    def _run_arima(
        self,
        series: pd.Series,
        config: Dict[str, Any],
        cache_context: Optional[Dict[str, Optional[str]]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        order = config.get("order") or (1, 1, 0)
        model = ARIMA(series, order=order)
        fit = model.fit()
        forecast = fit.forecast(steps=config["horizon"])
        details = self._extract_details(fit, order=order)
        return forecast.values, details

    def _run_sarima(
        self,
        series: pd.Series,
        config: Dict[str, Any],
        cache_context: Optional[Dict[str, Optional[str]]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        order = config.get("order") or (1, 1, 1)
        seasonal_order = config.get("seasonal_order") or (1, 1, 1, config.get("seasonal_periods") or 12)
        model = SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False)
        fit = model.fit(disp=False)
        forecast = fit.forecast(steps=config["horizon"])
        details = self._extract_details(fit, order=order, seasonal_order=seasonal_order)
        return forecast.values, details

    def _run_sarimax(
        self,
        series: pd.Series,
        config: Dict[str, Any],
        cache_context: Optional[Dict[str, Optional[str]]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        exog = self._sanitize_exogenous(config.get("exogenous"), config["horizon"])
        order = config.get("order") or (1, 1, 1)
        seasonal_order = config.get("seasonal_order") or (0, 0, 0, 0)
        model = SARIMAX(series, exog=exog["history"], order=order, seasonal_order=seasonal_order)
        fit = model.fit(disp=False)
        forecast = fit.get_forecast(steps=config["horizon"], exog=exog["future"])
        details = self._extract_details(
            fit, order=order, seasonal_order=seasonal_order, exogenous=bool(exog["history"])
        )
        return forecast.predicted_mean.values, details

    def _run_ets(
        self,
        series: pd.Series,
        config: Dict[str, Any],
        cache_context: Optional[Dict[str, Optional[str]]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        seasonal = config.get("params", {}).get("seasonal", None)
        trend = config.get("params", {}).get("trend", None)
        seasonal_periods = config.get("seasonal_periods")
        model = ExponentialSmoothing(
            series,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods if seasonal else None,
        )
        fit = model.fit()
        forecast = fit.forecast(steps=config["horizon"])
        details = self._extract_details(fit, trend=trend, seasonal=seasonal)
        return forecast.values, details

    def _run_prophet(
        self,
        series: pd.Series,
        config: Dict[str, Any],
        cache_context: Optional[Dict[str, Optional[str]]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        df = series.reset_index()
        df.columns = ["ds", "y"]
        ds = pd.to_datetime(df["ds"], errors="coerce")
        if ds.isna().any():
            fallback_index = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=len(df), freq="D")
            ds = fallback_index
        df["ds"] = ds
        prophet = Prophet()
        prophet.fit(df)
        future = prophet.make_future_dataframe(periods=config["horizon"])
        forecast = prophet.predict(future)["yhat"].iloc[-config["horizon"] :].values
        details = {"method": "Prophet", "components": list(prophet.component_cols)}
        return forecast, details

    def _run_holt_winters(
        self,
        series: pd.Series,
        config: Dict[str, Any],
        cache_context: Optional[Dict[str, Optional[str]]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        smoothing_level = config.get("params", {}).get("smoothing_level")
        smoothing_slope = config.get("params", {}).get("smoothing_slope")
        smoothing_seasonal = config.get("params", {}).get("smoothing_seasonal")
        seasonal_periods = config.get("seasonal_periods")
        model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal=config.get("params", {}).get("seasonal", "add") if seasonal_periods else None,
            seasonal_periods=seasonal_periods if seasonal_periods else None,
        )
        fit = model.fit(
            smoothing_level=smoothing_level,
            smoothing_slope=smoothing_slope,
            smoothing_seasonal=smoothing_seasonal,
        )
        forecast = fit.forecast(steps=config["horizon"])
        details = self._extract_details(
            fit,
            smoothing_level=smoothing_level,
            smoothing_slope=smoothing_slope,
            seasonal=config.get("params", {}).get("seasonal"),
        )
        return forecast.values, details

    def _run_auto_arima(
        self,
        series: pd.Series,
        config: Dict[str, Any],
        cache_context: Optional[Dict[str, Optional[str]]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        best = None
        best_aic = float("inf")
        search_space = [(p, d, q) for p in range(0, 3) for d in range(0, 2) for q in range(0, 3)]
        for order in search_space:
            try:
                model = ARIMA(series, order=order).fit()
            except Exception:
                continue
            if model.aic < best_aic:
                best_aic = model.aic
                best = model
                best_order = order
        if best is None:
            raise RuntimeError("Unable to fit AutoARIMA model for the provided series.")
        forecast = best.forecast(steps=config["horizon"])
        details = self._extract_details(best, order=best_order)
        details["aic"] = float(best_aic)
        return forecast.values, details

    def _run_ml_forecast(
        self,
        series: pd.Series,
        config: Dict[str, Any],
        cache_context: Optional[Dict[str, Optional[str]]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        dataset_id = cache_context.get("dataset_id") if cache_context else None
        file_hash = cache_context.get("file_hash") if cache_context else None
        method = config["method"]
        lookback = config.get("ml_lookback", 6)
        series_signature = self._hash_series(series)
        features_key = self._build_cache_key(
            namespace="forecasting",
            dataset_id=dataset_id,
            component="ml_lag_features",
            config=config,
            file_hash=file_hash,
            extra_signature=f"{series_signature}-{lookback}",
        )
        features_payload = self._caching_service.get_data_payload(
            features_key, config, dataset_id, file_hash
        )
        if features_payload:
            features = features_payload.value["features"]
            targets = features_payload.value["targets"]
        else:
            features, targets = self._build_lag_features(series, lookback)
            self._caching_service.set_data(
                features_key,
                {"features": features, "targets": targets},
                config,
                dataset_id,
                file_hash,
            )

        future_key = self._build_cache_key(
            namespace="forecasting",
            dataset_id=dataset_id,
            component="ml_future_frame",
            config=config,
            file_hash=file_hash,
            extra_signature=f"{series_signature}-{lookback}-{config['horizon']}",
        )
        future_payload = self._caching_service.get_data_payload(
            future_key, config, dataset_id, file_hash
        )
        if future_payload:
            future_feature = future_payload.value
        else:
            future_feature = self._build_future_frame(series, lookback, config["horizon"])
            self._caching_service.set_data(
                future_key,
                future_feature,
                config,
                dataset_id,
                file_hash,
            )

        estimator_signature = self._build_model_signature(
            series_signature, lookback, config["params"], method
        )
        model_key = self._build_cache_key(
            namespace="forecasting",
            dataset_id=dataset_id,
            component="ml_model",
            config=config,
            file_hash=file_hash,
            extra_signature=estimator_signature,
        )
        estimator = self._caching_service.get_resource(
            model_key, config, dataset_id, file_hash
        )
        if estimator is None:
            regressor = XGBRegressor if method == "xgboost" else RandomForestRegressor
            estimator = regressor(**config.get("params", {}))
            estimator.fit(features, targets)
            self._caching_service.set_resource(
                model_key, estimator, config, dataset_id, file_hash
            )
        forecast = estimator.predict(future_feature)
        details = {
            "method": estimator.__class__.__name__,
            "lookback": lookback,
            "n_samples": len(targets),
        }
        return forecast, details

    def _build_model_signature(
        self,
        series_signature: str,
        lookback: int,
        params: Dict[str, Any],
        method: str,
    ) -> str:
        normalized = normalize_config({
            "series_signature": series_signature,
            "lookback": lookback,
            "params": params,
            "method": method,
        })
        return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode("utf-8")).hexdigest()

    def _hash_series(self, series: pd.Series) -> str:
        cleaned = series.dropna()
        if cleaned.empty:
            return "empty_series"
        payload = cleaned.astype(float, errors="ignore").to_numpy().tobytes()
        meta = f"{len(cleaned)}-{cleaned.index.min()}-{cleaned.index.max()}".encode("utf-8")
        return hashlib.sha256(payload + meta).hexdigest()

    def _build_cache_context(
        self, dataset_id: Optional[str], file_hash: Optional[str]
    ) -> Dict[str, Optional[str]]:
        return {"dataset_id": dataset_id, "file_hash": file_hash}

    def _build_cache_key(
        self,
        namespace: str,
        dataset_id: Optional[str],
        component: str,
        config: Dict[str, Any],
        extra_signature: Optional[str] = None,
        file_hash: Optional[str] = None,
    ) -> str:
        return self._caching_service.build_key(
            namespace, dataset_id, component, config, extra_signature, file_hash
        )

    def _build_session_artifacts(
        self,
        result: Dict[str, Any],
        config: Dict[str, Any],
        dataset_id: Optional[str],
        file_hash: Optional[str],
    ) -> Dict[str, bytes]:
        payload = {
            "model": result["model"],
            "horizon": result["horizon"],
            "forecast": result["forecast"],
            "metadata": result["metadata"],
            "model_details": result.get("model_details", {}),
        }
        config_payload = {
            "dataset_id": dataset_id,
            "dataset_file_hash": file_hash,
            "config": config,
            "run_id": build_run_id(file_hash, compute_config_signature(config)),
        }
        return {
            "forecast_payload.json": self._serialize_to_bytes(payload),
            "configuration.json": self._serialize_to_bytes(config_payload),
        }

    def _serialize_to_bytes(self, payload: Any) -> bytes:
        normalized = json.dumps(payload, indent=2, sort_keys=True, default=str)
        return normalized.encode("utf-8")
