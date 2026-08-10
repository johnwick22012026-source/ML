"""Forecasting service that executes configured forecasting jobs entirely in memory."""

from __future__ import annotations

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

    def forecast(self, history: TimeSeries, config: Dict[str, Any]) -> Dict[str, Any]:
        """Produce forecasts driven entirely by the supplied configuration."""
        runtime_config = self._normalize_config(config)
        method = runtime_config["method"]
        horizon = runtime_config["horizon"]
        metadata: Dict[str, Any] = {
            "requested_at": datetime.utcnow().isoformat() + "Z",
            "method": method,
            "horizon": horizon,
            "seed": runtime_config.get("seed"),
        }

        series = self._prepare_series(history, runtime_config)
        forecast_values, details = self._dispatch_method(series, runtime_config)
        metadata["trained_samples"] = len(series)

        result = ForecastResult(
            model=method,
            horizon=horizon,
            forecast=[float(v) for v in forecast_values],
            metadata=metadata,
            model_details=details,
        )
        return {
            "model": result.model,
            "horizon": result.horizon,
            "forecast": result.forecast,
            "metadata": result.metadata,
            "model_details": result.model_details,
        }

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

    def _prepare_series(self, history: TimeSeries, runtime_config: Dict[str, Any]) -> pd.Series:
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
        return series

    def _dispatch_method(
        self, series: pd.Series, runtime_config: Dict[str, Any]
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
        forecast_series, details = handler(series, runtime_config)
        flatten = np.asarray(forecast_series).ravel()
        if flatten.shape[0] < runtime_config["horizon"]:
            flatten = np.resize(flatten, runtime_config["horizon"])
        return flatten[-runtime_config["horizon"] :], details

    def _run_arima(self, series: pd.Series, config: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        order = config.get("order") or (1, 1, 0)
        model = ARIMA(series, order=order)
        fit = model.fit()
        forecast = fit.forecast(steps=config["horizon"])
        details = self._extract_details(fit, order=order)
        return forecast.values, details

    def _run_sarima(self, series: pd.Series, config: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        order = config.get("order") or (1, 1, 1)
        seasonal_order = config.get("seasonal_order") or (1, 1, 1, config.get("seasonal_periods") or 12)
        model = SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False)
        fit = model.fit(disp=False)
        forecast = fit.forecast(steps=config["horizon"])
        details = self._extract_details(fit, order=order, seasonal_order=seasonal_order)
        return forecast.values, details

    def _run_sarimax(self, series: pd.Series, config: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
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

    def _run_ets(self, series: pd.Series, config: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
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

    def _run_prophet(self, series: pd.Series, config: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
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
        self, series: pd.Series, config: Dict[str, Any]
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
        self, series: pd.Series, config: Dict[str, Any]
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
        self, series: pd.Series, config: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        method = config["method"]
        lookback = config.get("ml_lookback", 6)
        features, targets = self._build_lag_features(series, lookback)
        regressor = XGBRegressor if method == "xgboost" else RandomForestRegressor
        estimator = regressor(**config.get("params", {}))
        estimator.fit(features, targets)
        future_feature = self._build_future_frame(series, lookback, config["horizon"])
        forecast = estimator.predict(future_feature)
        details = {
            "method": regressor.__name__,
            "lookback": lookback,
            "n_samples": len(targets),
        }
        return forecast, details

    def _build_lag_features(self, series: pd.Series, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        values = series.dropna().astype(float).values
        if len(values) <= lookback:
            raise ValueError("History must have more records than the requested lookback window.")
        X = []
        y = []
        for i in range(lookback, len(values)):
            X.append(values[i - lookback : i])
            y.append(values[i])
        return np.vstack(X), np.array(y)

    def _build_future_frame(self, series: pd.Series, lookback: int, horizon: int) -> np.ndarray:
        values = series.dropna().astype(float).values
        if len(values) < lookback:
            raise ValueError("Not enough history to generate future lag features.")
        frame = []
        last_window = values[-lookback:].tolist()
        for _ in range(horizon):
            frame.append(last_window.copy())
            last_window = last_window[1:] + [last_window[-1]]
        return np.vstack(frame)

    def _sanitize_exogenous(self, exogenous: Any, horizon: int) -> Dict[str, Optional[np.ndarray]]:
        if exogenous is None:
            return {"history": None, "future": None}
        history = exogenous.get("history")
        future = exogenous.get("future")
        history_array = np.asarray(history) if history is not None else None
        future_array = np.asarray(future) if future is not None else None
        if future_array is not None and len(future_array) < horizon:
            raise ValueError("Provided exogenous future data is shorter than the horizon.")
        return {"history": history_array, "future": future_array}

    def _extract_details(self, fit: Any, **kwargs: Any) -> Dict[str, Any]:
        details = {
            "params": dict(kwargs),
            "aic": float(getattr(fit, "aic", np.nan)) if hasattr(fit, "aic") else None,
            "bic": float(getattr(fit, "bic", np.nan)) if hasattr(fit, "bic") else None,
        }
        return {k: v for k, v in details.items() if v is not None}
