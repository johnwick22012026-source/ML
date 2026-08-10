"""Forecasting service with a simple last-value propagation strategy."""

from typing import Any, Dict, List

__all__ = ["ForecastingService"]


class ForecastingService:
    """Entry point for time series forecasting operations."""

    def forecast(self, history: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Returns a basic forecast that replicates the last known value."""
        forecast_horizon = config.get("horizon", 1)
        next_values: List[Any] = []

        if isinstance(history, list) and history:
            last_value = history[-1]
            next_values = [last_value] * forecast_horizon
        else:
            next_values = [config.get("default", 0)] * forecast_horizon

        return {"forecast": next_values, "horizon": forecast_horizon}
