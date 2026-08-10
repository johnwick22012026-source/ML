"""Machine learning service with a basic fit/predict simulation."""

from typing import Any, Dict, List

__all__ = ["MLService"]


class MLService:
    """Entry point for ML model training and inference."""

    def fit(self, features: Any, labels: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Simulates training by returning metadata about the dataset."""
        model = {
            "trained_on": len(features) if isinstance(features, (list, tuple)) else 1,
            "label_shape": len(labels) if isinstance(labels, (list, tuple)) else 1,
            "config": config,
        }

        return model

    def predict(self, model: Any, features: Any) -> List[Any]:
        """Returns a list of placeholder predictions based on feature count."""
        count = len(features) if isinstance(features, (list, tuple)) else 1
        return [model.get("config", {}).get("default_prediction", 0)] * count
