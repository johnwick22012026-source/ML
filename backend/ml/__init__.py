"""Machine learning service with configurable regression/classification training."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.svm import SVC, SVR

__all__ = ["MLService"]


FeatureMatrix = Union[List[List[float]], Sequence[Sequence[float]], np.ndarray]
TargetVector = Union[List[Any], Sequence[Any], np.ndarray]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator_cls: Any
    task_type: str


class ModelRegistry:
    """Registry of supported models for regression and classification."""

    _registry: Dict[str, ModelSpec] = {
        "linear_regression": ModelSpec(
            name="linear_regression",
            estimator_cls=LinearRegression,
            task_type="regression",
        ),
        "random_forest_regressor": ModelSpec(
            name="random_forest_regressor",
            estimator_cls=RandomForestRegressor,
            task_type="regression",
        ),
        "svr": ModelSpec(
            name="svr",
            estimator_cls=SVR,
            task_type="regression",
        ),
        "logistic_regression": ModelSpec(
            name="logistic_regression",
            estimator_cls=LogisticRegression,
            task_type="classification",
        ),
        "random_forest_classifier": ModelSpec(
            name="random_forest_classifier",
            estimator_cls=RandomForestClassifier,
            task_type="classification",
        ),
        "svc": ModelSpec(
            name="svc",
            estimator_cls=SVC,
            task_type="classification",
        ),
    }

    @classmethod
    def get(cls, name: str, task_type: str) -> ModelSpec:
        spec = cls._registry.get(name)
        if not spec or spec.task_type != task_type:
            raise ValueError(
                f"Model '{name}' not registered for task type '{task_type}'."
            )
        return spec

    @classmethod
    def available_models(cls, task_type: Optional[str] = None) -> List[str]:
        if task_type:
            return [name for name, spec in cls._registry.items() if spec.task_type == task_type]
        return list(cls._registry.keys())


class MLService:
    """Entry point for ML model training and inference."""

    def train_and_predict(
        self,
        config: Dict[str, Any],
        features: FeatureMatrix,
        labels: TargetVector,
    ) -> Dict[str, Any]:
        """Train configured models and return predictions with metadata."""
        runtime_config = self._normalize_config(config)
        task_type = runtime_config["task_type"]
        models = runtime_config["models"]

        X, y = self._prepare_data(features, labels)
        model_results = []

        for model_config in models:
            model_result = self._run_model(name=model_config["name"],
                                           task_type=task_type,
                                           hyperparameters=model_config.get("hyperparameters", {}),
                                           X=X,
                                           y=y)
            model_results.append(model_result)

        metadata = {
            "task_type": task_type,
            "model_count": len(model_results),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        return {"run_metadata": metadata, "models": model_results}

    def _run_model(
        self,
        name: str,
        task_type: str,
        hyperparameters: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        spec = ModelRegistry.get(name=name, task_type=task_type)
        estimator = spec.estimator_cls(**hyperparameters)
        estimator.fit(X, y)
        raw_predictions = estimator.predict(X)

        predictions = self._to_list(raw_predictions)
        training_metadata = self._build_metadata(X, y, task_type, raw_predictions)

        return {
            "name": name,
            "task_type": task_type,
            "hyperparameters": hyperparameters,
            "artifact": {
                "estimator": estimator,
                "model_class": estimator.__class__.__name__,
            },
            "predictions": predictions,
            "training_metadata": training_metadata,
        }

    def _build_metadata(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str,
        raw_predictions: np.ndarray,
    ) -> Dict[str, Any]:
        metrics: Dict[str, float] = {}
        if task_type == "regression":
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y, raw_predictions)))
        else:
            try:
                metrics["accuracy"] = float(accuracy_score(y, raw_predictions))
            except ValueError:
                metrics["accuracy"] = 0.0
        return {
            "trained_samples": len(X),
            "label_shape": len(y),
            "metrics": metrics,
        }

    def _prepare_data(
        self,
        features: FeatureMatrix,
        labels: TargetVector,
    ) -> Tuple[np.ndarray, np.ndarray]:
        X = np.array(features)
        y = np.array(labels)
        return X, y

    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        required_keys = {"task_type", "models"}
        if not config or not required_keys.issubset(config.keys()):
            raise ValueError(
                "Config must define 'task_type' and 'models'."
            )

        task_type = config["task_type"].lower()
        if task_type not in {"regression", "classification"}:
            raise ValueError("task_type must be 'regression' or 'classification'.")

        models = config["models"]
        if not isinstance(models, list) or not models:
            raise ValueError("'models' must be a non-empty list of model configs.")

        normalized_models = []
        for model in models:
            if not isinstance(model, dict) or "name" not in model:
                raise ValueError("Each model config must be a dict with a 'name'.")
            normalized_models.append({
                "name": model["name"],
                "hyperparameters": model.get("hyperparameters", {}),
            })

        return {"task_type": task_type, "models": normalized_models}

    @staticmethod
    def _to_list(values: Union[np.ndarray, Sequence[Any]]) -> List[Any]:
        if isinstance(values, np.ndarray):
            return values.tolist()
        if isinstance(values, (list, tuple)):
            return list(values)
        return [values]
