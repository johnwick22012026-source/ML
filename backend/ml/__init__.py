"""Machine learning service with configurable regression/classification training."""

from __future__ import annotations

import json
import pickle
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

from backend.exports import ArtifactDescriptor, ExportService

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
        export_service: Optional[ExportService] = None,
    ) -> Dict[str, Any]:
        """Train configured models and return predictions with metadata."""
        runtime_config = self._normalize_config(config)
        task_type = runtime_config["task_type"]
        models = runtime_config["models"]

        X, y = self._prepare_data(features, labels)
        model_results = []

        for model_config in models:
            model_result = self._run_model(
                name=model_config["name"],
                task_type=task_type,
                hyperparameters=model_config.get("hyperparameters", {}),
                X=X,
                y=y,
                export_service=export_service,
            )
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
        export_service: Optional[ExportService],
    ) -> Dict[str, Any]:
        spec = ModelRegistry.get(name=name, task_type=task_type)
        estimator = spec.estimator_cls(**hyperparameters)
        estimator.fit(X, y)
        raw_predictions = estimator.predict(X)

        predictions = self._to_list(raw_predictions)
        training_metadata = self._build_metadata(X, y, task_type, raw_predictions)
        artifacts = self._build_artifacts(
            export_service=export_service,
            estimator=estimator,
            predictions=predictions,
            task_type=task_type,
            model_name=name,
        )

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
            "artifacts": artifacts,
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

    def _build_artifacts(
        self,
        export_service: Optional[ExportService],
        estimator: Any,
        predictions: List[Any],
        task_type: str,
        model_name: str,
    ) -> Dict[str, Any]:
        serialized_model = self._serialize_estimator(estimator)
        serialized_predictions = self._serialize_predictions(predictions)

        model_artifact: Dict[str, Any] = {
            "payload": serialized_model,
            "payload_format": "pickle",
            "file_name": f"{model_name}_model.pkl",
            "artifact_descriptor": None,
        }
        prediction_artifact: Dict[str, Any] = {
            "payload": serialized_predictions,
            "payload_format": "json",
            "file_name": f"{model_name}_predictions.json",
            "artifact_descriptor": None,
        }

        if export_service:
            model_descriptor = self._register_artifact(
                export_service=export_service,
                payload=serialized_model,
                artifact_type="model",
                file_name=model_artifact["file_name"],
                metadata={
                    "model_name": model_name,
                    "task_type": task_type,
                    "model_class": estimator.__class__.__name__,
                },
            )
            prediction_descriptor = self._register_artifact(
                export_service=export_service,
                payload=serialized_predictions,
                artifact_type="predictions",
                file_name=prediction_artifact["file_name"],
                metadata={
                    "model_name": model_name,
                    "task_type": task_type,
                    "prediction_count": len(predictions),
                },
            )

            if model_descriptor:
                model_artifact["artifact_descriptor"] = model_descriptor
                model_artifact["payload"] = None
            if prediction_descriptor:
                prediction_artifact["artifact_descriptor"] = prediction_descriptor
                prediction_artifact["payload"] = None

        return {
            "model": model_artifact,
            "predictions": prediction_artifact,
        }

    def _serialize_estimator(self, estimator: Any) -> bytes:
        return pickle.dumps(estimator)

    def _serialize_predictions(self, predictions: Sequence[Any]) -> bytes:
        return json.dumps({"predictions": predictions}, default=str).encode("utf-8")

    def _register_artifact(
        self,
        export_service: ExportService,
        payload: bytes,
        artifact_type: str,
        file_name: str,
        metadata: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        try:
            descriptor = export_service.create(
                payload=payload,
                artifact_type=artifact_type,
                file_name=file_name,
                metadata=metadata,
            )
            return self._descriptor_to_dict(descriptor)
        except Exception:
            return None

    def _descriptor_to_dict(self, descriptor: ArtifactDescriptor) -> Dict[str, Any]:
        return {
            "artifact_id": descriptor.artifact_id,
            "artifact_type": descriptor.artifact_type,
            "file_name": descriptor.file_name,
            "payload_summary": descriptor.payload_summary,
            "created_at": descriptor.created_at,
            "metadata": descriptor.metadata,
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
