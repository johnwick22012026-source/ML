"""Configuration-driven orchestration for model optimization and comparison."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from backend.caching import CachingService
from backend.ml import MLService
from backend.utils.run_context import (
    build_run_id,
    compute_config_signature,
    normalize_config,
)


TrainerFn = Callable[
    [
        Dict[str, Any],
        np.ndarray,
        Optional[np.ndarray],
        str,
        Dict[str, Any],
        str,
    ],
    Dict[str, Any],
]


@dataclass(frozen=True)
class TrialResult:
    trial_id: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    status: str
    training_metadata: Dict[str, Any]


@dataclass(frozen=True)
class ModelComparisonEntry:
    name: str
    task_type: str
    trials: List[TrialResult]
    best_trial: TrialResult


class ModelOptimizationService:
    """Orchestrator that runs optimization + comparison workflows for configurable models."""

    def __init__(
        self,
        ml_service: MLService,
        caching_service: CachingService,
        trainer_handlers: Optional[Dict[str, TrainerFn]] = None,
    ) -> None:
        self._ml_service = ml_service
        self._caching_service = caching_service
        self._trainer_handlers: Dict[str, TrainerFn] = {}
        self._register_default_handlers()
        trainer_handlers = trainer_handlers or {}
        for task_type, handler in trainer_handlers.items():
            self.register_task_handler(task_type, handler)

    def register_task_handler(self, task_type: str, handler: TrainerFn) -> None:
        """Register a custom trainer for a new task type."""
        normalized = task_type.strip().lower()
        if not normalized:
            raise ValueError("task_type must be a non-empty string when registering a handler.")
        self._trainer_handlers[normalized] = handler

    def optimize(
        self,
        config: Dict[str, Any],
        features: Sequence[Sequence[Any]],
        targets: Optional[Sequence[Any]] = None,
        dataset_id: Optional[str] = None,
        file_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs the full optimization + comparison path and returns a structured payload."""
        runtime_config = config or {}
        normalized_config = normalize_config(runtime_config)
        task_type = self._task_type_from_config(runtime_config)
        if not task_type:
            raise ValueError("Configuration must include a task_type or task entry.")
        cache_key = self._caching_service.build_key(
            namespace="model_optimization",
            dataset_id=dataset_id,
            component="workflow",
            config=normalized_config,
            extra_signature=None,
            file_hash=file_hash,
        )

        cached = self._caching_service.get_data(
            key=cache_key,
            config=normalized_config,
            dataset_id=dataset_id,
            file_hash=file_hash,
        )
        if cached is not None:
            return cached

        features_matrix = self._prepare_matrix(features)
        targets_matrix = self._prepare_targets(targets)
        train_x, train_y, val_x, val_y = self._split_data(
            features_matrix, targets_matrix, runtime_config.get("split", {})
        )

        models_config = runtime_config.get("models")
        if not models_config or not isinstance(models_config, list):
            raise ValueError("'models' must be provided as a non-empty list in the configuration.")

        evaluation_config = runtime_config.get("evaluation", {}) or {}
        primary_metric_name, primary_metric_goal = self._determine_primary_metric(
            evaluation_config, task_type
        )
        metric_names = [m.lower() for m in evaluation_config.get("metrics", []) if isinstance(m, str)]
        if primary_metric_name not in metric_names:
            metric_names.insert(0, primary_metric_name)

        ranked_models: List[ModelComparisonEntry] = []
        for model_config in models_config:
            if not isinstance(model_config, dict) or "name" not in model_config:
                raise ValueError("Each model definition must be a dict with at least a 'name' entry.")
            entry = self._run_model_trials(
                model_config=model_config,
                task_type=task_type,
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y,
                evaluation=metric_names,
                primary_metric=primary_metric_name,
                trainer_handler=self._get_trainer(task_type),
            )
            ranked_models.append(entry)

        best_model = self._select_best_model(ranked_models, primary_metric_name, primary_metric_goal)

        run_metadata = {
            "task_type": task_type,
            "dataset_id": dataset_id,
            "run_id": build_run_id(file_hash, compute_config_signature(normalized_config)),
            "config_signature": compute_config_signature(normalized_config),
            "executed_at": datetime.utcnow().isoformat() + "Z",
        }

        payload = {
            "run_metadata": run_metadata,
            "models": [
                {
                    "name": model.name,
                    "task_type": model.task_type,
                    "trials": [
                        {
                            "trial_id": trial.trial_id,
                            "hyperparameters": trial.hyperparameters,
                            "metrics": trial.metrics,
                            "status": trial.status,
                            "training_metadata": trial.training_metadata,
                        }
                        for trial in model.trials
                    ],
                    "best_trial": {
                        "trial_id": model.best_trial.trial_id,
                        "hyperparameters": model.best_trial.hyperparameters,
                        "metrics": model.best_trial.metrics,
                        "status": model.best_trial.status,
                        "training_metadata": model.best_trial.training_metadata,
                    },
                }
                for model in ranked_models
            ],
            "best_model": {
                "model_name": best_model.name,
                "trial_id": best_model.best_trial.trial_id,
                "hyperparameters": best_model.best_trial.hyperparameters,
                "metrics": best_model.best_trial.metrics,
            },
        }

        self._caching_service.set_data(
            key=cache_key,
            value=payload,
            config=normalized_config,
            dataset_id=dataset_id,
            file_hash=file_hash,
        )

        return payload

    def _register_default_handlers(self) -> None:
        self.register_task_handler("regression", self._run_via_ml_service)
        self.register_task_handler("classification", self._run_via_ml_service)

    def _get_trainer(self, task_type: str) -> TrainerFn:
        normalized = task_type.strip().lower()
        handler = self._trainer_handlers.get(normalized)
        if handler is None:
            raise ValueError(
                f"No trainer registered for task_type '{task_type}'."
            )
        return handler

    def _run_via_ml_service(
        self,
        model_config: Dict[str, Any],
        train_x: np.ndarray,
        train_y: Optional[np.ndarray],
        task_type: str,
        runtime_config: Dict[str, Any],
        trial_id: str,
    ) -> Dict[str, Any]:
        runtime_payload = {
            "task_type": task_type,
            "models": [
                {
                    "name": model_config["name"],
                    "hyperparameters": model_config.get("hyperparameters", {}),
                }
            ],
        }
        models_output = self._ml_service.train_and_predict(
            runtime_payload,
            train_x,
            train_y,
        )
        model_output = models_output["models"][0]
        return {
            "estimator": model_output["artifact"]["estimator"],
            "training_metadata": model_output["training_metadata"],
            "training_predictions": model_output["predictions"],
        }

    def _run_model_trials(
        self,
        *,
        model_config: Dict[str, Any],
        task_type: str,
        train_x: np.ndarray,
        train_y: Optional[np.ndarray],
        val_x: Optional[np.ndarray],
        val_y: Optional[np.ndarray],
        evaluation: List[str],
        primary_metric: str,
        trainer_handler: TrainerFn,
    ) -> ModelComparisonEntry:
        base_hyperparameters = model_config.get("hyperparameters", {}) or {}
        search_space = model_config.get("search_space", {}) or {}
        trials_list: List[TrialResult] = []
        trial_combinations = self._enumerate_search_space(base_hyperparameters, search_space)
        for idx, hyperparameters in enumerate(trial_combinations, start=1):
            trial_id = f"{model_config['name']}_trial_{idx}"
            merged_model_config = {
                "name": model_config["name"],
                "hyperparameters": hyperparameters,
            }
            training_response = trainer_handler(
                merged_model_config,
                train_x,
                train_y,
                task_type,
                model_config,
                trial_id,
            )
            estimator = training_response.get("estimator")
            validation_metrics = self._compute_metrics(
                predictions=self._safe_predict(estimator, val_x),
                targets=self._to_list(val_y) if val_y is not None else [],
                metric_names=evaluation,
            )
            trial_result = TrialResult(
                trial_id=trial_id,
                hyperparameters=hyperparameters,
                metrics=validation_metrics,
                status="completed",
                training_metadata=training_response.get("training_metadata", {}),
            )
            trials_list.append(trial_result)

        if not trials_list:
            raise RuntimeError(
                f"No trials were generated for model '{model_config['name']}'."
            )

        best_trial = self._select_best_trial(trials_list, primary_metric)
        return ModelComparisonEntry(
            name=model_config["name"],
            task_type=task_type,
            trials=trials_list,
            best_trial=best_trial,
        )

    def _determine_primary_metric(
        self,
        evaluation_config: Dict[str, Any],
        task_type: str,
    ) -> Tuple[str, str]:
        primary = evaluation_config.get("primary_metric") or {}
        name = primary.get("name")
        goal = primary.get("goal", "min").lower()
        if not name:
            name = evaluation_config.get("goal", None)
        if not name:
            fallback = {
                "regression": ("rmse", "min"),
                "classification": ("accuracy", "max"),
            }
            name, goal = fallback.get(task_type, ("rmse", "min"))
        return name.lower(), goal if goal in {"min", "max"} else "min"

    def _compute_metrics(
        self,
        predictions: Sequence[Any],
        targets: Sequence[Any],
        metric_names: Iterable[str],
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for metric in metric_names:
            normalized = metric.lower()
            if not targets or len(targets) == 0 or not predictions:
                metrics[normalized] = 0.0
                continue
            try:
                if normalized == "rmse":
                    metrics[normalized] = float(
                        mean_squared_error(targets, predictions, squared=False)
                    )
                elif normalized == "mse":
                    metrics[normalized] = float(mean_squared_error(targets, predictions))
                elif normalized == "mae":
                    metrics[normalized] = float(mean_absolute_error(targets, predictions))
                elif normalized == "r2":
                    metrics[normalized] = float(r2_score(targets, predictions))
                elif normalized == "accuracy":
                    metrics[normalized] = float(accuracy_score(targets, predictions))
                else:
                    metrics[normalized] = 0.0
            except ValueError:
                metrics[normalized] = 0.0
        return metrics

    def _select_best_trial(self, trials: List[TrialResult], primary_metric: str) -> TrialResult:
        primary_metric = primary_metric.lower()
        def score_key(trial: TrialResult) -> float:
            metrics = trial.metrics
            return metrics.get(primary_metric, float("inf"))

        minimized = primary_metric in {"rmse", "mse", "mae"}
        if minimized:
            best = min(trials, key=score_key)
        else:
            best = max(trials, key=score_key)
        return best

    def _select_best_model(
        self,
        models: List[ModelComparisonEntry],
        primary_metric: str,
        goal: str,
    ) -> ModelComparisonEntry:
        best: Optional[ModelComparisonEntry] = None
        for entry in models:
            if best is None:
                best = entry
                continue
            current_value = entry.best_trial.metrics.get(primary_metric, 0.0)
            best_value = best.best_trial.metrics.get(primary_metric, 0.0)
            if goal == "min" and current_value < best_value:
                best = entry
            elif goal == "max" and current_value > best_value:
                best = entry
        if best is None:
            raise RuntimeError("Unable to determine best model from the supplied trials.")
        return best

    def _enumerate_search_space(
        self, base: Dict[str, Any], space: Dict[str, Sequence[Any]]
    ) -> List[Dict[str, Any]]:
        if not space:
            return [dict(base)]
        keys = sorted(space.keys())
        expanded = []
        iterables = [
            space[key]
            if isinstance(space[key], Iterable) and not isinstance(space[key], (str, bytes))
            else [space[key]]
            for key in keys
        ]
        if iterables:
            for combo in itertools.product(*iterables):
                entry = dict(base)
                entry.update({k: v for k, v in zip(keys, combo)})
                expanded.append(entry)
        return expanded or [dict(base)]

    def _split_data(
        self,
        features: np.ndarray,
        targets: Optional[np.ndarray],
        split_config: Dict[str, Any],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        if targets is None or len(targets) == 0:
            return features, targets, None, None

        strategy = (split_config.get("strategy") or "train_test").lower()
        if strategy == "none":
            return features, targets, None, None

        test_size = float(split_config.get("test_size", 0.2))
        shuffle = bool(split_config.get("shuffle", True))
        random_state = split_config.get("random_state")
        stratify = targets if split_config.get("stratify", False) else None
        if stratify is not None and len(np.unique(stratify)) < 2:
            stratify = None

        if len(targets) < 2 or not 0 < test_size < 1:
            return features, targets, None, None

        train_x, val_x, train_y, val_y = train_test_split(
            features,
            targets,
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state,
            stratify=stratify,
        )
        return train_x, train_y, val_x, val_y

    @staticmethod
    def _prepare_matrix(features: Sequence[Sequence[Any]]) -> np.ndarray:
        matrix = np.asarray(features)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        return matrix

    @staticmethod
    def _prepare_targets(targets: Optional[Sequence[Any]]) -> Optional[np.ndarray]:
        if targets is None:
            return None
        vector = np.asarray(targets)
        if vector.ndim > 1:
            vector = vector.ravel()
        return vector

    @staticmethod
    def _to_list(values: Optional[np.ndarray]) -> List[Any]:
        if values is None:
            return []
        if isinstance(values, np.ndarray):
            return values.tolist()
        if isinstance(values, (list, tuple)):
            return list(values)
        return [values]

    @staticmethod
    def _safe_predict(estimator: Any, features: Optional[np.ndarray]) -> List[Any]:
        if features is None:
            return []
        try:
            raw = estimator.predict(features)
            if isinstance(raw, np.ndarray):
                return raw.tolist()
            if isinstance(raw, (list, tuple)):
                return list(raw)
            return [raw]
        except Exception:
            return []

    @staticmethod
    def _task_type_from_config(config: Dict[str, Any]) -> str:
        task_type = config.get("task_type") or config.get("task")
        if not task_type:
            return ""
        return str(task_type).strip().lower()
