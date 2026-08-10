"""Orchestration layer that routes ML tasks to their proper pipelines."""

from __future__ import annotations

from typing import Any, Callable, Dict

__all__ = ["PipelineService"]


class PipelineService:
    """Routes configuration-driven task execution to the correct pipeline."""

    def __init__(
        self,
        handlers: Optional[Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]] = None,
    ) -> None:
        self._handlers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        self._register_default_handlers()
        if handlers:
            for task_type, handler in handlers.items():
                self.register_task(task_type, handler)

    def register_task(
        self, task_type: str, handler: Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> None:
        """Register or override a task pipeline handler."""
        normalized = task_type.strip().lower()
        if not normalized:
            raise ValueError("task_type must be a non-empty string")
        self._handlers[normalized] = handler

    def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the pipeline associated with the configured task type."""
        task_type = (config.get("task_type") or "").strip().lower()
        if not task_type:
            raise ValueError("Configuration must include a non-empty task_type session parameter.")

        handler = self._handlers.get(task_type)
        if handler is None:
            supported = ", ".join(sorted(self._handlers.keys()))
            raise ValueError(
                f"Unknown task_type '{config.get('task_type')}'. Supported tasks: {supported}."
            )

        return handler(config)

    @property
    def supported_tasks(self) -> Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]:
        """Expose the currently registered tasks for introspection."""
        return dict(self._handlers)

    def _register_default_handlers(self) -> None:
        self.register_task("regression", self._run_regression_pipeline)
        self.register_task("classification", self._run_classification_pipeline)
        self.register_task("clustering", self._run_clustering_pipeline)
        self.register_task("forecasting", self._run_forecasting_pipeline)
        self.register_task("anomaly_detection", self._run_anomaly_pipeline)

    def _base_pipeline_payload(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task_type": config.get("task_type"),
            "config_snapshot": config,
            "status": "executed",
        }

    def _run_regression_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        payload = self._base_pipeline_payload(config)
        payload["pipeline"] = "regression"
        payload["steps"] = ["ingestion", "validation", "preprocessing", "modeling", "evaluation"]
        return payload

    def _run_classification_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        payload = self._base_pipeline_payload(config)
        payload["pipeline"] = "classification"
        payload["steps"] = ["ingestion", "balancing", "preprocessing", "modeling", "scoring"]
        return payload

    def _run_clustering_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        payload = self._base_pipeline_payload(config)
        payload["pipeline"] = "clustering"
        payload["steps"] = ["ingestion", "preprocessing", "feature_engineering", "clustering", "interpretation"]
        return payload

    def _run_forecasting_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        payload = self._base_pipeline_payload(config)
        payload["pipeline"] = "forecasting"
        payload["steps"] = ["time_series_conversion", "feature_engineering", "forecast_execution", "evaluation"]
        return payload

    def _run_anomaly_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        payload = self._base_pipeline_payload(config)
        payload["pipeline"] = "anomaly_detection"
        payload["steps"] = ["ingestion", "baseline_modeling", "anomaly_scoring", "alerting"]
        return payload
