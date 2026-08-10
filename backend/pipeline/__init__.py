"""Orchestration layer that routes ML tasks to their proper pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

__all__ = [
    "PipelineService",
    "PipelineExecutionRecord",
    "RuntimeTaskSwitchValidator",
]


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
        payload["steps"] = [
            "ingestion",
            "preprocessing",
            "feature_engineering",
            "clustering",
            "interpretation",
        ]
        return payload

    def _run_forecasting_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        payload = self._base_pipeline_payload(config)
        payload["pipeline"] = "forecasting"
        payload["steps"] = [
            "time_series_conversion",
            "feature_engineering",
            "forecast_execution",
            "evaluation",
        ]
        return payload

    def _run_anomaly_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        payload = self._base_pipeline_payload(config)
        payload["pipeline"] = "anomaly_detection"
        payload["steps"] = ["ingestion", "baseline_modeling", "anomaly_scoring", "alerting"]
        return payload


@dataclass
class PipelineExecutionRecord:
    """Stores the details for each pipeline run executed as part of validation."""

    task_type: str
    config: Dict[str, Any]
    result: Dict[str, Any]
    executed_at: datetime = field(default_factory=datetime.utcnow)


class RuntimeTaskSwitchValidator:
    """Validates the stateless pipeline service across repeated task switches."""

    def __init__(self, pipeline_service: PipelineService) -> None:
        self.pipeline_service = pipeline_service
        self.history: List[PipelineExecutionRecord] = []

    def validate_sequence(
        self,
        configs: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run a supplied sequence of task configs and record the outcomes."""
        summary: Dict[str, Any] = {
            "status": "success",
            "executions": [],
            "errors": [],
            "last_task": None,
        }

        for config in configs:
            task_type = (config.get("task_type") or "").strip().lower()
            if not task_type:
                error = "Empty task_type encountered in sequence."
                summary["errors"].append(error)
                summary["status"] = "failed"
                continue

            try:
                result = self.pipeline_service.execute(config)
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                summary["errors"].append(
                    {
                        "task_type": task_type,
                        "error": str(exc),
                    }
                )
                summary["status"] = "failed"
                continue

            record = PipelineExecutionRecord(task_type=task_type, config=config, result=result)
            self.history.append(record)

            pipeline_value = result.get("pipeline")
            if pipeline_value != task_type:
                summary["errors"].append(
                    {
                        "task_type": task_type,
                        "pipeline": pipeline_value,
                        "message": "Pipeline output mismatch",
                    }
                )
                summary["status"] = "failed"

            summary["executions"].append(
                {
                    "task_type": task_type,
                    "pipeline": pipeline_value,
                    "timestamp": record.executed_at.isoformat(),
                }
            )
            summary["last_task"] = task_type

        return summary

    def validate_all_supported_tasks(
        self,
        repeat_each: int = 2,
        overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run every supported task one or more times to simulate user-triggered switching."""
        configs: List[Dict[str, Any]] = []
        supported = self.pipeline_service.supported_tasks.keys()
        for task_type in supported:
            base_config = self._build_runtime_config(task_type, overrides or {})
            for _ in range(max(1, repeat_each)):
                configs.append(base_config)
        return self.validate_sequence(configs)

    def _build_runtime_config(
        self, task_type: str, overrides: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        normalized = task_type.strip().lower()
        payload = {"task_type": normalized}
        payload.update(overrides.get(normalized, {}))
        return payload

    def last_executed(self) -> Optional[PipelineExecutionRecord]:
        """Expose the most recent execution for further assertions."""
        if not self.history:
            return None
        return self.history[-1]
