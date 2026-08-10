"""Export service building artifact descriptors."""

from dataclasses import dataclass
from datetime import datetime
import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

__all__ = [
    "ExportService",
    "ArtifactDescriptor",
    "EvaluationArtifactBundle",
]


@dataclass
class ArtifactDescriptor:
    artifact_id: str
    artifact_type: str
    file_name: Optional[str]
    payload: Any
    payload_summary: str
    created_at: str
    metadata: Dict[str, Any]


@dataclass
class EvaluationArtifactBundle:
    run_id: str
    task_type: str
    metrics_summary: ArtifactDescriptor
    comparison_table: ArtifactDescriptor
    configuration: ArtifactDescriptor
    pipeline_metadata: ArtifactDescriptor
    task_specific_outputs: List[ArtifactDescriptor]


class ExportService:
    """Entry point for creating downloadable artifacts in memory."""

    def __init__(self) -> None:
        self._artifacts: Dict[str, ArtifactDescriptor] = {}

    def create(
        self,
        payload: Any,
        artifact_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        file_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ArtifactDescriptor:
        """Registers an artifact payload for downstream download."""
        artifact_id = metadata.get("artifact_id") if metadata else None
        artifact_id = artifact_id or config.get("artifact_id") if config else artifact_id
        artifact_id = artifact_id or str(uuid4())
        descriptor = ArtifactDescriptor(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            file_name=file_name,
            payload=payload,
            payload_summary=str(payload)[:128],
            created_at=datetime.utcnow().isoformat(),
            metadata=metadata or {},
        )
        self._artifacts[artifact_id] = descriptor
        return descriptor

    def package_evaluation_artifacts(
        self,
        run_id: str,
        task_type: str,
        metrics_summary: Dict[str, float],
        comparison_table: List[Dict[str, Any]],
        configuration: Dict[str, Any],
        pipeline_metadata: Dict[str, Any],
        task_specific_outputs: Optional[Dict[str, Any]] = None,
    ) -> EvaluationArtifactBundle:
        """
        Packages every evaluation artifact generated during a run.

        All payloads stay in memory and are available for downstream download urls.
        """
        metrics_payload = self._dump_json(
            {"run_id": run_id, "task_type": task_type, "metrics": metrics_summary}
        )
        comparison_payload = self._dump_json(
            {
                "run_id": run_id,
                "task_type": task_type,
                "comparison": comparison_table,
            }
        )
        configuration_payload = self._dump_json(
            {"run_id": run_id, "task_type": task_type, "configuration": configuration}
        )
        pipeline_payload = self._dump_json(
            {
                "run_id": run_id,
                "task_type": task_type,
                "pipeline_metadata": pipeline_metadata,
            }
        )

        metrics_descriptor = self.create(
            payload=metrics_payload,
            artifact_type="evaluation-metrics",
            file_name=f"metrics_summary_{run_id}.json",
            metadata={"run_id": run_id, "task_type": task_type, "category": "metrics"},
        )
        comparison_descriptor = self.create(
            payload=comparison_payload,
            artifact_type="model-comparison",
            file_name=f"model_comparison_{run_id}.json",
            metadata={"run_id": run_id, "task_type": task_type, "category": "comparison"},
        )
        configuration_descriptor = self.create(
            payload=configuration_payload,
            artifact_type="configuration",
            file_name=f"configuration_{run_id}.json",
            metadata={"run_id": run_id, "task_type": task_type, "category": "configuration"},
        )
        pipeline_descriptor = self.create(
            payload=pipeline_payload,
            artifact_type="pipeline-metadata",
            file_name=f"pipeline_metadata_{run_id}.json",
            metadata={"run_id": run_id, "task_type": task_type, "category": "pipeline"},
        )

        task_specific_descriptors: List[ArtifactDescriptor] = []
        task_specific_outputs = task_specific_outputs or {}
        for label, output in sorted(task_specific_outputs.items()):
            descriptor = self.create(
                payload=self._dump_json({"label": label, "value": output, "run_id": run_id}),
                artifact_type=f"task-output-{task_type}",
                file_name=f"{label}_{run_id}.json",
                metadata={
                    "run_id": run_id,
                    "task_type": task_type,
                    "label": label,
                    "category": "task_specific",
                },
            )
            task_specific_descriptors.append(descriptor)

        return EvaluationArtifactBundle(
            run_id=run_id,
            task_type=task_type,
            metrics_summary=metrics_descriptor,
            comparison_table=comparison_descriptor,
            configuration=configuration_descriptor,
            pipeline_metadata=pipeline_descriptor,
            task_specific_outputs=task_specific_descriptors,
        )

    @staticmethod
    def _dump_json(payload: Any) -> str:
        try:
            return json.dumps(payload, indent=2, default=str)
        except TypeError:
            return json.dumps(payload, default=str)

    def list_artifacts(self) -> List[ArtifactDescriptor]:
        """Returns all registered artifacts for the current session."""
        return list(self._artifacts.values())

    def get_artifact(self, artifact_id: str) -> Optional[ArtifactDescriptor]:
        """Retrieves a descriptor by id."""
        return self._artifacts.get(artifact_id)

    def clear_artifacts(self) -> None:
        """Dropping artifacts ensures every run stays stateless."""
        self._artifacts.clear()
