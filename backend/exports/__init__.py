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
    "DiagnosticArtifactBundle",
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


@dataclass
class DiagnosticArtifactBundle:
    run_id: str
    dataset_id: str
    config_signature: str
    metrics_summary: ArtifactDescriptor
    diagnostics_overview: ArtifactDescriptor
    visualization_data: List[ArtifactDescriptor]
    supporting_artifacts: List[ArtifactDescriptor]


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

    def package_diagnostic_artifacts(
        self,
        run_id: str,
        dataset_id: str,
        config_signature: str,
        metrics_summary: Dict[str, Any],
        diagnostics: Dict[str, Any],
        visualization_payloads: Optional[Dict[str, Any]] = None,
        supporting_payloads: Optional[Dict[str, Any]] = None,
    ) -> DiagnosticArtifactBundle:
        """Registers diagnostics and visualization payloads for download within the session."""
        base_metadata = {
            "run_id": run_id,
            "dataset_id": dataset_id,
            "config_signature": config_signature,
            "category": "diagnostics",
        }

        metrics_descriptor = self.create(
            payload=self._dump_json({"metrics": metrics_summary, "run_id": run_id}),
            artifact_type="diagnostic-metrics",
            file_name=f"diagnostic_metrics_{run_id}.json",
            metadata={**base_metadata, "type": "metrics"},
        )

        diagnostics_descriptor = self.create(
            payload=self._dump_json({"diagnostics": diagnostics, "run_id": run_id}),
            artifact_type="diagnostic-overview",
            file_name=f"diagnostic_overview_{run_id}.json",
            metadata={**base_metadata, "type": "diagnostics"},
        )

        visualization_payloads = visualization_payloads or {}
        visualization_descriptors: List[ArtifactDescriptor] = []
        for label, payload in sorted(visualization_payloads.items()):
            descriptor = self.create(
                payload=self._dump_json({"label": label, "payload": payload, "run_id": run_id}),
                artifact_type="diagnostic-visualization",
                file_name=f"visualization_{label}_{run_id}.json",
                metadata={
                    **base_metadata,
                    "type": "visualization",
                    "label": label,
                },
            )
            visualization_descriptors.append(descriptor)

        supporting_payloads = supporting_payloads or {}
        supporting_descriptors: List[ArtifactDescriptor] = []
        for label, payload in sorted(supporting_payloads.items()):
            descriptor = self.create(
                payload=self._dump_json({"label": label, "payload": payload, "run_id": run_id}),
                artifact_type="diagnostic-support",
                file_name=f"supporting_{label}_{run_id}.json",
                metadata={
                    **base_metadata,
                    "type": "supporting",
                    "label": label,
                },
            )
            supporting_descriptors.append(descriptor)

        return DiagnosticArtifactBundle(
            run_id=run_id,
            dataset_id=dataset_id,
            config_signature=config_signature,
            metrics_summary=metrics_descriptor,
            diagnostics_overview=diagnostics_descriptor,
            visualization_data=visualization_descriptors,
            supporting_artifacts=supporting_descriptors,
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
