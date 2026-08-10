"""Export service building artifact descriptors."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

__all__ = ["ExportService","ArtifactDescriptor"]


@dataclass
class ArtifactDescriptor:
    artifact_id: str
    artifact_type: str
    file_name: Optional[str]
    payload: Any
    payload_summary: str
    created_at: str
    metadata: Dict[str, Any]


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

    def list_artifacts(self) -> List[ArtifactDescriptor]:
        """Returns all registered artifacts for the current session."""
        return list(self._artifacts.values())

    def get_artifact(self, artifact_id: str) -> Optional[ArtifactDescriptor]:
        """Retrieves a descriptor by id."""
        return self._artifacts.get(artifact_id)

    def clear_artifacts(self) -> None:
        """Dropping artifacts ensures every run stays stateless."""
        self._artifacts.clear()
