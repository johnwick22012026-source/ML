"""Export service building artifact descriptors."""

from typing import Any, Dict

__all__ = ["ExportService"]


class ExportService:
    """Entry point for creating downloadable artifacts in memory."""

    def create(self, payload: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Encodes a payload and returns an artifact descriptor."""
        artifact_id = config.get("artifact_id", "default-artifact")
        return {
            "artifact_id": artifact_id,
            "payload_summary": str(payload)[:128],
            "config": config,
        }
