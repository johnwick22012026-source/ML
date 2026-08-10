"""Simple ingestion service implementation for dataset payloads."""

import json
from typing import Any, Dict

__all__ = ["IngestionService"]


class IngestionService:
    """Entry point for lightweight dataset ingestion operations."""

    def ingest(self, dataset_id: str, payload: bytes, config: Dict[str, Any]) -> Dict[str, Any]:
        """Ingests raw payload bytes and returns a simple metadata record."""
        decoded_payload = None
        try:
            decoded_payload = payload.decode(config.get("encoding", "utf-8"))
        except (UnicodeDecodeError, AttributeError):
            decoded_payload = None

        summary: Dict[str, Any] = {
            "dataset_id": dataset_id,
            "size": len(payload),
            "ingestion_config": config,
            "decoded_payload": decoded_payload,
        }

        if config.get("parse_json", False) and decoded_payload:
            try:
                summary["parsed_payload"] = json.loads(decoded_payload)
            except json.JSONDecodeError:
                summary["parsed_payload_error"] = "Invalid JSON"

        return summary
