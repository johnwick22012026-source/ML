"""Preprocessing service implementation applying lightweight filters."""

from typing import Any, Dict, List

__all__ = ["PreprocessingService"]


class PreprocessingService:
    """Entry point for data preprocessing operations."""

    def run(self, data: Any, config: Dict[str, Any]) -> Any:
        """Applies configured filters and transformations to the input data."""
        if not isinstance(data, list):
            return data

        filters: List[str] = config.get("filters", [])
        transformed = data

        for entry_filter in filters:
            if entry_filter == "drop_null":
                transformed = [record for record in transformed if record is not None]
            elif entry_filter == "dedupe":
                seen = set()
                unique: List[Any] = []
                for record in transformed:
                    marker = tuple(sorted(record.items())) if isinstance(record, dict) else record
                    if marker not in seen:
                        seen.add(marker)
                        unique.append(record)
                transformed = unique

        return transformed
