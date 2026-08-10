"""Feature engineering service with simple derived field support."""

from typing import Any, Dict, List

__all__ = ["FeatureEngineeringService"]


class FeatureEngineeringService:
    """Entry point for feature engineering operations."""

    def transform(self, data: Any, config: Dict[str, Any]) -> Any:
        """Adds derived features based on the configured rules."""
        if not isinstance(data, list):
            return data

        derived: List[str] = config.get("derived_fields", [])
        transformed: List[Any] = []

        for item in data:
            if not isinstance(item, dict):
                transformed.append(item)
                continue

            enhanced = item.copy()
            for field in derived:
                if field not in enhanced:
                    enhanced[field] = config.get("default_value", None)
            transformed.append(enhanced)

        return transformed
