"""Validation service implementation with basic schema checks."""

from typing import Any, Dict

__all__ = ["ValidationService"]


class ValidationService:
    """Entry point for dataset validation operations."""

    def validate(self, data: Any, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validates that required keys exist and match expected types."""
        results: Dict[str, Any] = {
            "is_valid": True,
            "errors": [],
        }

        if not isinstance(data, dict):
            results["is_valid"] = False
            results["errors"].append("Data must be a dictionary.")
            return results

        for key, expected in schema.items():
            if key not in data:
                results["is_valid"] = False
                results["errors"].append(f"Missing key: {key}")
                continue

            value = data[key]
            if expected and not isinstance(value, expected):
                results["is_valid"] = False
                results["errors"].append(
                    f"Key '{key}' expected type {expected.__name__}, got {type(value).__name__}"
                )

        return results
