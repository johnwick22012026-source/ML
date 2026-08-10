"""EDA service providing lightweight data summaries."""

from typing import Any, Dict, List

__all__ = ["EDAService"]


class EDAService:
    """Entry point for generating exploratory data analysis artifacts."""

    def analyze(self, data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Returns basic statistics about the provided data."""
        summary: Dict[str, Any] = {"counts": 0, "unique_values": {}}

        if isinstance(data, list):
            summary["counts"] = len(data)
            for item in data:
                if isinstance(item, dict):
                    for key, value in item.items():
                        summary["unique_values"].setdefault(key, set()).add(value)
        elif isinstance(data, dict):
            summary["counts"] = 1
            for key, value in data.items():
                summary["unique_values"].setdefault(key, set()).add(value)

        summary["unique_values"] = {k: len(v) for k, v in summary["unique_values"].items()}
        summary["config"] = config
        return summary
