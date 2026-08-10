"""Explainability service providing simple attribution artifacts."""

from typing import Any, Dict

__all__ = ["ExplainabilityService"]


class ExplainabilityService:
    """Entry point for explainability and interpretability outputs."""

    def explain(self, model: Any, data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generates a basic explanation summary."""
        explanation: Dict[str, Any] = {
            "model_id": getattr(model, "id", "generic"),
            "input_size": len(data) if hasattr(data, "__len__") else None,
        }
        explanation.update(config)
        return explanation
