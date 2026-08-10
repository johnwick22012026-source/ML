"""Evaluation service providing simple metrics computations."""

from typing import Any, Dict, List

__all__ = ["EvaluationService"]


class EvaluationService:
    """Entry point for evaluation metric calculations."""

    def evaluate(self, predictions: Any, targets: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Computes basic accuracy and error counts."""
        evaluation: Dict[str, Any] = {"sample_size": 0, "matches": 0, "accuracy": 0.0}

        if isinstance(predictions, List) and isinstance(targets, List):
            common = min(len(predictions), len(targets))
            matches = sum(1 for i in range(common) if predictions[i] == targets[i])
            evaluation.update({"sample_size": common, "matches": matches})
            evaluation["accuracy"] = matches / common if common else 0.0
        return evaluation
