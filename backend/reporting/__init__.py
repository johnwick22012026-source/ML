"""Reporting service with basic aggregation capability."""

from typing import Any, Dict

__all__ = ["ReportingService"]


class ReportingService:
    """Entry point for generating session reports and artifacts."""

    def generate(self, results: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregates results and configuration into a report manifest."""
        return {
            "report_id": config.get("report_id", "default-report"),
            "summary": str(results),
            "config": config,
        }
