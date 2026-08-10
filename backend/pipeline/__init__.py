"""Pipeline orchestration layer for preprocessing driven entirely by configuration."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List

from backend.config.contracts import PreprocessingConfig, build_section
from backend.preprocessing import PreprocessingService
from backend.validation import ValidationService

__all__ = ["PipelineService"]


class PipelineService:
    """Orchestrates validation and preprocessing purely from a provided configuration."""

    _REQUIRED_CONFIG_SECTIONS: List[str] = [
        "data_settings",
        "missing_value",
        "outlier",
        "encoding",
        "scaling",
        "feature_selection",
    ]

    def __init__(self, validation_service: ValidationService, preprocessing_service: PreprocessingService):
        self._validation_service = validation_service
        self._preprocessing_service = preprocessing_service

    def run_preprocessing(
        self,
        config: Dict[str, Any],
        data: Any,
        dataset_id: str = "",
    ) -> Dict[str, Any]:
        """Validate the configuration and dispatch to the preprocessing service."""
        config = config or {}
        validation_payload = {section: dict for section in self._REQUIRED_CONFIG_SECTIONS}
        validation_result = self._validation_service.validate(config, validation_payload)
        if not validation_result.get("is_valid"):
            return {
                "status": "invalid_config",
                "errors": validation_result.get("errors", []),
                "schema": list(validation_payload.keys()),
            }

        preprocessing_section = build_section(PreprocessingConfig, config)
        if not preprocessing_section.enabled:
            return {
                "status": "skipped",
                "metadata": {
                    "reason": "Preprocessing is disabled in configuration.",
                    "data_settings": self._to_plain_dict(preprocessing_section.data_settings),
                },
            }

        payload = self._build_preprocessing_payload(preprocessing_section)
        resolved_dataset_id = dataset_id or preprocessing_section.data_settings.dataset_id or "streamlit"
        run_result = self._preprocessing_service.run(data, payload, dataset_id=resolved_dataset_id)
        metadata = run_result.get("metadata", {})
        metadata.setdefault("data_settings", self._to_plain_dict(preprocessing_section.data_settings))
        metadata.setdefault("feature_selection", self._to_plain_dict(preprocessing_section.feature_selection))
        run_result["metadata"] = metadata

        return {
            "status": "complete",
            "result": run_result,
            "config": self._to_plain_dict(preprocessing_section),
        }

    def _build_preprocessing_payload(self, section: PreprocessingConfig) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "missing_value": self._to_plain_dict(section.missing_value),
            "outlier": self._to_plain_dict(section.outlier),
            "encoding": self._to_plain_dict(section.encoding),
            "scaling": self._to_plain_dict(section.scaling),
            "feature_selection": self._to_plain_dict(section.feature_selection),
            "filters": list(section.filters),
            "derived_fields": list(section.derived_fields),
            "default_value": section.default_value,
        }
        return payload

    @staticmethod
    def _to_plain_dict(value: Any) -> Any:
        if is_dataclass(value):
            return asdict(value)
        return value
