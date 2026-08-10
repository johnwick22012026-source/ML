"""Feature engineering service with derived field support plus caching/export helpers."""

from io import BytesIO
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.caching import CachingService
from backend.exports import ExportService
from backend.utils.run_context import compute_config_signature, compute_file_signature

__all__ = ["FeatureEngineeringService"]


class FeatureEngineeringService:
    """Entry point for feature engineering operations with cache/export awareness."""

    def __init__(
        self,
        caching_service: CachingService,
        export_service: ExportService,
    ) -> None:
        self._cache = caching_service
        self._export = export_service

    def transform(
        self,
        dataset_id: Optional[str],
        data: Any,
        config: Dict[str, Any],
        file_bytes: Optional[bytes] = None,
    ) -> Any:
        """Adds derived features while leveraging in-memory caching and export registration."""
        config_signature = compute_config_signature(config)
        file_hash = compute_file_signature(file_bytes)
        cache_key = self._cache.build_key(
            namespace="feature_engineering",
            dataset_id=dataset_id,
            component="transform",
            config=config,
            extra_signature=file_hash,
        )

        derived_fields = self._resolve_derived_fields(config)

        cached = self._cache.get_data(cache_key, config, dataset_id, file_hash)
        if cached is not None:
            self._register_export_artifacts(
                cache_key=cache_key,
                dataset_id=dataset_id,
                data=cached,
                config=config,
                config_signature=config_signature,
            )
            return cached

        transformed = self._apply_derived_fields(data, derived_fields, config)

        self._cache.set_data(
            cache_key,
            transformed,
            config,
            dataset_id,
            file_hash,
        )

        self._register_export_artifacts(
            cache_key=cache_key,
            dataset_id=dataset_id,
            data=transformed,
            config=config,
            config_signature=config_signature,
        )

        return transformed

    def _resolve_derived_fields(self, config: Dict[str, Any]) -> List[str]:
        derived_fields = config.get("derived_fields", [])
        if derived_fields is None:
            return []
        if not isinstance(derived_fields, list):
            raise ValueError("derived_fields must be provided as a list of strings.")
        invalid_fields = [field for field in derived_fields if not isinstance(field, str)]
        if invalid_fields:
            raise ValueError("All derived field identifiers must be strings.")
        return derived_fields

    def _apply_derived_fields(
        self,
        data: Any,
        derived_fields: List[str],
        config: Dict[str, Any],
    ) -> Any:
        if not isinstance(data, list):
            return data

        transformed: List[Any] = []

        for item in data:
            if not isinstance(item, dict):
                transformed.append(item)
                continue

            enhanced = item.copy()
            for field in derived_fields:
                if field not in enhanced:
                    enhanced[field] = config.get("default_value", None)
            transformed.append(enhanced)

        return transformed

    def _register_export_artifacts(
        self,
        cache_key: str,
        dataset_id: Optional[str],
        data: Any,
        config: Dict[str, Any],
        config_signature: str,
    ) -> None:
        df = self._data_to_dataframe(data)
        if df is None or df.empty:
            return

        artifact_source = (
            config.get("artifact_file_name")
            or dataset_id
            or "feature_engineered_dataset"
        )
        metadata = {
            "dataset_id": dataset_id,
            "component": "feature_engineering",
            "config_signature": config_signature,
        }

        csv_id = f"{cache_key}-csv"
        if not self._export.get_artifact(csv_id):
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            self._export.create(
                payload=csv_bytes,
                artifact_type="feature_engineered_dataset",
                metadata={**metadata, "format": "csv", "artifact_id": csv_id},
                file_name=f"{artifact_source}.csv",
                config={"artifact_id": csv_id},
            )

        xlsx_id = f"{cache_key}-xlsx"
        if not self._export.get_artifact(xlsx_id):
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False)
            self._export.create(
                payload=buffer.getvalue(),
                artifact_type="feature_engineered_dataset",
                metadata={**metadata, "format": "xlsx", "artifact_id": xlsx_id},
                file_name=f"{artifact_source}.xlsx",
                config={"artifact_id": xlsx_id},
            )

    def _data_to_dataframe(self, data: Any) -> Optional[pd.DataFrame]:
        if isinstance(data, pd.DataFrame):
            return data.copy()

        if isinstance(data, list):
            if not data:
                return pd.DataFrame()
            try:
                return pd.DataFrame(data)
            except ValueError:
                return pd.DataFrame({"value": data})

        if hasattr(data, "to_dict"):
            try:
                records = data.to_dict("records")
                return pd.DataFrame(records)
            except Exception:
                return None

        try:
            return pd.DataFrame([data])
        except ValueError:
            return None
