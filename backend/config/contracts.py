"""Typed contracts describing the runtime configuration schema."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Type, get_args, get_origin


@dataclass
class SeedConfig:
    seed: Optional[int] = 42


@dataclass
class PreprocessingConfig:
    enabled: bool = True
    filters: List[str] = field(default_factory=list)
    derived_fields: List[str] = field(default_factory=list)
    default_value: Any = None


@dataclass
class FeatureEngineeringConfig:
    enabled: bool = True
    derived_fields: List[str] = field(default_factory=list)
    default_value: Any = None


@dataclass
class ModelConfig:
    task: str = "regression"
    family: str = "linear"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    default_prediction: Any = 0


@dataclass
class ForecastingConfig:
    enabled: bool = False
    horizon: int = 1
    default: Any = 0


@dataclass
class ValidationConfig:
    enabled: bool = True
    schema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualizationConfig:
    enabled: bool = True
    metrics: List[str] = field(default_factory=list)


@dataclass
class ReportingConfig:
    enabled: bool = True
    destination: str = "artifact"
    include_artifacts: List[str] = field(default_factory=lambda: ["preprocessing", "model"])


@dataclass
class RuntimeConfig:
    log_level: str = "info"
    timeout: int = 30
    seed: SeedConfig = field(default_factory=SeedConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    feature_engineering: FeatureEngineeringConfig = field(
        default_factory=FeatureEngineeringConfig
    )
    model: ModelConfig = field(default_factory=ModelConfig)
    forecasting: ForecastingConfig = field(default_factory=ForecastingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)


def build_section(cls: Type[Any], overrides: Optional[Dict[str, Any]] = None) -> Any:
    """
    Recursively build a dataclass section from overrides, enforcing nested contracts.

    Any nested dataclass fields provided as dicts in overrides will be converted
    into their respective dataclass instances.
    """
    overrides = overrides or {}
    # Instantiate with defaults
    section = cls()
    for f in fields(cls):
        if f.name not in overrides:
            continue
        value = overrides[f.name]
        # Determine the actual type, handling Optional[T]
        field_type = f.type
        origin = get_origin(field_type)
        args = get_args(field_type)
        # Unwrap Optional
        if origin is Optional and len(args) == 1:
            inner_type = args[0]
        else:
            inner_type = field_type
        # If it's a nested dataclass and override is a dict, recurse
        if is_dataclass(inner_type) and isinstance(value, dict):
            value = build_section(inner_type, value)
        setattr(section, f.name, value)
    return section
