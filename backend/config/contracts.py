"""Typed contracts describing the runtime configuration schema."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin


@dataclass
class SeedConfig:
    seed: Optional[int] = 42


@dataclass
class DataSettingsConfig:
    dataset_id: Optional[str] = None
    source: str = "memory"
    target_columns: List[str] = field(default_factory=list)
    index_columns: List[str] = field(default_factory=list)


@dataclass
class MissingValueConfig:
    strategy: str = "none"
    fill_value: Optional[Any] = None
    n_neighbors: int = 5
    max_iter: int = 10
    tol: float = 1e-3
    how: str = "any"


@dataclass
class OutlierConfig:
    method: str = "none"
    k: float = 1.5
    threshold: float = 3.0
    seed: int = 42
    n_neighbors: int = 20
    lower: float = 0.01
    upper: float = 0.99


@dataclass
class EncodingConfig:
    strategy: str = "none"
    drop: Optional[str] = None
    handle_unknown: str = "ignore"


@dataclass
class ScalingConfig:
    strategy: str = "none"


@dataclass
class FeatureSelectionConfig:
    method: str = "none"
    top_k: Optional[int] = None
    threshold: Optional[float] = None
    columns: List[str] = field(default_factory=list)


@dataclass
class PreprocessingConfig:
    enabled: bool = True
    data_settings: DataSettingsConfig = field(default_factory=DataSettingsConfig)
    missing_value: MissingValueConfig = field(default_factory=MissingValueConfig)
    outlier: OutlierConfig = field(default_factory=OutlierConfig)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    filters: List[str] = field(default_factory=list)
    derived_fields: List[str] = field(default_factory=list)
    default_value: Any = None


@dataclass
class PolynomialFeatureEngineeringConfig:
    enabled: bool = False
    columns: List[str] = field(default_factory=list)
    degree: int = 2
    include_bias: bool = False
    interaction_only: bool = False


@dataclass
class InteractionFeatureEngineeringConfig:
    enabled: bool = False
    columns: List[str] = field(default_factory=list)


@dataclass
class LagFeatureEngineeringConfig:
    enabled: bool = False
    columns: List[str] = field(default_factory=list)
    max_lag: int = 1
    fill_value: Optional[Any] = None


@dataclass
class RollingStatsFeatureEngineeringConfig:
    enabled: bool = False
    columns: List[str] = field(default_factory=list)
    windows: List[int] = field(default_factory=lambda: [3])
    stats: List[str] = field(default_factory=lambda: ["mean"])


@dataclass
class DateFeatureEngineeringConfig:
    enabled: bool = False
    column: Optional[str] = None
    features: List[str] = field(default_factory=lambda: ["year", "month", "day"])
    drop_original: bool = False


@dataclass
class HolidayFeatureEngineeringConfig:
    enabled: bool = False
    column: Optional[str] = None
    holidays: List[str] = field(default_factory=list)


@dataclass
class CyclicalFeatureEngineeringConfig:
    enabled: bool = False
    columns: List[str] = field(default_factory=list)
    max_values: Dict[str, int] = field(default_factory=dict)


@dataclass
class PCAFeatureEngineeringConfig:
    enabled: bool = False
    columns: List[str] = field(default_factory=list)
    n_components: Optional[int] = None
    whiten: bool = False


@dataclass
class FeatureEngineeringSelectionConfig:
    enabled: bool = False
    method: str = "variance"
    top_k: Optional[int] = None
    threshold: float = 0.0
    columns: List[str] = field(default_factory=list)


@dataclass
class FeatureEngineeringConfig:
    enabled: bool = True
    polynomial: PolynomialFeatureEngineeringConfig = field(default_factory=PolynomialFeatureEngineeringConfig)
    interaction: InteractionFeatureEngineeringConfig = field(
        default_factory=InteractionFeatureEngineeringConfig
    )
    lag: LagFeatureEngineeringConfig = field(default_factory=LagFeatureEngineeringConfig)
    rolling: RollingStatsFeatureEngineeringConfig = field(default_factory=RollingStatsFeatureEngineeringConfig)
    date: DateFeatureEngineeringConfig = field(default_factory=DateFeatureEngineeringConfig)
    holiday: HolidayFeatureEngineeringConfig = field(default_factory=HolidayFeatureEngineeringConfig)
    cyclical: CyclicalFeatureEngineeringConfig = field(default_factory=CyclicalFeatureEngineeringConfig)
    pca: PCAFeatureEngineeringConfig = field(default_factory=PCAFeatureEngineeringConfig)
    feature_selection: FeatureEngineeringSelectionConfig = field(
        default_factory=FeatureEngineeringSelectionConfig
    )
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


def _unwrap_optional(annotation: Any) -> Any:
    """Return the non-None type behind Optional/Union annotations."""
    origin = get_origin(annotation)
    if origin is Union:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


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
        inner_type = _unwrap_optional(f.type)
        # If it's a nested dataclass and override is a dict, recurse
        if is_dataclass(inner_type) and isinstance(value, dict):
            value = build_section(inner_type, value)
        setattr(section, f.name, value)
    return section
