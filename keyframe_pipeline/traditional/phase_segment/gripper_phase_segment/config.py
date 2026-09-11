"""Configuration objects and YAML loading for gripper segmentation."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

from .types import Direction, GripperSignalType


@dataclass
class SignalConfig:
    type: GripperSignalType = GripperSignalType.AUTO
    direction: Direction = Direction.AUTO
    max_interpolation_gap_sec: float = 0.2


@dataclass
class OutlierConfig:
    enabled: bool = True
    method: str = "hampel"
    window_sec: float = 0.15
    n_sigma: float = 3.0


@dataclass
class SmoothingConfig:
    method: str = "savgol"
    window_sec: float = 0.15
    polyorder: int = 2


@dataclass
class VelocitySmoothingConfig:
    enabled: bool = True
    method: str = "savgol"
    window_sec: float = 0.10
    polyorder: int = 2


@dataclass
class ExtremaConfig:
    enabled: bool = True
    prominence_mode: str = "auto"
    prominence: float | None = None
    auto_prominence_ratio: float = 0.03
    min_distance_sec: float = 0.05
    min_absolute_amplitude: float | None = None


@dataclass
class TrendConfig:
    amplitude_mode: str = "auto"
    amplitude_threshold: float | None = None
    amplitude_threshold_ratio: float = 0.015
    flat_min_duration_sec: float = 0.12
    short_duration_sec: float = 0.20
    small_amplitude_mode: str = "auto"
    small_amplitude_ratio: float = 0.02


@dataclass
class PlateauConfig:
    enabled: bool = True
    window_sec: float = 0.25
    range_mode: str = "auto"
    range_threshold: float | None = None
    range_ratio: float = 0.03
    normalized_slope_threshold: float = 0.03
    velocity_check_enabled: bool = False
    normalized_velocity_threshold: float = 0.03
    min_duration_sec: float = 0.50
    max_internal_gap_sec: float = 0.15
    merge_value_ratio: float = 0.03
    suppress_same_direction_internal_plateaus: bool = True


@dataclass
class MotionConfig:
    small_motion_ratio: float = 0.02


@dataclass
class AnomalyConfig:
    detect_extrema: bool = True
    detect_short_reversal: bool = True


@dataclass
class MotionStateConfig:
    threshold_mode: str = "auto"
    velocity_enter_threshold: float | None = None
    velocity_exit_threshold: float | None = None
    auto_percentile: float = 75.0
    auto_mad_k: float = 3.0
    exit_ratio: float = 0.5
    min_enter_threshold: float = 1e-6


@dataclass
class TemporalConfig:
    min_state_duration_sec: float = 0.10


@dataclass
class BoundaryConfig:
    merge_window_sec: float = 0.15


@dataclass
class FusionConfig:
    merge_window_sec: float = 0.15
    time_strategy: str = "weighted_mean"
    confidence_strategy: str = "max"
    one_to_one_matching: bool = True


@dataclass
class DualGripperConfig:
    enabled: bool = True
    fusion: FusionConfig = field(default_factory=FusionConfig)


@dataclass
class SegmentConfig:
    min_motion_event_duration_sec: float = 0.08
    min_segment_duration_sec: float = 0.20
    strong_motion_threshold_ratio: float = 1.25


@dataclass
class VisualizationConfig:
    enabled: bool = True
    save_png: bool = True


@dataclass
class GripperPhaseSegmentationConfig:
    enabled: bool = True
    signal: SignalConfig = field(default_factory=SignalConfig)
    outlier: OutlierConfig = field(default_factory=OutlierConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    velocity_smoothing: VelocitySmoothingConfig = field(default_factory=VelocitySmoothingConfig)
    extrema: ExtremaConfig = field(default_factory=ExtremaConfig)
    trend: TrendConfig = field(default_factory=TrendConfig)
    plateau: PlateauConfig = field(default_factory=PlateauConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    motion_state: MotionStateConfig = field(default_factory=MotionStateConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    dual_gripper: DualGripperConfig = field(default_factory=DualGripperConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GripperPhaseSegmentationConfig":
        if not data:
            return cls()
        data = dict(data)
        if "gripper_phase_segmentation" in data:
            data = dict(data["gripper_phase_segmentation"] or {})
        return _update_dataclass(cls(), data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GripperPhaseSegmentationConfig":
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)


T = TypeVar("T")


def _update_dataclass(obj: T, data: dict[str, Any]) -> T:
    field_map = {field.name: field for field in fields(obj)}
    for key, value in data.items():
        if key not in field_map:
            continue
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            setattr(obj, key, _update_dataclass(current, value))
        else:
            setattr(obj, key, _coerce_value(current, value))
    return obj


def _coerce_value(current: Any, value: Any) -> Any:
    if isinstance(current, Direction):
        return Direction(value)
    if isinstance(current, GripperSignalType):
        return GripperSignalType(value)
    return value
