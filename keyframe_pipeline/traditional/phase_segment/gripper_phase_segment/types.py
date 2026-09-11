"""Shared types for gripper phase segmentation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class MotionState(str, Enum):
    """Low-level, direction-neutral gripper motion state."""

    STABLE = "STABLE"
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    INVALID = "INVALID"


class TrendState(str, Enum):
    """Trajectory trend state used by the extrema-based segmenter."""

    FLAT = "FLAT"
    UP = "UP"
    DOWN = "DOWN"


class MainSegmentState(str, Enum):
    """Main plateau-driven phase state."""

    PLATEAU = "PLATEAU"
    MOTION = "MOTION"


class Direction(str, Enum):
    """Physical interpretation of positive gripper values."""

    AUTO = "auto"
    LARGER_IS_OPEN = "larger_is_open"
    LARGER_IS_CLOSED = "larger_is_closed"


class GripperSignalType(str, Enum):
    """Supported gripper signal families."""

    AUTO = "auto"
    ABSOLUTE_STATE = "absolute_state"
    RELATIVE_DELTA = "relative_delta"
    BINARY_COMMAND = "binary_command"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Thresholds:
    """Velocity thresholds used by the hysteresis state detector."""

    enter: float
    exit: float
    mode: str
    noise_level: float | None = None
    percentile_value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InvalidInterval:
    """Time interval where interpolation crossed a long missing span."""

    start_time: float
    end_time: float
    start_index: int
    end_index: int
    reason: str

    def contains(self, time_sec: float) -> bool:
        return self.start_time <= time_sec <= self.end_time

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExtremaPoint:
    """A detected local extremum on the smoothed gripper trajectory."""

    time: float
    frame_index: int
    index: int
    type: str
    value: float
    prominence: float
    accepted: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Boundary:
    """A low-level gripper motion-state transition."""

    time: float
    frame_index: int
    from_state: str
    to_state: str
    source: str = "gripper"
    confidence: float = 0.0
    velocity: float | None = None
    score: float | None = None
    side: str = "single"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Segment:
    """A contiguous time interval with a single low-level motion state."""

    segment_id: int
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    motion_state: str
    duration_sec: float
    trend: str | None = None
    amplitude: float | None = None
    net_delta: float | None = None
    max_value: float | None = None
    min_value: float | None = None
    mean_velocity: float | None = None
    peak_velocity: float | None = None
    is_short: bool = False
    is_small_amplitude: bool = False
    is_reversal: bool = False
    possible_jitter: bool = False
    segment_type: str | None = None
    start_value: float | None = None
    end_value: float | None = None
    value_range: float | None = None
    mean_slope: float | None = None
    direction: str | None = None
    small_motion: bool = False
    plateau_mean: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FusedBoundary:
    """A global gripper boundary from one side or a cross-side merge."""

    time: float
    frame_index: int
    source: str
    confidence: float
    source_sides: list[str]
    sources: list[dict[str, Any]]
    spread_sec: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GlobalSegment:
    """A global temporal segment split by fused gripper boundaries."""

    segment_id: int
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    duration_sec: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GripperPhaseSegmentationResult:
    """Full segmentation result plus intermediate debug signals."""

    boundaries: list[Boundary]
    segments: list[Segment]
    gripper_raw: np.ndarray
    gripper_clean: np.ndarray
    gripper_smooth: np.ndarray
    velocity_raw: np.ndarray
    velocity_smooth: np.ndarray
    motion_states: np.ndarray
    timestamps: np.ndarray
    frame_indices: np.ndarray
    thresholds: Thresholds
    extrema: list[ExtremaPoint] = field(default_factory=list)
    rejected_extrema: list[ExtremaPoint] = field(default_factory=list)
    trend_segments: list[Segment] = field(default_factory=list)
    plateau_mask: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=bool))
    plateau_local_range: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    plateau_normalized_slope: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    invalid_intervals: list[InvalidInterval] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def boundaries_dict(self) -> dict[str, Any]:
        return {"boundaries": [boundary.to_dict() for boundary in self.boundaries]}

    def segments_dict(self) -> dict[str, Any]:
        return {"segments": [segment.to_dict() for segment in self.segments]}

    def extrema_dict(self) -> dict[str, Any]:
        return {
            "extrema": [point.to_dict() for point in self.extrema],
            "rejected_extrema": [point.to_dict() for point in self.rejected_extrema],
        }

    def diagnostics_dict(self) -> dict[str, Any]:
        data = dict(self.diagnostics)
        data["thresholds"] = self.thresholds.to_dict()
        data["invalid_intervals"] = [interval.to_dict() for interval in self.invalid_intervals]
        return data


@dataclass
class DualGripperSegmentationResult:
    """Dual-gripper result preserving both single-side debug results."""

    left_result: GripperPhaseSegmentationResult | None
    right_result: GripperPhaseSegmentationResult | None
    global_boundaries: list[FusedBoundary]
    global_segments: list[GlobalSegment]
    fusion_diagnostics: dict[str, Any] = field(default_factory=dict)

    def global_boundaries_dict(self) -> dict[str, Any]:
        return {"global_boundaries": [boundary.to_dict() for boundary in self.global_boundaries]}

    def global_segments_dict(self) -> dict[str, Any]:
        return {"global_segments": [segment.to_dict() for segment in self.global_segments]}
