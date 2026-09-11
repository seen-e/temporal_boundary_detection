"""Gripper trajectory phase segmentation."""

from .config import GripperPhaseSegmentationConfig
from .segmenter import segment_gripper_trajectory
from .dual import segment_dual_gripper_trajectory
from .types import (
    Boundary,
    Direction,
    DualGripperSegmentationResult,
    ExtremaPoint,
    FusedBoundary,
    GlobalSegment,
    GripperPhaseSegmentationResult,
    GripperSignalType,
    MainSegmentState,
    MotionState,
    Segment,
    Thresholds,
    TrendState,
)

__all__ = [
    "Boundary",
    "Direction",
    "DualGripperSegmentationResult",
    "ExtremaPoint",
    "FusedBoundary",
    "GlobalSegment",
    "GripperPhaseSegmentationConfig",
    "GripperPhaseSegmentationResult",
    "GripperSignalType",
    "MainSegmentState",
    "MotionState",
    "Segment",
    "Thresholds",
    "TrendState",
    "segment_gripper_trajectory",
    "segment_dual_gripper_trajectory",
]
