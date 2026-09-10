"""Public gripper phase segmentation pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import GripperPhaseSegmentationConfig
from .preprocess import preprocess_gripper, validate_inputs
from .plateau import (
    build_plateau_segments,
    compute_plateau_diagnostics,
    detect_anomaly_events,
    detect_plateau_intervals,
    plateau_boundaries_from_segments,
    suppress_same_direction_internal_plateaus,
)
from .trend import detect_extrema
from .types import GripperPhaseSegmentationResult, GripperSignalType, MainSegmentState, Thresholds
from .velocity import compute_velocity, estimate_velocity_thresholds, smooth_velocity


def segment_gripper_trajectory(
    timestamps: np.ndarray,
    gripper_values: np.ndarray,
    config: GripperPhaseSegmentationConfig | dict[str, Any] | None = None,
    frame_indices: np.ndarray | None = None,
    side: str = "single",
) -> GripperPhaseSegmentationResult:
    """Segment one episode's gripper trajectory into low-level motion phases."""

    cfg = config if isinstance(config, GripperPhaseSegmentationConfig) else GripperPhaseSegmentationConfig.from_dict(config)
    if not cfg.enabled:
        raise ValueError("gripper phase segmentation is disabled by config")

    timestamps, gripper_values = validate_inputs(timestamps, gripper_values)
    if frame_indices is None:
        frame_indices = np.arange(len(timestamps), dtype=int)
    else:
        frame_indices = np.asarray(frame_indices, dtype=int)
        if frame_indices.shape != timestamps.shape:
            raise ValueError("frame_indices must have the same shape as timestamps")

    signal_type = _resolve_signal_type(cfg.signal.type, gripper_values)
    raw, clean, smooth, invalid_intervals = preprocess_gripper(timestamps, gripper_values, cfg)
    velocity_raw = compute_velocity(timestamps, smooth, signal_type=signal_type)
    velocity_smooth = smooth_velocity(velocity_raw, timestamps, cfg)
    velocity_thresholds = estimate_velocity_thresholds(velocity_smooth, cfg)
    extrema, rejected_extrema, extrema_thresholds = detect_extrema(timestamps, frame_indices, smooth, cfg)
    plateau_mask, plateau_range, normalized_slope, plateau_velocity, plateau_thresholds = compute_plateau_diagnostics(
        timestamps=timestamps,
        values=smooth,
        velocity=velocity_smooth,
        config=cfg,
    )
    plateau_intervals = detect_plateau_intervals(timestamps, smooth, plateau_mask, plateau_thresholds, cfg)
    segments = build_plateau_segments(timestamps, frame_indices, smooth, velocity_smooth, plateau_intervals, plateau_thresholds, cfg)
    segments, num_suppressed_internal_plateaus = suppress_same_direction_internal_plateaus(
        segments=segments,
        timestamps=timestamps,
        frame_indices=frame_indices,
        values=smooth,
        velocity=velocity_smooth,
        thresholds=plateau_thresholds,
        config=cfg,
    )
    motion_states = _expand_segment_states(segments, timestamps)
    boundaries = plateau_boundaries_from_segments(
        segments=segments,
        timestamps=timestamps,
        frame_indices=frame_indices,
        velocity=velocity_smooth,
        thresholds=plateau_thresholds,
        side=side,
    )
    anomaly_events = detect_anomaly_events(segments, extrema, timestamps, smooth, cfg)
    boundary_scale = Thresholds(
        enter=float(plateau_thresholds["range_threshold"]),
        exit=float(plateau_thresholds["small_motion_threshold"]),
        mode="plateau_edge",
        noise_level=float(plateau_thresholds["robust_gripper_range"]),
        percentile_value=None,
    )

    diagnostics = {
        "num_samples": int(len(timestamps)),
        "duration_sec": float(timestamps[-1] - timestamps[0]) if len(timestamps) else 0.0,
        "signal_type": signal_type.value,
        "direction": cfg.signal.direction.value,
        "num_boundaries": len(boundaries),
        "num_segments": len(segments),
        "num_extrema": len(extrema),
        "num_rejected_extrema": len(rejected_extrema),
        "num_plateaus": sum(1 for segment in segments if segment.motion_state == MainSegmentState.PLATEAU.value),
        "num_candidate_plateaus": len(plateau_intervals),
        "num_suppressed_internal_plateaus": num_suppressed_internal_plateaus,
        "num_anomaly_events": len(anomaly_events),
        "num_invalid_intervals": len(invalid_intervals),
        "gripper_min": float(np.nanmin(gripper_values)),
        "gripper_max": float(np.nanmax(gripper_values)),
        "velocity_abs_median": float(np.nanmedian(np.abs(velocity_smooth))),
        "velocity_abs_max": float(np.nanmax(np.abs(velocity_smooth))),
        "velocity_thresholds_for_diagnostics": velocity_thresholds.to_dict(),
        "prominence_threshold": float(extrema_thresholds["prominence_threshold"]),
        "plateau_thresholds": plateau_thresholds,
        "plateau_intervals": [interval.to_dict() for interval in plateau_intervals],
        "anomaly_events": anomaly_events,
        "segmentation_stage": "plateau_edge",
    }

    return GripperPhaseSegmentationResult(
        boundaries=boundaries,
        segments=segments,
        gripper_raw=raw,
        gripper_clean=clean,
        gripper_smooth=smooth,
        velocity_raw=velocity_raw,
        velocity_smooth=velocity_smooth,
        motion_states=motion_states,
        timestamps=timestamps,
        frame_indices=frame_indices,
        thresholds=boundary_scale,
        extrema=extrema,
        rejected_extrema=rejected_extrema,
        trend_segments=segments,
        plateau_mask=plateau_mask,
        plateau_local_range=plateau_range,
        plateau_normalized_slope=normalized_slope,
        invalid_intervals=invalid_intervals,
        diagnostics=diagnostics,
    )


def _resolve_signal_type(configured: GripperSignalType, values: np.ndarray) -> GripperSignalType:
    if configured != GripperSignalType.AUTO:
        return configured
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return GripperSignalType.UNKNOWN
    unique = np.unique(np.round(finite, decimals=6))
    if unique.size <= 2:
        return GripperSignalType.BINARY_COMMAND
    return GripperSignalType.ABSOLUTE_STATE


def _expand_segment_states(segments, timestamps: np.ndarray) -> np.ndarray:
    states = np.full(len(timestamps), "FLAT", dtype=object)
    for segment in segments:
        mask = (timestamps >= segment.start_time) & (timestamps <= segment.end_time)
        states[mask] = segment.motion_state
    return states
