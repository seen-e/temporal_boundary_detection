"""Velocity estimation and thresholding."""

from __future__ import annotations

import numpy as np

from .config import GripperPhaseSegmentationConfig
from .preprocess import smooth_signal
from .types import GripperSignalType, Thresholds


def compute_velocity(
    timestamps: np.ndarray,
    signal: np.ndarray,
    signal_type: GripperSignalType = GripperSignalType.ABSOLUTE_STATE,
) -> np.ndarray:
    if len(signal) < 2:
        return np.zeros_like(signal, dtype=float)
    if signal_type == GripperSignalType.RELATIVE_DELTA:
        dt = np.gradient(timestamps.astype(float))
        dt = np.maximum(dt, 1e-12)
        return signal.astype(float) / dt
    return np.gradient(signal.astype(float), timestamps.astype(float))


def smooth_velocity(velocity: np.ndarray, timestamps: np.ndarray, config: GripperPhaseSegmentationConfig) -> np.ndarray:
    if not config.velocity_smoothing.enabled:
        return velocity.copy()
    return smooth_signal(
        velocity,
        timestamps,
        config.velocity_smoothing.method,
        config.velocity_smoothing.window_sec,
        config.velocity_smoothing.polyorder,
    )


def estimate_velocity_thresholds(
    velocity: np.ndarray,
    config: GripperPhaseSegmentationConfig,
) -> Thresholds:
    motion = config.motion_state
    if motion.threshold_mode == "fixed":
        if motion.velocity_enter_threshold is None:
            raise ValueError("fixed threshold_mode requires velocity_enter_threshold")
        enter = abs(float(motion.velocity_enter_threshold))
        if motion.velocity_exit_threshold is None:
            exit_threshold = enter * float(motion.exit_ratio)
        else:
            exit_threshold = abs(float(motion.velocity_exit_threshold))
        if exit_threshold >= enter:
            exit_threshold = enter * min(float(motion.exit_ratio), 0.99)
        return Thresholds(enter=enter, exit=max(exit_threshold, 0.0), mode="fixed")

    abs_velocity = np.abs(velocity[np.isfinite(velocity)])
    if abs_velocity.size == 0:
        enter = float(motion.min_enter_threshold)
        return Thresholds(enter=enter, exit=enter * motion.exit_ratio, mode="auto", noise_level=enter)

    median = float(np.median(abs_velocity))
    mad = float(np.median(np.abs(abs_velocity - median)))
    noise_level = median + float(motion.auto_mad_k) * 1.4826 * mad
    percentile_value = float(np.percentile(abs_velocity, float(motion.auto_percentile)))
    enter = max(noise_level, percentile_value, float(motion.min_enter_threshold))
    exit_threshold = enter * float(motion.exit_ratio)
    if exit_threshold >= enter:
        exit_threshold = enter * 0.5
    return Thresholds(
        enter=float(enter),
        exit=float(max(exit_threshold, 0.0)),
        mode="auto",
        noise_level=float(noise_level),
        percentile_value=float(percentile_value),
    )

