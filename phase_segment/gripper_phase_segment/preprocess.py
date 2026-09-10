"""Validity checks, interpolation, outlier filtering, and smoothing."""

from __future__ import annotations

import numpy as np

from .config import GripperPhaseSegmentationConfig
from .types import InvalidInterval
from .utils import moving_average, moving_median, seconds_to_odd_window


def validate_inputs(timestamps: np.ndarray, gripper_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.asarray(timestamps, dtype=float)
    gripper_values = np.asarray(gripper_values, dtype=float)
    if timestamps.ndim != 1 or gripper_values.ndim != 1:
        raise ValueError("timestamps and gripper_values must be 1-D arrays")
    if timestamps.shape[0] != gripper_values.shape[0]:
        raise ValueError("timestamps and gripper_values must have the same length")
    if timestamps.shape[0] < 3:
        raise ValueError("at least 3 samples are required")
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("timestamps contain NaN or Inf")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    return timestamps, gripper_values


def interpolate_missing(
    timestamps: np.ndarray,
    values: np.ndarray,
    max_gap_sec: float,
) -> tuple[np.ndarray, list[InvalidInterval]]:
    values = values.astype(float, copy=True)
    invalid = ~np.isfinite(values)
    intervals: list[InvalidInterval] = []
    if not np.any(invalid):
        return values, intervals

    valid = ~invalid
    if valid.sum() < 2:
        raise ValueError("gripper_values must contain at least two finite samples")

    start = None
    for idx, is_invalid in enumerate(invalid):
        if is_invalid and start is None:
            start = idx
        if start is not None and (not is_invalid or idx == len(invalid) - 1):
            end = idx if not is_invalid else idx + 1
            left = max(start - 1, 0)
            right = min(end, len(values) - 1)
            gap_sec = float(timestamps[right] - timestamps[left])
            if gap_sec > max_gap_sec:
                intervals.append(
                    InvalidInterval(
                        start_time=float(timestamps[left]),
                        end_time=float(timestamps[right]),
                        start_index=left,
                        end_index=right,
                        reason="long_missing_gap",
                    )
                )
            start = None

    indices = np.arange(len(values))
    values[invalid] = np.interp(indices[invalid], indices[valid], values[valid])
    return values, intervals


def hampel_filter(values: np.ndarray, timestamps: np.ndarray, window_sec: float, n_sigma: float) -> np.ndarray:
    window = seconds_to_odd_window(window_sec, timestamps)
    if window < 3 or len(values) < 3:
        return values.copy()
    half = window // 2
    filtered = values.astype(float, copy=True)
    for idx in range(len(values)):
        start = max(0, idx - half)
        end = min(len(values), idx + half + 1)
        local = values[start:end]
        median = float(np.median(local))
        mad = float(np.median(np.abs(local - median)))
        scale = 1.4826 * mad
        if scale <= 1e-12:
            if abs(values[idx] - median) > 1e-12:
                filtered[idx] = median
            continue
        if abs(values[idx] - median) > n_sigma * scale:
            filtered[idx] = median
    return filtered


def remove_outliers(
    values: np.ndarray,
    timestamps: np.ndarray,
    config: GripperPhaseSegmentationConfig,
) -> np.ndarray:
    if not config.outlier.enabled:
        return values.copy()
    method = config.outlier.method.lower()
    if method != "hampel":
        raise ValueError(f"unsupported outlier method: {config.outlier.method}")
    return hampel_filter(values, timestamps, config.outlier.window_sec, config.outlier.n_sigma)


def smooth_signal(values: np.ndarray, timestamps: np.ndarray, method: str, window_sec: float, polyorder: int) -> np.ndarray:
    window = seconds_to_odd_window(window_sec, timestamps)
    if len(values) < 3 or window < 3:
        return values.copy()
    window = min(window, len(values) if len(values) % 2 == 1 else len(values) - 1)
    if window < 3:
        return values.copy()
    method = method.lower()
    if method == "savgol":
        try:
            from scipy.signal import savgol_filter

            order = min(max(int(polyorder), 0), window - 1)
            return savgol_filter(values.astype(float), window_length=window, polyorder=order, mode="interp")
        except Exception:
            return moving_average(values, window)
    if method in {"moving_average", "mean"}:
        return moving_average(values, window)
    if method in {"moving_median", "median"}:
        return moving_median(values, window)
    raise ValueError(f"unsupported smoothing method: {method}")


def preprocess_gripper(
    timestamps: np.ndarray,
    gripper_values: np.ndarray,
    config: GripperPhaseSegmentationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[InvalidInterval]]:
    timestamps, gripper_values = validate_inputs(timestamps, gripper_values)
    clean, invalid_intervals = interpolate_missing(
        timestamps,
        gripper_values,
        config.signal.max_interpolation_gap_sec,
    )
    clean = remove_outliers(clean, timestamps, config)
    smooth = smooth_signal(
        clean,
        timestamps,
        config.smoothing.method,
        config.smoothing.window_sec,
        config.smoothing.polyorder,
    )
    return gripper_values.copy(), clean, smooth, invalid_intervals
