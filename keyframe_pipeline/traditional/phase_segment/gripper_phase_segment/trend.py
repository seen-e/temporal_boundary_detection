"""Extrema and trend based gripper segmentation."""

from __future__ import annotations

import numpy as np

from .config import GripperPhaseSegmentationConfig
from .types import Boundary, ExtremaPoint, Segment, Thresholds, TrendState
from .utils import estimate_sample_period, run_duration, seconds_to_odd_window


def compute_gripper_range(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.percentile(finite, 95) - np.percentile(finite, 5))


def estimate_prominence_threshold(values: np.ndarray, config: GripperPhaseSegmentationConfig) -> float:
    extrema_cfg = config.extrema
    if extrema_cfg.prominence_mode == "fixed":
        return float(extrema_cfg.prominence or 0.0)
    value_range = compute_gripper_range(values)
    threshold = value_range * float(extrema_cfg.auto_prominence_ratio)
    if extrema_cfg.min_absolute_amplitude is not None:
        threshold = max(threshold, float(extrema_cfg.min_absolute_amplitude))
    return float(max(threshold, 1e-12))


def estimate_trend_amplitude_threshold(values: np.ndarray, config: GripperPhaseSegmentationConfig) -> float:
    trend_cfg = config.trend
    if trend_cfg.amplitude_mode == "fixed":
        return float(trend_cfg.amplitude_threshold or 0.0)
    return float(max(compute_gripper_range(values) * float(trend_cfg.amplitude_threshold_ratio), 1e-12))


def estimate_small_amplitude_threshold(values: np.ndarray, config: GripperPhaseSegmentationConfig) -> float:
    trend_cfg = config.trend
    if trend_cfg.small_amplitude_mode == "fixed" and trend_cfg.amplitude_threshold is not None:
        return float(trend_cfg.amplitude_threshold)
    return float(max(compute_gripper_range(values) * float(trend_cfg.small_amplitude_ratio), 1e-12))


def detect_extrema(
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    values: np.ndarray,
    config: GripperPhaseSegmentationConfig,
) -> tuple[list[ExtremaPoint], list[ExtremaPoint], dict[str, float]]:
    """Detect accepted and rejected local maxima/minima using prominence."""

    if not config.extrema.enabled or len(values) < 3:
        return [], [], {
            "prominence_threshold": estimate_prominence_threshold(values, config),
            "trend_amplitude_threshold": estimate_trend_amplitude_threshold(values, config),
            "small_amplitude_threshold": estimate_small_amplitude_threshold(values, config),
        }

    prominence_threshold = estimate_prominence_threshold(values, config)
    min_distance = seconds_to_odd_window(config.extrema.min_distance_sec, timestamps, min_window=1)
    min_distance = max(1, min_distance)

    maxima = _find_peaks(values, min_distance)
    minima = _find_peaks(-values, min_distance)

    accepted: list[ExtremaPoint] = []
    rejected: list[ExtremaPoint] = []
    for idx, prominence in maxima:
        point = _extrema_point(idx, "local_max", prominence, timestamps, frame_indices, values, prominence_threshold)
        (accepted if point.accepted else rejected).append(point)
    for idx, prominence in minima:
        point = _extrema_point(idx, "local_min", prominence, timestamps, frame_indices, values, prominence_threshold)
        (accepted if point.accepted else rejected).append(point)

    accepted.sort(key=lambda point: point.index)
    rejected.sort(key=lambda point: point.index)
    return accepted, rejected, {
        "prominence_threshold": prominence_threshold,
        "trend_amplitude_threshold": estimate_trend_amplitude_threshold(values, config),
        "small_amplitude_threshold": estimate_small_amplitude_threshold(values, config),
    }


def build_trend_segments(
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    values: np.ndarray,
    velocity: np.ndarray,
    extrema: list[ExtremaPoint],
    config: GripperPhaseSegmentationConfig,
) -> list[Segment]:
    """Build UP/DOWN/FLAT segments from accepted extrema and plateau edges."""

    amplitude_threshold = estimate_trend_amplitude_threshold(values, config)
    breakpoints = {0, len(values)}
    breakpoints.update(point.index for point in extrema if 0 < point.index < len(values))
    breakpoints.update(_plateau_edge_breakpoints(values, timestamps, sorted(breakpoints), amplitude_threshold, config))
    breakpoints = sorted(idx for idx in breakpoints if 0 <= idx <= len(values))

    segments: list[Segment] = []
    for start, end in zip(breakpoints, breakpoints[1:]):
        if end <= start:
            continue
        segment = _make_segment(
            segment_id=len(segments),
            start=start,
            end=end,
            timestamps=timestamps,
            frame_indices=frame_indices,
            values=values,
            velocity=velocity,
            amplitude_threshold=amplitude_threshold,
            config=config,
        )
        if segments and segments[-1].motion_state == segment.motion_state:
            segments[-1] = _merge_segments(segments[-1], segment, timestamps, frame_indices, values, velocity, amplitude_threshold, config)
            segments[-1].segment_id = len(segments) - 1
        else:
            segments.append(segment)

    _mark_reversals(segments, values, config)
    for idx, segment in enumerate(segments):
        segment.segment_id = idx
    return segments


def trend_boundaries_from_segments(
    segments: list[Segment],
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    velocity: np.ndarray,
    thresholds: Thresholds,
    side: str = "single",
) -> list[Boundary]:
    boundaries: list[Boundary] = []
    source = f"{side}_gripper" if side in {"left", "right"} else "gripper"
    for prev_segment, next_segment in zip(segments, segments[1:]):
        time = float(next_segment.start_time)
        frame = int(next_segment.start_frame)
        idx = int(np.argmin(np.abs(frame_indices.astype(int) - frame))) if len(frame_indices) else 0
        confidence = _trend_boundary_confidence(prev_segment, next_segment, thresholds)
        boundaries.append(
            Boundary(
                time=time,
                frame_index=frame,
                from_state=prev_segment.motion_state,
                to_state=next_segment.motion_state,
                source=source,
                confidence=confidence,
                velocity=float(velocity[idx]) if len(velocity) else None,
                score=confidence,
                side=side,
            )
        )
    return boundaries


def _find_peaks(values: np.ndarray, min_distance: int) -> list[tuple[int, float]]:
    try:
        from scipy.signal import find_peaks

        peaks, properties = find_peaks(values, distance=min_distance, prominence=0)
        prominences = properties.get("prominences", np.zeros_like(peaks, dtype=float))
        return [(int(idx), float(prominence)) for idx, prominence in zip(peaks, prominences)]
    except Exception:
        return _fallback_find_peaks(values, min_distance)


def _fallback_find_peaks(values: np.ndarray, min_distance: int) -> list[tuple[int, float]]:
    peaks: list[tuple[int, float]] = []
    last_idx = -min_distance
    for idx in range(1, len(values) - 1):
        if idx - last_idx < min_distance:
            continue
        if values[idx] >= values[idx - 1] and values[idx] > values[idx + 1]:
            left_min = float(np.min(values[max(0, idx - min_distance) : idx + 1]))
            right_min = float(np.min(values[idx : min(len(values), idx + min_distance + 1)]))
            prominence = float(values[idx] - max(left_min, right_min))
            peaks.append((idx, max(0.0, prominence)))
            last_idx = idx
    return peaks


def _extrema_point(
    idx: int,
    kind: str,
    prominence: float,
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    values: np.ndarray,
    threshold: float,
) -> ExtremaPoint:
    return ExtremaPoint(
        time=float(timestamps[idx]),
        frame_index=int(frame_indices[idx]),
        index=int(idx),
        type=kind,
        value=float(values[idx]),
        prominence=float(prominence),
        accepted=bool(prominence >= threshold),
    )


def _plateau_edge_breakpoints(
    values: np.ndarray,
    timestamps: np.ndarray,
    breakpoints: list[int],
    amplitude_threshold: float,
    config: GripperPhaseSegmentationConfig,
) -> list[int]:
    edges: set[int] = set()
    min_flat = float(config.trend.flat_min_duration_sec)
    for start, end in zip(breakpoints, breakpoints[1:]):
        if end - start < 3:
            continue
        delta = float(values[end - 1] - values[start])
        if abs(delta) <= amplitude_threshold:
            continue

        leading = None
        for idx in range(start + 1, end):
            if abs(float(values[idx] - values[start])) >= amplitude_threshold:
                leading = idx
                break
        if leading is not None and leading > start and run_duration(timestamps, start, leading) >= min_flat:
            edges.add(leading)

        trailing = None
        end_value = values[end - 1]
        for idx in range(end - 2, start, -1):
            if abs(float(end_value - values[idx])) >= amplitude_threshold:
                trailing = idx + 1
                break
        if trailing is not None and trailing < end and run_duration(timestamps, trailing, end) >= min_flat:
            edges.add(trailing)
    return sorted(edges)


def _make_segment(
    segment_id: int,
    start: int,
    end: int,
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    values: np.ndarray,
    velocity: np.ndarray,
    amplitude_threshold: float,
    config: GripperPhaseSegmentationConfig,
) -> Segment:
    segment_values = values[start:end]
    segment_velocity = velocity[start:end]
    net_delta = float(segment_values[-1] - segment_values[0])
    amplitude = float(np.max(segment_values) - np.min(segment_values))
    trend = _classify_trend(net_delta, amplitude_threshold)
    duration = run_duration(timestamps, start, end)
    small_threshold = estimate_small_amplitude_threshold(values, config)
    return Segment(
        segment_id=segment_id,
        start_time=float(timestamps[start]),
        end_time=float(timestamps[end - 1]),
        start_frame=int(frame_indices[start]),
        end_frame=int(frame_indices[end - 1]),
        motion_state=trend,
        duration_sec=duration,
        trend=trend,
        amplitude=amplitude,
        net_delta=net_delta,
        max_value=float(np.max(segment_values)),
        min_value=float(np.min(segment_values)),
        mean_velocity=float(np.mean(segment_velocity)) if len(segment_velocity) else 0.0,
        peak_velocity=float(np.max(np.abs(segment_velocity))) if len(segment_velocity) else 0.0,
        is_short=bool(duration < float(config.trend.short_duration_sec)),
        is_small_amplitude=bool(amplitude < small_threshold),
    )


def _merge_segments(
    left: Segment,
    right: Segment,
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    values: np.ndarray,
    velocity: np.ndarray,
    amplitude_threshold: float,
    config: GripperPhaseSegmentationConfig,
) -> Segment:
    start = int(np.where(frame_indices == left.start_frame)[0][0])
    end = int(np.where(frame_indices == right.end_frame)[0][0]) + 1
    return _make_segment(left.segment_id, start, end, timestamps, frame_indices, values, velocity, amplitude_threshold, config)


def _classify_trend(net_delta: float, threshold: float) -> str:
    if net_delta > threshold:
        return TrendState.UP.value
    if net_delta < -threshold:
        return TrendState.DOWN.value
    return TrendState.FLAT.value


def _mark_reversals(segments: list[Segment], values: np.ndarray, config: GripperPhaseSegmentationConfig) -> None:
    for idx, segment in enumerate(segments):
        prev_state = segments[idx - 1].motion_state if idx > 0 else None
        next_state = segments[idx + 1].motion_state if idx + 1 < len(segments) else None
        segment.is_reversal = bool(
            prev_state == next_state
            and segment.motion_state != prev_state
            and segment.motion_state in {TrendState.UP.value, TrendState.DOWN.value}
            and prev_state in {TrendState.UP.value, TrendState.DOWN.value}
        )
        segment.possible_jitter = bool(segment.is_short and segment.is_small_amplitude and segment.is_reversal)


def _trend_boundary_confidence(prev_segment: Segment, next_segment: Segment, thresholds: Thresholds) -> float:
    scale = max(thresholds.enter, 1e-12)
    amp = max(abs(prev_segment.net_delta or 0.0), abs(next_segment.net_delta or 0.0))
    duration_score = min(1.0, max(prev_segment.duration_sec, next_segment.duration_sec) / 0.30)
    amp_score = min(1.0, amp / scale)
    return float(0.65 * amp_score + 0.35 * duration_score)

