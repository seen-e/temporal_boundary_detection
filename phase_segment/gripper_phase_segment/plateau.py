"""Plateau-driven gripper phase segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import GripperPhaseSegmentationConfig
from .trend import compute_gripper_range, detect_extrema, estimate_trend_amplitude_threshold
from .types import Boundary, ExtremaPoint, MainSegmentState, Segment, Thresholds, TrendState
from .utils import estimate_sample_period, run_duration, seconds_to_odd_window


@dataclass(frozen=True)
class PlateauInterval:
    start_index: int
    end_index: int
    start_time: float
    end_time: float
    mean_value: float
    value_range: float
    duration_sec: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "mean_value": self.mean_value,
            "value_range": self.value_range,
            "duration_sec": self.duration_sec,
        }


def estimate_plateau_thresholds(values: np.ndarray, config: GripperPhaseSegmentationConfig) -> dict[str, float]:
    robust_range = max(compute_gripper_range(values), 1e-12)
    finite = values[np.isfinite(values)]
    value_scale = max(abs(float(np.median(finite))) if finite.size else 0.0, robust_range)
    if config.plateau.range_mode == "fixed" and config.plateau.range_threshold is not None:
        range_threshold = float(config.plateau.range_threshold)
    else:
        range_threshold = robust_range * float(config.plateau.range_ratio)
        range_threshold = max(range_threshold, value_scale * float(config.plateau.range_ratio))
    merge_value_threshold = robust_range * float(config.plateau.merge_value_ratio)
    motion_small_threshold = robust_range * float(config.motion.small_motion_ratio)
    return {
        "robust_gripper_range": float(robust_range),
        "range_threshold": float(max(range_threshold, 1e-12)),
        "normalized_slope_threshold": float(config.plateau.normalized_slope_threshold),
        "normalized_velocity_threshold": float(config.plateau.normalized_velocity_threshold),
        "merge_value_threshold": float(max(merge_value_threshold, 1e-12)),
        "small_motion_threshold": float(max(motion_small_threshold, 1e-12)),
        "trend_amplitude_threshold": estimate_trend_amplitude_threshold(values, config),
    }


def compute_plateau_diagnostics(
    timestamps: np.ndarray,
    values: np.ndarray,
    velocity: np.ndarray,
    config: GripperPhaseSegmentationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    thresholds = estimate_plateau_thresholds(values, config)
    window = seconds_to_odd_window(config.plateau.window_sec, timestamps)
    half = window // 2
    local_range = np.zeros(len(values), dtype=float)
    normalized_slope = np.zeros(len(values), dtype=float)
    median_abs_velocity = np.zeros(len(values), dtype=float)

    for idx in range(len(values)):
        start = max(0, idx - half)
        end = min(len(values), idx + half + 1)
        t = timestamps[start:end].astype(float)
        g = values[start:end].astype(float)
        local_range[idx] = float(np.max(g) - np.min(g))
        if len(g) >= 2 and np.ptp(t) > 1e-12:
            centered_t = t - float(np.mean(t))
            slope = float(np.polyfit(centered_t, g, 1)[0])
        else:
            slope = 0.0
        normalized_slope[idx] = abs(slope) / thresholds["robust_gripper_range"]
        median_abs_velocity[idx] = float(np.median(np.abs(velocity[start:end]))) if len(velocity) else 0.0

    small_range = local_range <= thresholds["range_threshold"]
    very_small_range = local_range <= thresholds["range_threshold"] * 0.25
    small_slope = normalized_slope <= thresholds["normalized_slope_threshold"]
    mask = small_range & (small_slope | very_small_range)
    if config.plateau.velocity_check_enabled:
        normalized_velocity = median_abs_velocity / thresholds["robust_gripper_range"]
        mask = mask & (normalized_velocity <= thresholds["normalized_velocity_threshold"])
    return mask, local_range, normalized_slope, median_abs_velocity, thresholds


def detect_plateau_intervals(
    timestamps: np.ndarray,
    values: np.ndarray,
    plateau_mask: np.ndarray,
    thresholds: dict[str, float],
    config: GripperPhaseSegmentationConfig,
) -> list[PlateauInterval]:
    runs = _mask_runs(plateau_mask)
    intervals = [_make_plateau_interval(start, end, timestamps, values) for start, end in runs]
    intervals = [item for item in intervals if item.duration_sec >= float(config.plateau.min_duration_sec)]
    intervals = _merge_plateau_gaps(intervals, timestamps, values, thresholds, config)
    intervals = _snap_edge_plateaus(intervals, timestamps, values, thresholds, config)
    return intervals


def build_plateau_segments(
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    values: np.ndarray,
    velocity: np.ndarray,
    plateau_intervals: list[PlateauInterval],
    thresholds: dict[str, float],
    config: GripperPhaseSegmentationConfig,
) -> list[Segment]:
    segments: list[Segment] = []
    cursor = 0
    for plateau in plateau_intervals:
        if plateau.start_index > cursor:
            segments.append(
                _make_main_segment(
                    len(segments),
                    cursor,
                    plateau.start_index,
                    MainSegmentState.MOTION.value,
                    timestamps,
                    frame_indices,
                    values,
                    velocity,
                    thresholds,
                )
            )
        segments.append(
            _make_main_segment(
                len(segments),
                plateau.start_index,
                plateau.end_index,
                MainSegmentState.PLATEAU.value,
                timestamps,
                frame_indices,
                values,
                velocity,
                thresholds,
            )
        )
        cursor = plateau.end_index
    if cursor < len(values):
        segments.append(
            _make_main_segment(
                len(segments),
                cursor,
                len(values),
                MainSegmentState.MOTION.value,
                timestamps,
                frame_indices,
                values,
                velocity,
                thresholds,
            )
        )

    segments = [segment for segment in segments if segment.duration_sec > 0]
    for idx, segment in enumerate(segments):
        segment.segment_id = idx
    return segments


def suppress_same_direction_internal_plateaus(
    segments: list[Segment],
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    values: np.ndarray,
    velocity: np.ndarray,
    thresholds: dict[str, float],
    config: GripperPhaseSegmentationConfig,
) -> tuple[list[Segment], int]:
    if not config.plateau.suppress_same_direction_internal_plateaus:
        return segments, 0

    total_suppressed = 0
    current = segments
    while True:
        current, suppressed = _suppress_same_direction_internal_plateaus_once(
            current,
            timestamps,
            frame_indices,
            values,
            velocity,
            thresholds,
        )
        total_suppressed += suppressed
        if suppressed == 0:
            break
    for idx, segment in enumerate(current):
        segment.segment_id = idx
    return current, total_suppressed


def plateau_boundaries_from_segments(
    segments: list[Segment],
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    velocity: np.ndarray,
    thresholds: dict[str, float],
    side: str,
) -> list[Boundary]:
    boundaries: list[Boundary] = []
    source = f"{side}_gripper" if side in {"left", "right"} else "gripper"
    scale = Thresholds(
        enter=thresholds["range_threshold"],
        exit=thresholds["small_motion_threshold"],
        mode="plateau_edge",
        noise_level=thresholds["robust_gripper_range"],
    )
    for prev_segment, next_segment in zip(segments, segments[1:]):
        time = float(next_segment.start_time)
        frame = int(next_segment.start_frame)
        idx = int(np.argmin(np.abs(frame_indices.astype(int) - frame))) if len(frame_indices) else 0
        confidence = _plateau_boundary_confidence(prev_segment, next_segment, scale)
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


def detect_anomaly_events(
    segments: list[Segment],
    extrema: list[ExtremaPoint],
    timestamps: np.ndarray,
    values: np.ndarray,
    config: GripperPhaseSegmentationConfig,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if config.anomaly.detect_extrema:
        for point in extrema:
            events.append(
                {
                    "type": point.type,
                    "time": point.time,
                    "frame_index": point.frame_index,
                    "value": point.value,
                    "prominence": point.prominence,
                    "source": "extrema_metadata",
                }
            )
    if not config.anomaly.detect_short_reversal:
        return events

    small_threshold = estimate_plateau_thresholds(values, config)["small_motion_threshold"]
    for segment in segments:
        if segment.motion_state != MainSegmentState.MOTION.value:
            continue
        inside = [point for point in extrema if segment.start_time < point.time < segment.end_time]
        for first, second in zip(inside, inside[1:]):
            if first.type == second.type:
                continue
            duration = float(second.time - first.time)
            amplitude = abs(float(second.value - first.value))
            if duration <= float(config.trend.short_duration_sec):
                events.append(
                    {
                        "type": "short_reversal",
                        "start_time": first.time,
                        "end_time": second.time,
                        "duration_sec": duration,
                        "amplitude": amplitude,
                        "possible_jitter": bool(amplitude <= small_threshold),
                        "source": "motion_internal_metadata",
                    }
                )
    return events


def _suppress_same_direction_internal_plateaus_once(
    segments: list[Segment],
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    values: np.ndarray,
    velocity: np.ndarray,
    thresholds: dict[str, float],
) -> tuple[list[Segment], int]:
    out: list[Segment] = []
    suppressed = 0
    idx = 0
    while idx < len(segments):
        if idx + 2 < len(segments) and _is_same_direction_motion_pause(segments[idx], segments[idx + 1], segments[idx + 2]):
            start = _nearest_frame_index(frame_indices, segments[idx].start_frame)
            end = _nearest_frame_index(frame_indices, segments[idx + 2].end_frame) + 1
            out.append(
                _make_main_segment(
                    len(out),
                    start,
                    end,
                    MainSegmentState.MOTION.value,
                    timestamps,
                    frame_indices,
                    values,
                    velocity,
                    thresholds,
                )
            )
            suppressed += 1
            idx += 3
            continue
        segment = segments[idx]
        segment.segment_id = len(out)
        out.append(segment)
        idx += 1
    return out, suppressed


def _is_same_direction_motion_pause(prev_segment: Segment, plateau_segment: Segment, next_segment: Segment) -> bool:
    if prev_segment.motion_state != MainSegmentState.MOTION.value:
        return False
    if plateau_segment.motion_state != MainSegmentState.PLATEAU.value:
        return False
    if next_segment.motion_state != MainSegmentState.MOTION.value:
        return False
    if prev_segment.direction not in {TrendState.UP.value, TrendState.DOWN.value}:
        return False
    return prev_segment.direction == next_segment.direction


def _nearest_frame_index(frame_indices: np.ndarray, frame: int) -> int:
    if len(frame_indices) == 0:
        return 0
    return int(np.argmin(np.abs(frame_indices.astype(int) - int(frame))))


def _mask_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask):
        if bool(value) and start is None:
            start = idx
        if start is not None and (not bool(value) or idx == len(mask) - 1):
            end = idx if not bool(value) else idx + 1
            runs.append((start, end))
            start = None
    return runs


def _merge_plateau_gaps(
    intervals: list[PlateauInterval],
    timestamps: np.ndarray,
    values: np.ndarray,
    thresholds: dict[str, float],
    config: GripperPhaseSegmentationConfig,
) -> list[PlateauInterval]:
    if not intervals:
        return []
    merged = [intervals[0]]
    for item in intervals[1:]:
        prev = merged[-1]
        gap_sec = max(0.0, float(timestamps[item.start_index] - timestamps[prev.end_index - 1]))
        close_values = abs(prev.mean_value - item.mean_value) <= thresholds["merge_value_threshold"]
        if gap_sec <= float(config.plateau.max_internal_gap_sec) and close_values:
            merged[-1] = _make_plateau_interval(prev.start_index, item.end_index, timestamps, values)
        else:
            merged.append(item)
    return merged


def _snap_edge_plateaus(
    intervals: list[PlateauInterval],
    timestamps: np.ndarray,
    values: np.ndarray,
    thresholds: dict[str, float],
    config: GripperPhaseSegmentationConfig,
) -> list[PlateauInterval]:
    if not intervals:
        return []
    out = list(intervals)
    first = out[0]
    if first.start_index > 0:
        gap_duration = run_duration(timestamps, 0, first.start_index)
        gap_mean = float(np.mean(values[: first.start_index]))
        if gap_duration <= float(config.plateau.min_duration_sec) and abs(gap_mean - first.mean_value) <= thresholds["merge_value_threshold"]:
            out[0] = _make_plateau_interval(0, first.end_index, timestamps, values)
    last = out[-1]
    if last.end_index < len(values):
        gap_duration = run_duration(timestamps, last.end_index, len(values))
        gap_mean = float(np.mean(values[last.end_index :]))
        if gap_duration <= float(config.plateau.min_duration_sec) and abs(gap_mean - last.mean_value) <= thresholds["merge_value_threshold"]:
            out[-1] = _make_plateau_interval(last.start_index, len(values), timestamps, values)
    return out


def _make_plateau_interval(start: int, end: int, timestamps: np.ndarray, values: np.ndarray) -> PlateauInterval:
    end = max(end, start + 1)
    vals = values[start:end]
    return PlateauInterval(
        start_index=int(start),
        end_index=int(end),
        start_time=float(timestamps[start]),
        end_time=float(timestamps[end - 1]),
        mean_value=float(np.mean(vals)),
        value_range=float(np.max(vals) - np.min(vals)),
        duration_sec=run_duration(timestamps, start, end),
    )


def _make_main_segment(
    segment_id: int,
    start: int,
    end: int,
    segment_type: str,
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    values: np.ndarray,
    velocity: np.ndarray,
    thresholds: dict[str, float],
) -> Segment:
    end = max(end, start + 1)
    vals = values[start:end]
    vel = velocity[start:end]
    net_delta = float(vals[-1] - vals[0])
    value_range = float(np.max(vals) - np.min(vals))
    duration = run_duration(timestamps, start, end)
    mean_slope = net_delta / max(duration, 1e-12)
    direction = _motion_direction(net_delta, value_range, thresholds["trend_amplitude_threshold"])
    return Segment(
        segment_id=segment_id,
        start_time=float(timestamps[start]),
        end_time=float(timestamps[end - 1]),
        start_frame=int(frame_indices[start]),
        end_frame=int(frame_indices[end - 1]),
        motion_state=segment_type,
        duration_sec=duration,
        trend=direction,
        amplitude=value_range,
        net_delta=net_delta,
        max_value=float(np.max(vals)),
        min_value=float(np.min(vals)),
        mean_velocity=float(np.mean(vel)) if len(vel) else 0.0,
        peak_velocity=float(np.max(np.abs(vel))) if len(vel) else 0.0,
        segment_type=segment_type,
        start_value=float(vals[0]),
        end_value=float(vals[-1]),
        value_range=value_range,
        mean_slope=float(mean_slope),
        direction=direction if segment_type == MainSegmentState.MOTION.value else None,
        small_motion=bool(segment_type == MainSegmentState.MOTION.value and abs(net_delta) < thresholds["small_motion_threshold"]),
        plateau_mean=float(np.mean(vals)) if segment_type == MainSegmentState.PLATEAU.value else None,
    )


def _motion_direction(net_delta: float, value_range: float, threshold: float) -> str:
    if abs(net_delta) <= threshold:
        return TrendState.FLAT.value if value_range <= threshold else "MIXED"
    return TrendState.UP.value if net_delta > 0 else TrendState.DOWN.value


def _plateau_boundary_confidence(prev_segment: Segment, next_segment: Segment, thresholds: Thresholds) -> float:
    plateau_duration = max(
        prev_segment.duration_sec if prev_segment.motion_state == MainSegmentState.PLATEAU.value else 0.0,
        next_segment.duration_sec if next_segment.motion_state == MainSegmentState.PLATEAU.value else 0.0,
    )
    duration_score = min(1.0, plateau_duration / 0.5)
    change = max(abs(prev_segment.net_delta or 0.0), abs(next_segment.net_delta or 0.0), prev_segment.value_range or 0.0, next_segment.value_range or 0.0)
    change_score = min(1.0, change / max(thresholds.enter, 1e-12))
    return float(0.55 * duration_score + 0.45 * change_score)
