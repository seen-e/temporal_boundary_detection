"""Boundary extraction and segment construction."""

from __future__ import annotations

import numpy as np

from .config import GripperPhaseSegmentationConfig
from .types import Boundary, InvalidInterval, Segment, Thresholds
from .utils import clamp01, run_duration, run_lengths


def detect_boundaries(
    states: np.ndarray,
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    velocity: np.ndarray,
    thresholds: Thresholds,
    invalid_intervals: list[InvalidInterval],
) -> list[Boundary]:
    boundaries: list[Boundary] = []
    runs = run_lengths(states)
    for prev_run, next_run in zip(runs, runs[1:]):
        prev_start, prev_end, prev_state = prev_run
        next_start, next_end, next_state = next_run
        idx = next_start
        time = float(timestamps[idx])
        if any(interval.contains(time) for interval in invalid_intervals):
            continue
        confidence = boundary_confidence(
            prev_start,
            prev_end,
            next_start,
            next_end,
            timestamps,
            velocity,
            thresholds,
        )
        boundaries.append(
            Boundary(
                time=time,
                frame_index=int(frame_indices[idx]),
                from_state=prev_state,
                to_state=next_state,
                confidence=confidence,
                velocity=float(velocity[idx]),
                score=confidence,
            )
        )
    return boundaries


def boundary_confidence(
    prev_start: int,
    prev_end: int,
    next_start: int,
    next_end: int,
    timestamps: np.ndarray,
    velocity: np.ndarray,
    thresholds: Thresholds,
) -> float:
    enter = max(thresholds.enter, 1e-12)
    next_peak = float(np.nanmax(np.abs(velocity[next_start:next_end]))) if next_end > next_start else 0.0
    prev_mean = float(np.nanmean(velocity[prev_start:prev_end])) if prev_end > prev_start else 0.0
    next_mean = float(np.nanmean(velocity[next_start:next_end])) if next_end > next_start else 0.0
    duration = run_duration(timestamps, next_start, next_end)
    strength_score = clamp01((next_peak - thresholds.exit) / enter)
    duration_score = clamp01(duration / 0.30)
    jump_score = clamp01(abs(next_mean - prev_mean) / (2.0 * enter))
    return clamp01(0.5 * strength_score + 0.3 * duration_score + 0.2 * jump_score)


def merge_boundaries(
    boundaries: list[Boundary],
    merge_window_sec: float,
) -> list[Boundary]:
    if not boundaries or merge_window_sec <= 0:
        return list(boundaries)
    merged: list[Boundary] = []
    for boundary in boundaries:
        if (
            merged
            and boundary.time - merged[-1].time <= merge_window_sec
            and boundary.from_state == merged[-1].from_state
            and boundary.to_state == merged[-1].to_state
        ):
            if boundary.confidence > merged[-1].confidence:
                merged[-1] = boundary
        else:
            merged.append(boundary)
    return merged


def build_segments(
    states: np.ndarray,
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
) -> list[Segment]:
    segments: list[Segment] = []
    for segment_id, (start, end, state) in enumerate(run_lengths(states)):
        segments.append(
            Segment(
                segment_id=segment_id,
                start_time=float(timestamps[start]),
                end_time=float(timestamps[end - 1]),
                start_frame=int(frame_indices[start]),
                end_frame=int(frame_indices[end - 1]),
                motion_state=state,
                duration_sec=run_duration(timestamps, start, end),
            )
        )
    return segments

