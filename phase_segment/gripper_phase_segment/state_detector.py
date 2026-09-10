"""Hysteresis state detection and duration filtering."""

from __future__ import annotations

import numpy as np

from .config import GripperPhaseSegmentationConfig
from .types import MotionState, Thresholds
from .utils import run_duration, run_lengths


def classify_motion_states(velocity: np.ndarray, thresholds: Thresholds) -> np.ndarray:
    states: list[str] = []
    current = MotionState.STABLE.value
    for raw_v in velocity:
        v = float(raw_v) if np.isfinite(raw_v) else 0.0
        if current == MotionState.STABLE.value:
            if v >= thresholds.enter:
                current = MotionState.POSITIVE.value
            elif v <= -thresholds.enter:
                current = MotionState.NEGATIVE.value
        elif current == MotionState.POSITIVE.value:
            if v <= -thresholds.enter:
                current = MotionState.NEGATIVE.value
            elif abs(v) <= thresholds.exit:
                current = MotionState.STABLE.value
        elif current == MotionState.NEGATIVE.value:
            if v >= thresholds.enter:
                current = MotionState.POSITIVE.value
            elif abs(v) <= thresholds.exit:
                current = MotionState.STABLE.value
        states.append(current)
    return np.asarray(states, dtype=object)


def apply_temporal_filters(
    states: np.ndarray,
    timestamps: np.ndarray,
    velocity: np.ndarray,
    thresholds: Thresholds,
    config: GripperPhaseSegmentationConfig,
) -> np.ndarray:
    filtered = states.astype(object, copy=True)
    filtered = _remove_short_runs(
        filtered,
        timestamps,
        velocity,
        thresholds,
        min_duration_sec=float(config.temporal.min_state_duration_sec),
        preserve_strong_motion=True,
        config=config,
    )
    filtered = _remove_short_runs(
        filtered,
        timestamps,
        velocity,
        thresholds,
        min_duration_sec=float(config.segment.min_segment_duration_sec),
        preserve_strong_motion=True,
        config=config,
    )
    return filtered


def _remove_short_runs(
    states: np.ndarray,
    timestamps: np.ndarray,
    velocity: np.ndarray,
    thresholds: Thresholds,
    min_duration_sec: float,
    preserve_strong_motion: bool,
    config: GripperPhaseSegmentationConfig,
) -> np.ndarray:
    if min_duration_sec <= 0:
        return states

    changed = True
    out = states.astype(object, copy=True)
    while changed:
        changed = False
        runs = run_lengths(out)
        for run_idx, (start, end, state) in enumerate(runs):
            duration = run_duration(timestamps, start, end)
            if duration >= min_duration_sec:
                continue
            if preserve_strong_motion and _is_strong_motion_run(
                state,
                start,
                end,
                duration,
                velocity,
                thresholds,
                config,
            ):
                continue
            replacement = _choose_replacement_state(runs, run_idx)
            if replacement is None or replacement == state:
                continue
            out[start:end] = replacement
            changed = True
            break
    return out


def _is_strong_motion_run(
    state: str,
    start: int,
    end: int,
    duration: float,
    velocity: np.ndarray,
    thresholds: Thresholds,
    config: GripperPhaseSegmentationConfig,
) -> bool:
    if state == MotionState.STABLE.value:
        return False
    if duration < float(config.segment.min_motion_event_duration_sec):
        return False
    peak = float(np.nanmax(np.abs(velocity[start:end]))) if end > start else 0.0
    return peak >= thresholds.enter * float(config.segment.strong_motion_threshold_ratio)


def _choose_replacement_state(runs: list[tuple[int, int, str]], run_idx: int) -> str | None:
    previous_state = runs[run_idx - 1][2] if run_idx > 0 else None
    next_state = runs[run_idx + 1][2] if run_idx + 1 < len(runs) else None
    if previous_state is not None and previous_state == next_state:
        return previous_state
    if previous_state is not None and next_state is not None:
        previous_len = runs[run_idx - 1][1] - runs[run_idx - 1][0]
        next_len = runs[run_idx + 1][1] - runs[run_idx + 1][0]
        return previous_state if previous_len >= next_len else next_state
    return previous_state or next_state

