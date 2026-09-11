"""Small numerical helpers."""

from __future__ import annotations

import numpy as np


def estimate_sample_period(timestamps: np.ndarray) -> float:
    diffs = np.diff(timestamps.astype(float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 1.0
    return float(np.median(diffs))


def seconds_to_odd_window(window_sec: float, timestamps: np.ndarray, min_window: int = 3) -> int:
    dt = estimate_sample_period(timestamps)
    samples = int(round(max(float(window_sec), 0.0) / max(dt, 1e-12)))
    samples = max(samples, min_window)
    if samples % 2 == 0:
        samples += 1
    return samples


def run_lengths(states: np.ndarray) -> list[tuple[int, int, str]]:
    if len(states) == 0:
        return []
    runs: list[tuple[int, int, str]] = []
    start = 0
    current = str(states[0])
    for idx in range(1, len(states)):
        state = str(states[idx])
        if state != current:
            runs.append((start, idx, current))
            start = idx
            current = state
    runs.append((start, len(states), current))
    return runs


def run_duration(timestamps: np.ndarray, start: int, end: int) -> float:
    if end <= start:
        return 0.0
    dt = estimate_sample_period(timestamps)
    return float(timestamps[end - 1] - timestamps[start] + dt)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values.copy()
    window = min(window, values.size)
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values.astype(float), pad, mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def moving_median(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values.copy()
    window = min(window, values.size)
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values.astype(float), pad, mode="edge")
    out = np.empty_like(values, dtype=float)
    for idx in range(values.size):
        out[idx] = np.median(padded[idx : idx + window])
    return out


def clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))

