"""Detect left/right base points for already accepted gripper extrema.

This module does not detect, remove, or relabel extrema. It only annotates
existing MAX/MIN keyframes with the structural range around each extremum.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


EXTREMA_TYPES = {"local_maximum", "local_minimum"}


@dataclass
class ExtremaBaseConfig:
    enabled: bool = True
    method: str = "monotonic_walk"
    flat_tolerance_abs: float | None = None
    flat_tolerance_ratio: float = 0.0
    min_extrema_prominence_abs: float | None = None
    min_extrema_prominence_ratio: float = 0.02
    min_base_width_samples: int = 1
    max_base_duration_sec: float | None = 12.0
    max_base_width_ratio: float = 0.65
    guard_neighbor_extrema: bool = True
    include_walk_profile: bool = True
    # Kept for backward config compatibility; no longer used by monotonic_walk.
    rel_height_min: float = 0.05
    rel_height_max: float = 0.95
    rel_height_step: float = 0.05
    width_curve_smooth_window: int = 3
    knee_min_persistence: int = 2
    min_knee_score: float = 0.035
    include_width_curve: bool = False

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> "ExtremaBaseConfig":
        values = values or {}
        if "extrema_base_detection" in values:
            values = values.get("extrema_base_detection") or {}
        allowed = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in allowed})


def annotate_extrema_bases(
    keyframes: list[dict[str, Any]],
    timestamps: np.ndarray,
    smooth_values: np.ndarray,
    frame_indices: np.ndarray | None = None,
    config: ExtremaBaseConfig | dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return copied keyframes with extrema_bases attached to MAX/MIN events."""

    cfg = config if isinstance(config, ExtremaBaseConfig) else ExtremaBaseConfig.from_mapping(config)
    copied = [dict(item) for item in keyframes]
    if not cfg.enabled:
        return copied, {"enabled": False, "config": asdict(cfg), "counts": {}}

    timestamps = np.asarray(timestamps, dtype=float)
    smooth_values = np.asarray(smooth_values, dtype=float)
    if frame_indices is None:
        frame_indices = np.arange(len(smooth_values), dtype=int)
    else:
        frame_indices = np.asarray(frame_indices, dtype=int)

    scale = _signal_scale(smooth_values)
    all_extrema_indices = _active_extrema_indices(copied, len(smooth_values))
    counts = {"processed": 0, "valid": 0, "low_confidence": 0, "failed": 0}

    for item in copied:
        if str(item.get("type")) not in EXTREMA_TYPES:
            continue
        counts["processed"] += 1
        result = detect_extremum_bases(
            event=item,
            timestamps=timestamps,
            smooth_values=smooth_values,
            frame_indices=frame_indices,
            scale=scale,
            all_extrema_indices=all_extrema_indices,
            cfg=cfg,
        )
        item["extrema_bases"] = result
        status = str(result.get("base_status", "FAILED")).lower()
        if status == "valid":
            counts["valid"] += 1
        elif status == "low_confidence":
            counts["low_confidence"] += 1
        else:
            counts["failed"] += 1

    return copied, {"enabled": True, "config": asdict(cfg), "counts": counts}


def detect_extremum_bases(
    event: dict[str, Any],
    timestamps: np.ndarray,
    smooth_values: np.ndarray,
    frame_indices: np.ndarray,
    scale: float,
    all_extrema_indices: list[int],
    cfg: ExtremaBaseConfig,
) -> dict[str, Any]:
    n = len(smooth_values)
    peak_index = _event_sample_index(event, n)
    if n < 3 or peak_index <= 0 or peak_index >= n - 1:
        return _failed(event, "extremum_at_signal_boundary_or_signal_too_short")
    if not np.all(np.isfinite(smooth_values)):
        return _failed(event, "non_finite_smoothed_trajectory")

    event_type = str(event.get("type"))
    signal = -smooth_values if event_type == "local_minimum" else smooth_values
    tolerance = _threshold(cfg.flat_tolerance_abs, cfg.flat_tolerance_ratio, scale)
    prom_threshold = _threshold(cfg.min_extrema_prominence_abs, cfg.min_extrema_prominence_ratio, scale)
    prominence = _event_prominence(event, signal, peak_index)
    low_confidence = prominence < prom_threshold

    left = _walk_one_side(signal, peak_index, -1, tolerance)
    right = _walk_one_side(signal, peak_index, 1, tolerance)
    left_ip = float(left["base_index"])
    right_ip = float(right["base_index"])
    reasons = []

    if cfg.guard_neighbor_extrema:
        left_ip, right_ip, guard_reasons = _apply_neighbor_guard(left_ip, right_ip, peak_index, all_extrema_indices, n)
        reasons.extend(guard_reasons)

    if not (left_ip < peak_index < right_ip):
        return _base_result(
            event=event,
            status="FAILED",
            reason="invalid_base_order",
            peak_index=peak_index,
            prominence=prominence,
            prom_threshold=prom_threshold,
            left_walk=left,
            right_walk=right,
            timestamps=timestamps,
            smooth_values=smooth_values,
            frame_indices=frame_indices,
            cfg=cfg,
            left_ip=left_ip,
            right_ip=right_ip,
        )

    min_width = max(1, int(cfg.min_base_width_samples))
    if peak_index - left_ip < min_width or right_ip - peak_index < min_width:
        reasons.append("base_too_close_to_extremum")
        low_confidence = True

    duration = _interp_time(timestamps, right_ip) - _interp_time(timestamps, left_ip)
    if cfg.max_base_duration_sec is not None and duration > float(cfg.max_base_duration_sec):
        reasons.append("base_duration_exceeds_limit")
        low_confidence = True
    if duration <= 0:
        return _failed(event, "non_positive_base_duration")

    max_width = max(1.0, float(n - 1) * float(cfg.max_base_width_ratio))
    if (peak_index - left_ip) > max_width or (right_ip - peak_index) > max_width:
        reasons.append("base_width_exceeds_ratio_limit")
        low_confidence = True

    status = "LOW_CONFIDENCE" if low_confidence else "VALID"
    return _base_result(
        event=event,
        status=status,
        reason=";".join(reasons) if reasons else None,
        peak_index=peak_index,
        prominence=prominence,
        prom_threshold=prom_threshold,
        left_walk=left,
        right_walk=right,
        timestamps=timestamps,
        smooth_values=smooth_values,
        frame_indices=frame_indices,
        cfg=cfg,
        left_ip=left_ip,
        right_ip=right_ip,
    )


def _walk_one_side(signal: np.ndarray, peak_index: int, step: int, tolerance: float) -> dict[str, Any]:
    """Walk outward while transformed signal keeps descending from the extremum."""

    n = len(signal)
    current = int(peak_index)
    profile = []
    stop_reason = "signal_boundary"
    while True:
        nxt = current + int(step)
        if nxt < 0 or nxt >= n:
            stop_reason = "signal_boundary"
            break
        delta = float(signal[nxt] - signal[current])
        profile.append({"from_index": current, "to_index": nxt, "delta": delta})
        if delta < -float(tolerance):
            current = nxt
            continue
        stop_reason = "not_descending"
        break
    return {
        "base_index": int(current),
        "width_samples": int(abs(int(peak_index) - int(current))),
        "stop_reason": stop_reason,
        "last_delta": profile[-1]["delta"] if profile else None,
        "profile": profile,
    }


def _event_prominence(event: dict[str, Any], signal: np.ndarray, peak_index: int) -> float:
    if event.get("prominence") is not None:
        try:
            return float(event["prominence"])
        except Exception:
            pass
    try:
        from scipy.signal import peak_prominences

        prominences, _, _ = peak_prominences(signal, [peak_index])
        return float(prominences[0]) if len(prominences) else 0.0
    except Exception:
        return 0.0


def _base_result(
    event: dict[str, Any],
    status: str,
    reason: str | None,
    peak_index: int,
    prominence: float,
    prom_threshold: float,
    left_walk: dict[str, Any],
    right_walk: dict[str, Any],
    timestamps: np.ndarray,
    smooth_values: np.ndarray,
    frame_indices: np.ndarray,
    cfg: ExtremaBaseConfig,
    left_ip: float,
    right_ip: float,
) -> dict[str, Any]:
    out = {
        "base_status": status,
        "reason": reason,
        "method": "monotonic_walk_from_extremum",
        "rule": "MAX walks outward while smoothed value keeps descending; MIN negates smoothed value then uses the same rule.",
        "type": _short_type(str(event.get("type"))),
        "sample_index": int(peak_index),
        "frame_index": int(event.get("frame_index", peak_index)),
        "time_sec": float(event.get("time", event.get("time_sec", _interp_time(timestamps, peak_index)))),
        "extremum_value": float(event.get("value_smooth", smooth_values[peak_index])),
        "prominence": float(prominence),
        "prominence_threshold": float(prom_threshold),
        "flat_tolerance": float(_threshold(cfg.flat_tolerance_abs, cfg.flat_tolerance_ratio, _signal_scale(smooth_values))),
        "left_base": _base_point(left_ip, peak_index - left_ip, peak_index, timestamps, smooth_values, frame_indices),
        "right_base": _base_point(right_ip, right_ip - peak_index, peak_index, timestamps, smooth_values, frame_indices),
        "left_walk": _walk_summary(left_walk, cfg),
        "right_walk": _walk_summary(right_walk, cfg),
    }
    return _json_safe(out)


def _walk_summary(walk: dict[str, Any], cfg: ExtremaBaseConfig) -> dict[str, Any]:
    out = {k: v for k, v in walk.items() if k != "profile"}
    if cfg.include_walk_profile:
        out["profile"] = walk.get("profile", [])
    return out


def _base_point(
    ip: float,
    width: float,
    peak_index: int,
    timestamps: np.ndarray,
    smooth_values: np.ndarray,
    frame_indices: np.ndarray,
) -> dict[str, Any]:
    sample_index = int(np.clip(round(float(ip)), 0, len(smooth_values) - 1))
    return {
        "ip": float(ip),
        "sample_index": sample_index,
        "frame_index": int(frame_indices[sample_index]) if len(frame_indices) else sample_index,
        "time_sec": float(_interp_time(timestamps, ip)),
        "value": float(_interp_value(smooth_values, ip)),
        "width_samples": float(width),
        "width_sec": abs(float(_interp_time(timestamps, peak_index) - _interp_time(timestamps, ip))) if len(timestamps) else None,
    }


def _failed(event: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "base_status": "FAILED",
        "reason": reason,
        "method": "monotonic_walk_from_extremum",
        "type": _short_type(str(event.get("type"))),
        "sample_index": int(event.get("sample_index", 0) or 0),
        "frame_index": int(event.get("frame_index", event.get("sample_index", 0)) or 0),
        "time_sec": float(event.get("time", event.get("time_sec", 0.0)) or 0.0),
        "extremum_value": _optional_float(event.get("value_smooth")),
        "left_base": None,
        "right_base": None,
    }


def _apply_neighbor_guard(left_ip: float, right_ip: float, peak_index: int, extrema_indices: list[int], n: int) -> tuple[float, float, list[str]]:
    reasons = []
    prev_candidates = [idx for idx in extrema_indices if idx < peak_index]
    next_candidates = [idx for idx in extrema_indices if idx > peak_index]
    if prev_candidates:
        lower = max(prev_candidates) + 0.51
        if left_ip <= lower:
            left_ip = lower
            reasons.append("left_base_clamped_by_neighbor_extremum")
    else:
        left_ip = max(0.0, left_ip)
    if next_candidates:
        upper = min(next_candidates) - 0.51
        if right_ip >= upper:
            right_ip = upper
            reasons.append("right_base_clamped_by_neighbor_extremum")
    else:
        right_ip = min(float(n - 1), right_ip)
    return left_ip, right_ip, reasons


def _active_extrema_indices(keyframes: list[dict[str, Any]], n: int) -> list[int]:
    indices = []
    for item in keyframes:
        if str(item.get("type")) in EXTREMA_TYPES:
            indices.append(_event_sample_index(item, n))
    return sorted(set(indices))


def _event_sample_index(event: dict[str, Any], n: int) -> int:
    return int(np.clip(int(event.get("sample_index", event.get("frame_index", 0)) or 0), 0, max(0, n - 1)))


def _signal_scale(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1e-12
    return float(max(np.percentile(finite, 95) - np.percentile(finite, 5), 1e-12))


def _threshold(abs_value: float | None, ratio: float, scale: float) -> float:
    threshold = float(scale) * float(ratio)
    if abs_value is not None:
        threshold = max(threshold, float(abs_value))
    return max(threshold, 1e-12)


def _interp_time(timestamps: np.ndarray, ip: float) -> float:
    if len(timestamps) == 0:
        return float(ip)
    x = np.arange(len(timestamps), dtype=float)
    return float(np.interp(float(ip), x, timestamps.astype(float)))


def _interp_value(values: np.ndarray, ip: float) -> float:
    if len(values) == 0:
        return 0.0
    x = np.arange(len(values), dtype=float)
    return float(np.interp(float(ip), x, values.astype(float)))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    return value if np.isfinite(value) else None


def _short_type(event_type: str) -> str:
    return {"local_maximum": "MAX", "local_minimum": "MIN"}.get(event_type, event_type)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
