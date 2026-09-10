"""Conservative post-processing for gripper keyframe candidates.

The detector upstream is intentionally left untouched: it still emits extrema
and plateau endpoints. This module only annotates, merges, or removes candidate
events after they have been created.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


EXTREMA_TYPES = {"local_maximum", "local_minimum"}
PLATEAU_TYPES = {"plateau_left_endpoint", "plateau_right_endpoint"}


@dataclass
class KeypointFilterConfig:
    """Thresholds and rule toggles for post-processing candidate events."""

    enabled: bool = True
    max_iterations: int = 4
    local_window_sec: float = 0.35
    min_segment_duration_sec: float = 0.25
    tiny_segment_duration_sec: float = 0.15
    event_cluster_sec: float = 0.25
    min_plateau_duration_sec: float = 0.35
    plateau_merge_gap_sec: float = 0.25
    tiny_amplitude_abs: float | None = 0.01
    tiny_amplitude_ratio: float = 0.015
    small_amplitude_abs: float | None = 0.04
    small_amplitude_ratio: float = 0.03
    min_extrema_prominence_abs: float | None = None
    min_extrema_prominence_ratio: float = 0.02
    local_noise_k: float = 3.0
    plateau_merge_value_abs: float | None = 0.02
    plateau_merge_value_ratio: float = 0.025
    plateau_max_displacement_abs: float | None = None
    plateau_max_displacement_ratio: float = 0.06
    plateau_max_mad_abs: float | None = None
    plateau_max_mad_ratio: float = 0.04
    same_trend_slope_ratio: float = 0.035
    uncertain_confidence: float = 0.55
    weak_confidence: float = 0.70
    enable_tiny_extrema: bool = True
    enable_short_reversal: bool = True
    enable_short_plateau: bool = True
    enable_false_plateau: bool = True
    enable_plateau_merge: bool = True
    enable_event_cluster: bool = True
    enable_short_segment: bool = True
    enable_same_trend_boundary: bool = True

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> "KeypointFilterConfig":
        """Build a config from a YAML/dict section while ignoring unknown keys."""

        values = values or {}
        if "keyframe_filter" in values:
            values = values.get("keyframe_filter") or {}
        allowed = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in allowed})


@dataclass
class CandidateEvent:
    """Normalized candidate event with mutable filtering metadata."""

    event_id: str
    type: str
    kind: str
    side: str
    time: float
    frame_index: int
    sample_index: int
    value_raw: float | None
    value_smooth: float | None
    source: dict[str, Any]
    features: dict[str, Any] = field(default_factory=dict)
    status: str = "keep"
    confidence: float = 1.0
    remove_reasons: list[str] = field(default_factory=list)
    review_reasons: list[str] = field(default_factory=list)
    merged_event_ids: list[str] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.status in {"keep", "merged_representative", "uncertain"}

    def mark_removed(self, reason: str) -> bool:
        """Remove this event and record a reason. Returns True on first removal."""

        if self.status == "removed":
            if reason not in self.remove_reasons:
                self.remove_reasons.append(reason)
            return False
        self.status = "removed"
        self.confidence = 0.0
        self.remove_reasons.append(reason)
        return True

    def mark_weak(self, reason: str, confidence: float) -> None:
        """Keep the event but lower confidence for later video/VLM review."""

        if self.status != "removed":
            self.status = "uncertain"
            self.confidence = min(self.confidence, float(confidence))
            if reason not in self.review_reasons:
                self.review_reasons.append(reason)

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-safe keyframe record with filtering metadata."""

        record = dict(self.source)
        record["filter"] = {
            "event_id": self.event_id,
            "status": self.status,
            "confidence": float(self.confidence),
            "remove_reasons": list(self.remove_reasons),
            "review_reasons": list(self.review_reasons),
            "merged_event_ids": list(self.merged_event_ids),
            "features": _json_safe(self.features),
        }
        return _json_safe(record)


def filter_keyframes(
    keyframes: list[dict[str, Any]],
    timestamps: np.ndarray,
    smooth_values: np.ndarray,
    config: KeypointFilterConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Clean a single gripper's candidate keyframes without cross-hand context."""

    cfg = config if isinstance(config, KeypointFilterConfig) else KeypointFilterConfig.from_mapping(config)
    raw_counts = _count_types(keyframes)
    if not cfg.enabled:
        return {
            "enabled": False,
            "config": asdict(cfg),
            "raw_counts": raw_counts,
            "cleaned_counts": raw_counts,
            "removed": {"total": 0, "by_reason": {}},
            "merged": {"total": 0, "records": []},
            "uncertain": {"total": 0, "by_reason": {}},
            "removal_ratio": 0.0,
            "merge_ratio": 0.0,
            "removed_events": [],
            "cleaned_keyframes": list(keyframes),
            "events_debug": [],
        }

    timestamps = np.asarray(timestamps, dtype=float)
    smooth_values = np.asarray(smooth_values, dtype=float)
    events = _build_events(keyframes)
    scale = _signal_scale(smooth_values)
    merge_records: list[dict[str, Any]] = []
    _refresh_features(events, timestamps, smooth_values, cfg, scale)

    for _ in range(max(1, int(cfg.max_iterations))):
        changed = False
        if cfg.enable_short_plateau:
            changed |= _remove_short_plateaus(events, cfg)
        if cfg.enable_false_plateau:
            changed |= _remove_false_plateaus(events, cfg, scale)
        if cfg.enable_plateau_merge:
            changed |= _merge_fragmented_plateaus(events, cfg, scale, merge_records)
        if cfg.enable_event_cluster:
            changed |= _merge_event_clusters(events, cfg, scale, merge_records)
        if cfg.enable_tiny_extrema:
            changed |= _remove_tiny_extrema(events, cfg, scale)
        if cfg.enable_short_reversal:
            changed |= _handle_short_reversals(events, cfg, scale)
        if cfg.enable_short_segment:
            changed |= _remove_short_small_segments(events, cfg, scale)
        if cfg.enable_same_trend_boundary:
            changed |= _downgrade_same_trend_boundaries(events, cfg, scale)
        _refresh_features(events, timestamps, smooth_values, cfg, scale)
        if not changed:
            break

    _remove_orphan_plateau_endpoints(events)
    cleaned = [event.to_record() for event in _active_events(events)]
    removed = [event.to_record() for event in events if event.status == "removed"]
    stats = _filter_stats(keyframes, cleaned, removed, events, merge_records, cfg)
    stats["cleaned_keyframes"] = cleaned
    stats["removed_events"] = removed
    stats["events_debug"] = [event.to_record() for event in events]
    return stats


def _build_events(keyframes: list[dict[str, Any]]) -> list[CandidateEvent]:
    events = []
    for idx, item in enumerate(sorted(keyframes, key=lambda x: (float(x["time"]), int(x["sample_index"]), str(x["type"])))):
        events.append(
            CandidateEvent(
                event_id=f"{item.get('side', 'gripper')}_{idx:04d}_{_short_type(str(item['type']))}",
                type=str(item["type"]),
                kind=str(item.get("kind", "")),
                side=str(item.get("side", "")),
                time=float(item["time"]),
                frame_index=int(item["frame_index"]),
                sample_index=int(item["sample_index"]),
                value_raw=_optional_float(item.get("value_raw")),
                value_smooth=_optional_float(item.get("value_smooth")),
                source=dict(item),
            )
        )
    return events


def _refresh_features(
    events: list[CandidateEvent],
    timestamps: np.ndarray,
    smooth_values: np.ndarray,
    cfg: KeypointFilterConfig,
    scale: float,
) -> None:
    active = _active_events(events)
    for i, event in enumerate(active):
        prev_event = active[i - 1] if i > 0 else None
        next_event = active[i + 1] if i + 1 < len(active) else None
        idx = int(np.clip(event.sample_index, 0, max(0, len(smooth_values) - 1)))
        left_slice = _time_window_indices(timestamps, event.time - cfg.local_window_sec, event.time)
        right_slice = _time_window_indices(timestamps, event.time, event.time + cfg.local_window_sec)
        event.features.update(
            {
                "dynamic_range": scale,
                "prev_event_dt": None if prev_event is None else float(event.time - prev_event.time),
                "next_event_dt": None if next_event is None else float(next_event.time - event.time),
                "prev_event_dx": None if prev_event is None else _safe_dx(prev_event.value_smooth, event.value_smooth),
                "next_event_dx": None if next_event is None else _safe_dx(event.value_smooth, next_event.value_smooth),
                "local_mad": _local_mad(timestamps, smooth_values, event.time, cfg.local_window_sec),
                "left_slope": _slope(timestamps[left_slice], smooth_values[left_slice]),
                "right_slope": _slope(timestamps[right_slice], smooth_values[right_slice]),
                "local_value": float(smooth_values[idx]) if len(smooth_values) else event.value_smooth,
            }
        )
        if event.type in PLATEAU_TYPES:
            _add_plateau_features(event, timestamps, smooth_values)


def _add_plateau_features(event: CandidateEvent, timestamps: np.ndarray, smooth_values: np.ndarray) -> None:
    start_time = _optional_float(event.source.get("plateau_start_time"))
    end_time = _optional_float(event.source.get("plateau_end_time"))
    if start_time is None or end_time is None:
        return
    sl = _time_window_indices(timestamps, start_time, end_time)
    vals = smooth_values[sl]
    event.features.update(
        {
            "plateau_duration": float(max(0.0, end_time - start_time)),
            "plateau_median": float(np.median(vals)) if len(vals) else event.source.get("plateau_mean"),
            "plateau_mad": _mad(vals),
            "plateau_slope": _slope(timestamps[sl], vals),
            "plateau_displacement": float(vals[-1] - vals[0]) if len(vals) >= 2 else 0.0,
            "plateau_value_range": _optional_float(event.source.get("plateau_value_range")),
            "plateau_id": event.source.get("plateau_id"),
        }
    )


def _remove_short_plateaus(events: list[CandidateEvent], cfg: KeypointFilterConfig) -> bool:
    changed = False
    for pair in _plateau_pairs(events):
        duration = float(pair[0].features.get("plateau_duration") or pair[0].source.get("plateau_duration_sec") or 0.0)
        value_range = float(pair[0].features.get("plateau_value_range") or 0.0)
        tiny_amp = _threshold(cfg.tiny_amplitude_abs, cfg.tiny_amplitude_ratio, float(pair[0].features["dynamic_range"]))
        if duration < cfg.min_plateau_duration_sec and value_range <= max(tiny_amp, 1e-12):
            for event in pair:
                changed |= event.mark_removed("short_plateau")
    return changed


def _remove_false_plateaus(events: list[CandidateEvent], cfg: KeypointFilterConfig, scale: float) -> bool:
    changed = False
    max_disp = _threshold(cfg.plateau_max_displacement_abs, cfg.plateau_max_displacement_ratio, scale)
    max_mad = _threshold(cfg.plateau_max_mad_abs, cfg.plateau_max_mad_ratio, scale)
    for pair in _plateau_pairs(events):
        left = pair[0]
        duration = max(float(left.features.get("plateau_duration") or 0.0), 1e-12)
        displacement = abs(float(left.features.get("plateau_displacement") or 0.0))
        mad = float(left.features.get("plateau_mad") or 0.0)
        slope_motion = displacement / duration
        slope_limit = cfg.same_trend_slope_ratio * scale
        if displacement > max_disp and mad > max_mad and slope_motion > slope_limit:
            for event in pair:
                changed |= event.mark_removed("false_plateau")
    return changed


def _merge_fragmented_plateaus(
    events: list[CandidateEvent],
    cfg: KeypointFilterConfig,
    scale: float,
    merge_records: list[dict[str, Any]],
) -> bool:
    changed = False
    pairs = _plateau_pairs(events)
    value_threshold = _threshold(cfg.plateau_merge_value_abs, cfg.plateau_merge_value_ratio, scale)
    for first, second in zip(pairs, pairs[1:]):
        pr1, pl2 = first[1], second[0]
        if pr1.status == "removed" or pl2.status == "removed":
            continue
        gap = max(0.0, pl2.time - pr1.time)
        mean1 = _optional_float(first[0].source.get("plateau_mean"))
        mean2 = _optional_float(second[0].source.get("plateau_mean"))
        if mean1 is None or mean2 is None:
            continue
        if gap <= cfg.plateau_merge_gap_sec and abs(mean1 - mean2) <= value_threshold:
            changed |= pr1.mark_removed("plateau_fragmentation")
            changed |= pl2.mark_removed("plateau_fragmentation")
            first[0].source["plateau_end_time"] = second[1].source.get("plateau_end_time")
            first[0].source["plateau_duration_sec"] = max(0.0, float(second[1].time - first[0].time))
            first[0].source["merged_plateau_ids"] = [first[0].source.get("plateau_id"), second[1].source.get("plateau_id")]
            second[1].source["plateau_id"] = first[0].source.get("plateau_id")
            second[1].source["plateau_start_time"] = first[0].source.get("plateau_start_time")
            second[1].source["plateau_duration_sec"] = max(0.0, float(second[1].time - first[0].time))
            second[1].source["merged_plateau_ids"] = [first[0].source.get("plateau_id"), pl2.source.get("plateau_id")]
            first[0].merged_event_ids.extend([pr1.event_id, pl2.event_id])
            second[1].merged_event_ids.extend([pr1.event_id, pl2.event_id])
            merge_records.append(
                {
                    "reason": "plateau_fragmentation",
                    "kept_event_ids": [first[0].event_id, second[1].event_id],
                    "removed_event_ids": [pr1.event_id, pl2.event_id],
                    "gap_sec": gap,
                    "plateau_mean_diff": abs(mean1 - mean2),
                }
            )
    return changed


def _merge_event_clusters(
    events: list[CandidateEvent],
    cfg: KeypointFilterConfig,
    scale: float,
    merge_records: list[dict[str, Any]],
) -> bool:
    changed = False
    for cluster in _event_clusters(_active_events(events), cfg.event_cluster_sec):
        if len(cluster) < 2:
            continue
        values = [event.value_smooth for event in cluster if event.value_smooth is not None]
        value_span = max(values) - min(values) if values else 0.0
        same_boundary = any(event.type in EXTREMA_TYPES for event in cluster) and any(event.type in PLATEAU_TYPES for event in cluster)
        small_value_span = value_span <= _threshold(cfg.small_amplitude_abs, cfg.small_amplitude_ratio, scale)
        pure_extrema_cluster = all(event.type in EXTREMA_TYPES for event in cluster)
        if not (same_boundary or (pure_extrema_cluster and small_value_span)):
            continue
        keep = max(cluster, key=_event_strength)
        keep.status = "merged_representative"
        removed_ids = []
        for event in cluster:
            if event is keep:
                continue
            changed |= event.mark_removed("duplicate_event_cluster")
            keep.merged_event_ids.append(event.event_id)
            removed_ids.append(event.event_id)
        if removed_ids:
            merge_records.append(
                {
                    "reason": "duplicate_event_cluster",
                    "kept_event_ids": [keep.event_id],
                    "removed_event_ids": removed_ids,
                    "cluster_start_time": min(event.time for event in cluster),
                    "cluster_end_time": max(event.time for event in cluster),
                }
            )
    return changed


def _remove_tiny_extrema(events: list[CandidateEvent], cfg: KeypointFilterConfig, scale: float) -> bool:
    changed = False
    prom_threshold = _threshold(cfg.min_extrema_prominence_abs, cfg.min_extrema_prominence_ratio, scale)
    amp_threshold = _threshold(cfg.tiny_amplitude_abs, cfg.tiny_amplitude_ratio, scale)
    for event in _active_events(events):
        if event.type not in EXTREMA_TYPES:
            continue
        prominence = float(event.source.get("prominence") or 0.0)
        local_noise = float(event.features.get("local_mad") or 0.0) * cfg.local_noise_k
        local_amp = _local_event_amplitude(event)
        if prominence < prom_threshold and local_amp <= max(amp_threshold, local_noise):
            changed |= event.mark_removed("tiny_extrema")
    return changed


def _handle_short_reversals(events: list[CandidateEvent], cfg: KeypointFilterConfig, scale: float) -> bool:
    changed = False
    active = _active_events(events)
    amp_threshold = _threshold(cfg.small_amplitude_abs, cfg.small_amplitude_ratio, scale)
    for prev_event, event, next_event in zip(active, active[1:], active[2:]):
        if event.type not in EXTREMA_TYPES:
            continue
        duration = next_event.time - prev_event.time
        amplitude = _local_event_amplitude(event)
        if duration <= cfg.min_segment_duration_sec and amplitude <= amp_threshold:
            changed |= event.mark_removed("short_reversal")
        elif duration <= cfg.min_segment_duration_sec and amplitude > amp_threshold:
            event.mark_weak("short_but_significant_reversal", cfg.uncertain_confidence)
    return changed


def _remove_short_small_segments(events: list[CandidateEvent], cfg: KeypointFilterConfig, scale: float) -> bool:
    changed = False
    active = _active_events(events)
    amp_threshold = _threshold(cfg.small_amplitude_abs, cfg.small_amplitude_ratio, scale)
    for left, right in zip(active, active[1:]):
        dt = right.time - left.time
        dx = abs(_safe_dx(left.value_smooth, right.value_smooth) or 0.0)
        if dt <= cfg.tiny_segment_duration_sec and dx <= amp_threshold:
            extrema = [event for event in (left, right) if event.type in EXTREMA_TYPES]
            weaker = min(extrema or [left, right], key=_event_strength)
            changed |= weaker.mark_removed("small_dt_small_dx")
    return changed


def _downgrade_same_trend_boundaries(events: list[CandidateEvent], cfg: KeypointFilterConfig, scale: float) -> bool:
    changed = False
    slope_threshold = cfg.same_trend_slope_ratio * scale
    amp_threshold = _threshold(cfg.small_amplitude_abs, cfg.small_amplitude_ratio, scale)
    for event in _active_events(events):
        left_slope = _optional_float(event.features.get("left_slope")) or 0.0
        right_slope = _optional_float(event.features.get("right_slope")) or 0.0
        same_direction = abs(left_slope) > slope_threshold and abs(right_slope) > slope_threshold and np.sign(left_slope) == np.sign(right_slope)
        if same_direction and _local_event_amplitude(event) <= amp_threshold:
            if event.type in EXTREMA_TYPES:
                changed |= event.mark_removed("same_trend_boundary")
            else:
                event.mark_weak("same_trend_boundary", cfg.weak_confidence)
    return changed


def _plateau_pairs(events: list[CandidateEvent]) -> list[tuple[CandidateEvent, CandidateEvent]]:
    by_id: dict[Any, dict[str, CandidateEvent]] = {}
    for event in _active_events(events):
        if event.type not in PLATEAU_TYPES:
            continue
        plateau_id = event.source.get("plateau_id")
        if plateau_id is None:
            continue
        by_id.setdefault(plateau_id, {})[event.type] = event
    pairs = []
    for items in by_id.values():
        left = items.get("plateau_left_endpoint")
        right = items.get("plateau_right_endpoint")
        if left is not None and right is not None and left.time <= right.time:
            pairs.append((left, right))
    return sorted(pairs, key=lambda pair: pair[0].time)


def _event_clusters(events: list[CandidateEvent], threshold_sec: float) -> list[list[CandidateEvent]]:
    clusters = []
    current: list[CandidateEvent] = []
    for event in events:
        if not current or event.time - current[-1].time <= threshold_sec:
            current.append(event)
        else:
            clusters.append(current)
            current = [event]
    if current:
        clusters.append(current)
    return clusters


def _filter_stats(
    raw_keyframes: list[dict[str, Any]],
    cleaned: list[dict[str, Any]],
    removed: list[dict[str, Any]],
    events: list[CandidateEvent],
    merge_records: list[dict[str, Any]],
    cfg: KeypointFilterConfig,
) -> dict[str, Any]:
    removed_reasons = Counter(reason for event in events for reason in event.remove_reasons)
    review_reasons = Counter(reason for event in events for reason in event.review_reasons if event.status != "removed")
    raw_total = len(raw_keyframes)
    return {
        "enabled": True,
        "config": asdict(cfg),
        "raw_counts": _count_types(raw_keyframes),
        "cleaned_counts": _count_types(cleaned),
        "removed": {"total": len(removed), "by_reason": dict(sorted(removed_reasons.items()))},
        "merged": {"total": len(merge_records), "records": merge_records},
        "uncertain": {
            "total": sum(1 for event in events if event.status == "uncertain"),
            "by_reason": dict(sorted(review_reasons.items())),
        },
        "removal_ratio": float(len(removed) / raw_total) if raw_total else 0.0,
        "merge_ratio": float(len(merge_records) / raw_total) if raw_total else 0.0,
    }


def _count_types(events: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(item.get("type")) for item in events)
    return {
        "local_maximum": int(counts.get("local_maximum", 0)),
        "local_minimum": int(counts.get("local_minimum", 0)),
        "plateau_left_endpoint": int(counts.get("plateau_left_endpoint", 0)),
        "plateau_right_endpoint": int(counts.get("plateau_right_endpoint", 0)),
        "total": int(sum(counts.values())),
    }


def _active_events(events: list[CandidateEvent]) -> list[CandidateEvent]:
    return sorted([event for event in events if event.active], key=lambda event: (event.time, event.sample_index, event.type))


def _event_strength(event: CandidateEvent) -> float:
    score = 0.0
    if event.type in EXTREMA_TYPES:
        score += 10.0 + float(event.source.get("prominence") or 0.0)
    if event.type in PLATEAU_TYPES:
        score += 12.0 + float(event.source.get("plateau_duration_sec") or 0.0)
    score += abs(float(event.value_smooth or 0.0)) * 1e-3
    return score


def _local_event_amplitude(event: CandidateEvent) -> float:
    values = [abs(v) for v in [event.features.get("prev_event_dx"), event.features.get("next_event_dx")] if v is not None]
    return float(min(values)) if values else 0.0


def _signal_scale(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1e-12
    return float(max(np.percentile(finite, 95) - np.percentile(finite, 5), 1e-12))


def _remove_orphan_plateau_endpoints(events: list[CandidateEvent]) -> None:
    """Final consistency pass: cleaned plateaus must keep legal PL/PR pairs."""

    by_id: dict[Any, list[CandidateEvent]] = {}
    for event in _active_events(events):
        if event.type in PLATEAU_TYPES:
            by_id.setdefault(event.source.get("plateau_id"), []).append(event)
    for items in by_id.values():
        types = {event.type for event in items}
        if types == PLATEAU_TYPES and len(items) == 2:
            left = next(event for event in items if event.type == "plateau_left_endpoint")
            right = next(event for event in items if event.type == "plateau_right_endpoint")
            if left.time <= right.time:
                continue
        for event in items:
            event.mark_removed("orphan_plateau_endpoint")


def _threshold(abs_value: float | None, ratio: float, scale: float) -> float:
    threshold = float(scale) * float(ratio)
    if abs_value is not None:
        threshold = max(threshold, float(abs_value))
    return max(threshold, 1e-12)


def _time_window_indices(timestamps: np.ndarray, start: float, end: float) -> np.ndarray:
    if len(timestamps) == 0:
        return np.asarray([], dtype=bool)
    lo = min(float(start), float(end))
    hi = max(float(start), float(end))
    mask = (timestamps >= lo) & (timestamps <= hi)
    if not np.any(mask):
        nearest = int(np.argmin(np.abs(timestamps - ((lo + hi) / 2.0))))
        mask[nearest] = True
    return mask


def _slope(timestamps: np.ndarray, values: np.ndarray) -> float:
    if len(values) < 2 or np.ptp(timestamps) <= 1e-12:
        return 0.0
    centered_t = timestamps.astype(float) - float(np.mean(timestamps))
    return float(np.polyfit(centered_t, values.astype(float), 1)[0])


def _local_mad(timestamps: np.ndarray, values: np.ndarray, time: float, window_sec: float) -> float:
    mask = _time_window_indices(timestamps, time - window_sec, time + window_sec)
    return _mad(values[mask])


def _mad(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def _safe_dx(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(right) - float(left)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    return value if np.isfinite(value) else None


def _short_type(event_type: str) -> str:
    return {
        "local_maximum": "MAX",
        "local_minimum": "MIN",
        "plateau_left_endpoint": "PL",
        "plateau_right_endpoint": "PR",
    }.get(event_type, event_type)


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
