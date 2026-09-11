"""Second-pass filtering after extrema base detection.

This pass only removes extrema. It is intended for cases where MAX/MIN remain
adjacent after the traditional filter and base detection shows that one of them
has no valid monotonic structure.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


EXTREMA_TYPES = {"local_maximum", "local_minimum"}


@dataclass
class PostBaseFilterConfig:
    enabled: bool = True
    remove_failed_bases: bool = True
    enable_adjacent_extrema_filter: bool = True
    adjacent_extrema_sec: float = 0.30
    adjacent_base_overlap_sec: float = 0.0
    max_iterations: int = 3

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> "PostBaseFilterConfig":
        values = values or {}
        if "post_base_filter" in values:
            values = values.get("post_base_filter") or {}
        allowed = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in allowed})


def filter_after_extrema_bases(
    keyframes: list[dict[str, Any]],
    config: PostBaseFilterConfig | dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove weak/invalid adjacent extrema after extrema_bases are available."""

    cfg = config if isinstance(config, PostBaseFilterConfig) else PostBaseFilterConfig.from_mapping(config)
    events = [_event_record(item) for item in sorted(keyframes, key=_sort_key)]
    raw_counts = _count_types([event["record"] for event in events])
    if not cfg.enabled:
        return [event["record"] for event in events], {
            "enabled": False,
            "config": asdict(cfg),
            "input_counts": raw_counts,
            "output_counts": raw_counts,
            "removed": {"total": 0, "by_reason": {}},
            "removed_events": [],
        }

    if cfg.remove_failed_bases:
        for event in events:
            if event["type"] in EXTREMA_TYPES and _base_status(event["record"]) == "FAILED":
                _mark_removed(event, "invalid_extrema_base")

    for _ in range(max(1, int(cfg.max_iterations))):
        changed = False
        if cfg.enable_adjacent_extrema_filter:
            active_extrema = [event for event in _active(events) if event["type"] in EXTREMA_TYPES]
            for left, right in zip(active_extrema, active_extrema[1:]):
                if left["removed"] or right["removed"]:
                    continue
                reason = _adjacent_reason(left["record"], right["record"], cfg)
                if not reason:
                    continue
                weaker = _weaker_extremum(left, right)
                changed |= _mark_removed(weaker, reason)
        if not changed:
            break

    cleaned = [event["record"] for event in _active(events)]
    removed = [event["record"] for event in events if event["removed"]]
    summary = {
        "enabled": True,
        "config": asdict(cfg),
        "input_counts": raw_counts,
        "output_counts": _count_types(cleaned),
        "removed": {
            "total": len(removed),
            "by_reason": dict(sorted(Counter(reason for event in events for reason in event["remove_reasons"]).items())),
        },
        "removed_events": removed,
    }
    return cleaned, _json_safe(summary)


def _event_record(item: dict[str, Any]) -> dict[str, Any]:
    record = dict(item)
    return {"record": record, "type": str(record.get("type")), "removed": False, "remove_reasons": []}


def _mark_removed(event: dict[str, Any], reason: str) -> bool:
    if event["removed"]:
        if reason not in event["remove_reasons"]:
            event["remove_reasons"].append(reason)
        return False
    event["removed"] = True
    event["remove_reasons"].append(reason)
    record = event["record"]
    record["post_base_filter"] = {
        "status": "removed",
        "remove_reasons": list(event["remove_reasons"]),
    }
    return True


def _active(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in sorted(events, key=lambda e: _sort_key(e["record"])) if not event["removed"]]


def _adjacent_reason(left: dict[str, Any], right: dict[str, Any], cfg: PostBaseFilterConfig) -> str | None:
    dt = abs(_time(right) - _time(left))
    if dt <= float(cfg.adjacent_extrema_sec):
        return "adjacent_extrema_after_base"
    if _base_status(left) in {"VALID", "LOW_CONFIDENCE"} and _base_status(right) in {"VALID", "LOW_CONFIDENCE"}:
        overlap = _base_interval_overlap(left, right)
        if overlap is not None and overlap > float(cfg.adjacent_base_overlap_sec):
            return "overlapping_extrema_base_range"
    return None


def _weaker_extremum(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_strength = _extremum_strength(left["record"])
    right_strength = _extremum_strength(right["record"])
    if left_strength < right_strength:
        return left
    if right_strength < left_strength:
        return right
    return left if _time(left["record"]) > _time(right["record"]) else right


def _extremum_strength(record: dict[str, Any]) -> float:
    bases = record.get("extrema_bases") or {}
    status_bonus = {"VALID": 2.0, "LOW_CONFIDENCE": 1.0, "FAILED": 0.0}.get(str(bases.get("base_status")), 0.0)
    prominence = _optional_float(bases.get("prominence"))
    if prominence is None:
        prominence = _optional_float(record.get("prominence")) or 0.0
    left = bases.get("left_base") or {}
    right = bases.get("right_base") or {}
    width = float(left.get("width_sec") or 0.0) + float(right.get("width_sec") or 0.0)
    return status_bonus + float(prominence) + 0.01 * width


def _base_interval_overlap(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    li = _base_interval(left)
    ri = _base_interval(right)
    if li is None or ri is None:
        return None
    lo = max(li[0], ri[0])
    hi = min(li[1], ri[1])
    return max(0.0, hi - lo)


def _base_interval(record: dict[str, Any]) -> tuple[float, float] | None:
    bases = record.get("extrema_bases") or {}
    left = bases.get("left_base") or {}
    right = bases.get("right_base") or {}
    if left.get("time_sec") is None or right.get("time_sec") is None:
        return None
    return (float(left["time_sec"]), float(right["time_sec"]))


def _base_status(record: dict[str, Any]) -> str:
    return str((record.get("extrema_bases") or {}).get("base_status", "MISSING"))


def _sort_key(item: dict[str, Any]) -> tuple[float, int, str]:
    return (_time(item), int(item.get("sample_index", item.get("frame_index", 0)) or 0), str(item.get("type")))


def _time(item: dict[str, Any]) -> float:
    return float(item.get("time", item.get("time_sec", 0.0)) or 0.0)


def _count_types(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(item.get("type")) for item in items)
    return {
        "local_maximum": int(counts.get("local_maximum", 0)),
        "local_minimum": int(counts.get("local_minimum", 0)),
        "plateau_left_endpoint": int(counts.get("plateau_left_endpoint", 0)),
        "plateau_right_endpoint": int(counts.get("plateau_right_endpoint", 0)),
        "total": int(sum(counts.values())),
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    return value if np.isfinite(value) else None


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
