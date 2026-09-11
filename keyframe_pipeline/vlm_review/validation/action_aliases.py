from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..core.models import ReviewSlice, display_id_from_event_id


EVENT_ACTION_ALIASES = {
    "VALID": "KEEP",
    "INVALID": "REMOVE",
}

INTERVAL_ACTION_ALIASES = {
    "NO_ADD": "NO_MISSING_EVENT",
}

STATE_ALIASES = {
    "STABLE_WITH_FLUCTUATION": "STABLE",
    "QUASI_PLATEAU": "STABLE",
    "PLATEAU": "STABLE",
}

SEGMENT_TYPES = {
    "稳定打开",
    "稳定关闭",
    "打开中",
    "关闭中",
    "稳定打开-带波动",
    "稳定关闭-带波动",
    "打开中-带波动",
    "关闭中-带波动",
}

ADDITION_TYPE_MAP = {
    "平台左端点": "PL",
    "平台右端点": "PR",
    "极大值": "MAX",
    "极小值": "MIN",
}


def normalize_review_actions(
    review: Dict[str, Any],
    review_slice: Optional[ReviewSlice] = None,
    frozen_segments: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Normalize supported VLM schemas into the internal review format."""

    if isinstance(review, dict) and "segments" in review and "region" not in review:
        if review_slice is None:
            raise ValueError("review_slice is required for pass2 segment selection output")
        return _normalize_pass2_schema(review, review_slice, frozen_segments or [])

    if isinstance(review, dict) and "region" in review:
        if review_slice is None:
            raise ValueError("review_slice is required for segment-style VLM output")
        return _normalize_segment_schema(review, review_slice)

    normalized = deepcopy(review)

    if "keyframe_results" in normalized and "event_reviews" not in normalized:
        normalized["event_reviews"] = [_normalize_event_result(item) for item in normalized.get("keyframe_results") or []]
    elif "event_results" in normalized and "event_reviews" not in normalized:
        normalized["event_reviews"] = [_normalize_event_result(item) for item in normalized.get("event_results") or []]
    else:
        normalized["event_reviews"] = [_normalize_event_review(item) for item in normalized.get("event_reviews") or []]

    if "add_results" in normalized and "interval_reviews" not in normalized:
        normalized["interval_reviews"] = [_normalize_add_result(item) for item in normalized.get("add_results") or []]
    else:
        normalized["interval_reviews"] = [_normalize_interval_review(item) for item in normalized.get("interval_reviews") or []]

    return normalized


def normalize_pass1_structure(review: Dict[str, Any]) -> Dict[str, Any]:
    raw = deepcopy(review)
    warnings: List[Dict[str, str]] = []
    region = raw.get("region")
    if not isinstance(region, dict):
        warnings.append(_warning("region_not_object", "region must be an object", "region"))
        region = {}
    if "keyframes" in region:
        warnings.append(_warning("pass1_keyframes_forbidden", "Pass 1 must not output keyframes", "region.keyframes"))
        region.pop("keyframes", None)
    if "events" in region:
        warnings.append(_warning("pass1_events_forbidden", "Pass 1 must not output events", "region.events"))
        region.pop("events", None)
    if "additions" in raw:
        warnings.append(_warning("pass1_additions_forbidden", "Pass 1 must not output additions", "additions"))
    normalized_segments: List[Dict[str, Any]] = []
    for idx, segment in enumerate(_as_list(region.get("segments"))):
        path = f"region.segments[{idx}]"
        if not isinstance(segment, dict):
            warnings.append(_warning("segment_not_object", "segment must be an object", path))
            continue
        seg = deepcopy(segment)
        if seg.get("type") not in SEGMENT_TYPES:
            warnings.append(_warning("unknown_segment_type", f"unknown segment type {seg.get('type')}", f"{path}.type"))
        if "events" in seg:
            warnings.append(_warning("pass1_segment_events_forbidden", "Pass 1 segment must not contain events", f"{path}.events"))
            seg.pop("events", None)
        seg.setdefault("id", f"{idx + 1:03d}")
        _normalize_segment_time(seg, path, warnings)
        normalized_segments.append(seg)
    region["segments"] = normalized_segments
    needs_review = bool(raw.get("needs_review")) or bool(warnings) or not normalized_segments
    return {
        "schema_version": "pass1_structure_v1",
        "region": region,
        "warnings": warnings,
        "needs_review": needs_review,
        "manual_review": needs_review,
        "raw_vlm_review": raw,
    }


def _normalize_segment_time(segment: Dict[str, Any], path: str, warnings: List[Dict[str, str]]) -> None:
    for key in ("start_time_sec", "end_time_sec"):
        value, ok = _coerce_float(segment.get(key))
        if not ok:
            warnings.append(_warning("pass1_segment_time_missing", f"Pass 1 segment requires numeric {key}", f"{path}.{key}"))
            continue
        segment[key] = value
    if isinstance(segment.get("start_time_sec"), (int, float)) and isinstance(segment.get("end_time_sec"), (int, float)):
        if float(segment["end_time_sec"]) < float(segment["start_time_sec"]):
            warnings.append(_warning("pass1_segment_time_reversed", "Pass 1 segment end_time_sec is before start_time_sec", path))
            segment["start_time_sec"], segment["end_time_sec"] = segment["end_time_sec"], segment["start_time_sec"]


def _normalize_pass2_schema(
    review: Dict[str, Any],
    review_slice: ReviewSlice,
    frozen_segments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    raw = deepcopy(review)
    warnings: List[Dict[str, str]] = []
    frozen_by_id = {str(seg.get("id")): seg for seg in frozen_segments if isinstance(seg, dict)}
    output_by_id = {str(seg.get("id")): seg for seg in _as_list(raw.get("segments")) if isinstance(seg, dict)}
    for seg_id in sorted(set(frozen_by_id) - set(output_by_id)):
        warnings.append(_warning("pass2_missing_frozen_segment", f"Pass 2 missing frozen segment {seg_id}", "segments"))
    for seg_id in sorted(set(output_by_id) - set(frozen_by_id)):
        warnings.append(_warning("pass2_extra_segment", f"Pass 2 output extra segment {seg_id}", "segments"))
    merged_segments: List[Dict[str, Any]] = []
    for frozen in frozen_segments:
        if not isinstance(frozen, dict):
            continue
        seg_id = str(frozen.get("id"))
        out_seg = output_by_id.get(seg_id, {})
        if out_seg.get("type") is not None and out_seg.get("type") != frozen.get("type"):
            warnings.append(_warning("pass2_changed_segment_type", f"Pass 2 changed frozen segment {seg_id} type", f"segments.{seg_id}.type"))
        merged = deepcopy(frozen)
        merged["events"] = out_seg.get("events", []) if isinstance(out_seg, dict) else []
        merged_segments.append(merged)
    wrapped = {
        "region": {
            "segments": merged_segments,
            "keyframes": raw.get("keyframes", []),
        },
        "additions": [],
        "needs_review": bool(raw.get("needs_review")) or bool(warnings),
    }
    normalized = _normalize_segment_schema(wrapped, review_slice)
    normalized["schema_version"] = "pass2_selection_v1"
    normalized["pass2_keyframes"] = raw.get("keyframes", [])
    normalized["frozen_segments"] = deepcopy(frozen_segments)
    normalized["raw_pass2_review"] = raw
    normalized["warnings"] = [*warnings, *normalized.get("warnings", [])]
    normalized["needs_review"] = bool(normalized.get("needs_review")) or bool(normalized["warnings"])
    normalized["manual_review"] = normalized["needs_review"]
    return normalized


def _normalize_segment_schema(review: Dict[str, Any], review_slice: ReviewSlice) -> Dict[str, Any]:
    raw = deepcopy(review)
    warnings: List[Dict[str, str]] = []
    target_by_display = {event.display_id: event for event in review_slice.targets}
    left_overlap_by_display = {event.display_id: event for event in review_slice.context_left}
    right_overlap_by_display = {event.display_id: event for event in review_slice.context_right}
    context_by_display = {**left_overlap_by_display, **right_overlap_by_display}
    local_by_display = {**left_overlap_by_display, **target_by_display, **right_overlap_by_display}
    visible_display_ids = set(local_by_display)
    owned_intervals = {_display_interval_from_internal(interval.interval_id): interval for interval in review_slice.owned_intervals}

    region = raw.get("region")
    if not isinstance(region, dict):
        warnings.append(_warning("region_not_object", "region must be an object", "region"))
        region = {}

    _validate_descriptive_keyframes(region.get("keyframes"), target_by_display, context_by_display, warnings)

    selected_local_display_ids: set[str] = set()
    normalized_segments: List[Dict[str, Any]] = []
    for seg_idx, segment in enumerate(_as_list(region.get("segments"))):
        path = f"region.segments[{seg_idx}]"
        if not isinstance(segment, dict):
            warnings.append(_warning("segment_not_object", "segment must be an object", path))
            continue
        segment_type = segment.get("type")
        if segment_type not in SEGMENT_TYPES:
            warnings.append(_warning("unknown_segment_type", f"unknown segment type {segment_type}", f"{path}.type"))
        segment_events: List[str] = []
        for event_idx, raw_id in enumerate(_as_list(segment.get("events"))):
            id_path = f"{path}.events[{event_idx}]"
            display_id, id_warning = _normalize_display_id(raw_id)
            if id_warning:
                warnings.append(_warning("non_numeric_display_id", id_warning, id_path))
            if display_id in local_by_display:
                selected_local_display_ids.add(display_id)
                segment_events.append(display_id)
            elif display_id:
                warnings.append(_warning("unknown_event_id", f"unknown keyframe id {display_id}", id_path))
            else:
                warnings.append(_warning("bad_event_id", f"bad keyframe id {raw_id!r}", id_path))
        normalized_segment = deepcopy(segment)
        normalized_segment["events"] = segment_events
        normalized_segments.append(normalized_segment)

    target_display_ids = [event.display_id for event in review_slice.targets]
    kept_display_ids = selected_local_display_ids & set(target_display_ids)
    deleted_display_ids = [display_id for display_id in target_display_ids if display_id not in kept_display_ids]
    kept_event_ids = [target_by_display[display_id].event_id for display_id in target_display_ids if display_id in kept_display_ids]
    deleted_event_ids = [target_by_display[display_id].event_id for display_id in deleted_display_ids]

    event_reviews = []
    for event in review_slice.targets:
        keep = event.display_id in kept_display_ids
        event_reviews.append(
            {
                "event_id": event.event_id,
                "keyframe_id": event.event_id,
                "display_id": event.display_id,
                "action": "KEEP" if keep else "REMOVE",
                "state_before": "UNCERTAIN",
                "state_after": "UNCERTAIN",
                "confidence": "MEDIUM",
                "reason": (
                    "Kept because this keyframe id appears in region.segments[].events."
                    if keep
                    else "Removed because this target keyframe id does not appear in any region.segments[].events."
                ),
            }
        )

    normalized_additions, interval_reviews = _normalize_additions(raw.get("additions"), owned_intervals, visible_display_ids, warnings)

    region_out = deepcopy(region)
    region_out["segments"] = normalized_segments
    selected_local_ids = [
        local_by_display[display_id].event_id
        for display_id in [event.display_id for event in review_slice.all_events]
        if display_id in selected_local_display_ids
    ]
    needs_review = bool(raw.get("needs_review")) or bool(warnings)
    return {
        "schema_version": "segment_events_v1",
        "region": region_out,
        "left_overlap_ids": [event.event_id for event in review_slice.context_left],
        "target_ids": [event.event_id for event in review_slice.targets],
        "right_overlap_ids": [event.event_id for event in review_slice.context_right],
        "left_overlap_display_ids": [event.display_id for event in review_slice.context_left],
        "target_display_ids": target_display_ids,
        "right_overlap_display_ids": [event.display_id for event in review_slice.context_right],
        "selected_local_ids": selected_local_ids,
        "selected_local_display_ids": [
            event.display_id for event in review_slice.all_events if event.display_id in selected_local_display_ids
        ],
        "kept_event_ids": kept_event_ids,
        "deleted_event_ids": deleted_event_ids,
        "kept_keyframe_ids": kept_event_ids,
        "deleted_keyframe_ids": deleted_event_ids,
        "kept_display_ids": [display_id for display_id in target_display_ids if display_id in kept_display_ids],
        "deleted_display_ids": deleted_display_ids,
        "additions": normalized_additions,
        "event_reviews": event_reviews,
        "interval_reviews": interval_reviews,
        "warnings": warnings,
        "needs_review": needs_review,
        "manual_review": needs_review,
        "raw_vlm_review": raw,
    }


def _validate_descriptive_keyframes(
    keyframes: Any,
    target_by_display: Dict[str, Any],
    context_by_display: Dict[str, Any],
    warnings: List[Dict[str, str]],
) -> None:
    if keyframes is None:
        return
    for idx, item in enumerate(_as_list(keyframes)):
        path = f"region.keyframes[{idx}].id"
        if not isinstance(item, dict):
            warnings.append(_warning("keyframe_not_object", "region.keyframes item must be an object", f"region.keyframes[{idx}]"))
            continue
        display_id, id_warning = _normalize_display_id(item.get("id"))
        if id_warning:
            warnings.append(_warning("non_numeric_display_id", id_warning, path))
        if display_id in target_by_display or display_id in context_by_display:
            continue
        if display_id:
            warnings.append(_warning("unknown_keyframe_description", f"unknown keyframe id {display_id}", path))
        else:
            warnings.append(_warning("bad_keyframe_description_id", f"bad keyframe id {item.get('id')!r}", path))


def _normalize_additions(
    additions: Any,
    owned_intervals: Dict[str, Any],
    visible_display_ids: Iterable[str],
    warnings: List[Dict[str, str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    normalized_additions: List[Dict[str, Any]] = []
    interval_reviews: List[Dict[str, Any]] = []
    visible_display_ids = set(visible_display_ids)
    for idx, item in enumerate(_as_list(additions)):
        path = f"additions[{idx}]"
        if not isinstance(item, dict):
            warnings.append(_warning("addition_not_object", "addition must be an object", path))
            continue
        interval_display = str(item.get("interval", item.get("interval_id", ""))).strip()
        left_display, right_display = _parse_display_interval(interval_display)
        if not left_display or not right_display:
            warnings.append(_warning("bad_addition_interval", f"bad addition interval {interval_display!r}", f"{path}.interval"))
            continue
        if left_display not in visible_display_ids or right_display not in visible_display_ids:
            warnings.append(_warning("addition_interval_unknown_event", f"addition interval {interval_display} contains unknown keyframe", f"{path}.interval"))
            continue
        canonical_interval = f"{left_display}_{right_display}"
        interval = owned_intervals.get(canonical_interval)
        if interval is None:
            warnings.append(_warning("addition_interval_not_owned", f"addition interval {canonical_interval} is not owned by this target slice", f"{path}.interval"))
            continue
        suggested_type = ADDITION_TYPE_MAP.get(str(item.get("type", "")).strip())
        if suggested_type is None:
            warnings.append(_warning("unknown_addition_type", f"unknown addition type {item.get('type')}", f"{path}.type"))
            continue
        approx_time, time_ok = _coerce_float(item.get("approx_time_sec"))
        if not time_ok:
            warnings.append(_warning("bad_addition_time", f"bad approx_time_sec {item.get('approx_time_sec')!r}", f"{path}.approx_time_sec"))
            continue
        lo, hi = sorted((float(interval.start_time), float(interval.end_time)))
        if approx_time < lo or approx_time > hi:
            warnings.append(_warning("addition_time_outside_interval", f"addition time {approx_time} outside interval [{lo}, {hi}]", f"{path}.approx_time_sec"))
            continue
        normalized_addition = {
            "interval": canonical_interval,
            "interval_id": interval.interval_id,
            "display_interval": canonical_interval,
            "type": item.get("type"),
            "suggested_type": suggested_type,
            "approx_time_sec": approx_time,
            "description": item.get("description", ""),
            "source_addition": deepcopy(item),
        }
        normalized_additions.append(normalized_addition)
        interval_reviews.append(
            {
                "interval_id": interval.interval_id,
                "display_interval": canonical_interval,
                "action": "ADD",
                "suggested_type": suggested_type,
                "approx_time_sec": approx_time,
                "confidence": "MEDIUM",
                "reason": item.get("description", ""),
                "source_addition": deepcopy(item),
            }
        )
    return normalized_additions, interval_reviews


def _parse_display_interval(value: str) -> Tuple[Optional[str], Optional[str]]:
    parts = [part.strip() for part in value.split("_")]
    if len(parts) != 2:
        return None, None
    left, left_warning = _normalize_display_id(parts[0])
    right, right_warning = _normalize_display_id(parts[1])
    if left_warning or right_warning:
        return None, None
    return left, right


def _display_interval_from_internal(interval_id: str) -> str:
    parts = interval_id.split("_")
    if len(parts) != 2:
        return interval_id
    return f"{display_id_from_event_id(parts[0])}_{display_id_from_event_id(parts[1])}"


def _normalize_display_id(value: Any) -> Tuple[str, Optional[str]]:
    if value is None:
        return "", "missing keyframe id"
    if isinstance(value, int):
        return f"{value:03d}", None
    text = str(value).strip()
    if text.isdigit():
        return f"{int(text):03d}", None
    # Accept prefixed ids only to recover gracefully; still warn because the schema says no prefix.
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return f"{int(digits):03d}", f"keyframe id {text!r} should be a plain three-digit string"
    return "", f"keyframe id {text!r} is not numeric"


def _coerce_float(value: Any) -> Tuple[float, bool]:
    try:
        return float(value), True
    except (TypeError, ValueError):
        return 0.0, False


def _warning(code: str, message: str, path: str) -> Dict[str, str]:
    return {"code": code, "message": message, "path": path}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _normalize_event_result(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    action_original = item.get("action")
    action = EVENT_ACTION_ALIASES.get(action_original, action_original)
    operation = item.get("operation") if isinstance(item.get("operation"), dict) else {}
    keyframe_id = item.get("keyframe_id") or item.get("event_id")
    out: Dict[str, Any] = {
        "event_id": keyframe_id,
        "keyframe_id": keyframe_id,
        "reason": item.get("reason", ""),
        "action": action,
        "state_before": _normalize_state(item.get("state_before", "UNCERTAIN")),
        "state_after": _normalize_state(item.get("state_after", "UNCERTAIN")),
        "confidence": item.get("confidence", "MEDIUM"),
        "operation": item.get("operation"),
    }
    if action_original != action:
        out["action_original"] = action_original
    if action == "RELABEL":
        out["new_type"] = operation.get("new_type") or item.get("new_type")
    if action == "MERGE":
        merge_into = operation.get("merge_into") or item.get("merge_into")
        out["merge_into"] = merge_into
        out["merge_with"] = [merge_into] if merge_into else []
    if action == "RELOCALIZE":
        out["relocalize"] = operation or item.get("relocalize")
    return out


def _normalize_event_review(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    out = deepcopy(item)
    if "event_id" not in out and "keyframe_id" in out:
        out["event_id"] = out.get("keyframe_id")
    if "keyframe_id" not in out and "event_id" in out:
        out["keyframe_id"] = out.get("event_id")
    action = out.get("action")
    if action in EVENT_ACTION_ALIASES:
        out["action_original"] = action
        out["action"] = EVENT_ACTION_ALIASES[action]
    for key in ("state_before", "state_after"):
        out[key] = _normalize_state(out.get(key, "UNCERTAIN"))
    out.setdefault("confidence", "MEDIUM")
    operation = out.get("operation") if isinstance(out.get("operation"), dict) else {}
    if out.get("action") == "RELABEL" and "new_type" not in out:
        out["new_type"] = operation.get("new_type")
    if out.get("action") == "MERGE" and "merge_into" not in out:
        refs = out.get("merge_with") or []
        out["merge_into"] = refs[0] if refs else operation.get("merge_into")
    return out


def _normalize_add_result(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    action_original = item.get("action")
    action = INTERVAL_ACTION_ALIASES.get(action_original, action_original)
    operation = item.get("operation") if isinstance(item.get("operation"), dict) else {}
    out: Dict[str, Any] = {
        "interval_id": item.get("interval_id"),
        "reason": item.get("reason", ""),
        "action": action,
        "suggested_type": operation.get("type") or item.get("suggested_type", "UNKNOWN"),
        "approx_time_sec": operation.get("approx_time_sec", item.get("approx_time_sec")),
        "confidence": item.get("confidence", "MEDIUM"),
        "operation": item.get("operation"),
    }
    if action_original != action:
        out["action_original"] = action_original
    return out


def _normalize_interval_review(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    out = deepcopy(item)
    action = out.get("action")
    if action in INTERVAL_ACTION_ALIASES:
        out["action_original"] = action
        out["action"] = INTERVAL_ACTION_ALIASES[action]
    out.setdefault("confidence", "MEDIUM")
    operation = out.get("operation") if isinstance(out.get("operation"), dict) else {}
    if out.get("action") == "ADD":
        out.setdefault("suggested_type", operation.get("type", "UNKNOWN"))
        out.setdefault("approx_time_sec", operation.get("approx_time_sec"))
    return out


def _normalize_state(state: Any) -> Any:
    return STATE_ALIASES.get(state, state)
